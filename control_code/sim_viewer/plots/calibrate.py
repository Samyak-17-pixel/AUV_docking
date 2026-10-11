#!/usr/bin/env python3
"""Fit vehicle-model parameters to recorded runs (least squares) and say which ones the data can actually pin down.

  python3 plots/calibrate.py RUN.csv [RUN2.csv ...]
  python3 plots/calibrate.py RUN.csv --params rpm_to_rps,drag_quad_X,net_up_n --segment-s 4 --json fit.json

How it works: every run is cut into short SEGMENTS (default 4 s). The model starts each segment from the recorded pose and velocity, is driven by the
recorded actuator commands, and the predicted states are compared with the recorded ones (the same measure as `compare.py --mode segments`). The chosen
parameters are changed until the sum of squared, scaled errors is smallest (scipy least_squares, bounded). Short segments make the fit about the
dynamics (thrust, drag, buoyancy, lag) instead of about accumulated drift.

Read the "status" column before trusting a number:
  ok           the data moved this parameter's prediction enough to measure it (relative standard error below 30 %).
  weak         it has some effect but is entangled with another parameter or noisy (30-100 %): treat as a hint only.
  NOT IDENTIFIABLE  the run barely depends on it (e.g. fin lift slope in a run without speed + fin commands): the value is meaningless; record a run that excites it.
Nothing is written automatically. The tool prints the lines to change in common/mako_geometry.yaml; you decide.
This calibrates the OFFLINE MODEL against whatever produced the recording. On a real-sim recording, a bad fit (large residual left after fitting) means the model
STRUCTURE is wrong (e.g. the pitch pendulum seen on 2026-10-01), not just its numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE)] + [str(_d) for _d in sorted(HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(HERE.parent / "sim_offline"))
sys.path.insert(0, str(HERE.parent / "common"))

from allocation import load_geometry  # noqa: E402
from compare import CHANNELS, PARAMS, build_model, errors, simulate  # noqa: E402
from replay import Run, UnsupportedLog, load_run  # noqa: E402

DEFAULT_PARAMS = ["rpm_to_rps", "thruster_tau_s", "drag_quad_X", "drag_quad_Z", "drag_quad_M", "net_up_n", "added_mass_frac_w"]
# Residual scale per channel (the error that counts as "1"): roughly the measurement noise / the size that matters.
SCALE = {"x": 0.10, "y": 0.10, "depth": 0.05, "roll": 2.0, "pitch": 1.0, "yaw": 2.0, "u": 0.03, "v": 0.03, "w": 0.03, "p": 3.0, "q": 2.0, "r": 2.0}
DEFAULT_CHANNELS = ["depth", "pitch", "yaw", "u", "w", "q", "r"]
NET_UP_SCALE_N = 1.0                      # net buoyancy is fitted as nominal + theta * 1 N (its nominal is 0, so a multiplier is meaningless)
SUGGEST = {
    "rpm_to_rps": "thrusters: rpm_to_rps: {v:.6g}",
    "thruster_tau_s": "thruster lag: sim_offline/vehicle_model.py thruster_tau_s (and fake_vehicle) = {v:.4g}",
    "drag_quad_X": "vehicle: drag_quad: X: {v:.6g}", "drag_quad_Y": "vehicle: drag_quad: Y: {v:.6g}", "drag_quad_Z": "vehicle: drag_quad: Z: {v:.6g}",
    "drag_quad_M": "vehicle: drag_quad: M: {v:.6g}", "drag_quad_N": "vehicle: drag_quad: N: {v:.6g}", "drag_quad_K": "vehicle: drag_quad: K: {v:.6g}",
    "net_up_n": "vehicle: buoyancy_mass_kg: {bm:.5f}   (= mass + net_up/g; net_up = {v:+.3f} N, + floats)",
    "cl_alpha_per_rad": "fins: cl_alpha_per_rad: {v:.6g}",
    "added_mass_frac_w": "vehicle: added_mass_frac: [_, _, {v:.4g}, _, _, _]  (3rd entry, heave)",
    "added_mass_frac_u": "vehicle: added_mass_frac: [{v:.4g}, _, _, _, _, _]  (1st entry, surge)",
    "added_mass_frac_q": "vehicle: added_mass_frac: [_, _, _, _, {v:.4g}, _]  (5th entry, pitch)",
}
BOUNDS_MULT = (0.05, 20.0)
BOUNDS_ADD = (-15.0, 15.0)


def nominal_values(geom: Optional[dict] = None) -> Dict[str, float]:
    g = geom or load_geometry()
    v = g["vehicle"]
    am = v["added_mass_frac"]
    nom = {"rpm_to_rps": float(g["thrusters"]["rpm_to_rps"]), "thruster_tau_s": 0.2, "net_up_n": (float(v["buoyancy_mass_kg"]) - float(v["mass_kg"])) * float(v["gravity"]),
           "cl_alpha_per_rad": float(g["fins"]["cl_alpha_per_rad"]), "added_mass_frac_u": float(am[0]), "added_mass_frac_w": float(am[2]), "added_mass_frac_q": float(am[4])}
    for k, val in v["drag_quad"].items():
        nom["drag_quad_" + k] = float(val)
    return nom


def to_values(theta: np.ndarray, names: Sequence[str], nom: Dict[str, float]) -> Dict[str, float]:
    return {n: (nom[n] + t * NET_UP_SCALE_N if n == "net_up_n" else nom[n] * t) for n, t in zip(names, theta)}


def residuals(theta: np.ndarray, names: Sequence[str], runs: Sequence[Run], nom: Dict[str, float], segment_s: float, channels: Sequence[str], dt: float) -> np.ndarray:
    ov = to_values(theta, names, nom)
    out: List[np.ndarray] = []
    try:
        model = build_model(ov)
    except Exception:                                             # noqa: BLE001  (a parameter set the model rejects)
        return np.full(sum(r.n for r in runs) * len(channels), 1e3)
    for run in runs:
        pred = simulate(run, mode="segments", segment_s=segment_s, dt=dt, model=model)
        e = errors(run, pred)
        for c in channels:
            r = e[c] / SCALE[c]
            out.append(np.clip(np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3), -1e3, 1e3))
    return np.concatenate(out)


def calibrate(runs: Sequence[Run], params: Sequence[str] = DEFAULT_PARAMS, segment_s: float = 4.0, channels: Sequence[str] = DEFAULT_CHANNELS,
              dt: float = 0.02, max_nfev: int = 60, start: Optional[Dict[str, float]] = None) -> dict:
    from scipy.optimize import least_squares

    bad = [p for p in params if p not in PARAMS]
    if bad:
        raise KeyError(f"unknown parameter(s) {bad}; choose from {list(PARAMS)}")
    nom = nominal_values()
    names = list(params)
    th0 = np.array([0.0 if n == "net_up_n" else 1.0 for n in names])
    if start:
        th0 = np.array([(start[n] - nom[n]) / NET_UP_SCALE_N if n == "net_up_n" else start[n] / nom[n] for n in names])
    lo = np.array([BOUNDS_ADD[0] if n == "net_up_n" else BOUNDS_MULT[0] for n in names])
    hi = np.array([BOUNDS_ADD[1] if n == "net_up_n" else BOUNDS_MULT[1] for n in names])
    th0 = np.clip(th0, lo + 1e-6, hi - 1e-6)
    f = lambda th: residuals(th, names, runs, nom, segment_s, channels, dt)           # noqa: E731
    r0 = f(np.array([0.0 if n == "net_up_n" else 1.0 for n in names]))
    sol = least_squares(f, th0, bounds=(lo, hi), x_scale=np.where(np.array(names) == "net_up_n", 1.0, 0.3), diff_step=1e-3, max_nfev=max_nfev, method="trf")
    J, r = sol.jac, sol.fun
    n, k = len(r), len(names)
    s2 = float(r @ r) / max(n - k, 1)
    # parameter units: d value / d theta  (multiplier -> nominal, additive -> 1 N)
    unit = np.array([NET_UP_SCALE_N if p == "net_up_n" else nom[p] for p in names])
    colnorm = np.linalg.norm(J, axis=0)
    try:
        cov = s2 * np.linalg.inv(J.T @ J + 1e-12 * np.eye(k))
        se_theta = np.sqrt(np.clip(np.diag(cov), 0, None))
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        se_theta = np.full(k, np.inf)
        corr = np.ones((k, k))
    np.fill_diagonal(corr, 0.0)
    vals = to_values(sol.x, names, nom)
    rows = []
    for i, p in enumerate(names):
        se = se_theta[i] * unit[i]
        ref = max(abs(vals[p]), 1e-12) if p != "net_up_n" else 1.0                      # net buoyancy: standard error compared with 1 N
        rel = se / ref
        weak_effect = colnorm[i] < 1e-3 * max(colnorm.max(), 1e-12) or colnorm[i] < 0.5
        status = "NOT IDENTIFIABLE" if (weak_effect or rel > 1.0 or not np.isfinite(rel)) else ("weak" if rel > 0.3 else "ok")
        j = int(np.argmax(np.abs(corr[i]))) if k > 1 else 0
        cmax = float(abs(corr[i, j])) if k > 1 else 0.0
        if status == "ok" and cmax > 0.9:
            status = "weak"                                                          # entangled with another parameter: the pair is not separable
        rows.append({"name": p, "entangled_with": names[j] if k > 1 and cmax > 0.9 else "", "max_corr": cmax, "nominal": nom[p], "fitted": vals[p], "se": float(se), "rel_se": float(rel), "sensitivity": float(colnorm[i]), "status": status})
    sing = np.linalg.svd(J, compute_uv=False)
    return {"params": rows, "rms_before": float(np.sqrt(np.mean(r0 ** 2))), "rms_after": float(np.sqrt(np.mean(r ** 2))), "n_residuals": n,
            "cond": float(sing[0] / max(sing[-1], 1e-30)), "nfev": int(sol.nfev), "success": bool(sol.success), "segment_s": segment_s, "channels": list(channels),
            "note": "rms is in units of the channel scales (1.0 = errors of about the noise floor); see SCALE in calibrate.py"}


def format_report(res: dict) -> str:
    L = [f"{'parameter':20s} {'nominal':>12s} {'fitted':>12s} {'+/- SE':>11s} {'rel SE':>8s}  status", "-" * 82]
    for r in res["params"]:
        L.append(f"{r['name']:20s} {r['nominal']:12.5g} {r['fitted']:12.5g} {r['se']:11.3g} {r['rel_se'] * 100:7.1f}%  {r['status']}" + (f" (entangled with {r['entangled_with']}, |corr| {r['max_corr']:.2f})" if r["entangled_with"] else ""))
    L += ["", f"scaled residual RMS: {res['rms_before']:.3f} (nominal model) -> {res['rms_after']:.3f} (fitted); {res['n_residuals']} residuals, {res['nfev']} evaluations, "
          f"condition number {res['cond']:.3g}{'' if res['success'] else '  (optimiser did not converge)'}"]
    if res["rms_after"] > 3.0:
        L.append("The fit is still poor: the model STRUCTURE probably disagrees with the recording (something not in the model), not just its parameters.")
    ok = [r for r in res["params"] if r["status"] == "ok"]
    if ok:
        L += ["", "Suggested edits (YAML in common/mako_geometry.yaml unless noted) - only the parameters marked 'ok':"]
        g = load_geometry()
        for r in ok:
            bm = float(g["vehicle"]["mass_kg"]) + r["fitted"] / float(g["vehicle"]["gravity"])
            L.append("  " + SUGGEST[r["name"]].format(v=r["fitted"], bm=bm))
    L.append("The standard errors only include the noise left in the residuals, not model-structure error or noise in the segment start states: treat them as optimistic.")
    L.append("\nNothing was written. Re-run compare.py with --set NAME=VALUE to check a change before editing the file.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--params", default=",".join(DEFAULT_PARAMS), help="comma list; available: " + ", ".join(PARAMS))
    ap.add_argument("--segment-s", type=float, default=4.0)
    ap.add_argument("--channels", default=",".join(DEFAULT_CHANNELS), help="comma list of: " + ", ".join(c[0] for c in CHANNELS))
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--json", type=Path, help="also save the result as JSON")
    a = ap.parse_args(argv)
    try:
        runs = [load_run(p) for p in a.runs]
        for r in runs:
            if not r.cmd:
                raise UnsupportedLog(f"{Path(r.path).name}: no actuator commands in this log")
        res = calibrate(runs, [p for p in a.params.split(",") if p], a.segment_s, [c for c in a.channels.split(",") if c], a.dt, a.max_nfev)
    except (UnsupportedLog, KeyError) as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 2
    print(f"{len(runs)} run(s), {sum(r.duration for r in runs):.0f} s in total, {a.segment_s:g} s segments")
    print(format_report(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
