#!/usr/bin/env python3
"""Real-vs-model comparison: replay a recorded run's ACTUATOR COMMANDS through the offline VehicleModel and compare the states.

  python3 plots/compare.py RUN.csv                      # free run from the first sample
  python3 plots/compare.py RUN.csv --mode segments --segment-s 5     # re-start the model from the recorded state every 5 s
  python3 plots/compare.py RUN.csv --out model.csv --plot overlay.png
  python3 run_sim_viewer.sh --replay RUN.csv --overlay           # same thing, ghost vehicle in the 3D view

How to read it: a FREE run shows how far the model drifts from reality over the whole run (errors accumulate, so it is the harsh test). SEGMENTS
restarts the model from the recorded pose and velocity every N seconds, so each segment only tests the short-term dynamics (thrust, drag, lag).
A small RMS here means the numbers in common/mako_geometry.yaml describe the vehicle; a big one points to the wrong assumption (see calibrate.py).
The recorded actuator command is held until the next row (zero-order hold); the real sim's command delay is unknown and not modelled.
Pure Python (numpy + the offline model): no ROS, no Qt.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE)] + [str(_d) for _d in sorted(HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(HERE.parent / "sim_offline"))
sys.path.insert(0, str(HERE.parent / "common"))

from allocation import load_geometry  # noqa: E402
from replay import Run, UnsupportedLog, load_run  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

# parameter name -> how it is applied to the geometry dict / model. These are the numbers calibrate.py can fit.
PARAMS = {
    "rpm_to_rps": "thrust per RPM^2 scale (thrusters.rpm_to_rps)",
    "thruster_tau_s": "thruster first-order lag [s]",
    "drag_quad_X": "quadratic surge drag", "drag_quad_Y": "quadratic sway drag", "drag_quad_Z": "quadratic heave drag",
    "drag_quad_M": "quadratic pitch drag", "drag_quad_N": "quadratic yaw drag", "drag_quad_K": "quadratic roll drag",
    "net_up_n": "net buoyancy [N, + floats]",
    "cl_alpha_per_rad": "fin lift slope",
    "added_mass_frac_w": "heave added-mass fraction", "added_mass_frac_u": "surge added-mass fraction", "added_mass_frac_q": "pitch added-mass fraction",
}


def build_model(overrides: Optional[Dict[str, float]] = None, geom: Optional[dict] = None) -> VehicleModel:
    """VehicleModel with some of the PARAMS replaced. The geometry file itself is never modified."""
    g = copy.deepcopy(geom or load_geometry())
    o = dict(overrides or {})
    tau = float(o.pop("thruster_tau_s", 0.2))
    net_up = o.pop("net_up_n", None)
    if "rpm_to_rps" in o:
        g["thrusters"]["rpm_to_rps"] = float(o.pop("rpm_to_rps"))
    if "cl_alpha_per_rad" in o:
        g["fins"]["cl_alpha_per_rad"] = float(o.pop("cl_alpha_per_rad"))
    idx = {"u": 0, "w": 2, "q": 4}
    for k in list(o):
        if k.startswith("drag_quad_"):
            g["vehicle"]["drag_quad"][k[-1]] = float(o.pop(k))
        elif k.startswith("added_mass_frac_"):
            g["vehicle"]["added_mass_frac"][idx[k[-1]]] = float(o.pop(k))
    if o:
        raise KeyError(f"unknown parameter(s): {sorted(o)}")
    if net_up is not None:                                             # buoyancy mass chosen so that net = net_up
        v = g["vehicle"]
        v["buoyancy_mass_kg"] = float(v["mass_kg"]) + float(net_up) / float(v["gravity"])
    return VehicleModel(g, thruster_tau_s=tau)


def _init_from(model: VehicleModel, run: Run, i: int, warm_s: float = 1.5) -> None:
    """Start the model at recorded sample i. The actuator state is not recorded, so it is rebuilt by feeding the commands of the previous `warm_s` seconds
    through the same lag the model uses (assuming the actuators were idle `warm_s` earlier would be wrong after a long command; steady state is the best guess there)."""
    model.reset(run.pos[i], np.degrees(run.eul[i]))
    model.nu = run.nu[i].copy()
    j = run.index_at(run.t[i] - warm_s)
    cmd0 = {k: float(v[j]) for k, v in run.cmd.items()}
    for k in model.rpm:
        model.rpm[k] = float(cmd0.get(k, 0.0))
    for k in model.fin_deg:
        model.fin_deg[k] = float(cmd0.get(k, 0.0))
    for r in range(j + 1, i + 1):
        dt = float(run.t[r] - run.t[r - 1])
        a = dt / (model.tau_th + dt)
        for k in model.rpm:
            model.rpm[k] += a * (float(run.cmd[k][r - 1]) - model.rpm[k]) if k in run.cmd else 0.0
        for k in model.fin_deg:
            if k in run.cmd:
                step = math.degrees(model.fin_rate) * dt
                model.fin_deg[k] += float(np.clip(float(run.cmd[k][r - 1]) - model.fin_deg[k], -step, step))
    model.set_command({k: float(v[i]) for k, v in run.cmd.items()})


def simulate(run: Run, overrides: Optional[Dict[str, float]] = None, mode: str = "free", segment_s: float = 5.0, dt: float = 0.01,
             geom: Optional[dict] = None, model: Optional[VehicleModel] = None) -> Run:
    """Model prediction on the SAME time grid as `run` (a Run with the model's pos / eul / nu and the commands that drove it)."""
    if not run.cmd:
        raise UnsupportedLog(f"{Path(run.path).name}: has no actuator commands, so there is nothing to replay through the model")
    m = model or build_model(overrides, geom)
    n = run.n
    pos, eul, nu = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 6))
    seg_start_t = run.t[0]
    _init_from(m, run, 0)
    pos[0], eul[0], nu[0] = run.pos[0], run.eul[0], run.nu[0]
    for i in range(1, n):
        if mode == "segments" and run.t[i - 1] - seg_start_t >= segment_s - 1e-9:
            _init_from(m, run, i - 1)
            seg_start_t = run.t[i - 1]
        m.set_command({k: float(v[i - 1]) for k, v in run.cmd.items()})
        span = run.t[i] - run.t[i - 1]
        k = max(1, int(math.ceil(span / dt)))
        h = span / k
        for _ in range(k):
            m.step(h)
            if not np.isfinite(m.nu).all() or np.abs(m.nu).max() > 1e3:          # model blew up: freeze, the RMS will say so
                break
        pos[i], eul[i], nu[i] = m.pos, m.eul, m.nu
    eul[:, 2] = np.unwrap(eul[:, 2])
    return Run(run.t.copy(), pos, eul, nu, {k: v.copy() for k, v in run.cmd.items()}, None, {}, {"model": "VehicleModel", "mode": mode, "overrides": str(overrides or {})},
               "model", run.path)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


CHANNELS = [("x", "m", lambda r: r.pos[:, 0]), ("y", "m", lambda r: r.pos[:, 1]), ("depth", "m", lambda r: r.pos[:, 2]),
            ("roll", "deg", lambda r: np.degrees(r.eul[:, 0])), ("pitch", "deg", lambda r: np.degrees(r.eul[:, 1])), ("yaw", "deg", lambda r: np.degrees(r.eul[:, 2])),
            ("u", "m/s", lambda r: r.nu[:, 0]), ("v", "m/s", lambda r: r.nu[:, 1]), ("w", "m/s", lambda r: r.nu[:, 2]),
            ("p", "deg/s", lambda r: np.degrees(r.nu[:, 3])), ("q", "deg/s", lambda r: np.degrees(r.nu[:, 4])), ("r", "deg/s", lambda r: np.degrees(r.nu[:, 5]))]


def errors(real: Run, model: Run) -> Dict[str, np.ndarray]:
    out = {}
    for name, _, f in CHANNELS:
        e = f(model) - f(real)
        if name in ("roll", "pitch", "yaw"):
            e = np.degrees(_wrap(np.radians(e)))
        out[name] = e
    return out


def rms_table(real: Run, model: Run) -> List[dict]:
    rows = []
    e = errors(real, model)
    for name, unit, f in CHANNELS:
        s = f(real)
        rows.append({"channel": name, "unit": unit, "rms": float(np.sqrt(np.nanmean(e[name] ** 2))), "bias": float(np.nanmean(e[name])),
                     "final": float(e[name][-1]), "real_span": float(np.nanmax(s) - np.nanmin(s))})
    return rows


def format_table(rows: List[dict]) -> str:
    lines = [f"{'channel':8s} {'unit':6s} {'RMS err':>10s} {'bias':>10s} {'final err':>10s} {'real range':>11s}  RMS / range"]
    for r in rows:
        frac = f"{r['rms'] / r['real_span'] * 100:6.1f} %" if r["real_span"] > 1e-9 else "   (did not move)"
        lines.append(f"{r['channel']:8s} {r['unit']:6s} {r['rms']:10.4f} {r['bias']:10.4f} {r['final']:10.4f} {r['real_span']:11.4f}  {frac}")
    return "\n".join(lines)


def write_model_csv(model: Run, path: Path) -> None:
    from recording import CsvRecorder
    rec = CsvRecorder(path, {"recorder": "compare.py (MODEL prediction, not a real run)", "source": model.path, **model.meta})
    for i in range(model.n):
        rec.set_cmd(list(model.cmd), [model.cmd[k][i] for k in model.cmd], float(model.t[i]))
        rec.add_state(float(model.t[i]), model.pos[i], model.eul[i], model.nu[i])
    rec.close()


def plot_overlay(real: Run, model: Run, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chans = [c for c in CHANNELS if c[0] in ("x", "depth", "pitch", "yaw", "u", "w", "q", "r")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for ax, (name, unit, f) in zip(axes.ravel(), chans):
        ax.plot(real.t, f(real), lw=1.3, label="recorded")
        ax.plot(model.t, f(model), lw=1.1, ls="--", label="model")
        ax.set_title(f"{name} [{unit}]", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--mode", choices=["free", "segments"], default="free")
    ap.add_argument("--segment-s", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a model parameter, e.g. --set rpm_to_rps=0.0333 (names: %s)" % ", ".join(PARAMS))
    ap.add_argument("--out", type=Path, help="write the model prediction as a recorder CSV")
    ap.add_argument("--plot", type=Path, help="write an overlay PNG")
    a = ap.parse_args(argv)
    try:
        real = load_run(a.run)
        ov = {k: float(v) for k, v in (s.split("=", 1) for s in a.set)}
        model = simulate(real, ov, a.mode, a.segment_s, a.dt)
    except (UnsupportedLog, KeyError, ValueError) as exc:
        print(f"compare: {exc}", file=sys.stderr)
        return 2
    print(f"{a.run.name}: {real.fmt}, {real.n} rows, {real.duration:.1f} s, mode={a.mode}" + (f" ({a.segment_s:g} s segments)" if a.mode == "segments" else ""))
    print(format_table(rms_table(real, model)))
    if a.out:
        write_model_csv(model, a.out)
        print("wrote", a.out)
    if a.plot:
        plot_overlay(real, model, a.plot)
        print("wrote", a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
