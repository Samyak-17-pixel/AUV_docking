#!/usr/bin/env python3
"""Per-run summary of a recorded run against its plan: numbers and one figure (no ROS, no Qt).

  python3 plots/run_summary.py RUN.csv [--mission mission.yaml | --waypoints waypoint_tracking.yaml] [--swath 3.0] [--out summary.png]

RUN.csv is a sim_viewer recording (outputs/sim_viewer_runs/auto_*.csv, record_run.py, or a dof_testing / station_keeping CSV). The plan is expanded from the mission YAML
(or the waypoint list); without a plan only speed, depth and actuator numbers are given. Numbers: path error to the planned polyline (RMS, 95 %, max), depth error against the
planned depth, mean and top speed, path length, swath coverage, actuator use. The figure shows the path coloured by its error, the error and speed over time, depth vs plan, and the
actuators. The viewer saves this automatically next to a finished mission recording.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE)] + [str(_d) for _d in sorted(HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders stay flat-importable
for _p in (HERE, HERE.parent / "common", HERE.parent / "mission", HERE.parent / "waypoint_tracking"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import coverage as cov  # noqa: E402
from replay import Run, load_run  # noqa: E402


def dist_to_polyline(pts: np.ndarray, poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(distance, index of the nearest segment's start) of every row of pts (N,2) to the polyline poly (M,2)."""
    if len(poly) == 1:
        return np.hypot(*(pts - poly[0]).T), np.zeros(len(pts), int)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    best = np.full(len(pts), np.inf)
    arg = np.zeros(len(pts), int)
    for k in range(len(a)):
        t = np.clip(((pts - a[k]) @ ab[k]) / L2[k], 0.0, 1.0)
        d = np.hypot(*(pts - (a[k] + t[:, None] * ab[k])).T)
        m = d < best
        best[m], arg[m] = d[m], k
    return best, arg


def plan_from_yaml(path: Path, mission: bool, start_xy: Tuple[float, float]) -> List[Tuple[float, float, float, str]]:
    cfg = yaml.safe_load(Path(path).read_text())
    if mission:
        from mission_core import plan_polyline
        return plan_polyline(cfg, start_xy)
    from waypoint_tracking_core import parse_waypoints
    return [(w[0], w[1], w[2], f"wp{i}") for i, w in enumerate(parse_waypoints(cfg["waypoints"], float(cfg.get("mission", {}).get("hold_depth_m", 5.0))))]


def summarize(run: Run, plan: Optional[Sequence[Tuple[float, float, float, str]]] = None, swath_m: float = 0.0) -> Dict[str, float]:
    t = run.t - run.t[0]
    xy = run.pos[:, :2]
    out: Dict[str, float] = {"duration_s": float(t[-1]) if len(t) else 0.0}
    out["path_length_m"] = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
    sp = np.hypot(*(np.diff(xy, axis=0) / np.maximum(np.diff(t), 1e-6)[:, None]).T) if len(xy) > 1 else np.zeros(1)
    out["mean_speed_mps"], out["max_speed_mps"] = float(sp.mean()), float(sp.max())
    for name in ("th_01", "th_02", "th_03"):
        if name in run.cmd:
            out[f"mean_abs_{name}_rpm"] = float(np.abs(run.cmd[name]).mean())
    fins = [np.abs(v) for k, v in run.cmd.items() if k.startswith("cs_")]
    if fins:
        out["mean_abs_fin_deg"] = float(np.mean([f.mean() for f in fins]))
    if plan and len(plan) >= 2:
        poly = np.array([(p[0], p[1]) for p in plan])
        d, seg = dist_to_polyline(xy, poly)
        out["path_error_rms_m"], out["path_error_p95_m"], out["path_error_max_m"] = float(np.sqrt((d ** 2).mean())), float(np.percentile(d, 95)), float(d.max())
        z_plan = np.array([p[2] for p in plan])
        dz = run.pos[:, 2] - np.interp(seg + 0.5, np.arange(len(z_plan)), z_plan)
        out["depth_error_rms_m"] = float(np.sqrt((dz ** 2).mean()))
        if swath_m > 0:
            out["coverage_pct"] = 100.0 * cov.coverage_fraction([(p[0], p[1]) for p in plan], [tuple(q) for q in xy[:: max(1, len(xy) // 800)]], swath_m, 0.5)
    return out


NAMES = {"duration_s": "duration [s]", "path_length_m": "path length [m]", "mean_speed_mps": "mean speed [m/s]", "max_speed_mps": "top speed [m/s]",
             "path_error_rms_m": "path error RMS [m]", "path_error_p95_m": "path error 95% [m]", "path_error_max_m": "path error max [m]",
             "depth_error_rms_m": "depth error RMS [m]", "coverage_pct": "swath coverage [%]", "mean_abs_fin_deg": "mean |fin| [deg]"}


def format_summary(s: Dict[str, float]) -> str:
    return "\n".join(f"{NAMES.get(k, k):<22} {v:8.2f}" for k, v in s.items())


def save_figure(run: Run, plan: Optional[Sequence[Tuple[float, float, float, str]]], path: Path, swath_m: float = 0.0, title: str = "") -> Dict[str, float]:
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = summarize(run, plan, swath_m)
    t = run.t - run.t[0]
    fig = Figure(figsize=(12, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots(2, 3).ravel()
    xy = run.pos[:, :2]
    if plan and len(plan) >= 2:
        poly = np.array([(p[0], p[1]) for p in plan])
        d, seg = dist_to_polyline(xy, poly)
        ax[0].plot(poly[:, 1], poly[:, 0], color="#7f7f7f", ls="--", lw=1.2, label="plan")
        sc = ax[0].scatter(xy[:, 1], xy[:, 0], c=d, s=6, cmap="viridis", label="actual")
        fig.colorbar(sc, ax=ax[0], label="distance to plan [m]", shrink=0.8)
        ax[1].plot(t, d, color="#d62728")
        ax[1].set_ylabel("distance to the planned path [m]")
        z_plan = np.array([p[2] for p in plan])
        ax[3].plot(t, np.interp(seg + 0.5, np.arange(len(z_plan)), z_plan), color="#7f7f7f", ls="--", label="planned depth")
    else:
        ax[0].plot(xy[:, 1], xy[:, 0], color="#2ca02c")
        ax[1].text(0.5, 0.5, "no plan given", ha="center", transform=ax[1].transAxes)
    ax[0].plot([0.0], [10.0], "s", color="#444444", label="dock")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set_xlabel("East y [m]")
    ax[0].set_ylabel("North x [m]")
    ax[0].set_title("Path vs plan")
    ax[0].legend(fontsize=7)
    ax[1].set_title("Path error")
    sp = np.hypot(*(np.diff(xy, axis=0) / np.maximum(np.diff(t), 1e-6)[:, None]).T) if len(xy) > 1 else np.zeros(1)
    ax[2].plot(t[1:], sp, color="#1f77b4")
    ax[2].set_title("Speed over ground [m/s]")
    ax[3].plot(t, run.pos[:, 2], color="#2ca02c", label="depth")
    ax[3].invert_yaxis()
    ax[3].set_title("Depth [m]")
    ax[3].legend(fontsize=7)
    for name in ("th_01", "th_02", "th_03"):
        if name in run.cmd:
            ax[4].plot(t, run.cmd[name], lw=1, label=name)
    ax[4].set_title("Thrusters [RPM]")
    ax[4].legend(fontsize=7)
    for name, v in run.cmd.items():
        if name.startswith("cs_"):
            ax[5].plot(t, v, lw=1, label=name)
    ax[5].set_title("Fins [deg]")
    ax[5].legend(fontsize=7)
    for a in ax[1:]:
        a.set_xlabel("time [s]")
        a.grid(True, alpha=0.3)
    keys = [k for k in ("duration_s", "path_length_m", "mean_speed_mps", "path_error_rms_m", "path_error_max_m", "depth_error_rms_m", "coverage_pct") if k in s]
    fig.suptitle((title + "   " if title else "") + "   ".join(f"{NAMES.get(k, k)} {s[k]:.2f}" for k in keys), fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=110)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--mission", type=Path, default=None)
    ap.add_argument("--waypoints", type=Path, default=None)
    ap.add_argument("--swath", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    run = load_run(a.run)
    start = (float(run.pos[0, 0]), float(run.pos[0, 1]))
    plan = plan_from_yaml(a.mission, True, start) if a.mission else plan_from_yaml(a.waypoints, False, start) if a.waypoints else None
    out = a.out or a.run.with_name(a.run.stem + "_summary.png")
    s = save_figure(run, plan, out, a.swath, a.run.name)
    print(format_summary(s))
    print(f"figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
