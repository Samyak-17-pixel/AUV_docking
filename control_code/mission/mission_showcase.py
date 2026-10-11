#!/usr/bin/env python3
"""Fly the mission legs (lawnmower, orbit, spiral, yo-yo, go-to + hold, and the shipped chained mission) closed loop on the offline vehicle with REALISTIC odometry
(4.5 Hz, 0.25 s late, noisy, through the pose filter, like the mission node sees it), judge each run on the simulated ground truth, and SAVE the trajectories.

  python3 mission_showcase.py                        # all missions -> outputs/sim_viewer_runs/missions_showcase/
  python3 mission_showcase.py --only lawnmower,orbit --freezes 40
Per mission: trajectory.csv (replayable in the viewer: History tab or `run_sim_viewer.sh --replay ...`; all seven actuator commands), top_view.png (path, plan, geofence, dock
keep-out), timeline.png (cross-track, speed, depth, progress, fins), summary.json; plus index.html and README.txt. Passing runs are the 'good' ones; a failing run is saved too and
named FAILED_*. Everything is OFFLINE and on the ASSUMED vehicle model (README section 9): nothing here has run on the real mavsim.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_CTRL = _HERE.parent
sys.path[:0] = [str(_d) for _d in [_CTRL / "sim_viewer", *sorted((_CTRL / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # sim_viewer and its sub-folders
for p in (_HERE, _CTRL / "common", _CTRL / "sim_offline", _CTRL / "sim_viewer"):
    sys.path.insert(0, str(p))

from mission_core import MissionRunner, plan_polyline, validate_mission  # noqa: E402
from outdirs import out_dir  # noqa: E402
from pose_filter import PoseFilterConfig, PoseTracker  # noqa: E402
from sensors import OdometryConfig, OdometrySensor  # noqa: E402
from state import State  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

CFG = yaml.safe_load((_HERE / "mission.yaml").read_text())
DOCK = (10.0, 0.0)

MISSIONS: Dict[str, List[dict]] = {
    "lawnmower": [{"type": "lawnmower", "origin": [0.0, 5.0], "heading_deg": 90.0, "length_m": 10.0, "width_m": 16.0, "spacing_m": 8.0, "z": 3.0, "leadin_m": 8.0}, {"type": "return_home"}],
    "orbit": [{"type": "goto", "x": -16.0, "y": 12.0, "z": 3.0}, {"type": "orbit", "centre": [-20.0, 19.0], "radius_m": 4.0, "revs": 1.0, "clockwise": True, "start_deg": 0.0, "points_per_rev": 12, "z": 3.0},
              {"type": "return_home"}],
    "spiral": [{"type": "goto", "x": -10.0, "y": 8.0, "z": 3.0}, {"type": "spiral", "centre": [-14.0, 12.0], "r_start_m": 4.0, "r_end_m": 9.0, "pitch_m": 5.0, "start_deg": 0.0, "z": 3.0}, {"type": "return_home"}],
    "yoyo": [{"type": "goto", "x": -10.0, "y": 5.0, "z": 3.0}, {"type": "yoyo", "from": [-10.0, 5.0], "to": [-10.0, 22.0], "z_top": 2.0, "z_bottom": 4.0, "cycles": 2}, {"type": "return_home"}],
    "goto_hold": [{"type": "goto", "x": -6.0, "y": 8.0, "z": 3.0}, {"type": "hold", "seconds": 8.0}, {"type": "goto", "x": -6.0, "y": -4.0, "z": 3.5}, {"type": "return_home"}],
    "chained_shipped": copy.deepcopy(CFG["legs"]),
}


def fly(legs: List[dict], T: float = 420.0, dt: float = 0.05, seed: int = 3, freeze_s: float = 0.0, z0: float = 3.0):
    cfg = copy.deepcopy(CFG)
    cfg["legs"] = legs
    warn = validate_mission(cfg)
    mr = MissionRunner(cfg)
    m = VehicleModel(pos=(0.0, 0.0, z0), eul_deg=(0.0, 0.0, 0.0))
    sensor = OdometrySensor(OdometryConfig(rate_hz=4.5, latency_s=0.25, freeze_mean_interval_s=freeze_s, seed=seed))
    trk = PoseTracker(0.25, 1.0, PoseFilterConfig())
    rows: List[list] = []
    out: Dict[str, float] = {}
    t = 0.0
    while t < T and not mr.done:
        for _ in range(int(round(dt / 0.01))):
            m.step(0.01)
            for o in sensor.update(t, m.pos, m.eul, m.nu):
                trk.push(State(pos=np.array(o.pos), eul=np.array(o.eul), nu=np.array(o.nu)), t)
            t += 0.01
        st = trk.at(t)
        if st is None:
            continue
        if trk.frozen(t, expect_motion=True):
            out = mr.neutral()
            mr.report_odom_loss(dt)
            s = dict(mr.status or {})
        else:
            out, s = mr.update(st, dt)
        m.set_command(out)
        rows.append([t, *m.pos, *m.eul, *m.nu, out.get("th_01", 0.0), out.get("th_02", 0.0), out.get("th_03", 0.0), out.get("cs_04", 0.0), out.get("cs_06", 0.0), out.get("cs_07", 0.0),
                     out.get("cs_08", 0.0), float(s.get("segment", -1)), float(s.get("cross_track_m", 0.0) or 0.0), float(s.get("progress", 0.0) or 0.0), float(s.get("u_sp", 0.0) or 0.0)])
    return mr, np.array(rows), warn, cfg


COLS = ["t", "x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r", "th_01", "th_02", "th_03", "cs_04", "cs_06", "cs_07", "cs_08", "segment", "cross_track", "progress", "u_sp"]


def judge(name: str, mr: MissionRunner, L: np.ndarray, cfg: dict) -> Tuple[bool, List[str], Dict[str, float]]:
    fails: List[str] = []
    x, y, z = L[:, 1], L[:, 2], L[:, 3]
    geo = cfg["safety"].get("geofence", {})
    keep = float(cfg["safety"].get("dock_keepout_m", 0.0))
    d_dock = float(np.hypot(x - DOCK[0], y - DOCK[1]).min())
    if not mr.done:
        fails.append("did not finish in time")
    if mr.aborted:
        fails.append(f"aborted: {mr.aborted}")
    if keep and d_dock < keep:
        fails.append(f"came within {d_dock:.1f} m of the dock (keep-out {keep:g} m)")
    for k, v, lo in (("x", x, "x_min"), ("y", y, "y_min"), ("z", z, "z_min")):
        hi = lo.replace("min", "max")
        if lo in geo and v.min() < geo[lo] - 0.3:
            fails.append(f"left the geofence ({lo})")
        if hi in geo and v.max() > geo[hi] + 0.3:
            fails.append(f"left the geofence ({hi})")
    home = float(np.hypot(x[-1], y[-1]))
    if cfg["legs"][-1].get("type") == "return_home" and home > 1.5:
        fails.append(f"ended {home:.1f} m from home")
    # lane keeping, measured on the TRUE path: the middle-to-end part (40-95% of the way along) of every straight lane of a lawnmower leg (the arcs and run-in are not lanes)
    errs: List[float] = []
    seg_col = L[:, COLS.index("segment")].astype(int)
    for si, sg in enumerate(mr.segments):
        if sg.kind != "lawnmower" or len(sg.wps) < 2:
            continue
        P = L[seg_col == si][:, 1:3]
        if len(P) == 0:
            continue
        for w0, w1 in zip(sg.wps[:-1], sg.wps[1:]):                          # every long straight piece of the path is a lane (the short pieces are the U-turn arcs)
            A, B = np.array(w0[:2], float), np.array(w1[:2], float)
            n = float(np.linalg.norm(B - A))
            if n < 6.0:
                continue
            along = ((P - A) @ (B - A)) / (n * n)
            cross = ((P - A)[:, 0] * (B - A)[1] - (P - A)[:, 1] * (B - A)[0]) / n
            sel = (along >= 0.4) & (along <= 0.95) & (np.abs(cross) < 4.0)       # inside this lane's corridor (the next lane is 8 m away)
            errs += list(np.abs(cross[sel]))
    ct95 = float(np.percentile(errs, 95)) if errs else 0.0
    if errs and ct95 > 1.5:
        fails.append(f"lane error {ct95:.2f} m (95th percentile of the straight lane parts) above 1.5 m")
    met = {"time_s": float(L[-1, 0]), "min_dock_distance_m": d_dock, "end_from_home_m": home, "lane_error_p95_m": ct95, "depth_min_m": float(z.min()), "depth_max_m": float(z.max()),
           "progress": float(L[-1, COLS.index("progress")])}
    return (not fails), fails, met


def top_view(L, cfg, name, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    geo = cfg["safety"].get("geofence", {})
    if geo:
        ax.add_patch(Rectangle((geo["y_min"], geo["x_min"]), geo["y_max"] - geo["y_min"], geo["x_max"] - geo["x_min"], fill=False, ec="#c33", ls="--", lw=1, label="geofence"))
    if cfg["safety"].get("dock_keepout_m"):
        ax.add_patch(Circle((DOCK[1], DOCK[0]), float(cfg["safety"]["dock_keepout_m"]), fill=True, fc="#f3dede", ec="#c33", alpha=0.5, label="dock keep-out"))
    ax.plot([0], [DOCK[0]], "s", color="#7a5c3a", ms=9, label="dock (10, 0)")
    try:
        pl = np.array([(p[0], p[1]) for p in plan_polyline(cfg)])
        ax.plot(pl[:, 1], pl[:, 0], ":", color="#888", lw=1.4, label="plan")
    except Exception:                                                              # noqa: BLE001
        pass
    seg = L[:, COLS.index("segment")].astype(int)
    sc = ax.scatter(L[:, 2], L[:, 1], c=seg, s=4, cmap="viridis", zorder=4)
    ax.plot(L[0, 2], L[0, 1], "o", color="#1f4e9c", ms=8, zorder=6)
    ax.plot(L[-1, 2], L[-1, 1], "^", color="#146c43", ms=9, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(title, fontsize=9)
    fig.colorbar(sc, ax=ax, label="segment", shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def timeline(L, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = L[:, 0]
    fig, axs = plt.subplots(5, 1, figsize=(8.5, 9.5), sharex=True)
    axs[0].plot(t, L[:, COLS.index("cross_track")], color="#3f8cff")
    axs[0].set_ylabel("cross-track [m]")
    axs[1].plot(t, L[:, COLS.index("u")], color="#222", label="speed u")
    axs[1].plot(t, L[:, COLS.index("u_sp")], "--", color="#e0a526", label="setpoint")
    axs[1].set_ylabel("speed [m/s]")
    axs[1].legend(fontsize=7)
    axs[2].plot(t, L[:, 3], color="#222")
    axs[2].invert_yaxis()
    axs[2].set_ylabel("depth [m]")
    axs[3].plot(t, 100 * L[:, COLS.index("progress")], color="#2e9d5b")
    axs[3].set_ylabel("progress [%]")
    for c, col, lw in (("cs_04", "#222", 2.2), ("cs_06", "#e0a526", 1.5), ("cs_07", "#3f8cff", 1.1), ("cs_08", "#d63b3b", 0.8)):
        axs[4].plot(t, L[:, COLS.index(c)], label=c, color=col, lw=lw)
    axs[4].set_ylabel("fins [deg]")
    axs[4].set_xlabel("time [s]")
    axs[4].legend(fontsize=7, ncol=4)
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_csv(L, path, meta):
    from recording import CsvRecorder
    rec = CsvRecorder(path, meta)
    nxt = 0.0
    for r in L:
        if r[0] + 1e-9 < nxt:
            continue
        nxt += 0.1
        rec.set_cmd(["th_01", "th_02", "th_03", "cs_04", "cs_06", "cs_07", "cs_08"], [r[13], r[14], r[15], r[16], r[17], r[18], r[19]], r[0])
        rec.add_state(r[0], r[1:4], r[4:7], r[7:13])
    rec.close()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--freezes", type=float, default=0.0, help="mean seconds between odometry freezes (0 = none)")
    ap.add_argument("--out", type=Path, default=out_dir("sim_viewer_runs") / "missions_showcase")
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--T", type=float, default=420.0)
    a = ap.parse_args(argv)
    names = [n for n in MISSIONS if not a.only or n in a.only.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for n in names:
        mr, L, warn, cfg = fly(MISSIONS[n], T=a.T, freeze_s=a.freezes)
        ok, fails, met = judge(n, mr, L, cfg)
        tag = n if ok else "FAILED_" + n
        d = a.out / tag
        d.mkdir(exist_ok=True)
        title = f"{n}: {'PASS' if ok else 'FAILED'}  ({met['time_s']:.0f} s, closest to the dock {met['min_dock_distance_m']:.1f} m, ends {met['end_from_home_m']:.2f} m from home)"
        top_view(L, cfg, n, title, d / "top_view.png")
        timeline(L, title, d / "timeline.png")
        write_csv(L, d / "trajectory.csv", {"recorder": "mission/mission_showcase.py", "controller": "mission", "scenario": title})
        if not a.no_copy:
            shutil.copy(d / "trajectory.csv", out_dir("sim_viewer_runs") / f"mission_{tag}.csv")
        info = {"name": tag, "mission": n, "pass": ok, "failures": fails, "warnings": warn, **{k: round(v, 3) for k, v in met.items()}, "legs": [lg.get("type") for lg in cfg["legs"]]}
        (d / "summary.json").write_text(json.dumps(info, indent=2))
        entries.append(info)
        print(f"  {tag:28s} {'PASS' if ok else 'FAIL'}  {met['time_s']:6.1f} s  dock {met['min_dock_distance_m']:5.1f} m  home {met['end_from_home_m']:5.2f} m  lane p95 {met['lane_error_p95_m']:.2f} m  {'; '.join(fails)}")
    rows = "".join(f"<tr><td><a href='{e['name']}/top_view.png'><img src='{e['name']}/top_view.png' width='260'></a></td><td><b>{html.escape(e['mission'])}</b><br>legs: {html.escape(', '.join(e['legs']))}</td>"
                   f"<td>{'PASS' if e['pass'] else 'FAILED: ' + html.escape('; '.join(e['failures']))}<br><small>{e['time_s']} s, dock {e['min_dock_distance_m']} m, home {e['end_from_home_m']} m, lane p95 {e['lane_error_p95_m']} m</small></td>"
                   f"<td><a href='{e['name']}/timeline.png'>timeline</a><br><a href='{e['name']}/trajectory.csv'>trajectory.csv</a></td></tr>" for e in entries)
    (a.out / "index.html").write_text("<html><head><meta charset='utf-8'><title>Mission showcase</title><style>body{font-family:sans-serif;margin:20px} td{border-bottom:1px solid #ddd;padding:8px;vertical-align:top}</style></head>"
                                      f"<body><h2>Mission showcase (offline sim, realistic odometry)</h2><p>Synthetic: the vehicle model and the 2.5 m turning circle are assumptions.</p><table>{rows}</table></body></html>")
    (a.out / "README.txt").write_text("Mission showcase (offline simulation). index.html = gallery; <mission>/trajectory.csv replays in the viewer (History tab, or run_sim_viewer.sh --replay FILE). FAILED_* = a run that did not pass.\n")
    print(f"{sum(e['pass'] for e in entries)}/{len(entries)} passed -> {a.out}")
    return 0 if all(e["pass"] for e in entries) else 1


if __name__ == "__main__":
    sys.exit(main())
