#!/usr/bin/env python3
"""Save the docking runs that went well (and a few that did not) as trajectories you can look at afterwards.

  python3 showcase.py --from-csv outputs/docking_sweeps/full_hint/results.csv --hint near --out outputs/sim_viewer_runs/docking_showcase
  python3 showcase.py --starts "7,0,0;4,1.5,30;3,0,45" --hint near          # chosen starts: range,lateral,heading separated by ';'

For every chosen start it re-runs the closed loop (same seeds, so the same result as in the sweep) with the real detector and controller and writes, in <out>/<name>/:
  trajectory.csv   replayable in the viewer (History tab: open it, or ./run_sim_viewer.sh then Replay); the full state, commands and detector values
  top_view.png     the path seen from above (north up) coloured by controller phase, the dock and funnel, start and end poses, the gates
  timeline.png     range, cross-track, heading error, speed and thrust against time, with the phase bands
  frames.png       four camera pictures with the detector pop-up drawing at key moments (first lock, mid-approach, close, last lock)
  summary.json     start pose, outcome, closest approach, time, retries, repositions
and <out>/index.html (a gallery), <out>/README.txt (what the folders are) and copies of the CSVs as outputs/sim_viewer_runs/showcase_<tag>_<name>.csv for the History tab.
Everything is SYNTHETIC: the offline vehicle model and camera look are assumptions (README section 9), not the real mavsim.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "common"))
from outdirs import out_dir  # noqa: E402
sys.path[:0] = [str(_d) for _d in [_HERE.parent / "sim_viewer", *sorted((_HERE.parent / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
sys.path.insert(0, str(_HERE.parents[1] / "dock_detection_algo"))
from docking_sim import DOCK_POS, DockingSim, Scenario, load_cfg  # noqa: E402
from docking_sweep import classify, view_class  # noqa: E402

PHASES = ["WAIT", "SEARCH", "APPROACH", "TERMINAL", "DOCKED", "RETRY", "SAFE_STOP", "ABORT"]
PHASE_COL = {"WAIT": "#9aa0a6", "SEARCH": "#3f8cff", "APPROACH": "#2e9d5b", "TERMINAL": "#146c43", "DOCKED": "#d63b3b", "RETRY": "#f0a020", "SAFE_STOP": "#8e44ad", "ABORT": "#000000"}


class FrameKeeper:
    """Keeps every detector frame (small) so the key moments can be drawn afterwards."""

    def __init__(self) -> None:
        self.items: List[Tuple[float, np.ndarray, object, float]] = []

    def __call__(self, t, bgr, res, roll) -> None:
        self.items.append((float(t), bgr.copy(), res, float(roll)))


def run_one(start: Tuple[float, float, float], overrides: dict, plant: str = "nominal", T: float = 150.0, seed: int = 1, stress: Optional[dict] = None):
    r, e, h = start
    scn = Scenario(range_m=r, lateral_m=e, heading_deg=h, plant=plant, vision="detector", seed=seed, T=T)
    sim = DockingSim(scn, load_cfg(overrides=overrides))
    if stress:
        sim.cam.stress = dict(stress)
    keeper = FrameKeeper()
    sim.frame_cb = keeper
    res = sim.run(keep_log=True)
    ok_after = (not res.failures) and sim.core.phase == "DOCKED"
    outcome = "RECOVERED" if ok_after and not res.passed else classify(res.passed, res.failures, res.metrics)
    return scn, sim, res, outcome, keeper


# ------------------------------------------------------------------------------------------------------------------------------- figures
def top_view(log: np.ndarray, scn: Scenario, outcome: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    x, y = log[:, 1], log[:, 2]
    ph = log[:, 16].astype(int)
    lo_x, hi_x = min(x.min() - 1.5, 6.0), max(x.max() + 1.0, 13.0)
    lo_y, hi_y = min(y.min() - 1.5, -3.0), max(y.max() + 1.5, 3.0)
    # funnel (inner radius 0.75 m at the mouth, 0.15 m at depth 1.25 m; drawn as a trapezoid) and the ring
    d0 = DOCK_POS[0]
    ax.add_patch(Polygon([[-0.75, d0], [-0.15, d0 + 1.25], [-0.15, d0 + 2.5], [0.15, d0 + 2.5], [0.15, d0 + 1.25], [0.75, d0]], closed=True, fc="#e9d8c4", ec="#7a5c3a", lw=1.2, zorder=2))
    ax.plot([-1.0, 1.0], [d0, d0], color="#7a5c3a", lw=3, zorder=3)
    for gate, lab in ((DOCK_POS[0] - 1.093, "gate"),):
        ax.axhline(gate, color="#888", ls=":", lw=1)
        ax.text(lo_y + 0.1, gate + 0.08, lab, fontsize=7, color="#666")
    ax.plot([0, 0], [lo_x, DOCK_POS[0]], color="#bbb", lw=1, ls="--", zorder=1)          # the dock axis
    for k in range(len(x) - 1):
        ax.plot(y[k:k + 2], x[k:k + 2], color=PHASE_COL[PHASES[ph[k]]], lw=2.2, solid_capstyle="round", zorder=4)
    def vehicle(i, col):
        yaw = math.radians(log[i, 6])
        nose = (y[i] + 0.66 * math.sin(yaw), x[i] + 0.66 * math.cos(yaw))
        tail = (y[i] - 0.69 * math.sin(yaw), x[i] - 0.69 * math.cos(yaw))
        ax.plot([tail[0], nose[0]], [tail[1], nose[1]], color=col, lw=4, solid_capstyle="round", zorder=6)
        ax.plot(nose[0], nose[1], "o", color=col, ms=4, zorder=7)
    for i in range(0, len(x), max(1, len(x) // 14)):
        yaw = math.radians(log[i, 6])
        ax.arrow(y[i], x[i], 0.45 * math.sin(yaw), 0.45 * math.cos(yaw), head_width=0.14, color=PHASE_COL[PHASES[ph[i]]], alpha=0.65, zorder=5, length_includes_head=True)
    vehicle(0, "#1f4e9c")
    vehicle(len(x) - 1, "#d63b3b" if outcome in ("COLLISION", "FAIL", "NOT_SEEN") else "#146c43")
    ax.text(y[0], x[0] - 0.9, "start", ha="center", fontsize=8, color="#1f4e9c")
    ax.set_xlim(lo_y, hi_y)
    ax.set_ylim(lo_x, hi_x)
    ax.set_aspect("equal")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]  (dock mouth at 10 m, funnel pointing north)")
    ax.grid(alpha=0.25)
    used = sorted(set(ph.tolist()))
    for p in used:
        ax.plot([], [], color=PHASE_COL[PHASES[p]], lw=3, label=PHASES[p])
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def timeline(log: np.ndarray, outcome: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log[:, 0]
    s_true = DOCK_POS[0] - log[:, 1]
    e_true = log[:, 2]
    fig, axs = plt.subplots(6, 1, figsize=(8.5, 10.5), sharex=True)
    ph = log[:, 16].astype(int)
    def bands(ax):
        start = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or ph[k] != ph[start]:
                ax.axvspan(t[start], t[min(k, len(t) - 1)], color=PHASE_COL[PHASES[ph[start]]], alpha=0.10, lw=0)
                start = k
    for ax in axs:
        bands(ax)
        ax.grid(alpha=0.3)
    axs[0].plot(t, s_true, label="true range to the mouth plane", color="#222")
    axs[0].plot(t, log[:, 8], label="controller's estimate", color="#3f8cff", lw=1)
    axs[0].set_ylabel("range [m]")
    axs[0].legend(fontsize=7)
    axs[1].plot(t, e_true, color="#222", label="true lateral")
    axs[1].plot(t, log[:, 9], color="#3f8cff", lw=1, label="estimate")
    axs[1].set_ylabel("cross-track [m]")
    axs[1].legend(fontsize=7)
    axs[2].plot(t, log[:, 6], color="#222", label="yaw [deg]")
    axs[2].plot(t, log[:, 10], color="#3f8cff", lw=1, label="heading error vs the axis (est)")
    axs[2].set_ylabel("heading [deg]")
    axs[2].legend(fontsize=7)
    axs[3].plot(t, log[:, 7], color="#222")
    axs[3].set_ylabel("surge u [m/s]")
    axs[4].plot(t, log[:, 12], label="th_01 axial", color="#222")
    axs[4].plot(t, log[:, 13], label="th_02", color="#e0a526", lw=1)
    axs[4].plot(t, log[:, 14], label="th_03", color="#3f8cff", lw=1)
    axs[4].set_ylabel("thrusters [RPM]")
    axs[4].legend(fontsize=7, ncol=3)
    for k, (col, name, c, lw) in enumerate(((15, "cs_04", "#222", 2.4), (18, "cs_06", "#e0a526", 1.6), (19, "cs_07", "#3f8cff", 1.2), (20, "cs_08", "#d63b3b", 0.8))):
        axs[5].plot(t, log[:, col], label=name, color=c, lw=lw)
    axs[5].set_ylabel("fins [deg]")
    axs[5].set_xlabel("time [s]")
    axs[5].legend(fontsize=7, ncol=4)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def frames_montage(keeper: FrameKeeper, log: np.ndarray, path: Path, vfov: float = 60.0) -> int:
    import cv2
    import dock_hud
    if not keeper.items:
        return 0
    t_all = np.array([f[0] for f in keeper.items])
    valid = np.array([bool(f[2].msg.valid) for f in keeper.items])
    cand = []
    if valid.any():
        idx_valid = np.nonzero(valid)[0]
        radius = np.array([f[2].msg.radius_px if f[2].msg.valid else 0.0 for f in keeper.items])
        cand.append(int(idx_valid[0]))                                   # first lock
        cand.append(int(idx_valid[len(idx_valid) // 2]))                 # middle of the locked part
        near = [i for i in idx_valid if radius[i] > 0.6 * radius[idx_valid].max()]
        cand.append(int(near[0]) if near else int(idx_valid[-2]))        # the ring is big: close
        cand.append(int(idx_valid[-1]))                                  # last lock
    else:
        cand = [0, len(keeper.items) // 3, 2 * len(keeper.items) // 3, len(keeper.items) - 1]
    tiles = []
    h = dock_hud.History(10.0)
    seen = 0
    for i, (t, bgr, res, roll) in enumerate(keeper.items):
        info = dock_hud.info_from_result(res, vfov, t)
        h.add(info)
        if i in cand and len(tiles) < 4:
            hh, ww = bgr.shape[:2]
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), -math.degrees(roll), 1.0)
            level = cv2.warpAffine(bgr, M, (ww, hh))
            img = dock_hud.draw_camera_view(level, info, h)
            tile = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"t = {t:.1f} s", (tile.shape[1] - 150, tile.shape[0] - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
            seen += 1
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    cv2.imwrite(str(path), np.vstack([top, bot]))
    return seen


def write_csv(log: np.ndarray, path: Path, meta: dict) -> None:
    sys.path[:0] = [str(_d) for _d in [_HERE.parent / "sim_viewer", *sorted((_HERE.parent / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
    from recording import CsvRecorder
    rec = CsvRecorder(path, meta)
    step = 0.1
    next_t = 0.0
    for row in log:
        if row[0] + 1e-9 < next_t:
            continue
        next_t += step
        rec.set_cmd(["th_01", "th_02", "th_03", "cs_04", "cs_06", "cs_07", "cs_08"], [row[12], row[13], row[14], row[15], row[18], row[19], row[20]], row[0])      # ALL four fins (an earlier version kept only cs_04)
        rec.set_align(bool(row[17]), 4 if row[17] else 0, 0.0, 0.0, 0.0, 0.0)
        rec.add_state(row[0], [row[1], row[2], row[3]], np.radians(row[4:7]), [row[7], 0.0, 0.0, 0.0, 0.0, 0.0])
    rec.close()


# ------------------------------------------------------------------------------------------------------------------------------- selection
def pick_from_csv(csv_path: Path, n_good: int = 12, n_bad: int = 3) -> List[Tuple[Tuple[float, float, float], str]]:
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["range"], r["lateral"], r["heading"] = float(r["range"]), float(r["lateral"]), float(r["heading"])
    good = [r for r in rows if r["outcome"] in ("DOCKED", "RECOVERED")]
    bad = [r for r in rows if r["outcome"] not in ("DOCKED", "RECOVERED")]
    chosen: List[Tuple[Tuple[float, float, float], str]] = []
    used = set()

    def take(pred, label, n=1, key=None, rev=False, pool=None):
        c = [r for r in (pool or good) if pred(r) and (r["range"], r["lateral"], r["heading"]) not in used]
        if key:
            c.sort(key=key, reverse=rev)
        for r in c[:n]:
            used.add((r["range"], r["lateral"], r["heading"]))
            chosen.append(((r["range"], r["lateral"], r["heading"]), label))
    take(lambda r: r["view"] == "IN_VIEW" and abs(r["lateral"]) < 0.1 and abs(r["heading"]) < 1 and r["outcome"] == "DOCKED", "straight_in", key=lambda r: -r["range"])
    take(lambda r: r["view"] == "IN_VIEW" and r["heading"] >= 30, "angled_right", key=lambda r: -r["range"])
    take(lambda r: r["view"] == "IN_VIEW" and r["heading"] <= -30, "angled_left", key=lambda r: -r["range"])
    take(lambda r: r["lateral"] >= 1.5 and r["range"] >= 6, "offset_right", key=lambda r: abs(r["heading"]))
    take(lambda r: r["lateral"] <= -1.5 and r["range"] >= 6, "offset_left", key=lambda r: abs(r["heading"]))
    take(lambda r: r["outcome"] == "RECOVERED" and r["range"] <= 4 and r["view"] == "IN_VIEW", "close_backs_out", n=2, key=lambda r: abs(r["heading"]) + abs(r["lateral"]), rev=True)
    take(lambda r: r["outcome"] == "RECOVERED" and r["range"] >= 6, "backs_out_then_approaches", n=2, key=lambda r: abs(r["lateral"]) + abs(r["heading"]) / 30, rev=True)
    take(lambda r: r["view"] == "HIDDEN" and r["outcome"] in ("DOCKED", "RECOVERED"), "searches_then_docks", n=2, key=lambda r: abs(r["heading"]), rev=True)
    take(lambda r: r["view"] == "PARTIAL", "partial_view", n=1)
    take(lambda r: True, "extra", n=max(0, n_good - len(chosen)), key=lambda r: -r["range"])
    for r in sorted(bad, key=lambda r: (r["outcome"] != "COLLISION", r["range"]))[:n_bad]:
        chosen.append(((r["range"], r["lateral"], r["heading"]), "NOT_DOCKED_" + r["outcome"].lower()))
    return chosen


def safe_name(label: str, start) -> str:
    return f"{label}_r{start[0]:g}_l{start[1]:+g}_h{start[2]:+g}".replace("+", "p").replace("-", "m").replace(".", "_")


def _build_one(args):
    start, label, overrides, out, copy_csv_dir, plant, T, tag = args
    import cv2
    cv2.setNumThreads(1)
    name = safe_name(label, start)
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    scn, sim, res, outcome, keeper = run_one(start, overrides, plant, T)
    L = res.log
    title = f"{label}: start range {start[0]:g} m, lateral {start[1]:+g} m, heading {start[2]:+g} deg -> {outcome}"
    top_view(L, scn, outcome, title, d / "top_view.png")
    timeline(L, outcome, title, d / "timeline.png")
    nfr = frames_montage(keeper, L, d / "frames.png")
    write_csv(L, d / "trajectory.csv", {"recorder": "terminal_docking_control/showcase.py", "controller": "terminal_docking", "scenario": title, "outcome": outcome})
    if copy_csv_dir is not None:
        Path(copy_csv_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(d / "trajectory.csv", Path(copy_csv_dir) / (f"showcase_{tag}_{name}.csv" if tag else f"showcase_{name}.csv"))
    m = res.metrics
    info = {"name": name, "label": label, "range_m": start[0], "lateral_m": start[1], "heading_deg": start[2], "plant": plant, "outcome": outcome, "view": view_class(*start),
            "failures": res.failures, "time_s": round(float(m.get("t_end", 0.0)), 1), "retries": int(m.get("retries", 0)), "repositions": int(m.get("repositions", 0)),
            "min_wall_clearance_m": None if m.get("min_clearance_m") is None or math.isnan(m.get("min_clearance_m", float("nan"))) else round(float(m["min_clearance_m"]), 3),
            "entry_lateral_m": m.get("cross_lat"), "entry_heading_deg": m.get("cross_heading_deg"), "entry_speed_mps": m.get("cross_speed"), "frames_saved": nfr}
    (d / "summary.json").write_text(json.dumps(info, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    return info


def build(chosen, overrides, out: Path, copy_csv_dir: Optional[Path], plant="nominal", T=150.0, log=print, workers: int = 1, tag: str = "") -> List[dict]:
    import multiprocessing as mp
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(start, label, overrides, str(out), str(copy_csv_dir) if copy_csv_dir else None, plant, T, tag) for start, label in chosen]
    if workers > 1:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            entries = pool.map(_build_one, jobs, chunksize=1)
    else:
        entries = [_build_one(j) for j in jobs]
    for info in entries:
        log(f"  {info['name']:55s} {info['outcome']:10s} {info['time_s']:6.1f} s  repositions {info['repositions']}  retries {info['retries']}")
    write_index(out, entries, overrides)
    return entries


def write_index(out: Path, entries: List[dict], overrides: dict) -> None:
    badge = {"DOCKED": "#2e9d5b", "RECOVERED": "#6cc070", "COLLISION": "#d63b3b", "FAIL": "#e0a526", "NOT_SEEN": "#888"}
    rows = []
    for e in entries:
        c = badge.get(e["outcome"], "#888")
        extra = (f"<br><small>backed out {e['repositions']}x</small>" if e["repositions"] else "") + (f"<br><small>{html.escape('; '.join(e['failures'])[:140])}</small>" if e["failures"] else "")
        rows.append(f"<tr><td><a href='{e['name']}/top_view.png'><img src='{e['name']}/top_view.png' width='230'></a></td>"
                    f"<td><b>{html.escape(e['label'])}</b><br>start: {e['range_m']:g} m ahead, {e['lateral_m']:+g} m sideways, heading {e['heading_deg']:+g}&deg;<br>dock in view at start: {e['view']}</td>"
                    f"<td><span style='background:{c};color:#fff;padding:2px 8px;border-radius:9px'>{e['outcome']}</span>{extra}</td>"
                    f"<td>{e['time_s']} s<br>wall clearance {e['min_wall_clearance_m']} m</td>"
                    f"<td><a href='{e['name']}/top_view.png'>top view</a><br><a href='{e['name']}/timeline.png'>timeline</a><br><a href='{e['name']}/frames.png'>camera frames</a><br>"
                    f"<a href='{e['name']}/trajectory.csv'>trajectory.csv</a></td></tr>")
    css = "body{font-family:sans-serif;margin:20px;background:#fafafa} table{border-collapse:collapse} td{border-bottom:1px solid #ddd;padding:8px;vertical-align:top} small{color:#555}"
    (out / "index.html").write_text(f"<html><head><meta charset='utf-8'><title>Docking showcase</title><style>{css}</style></head><body><h2>Docking showcase (offline sim)</h2>"
                                    f"<p>The real detector on rendered frames and the real controller, judged on the simulated ground truth. <b>Synthetic</b>: the vehicle model, the camera look and the 2.4 m turning "
                                    f"circle are assumptions, not the real mavsim.</p><table>{''.join(rows)}</table></body></html>")
    (out / "README.txt").write_text(
        "Docking showcase (offline simulation)\n=====================================\n"
        "index.html        gallery: open it in a browser\n<run>/top_view.png      path from above, coloured by controller phase (SEARCH blue, APPROACH green, TERMINAL dark green, RETRY orange, DOCKED red)\n"
        "<run>/timeline.png      range, cross-track, heading, speed, thrust vs time\n<run>/frames.png        camera pictures with the detector pop-up at four key moments\n"
        "<run>/trajectory.csv    replayable in the viewer (History tab); copies are in outputs/sim_viewer_runs/showcase_<tag>_<run>.csv\n<run>/summary.json     numbers\n"
        "All runs: nominal plant, synthetic camera, controller overrides: " + json.dumps(overrides) + "\nBackup/failed examples are named NOT_DOCKED_* so they are not mistaken for successes.\n")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-csv", type=Path)
    ap.add_argument("--starts", default="", help="'range,lateral,heading;...' instead of --from-csv")
    ap.add_argument("--hint", default="", help="'near' or 'x,y,z' rough dock position for the search (as in docking_sweep.py)")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--good", type=int, default=12)
    ap.add_argument("--bad", type=int, default=3)
    ap.add_argument("--out", type=Path, default=out_dir("sim_viewer_runs") / "docking_showcase")
    ap.add_argument("--no-copy", action="store_true", help="do not copy the CSVs into outputs/sim_viewer_runs")
    ap.add_argument("--T", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="name added to the CSV copies in outputs/sim_viewer_runs (showcase_<tag>_<run>.csv), e.g. odometry / dead_reckoning")
    a = ap.parse_args(argv)
    from terminal_docking_eval import parse_overrides
    ov = parse_overrides(a.set)
    if a.hint:
        ov.setdefault("recover", {})["dock_hint_ned"] = [10.6, -0.5, 3.1] if a.hint == "near" else [float(v) for v in a.hint.split(",")]
    if a.starts:
        chosen = [(tuple(float(v) for v in s.split(",")), "chosen") for s in a.starts.split(";")]
    elif a.from_csv:
        chosen = pick_from_csv(a.from_csv, a.good, a.bad)
    else:
        ap.error("give --from-csv or --starts")
    print(f"{len(chosen)} runs -> {a.out}")
    build(chosen, ov, a.out, None if a.no_copy else out_dir("sim_viewer_runs"), T=a.T, workers=a.workers, tag=a.tag)
    print(f"open {a.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
