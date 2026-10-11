#!/usr/bin/env python3
"""Dock from EVERY kind of start: range x lateral offset x heading error, with the real detector on rendered frames and the real controller core, judged on ground truth.

  python3 docking_sweep.py --grid full --tag base                 # 6 ranges x 5 lateral offsets x 11 headings = 330 starts, nominal plant
  python3 docking_sweep.py --grid quick                           # a small grid (about 40 starts)
  python3 docking_sweep.py --grid full --plants nominal,stress --set recover.enabled=false --tag no_recovery
  python3 docking_sweep.py --showcase DIR --from-csv results.csv  # re-run chosen starts with logs and save trajectories (see showcase.py)
Outcome of a start:
  DOCKED      docked on the first try;      RECOVERED  docked after one or more back-offs / recovery manoeuvres;
  NOT_SEEN    the dock was never in view (the controller never got an estimate): nothing to dock on;
  COLLISION   wall contact / bad entry / overshoot (the dangerous ones);       FAIL  anything else (timeout, lost dock, limits ...).
Each start is also tagged with how the dock looked at t=0: IN_VIEW (all four lights inside the image), PARTIAL (some) or HIDDEN (none): the controller can only
dock a start that it can see, or that it can learn to see by turning.
Writes <out>/<tag>/results.csv, summary.txt and heatmap PNGs. SYNTHETIC evidence on the assumed plant/camera (README section 9).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from docking_sim import DOCK_POS, PLANTS, DockingSim, Scenario, load_cfg  # noqa: E402

GRIDS = {
    "full": dict(ranges=(2.0, 3.0, 4.0, 6.0, 8.0, 10.0), lats=(-3.0, -1.5, 0.0, 1.5, 3.0), heads=tuple(range(-75, 76, 15))),
    "quick": dict(ranges=(3.0, 6.0, 9.0), lats=(-1.5, 0.0, 1.5), heads=(-60, -30, 0, 30, 60)),
    "close": dict(ranges=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0), lats=(-1.0, 0.0, 1.0), heads=(-45, -30, -15, 0, 15, 30, 45)),
}


def view_class(rng_m: float, lat: float, head: float) -> str:
    """How much of the dock ring is inside the image at the start (from the synthetic camera's ground truth)."""
    sys.path[:0] = [str(_d) for _d in [_HERE.parent / "sim_viewer", *sorted((_HERE.parent / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
    from synthetic_camera import SyntheticCamera
    cam = _cam()
    pos = np.array([DOCK_POS[0] - rng_m, lat, DOCK_POS[2]])
    inside = 0
    for L in cam.lights:
        if not L["ring"]:
            continue
        pr = cam.project(pos, [0.0, 0.0, head], L["p"])
        if pr is not None and 8 < pr[0] < cam.W - 8 and 8 < pr[1] < cam.H - 8 and cam.brightness(pos, [0.0, 0.0, head], L) > 0.02:
            inside += 1
    return "IN_VIEW" if inside == 4 else ("PARTIAL" if inside > 0 else "HIDDEN")


_C = None


def _cam():
    global _C
    if _C is None:
        from synthetic_camera import SyntheticCamera
        _C = SyntheticCamera()
    return _C


def classify(passed: bool, failures: List[str], metrics: dict) -> str:
    f = " ".join(failures)
    if passed:
        return "DOCKED"
    if not failures:
        return "RECOVERED" if metrics.get("retries", 0) > 0 else "DOCKED"
    if "wall contact" in f or "bad entry" in f or "overshoot" in f:
        return "COLLISION"
    if "missed" in f and "WAIT" in f:
        return "NOT_SEEN"
    return "FAIL"


def _one(args):
    """one run; `overrides` may carry recover.dock_hint_ned as a list (see --hint)"""
    scn, overrides, det_over, stress = args
    import cv2
    cv2.setNumThreads(1)
    sys.path[:0] = [str(_d) for _d in [_HERE.parent / "sim_viewer", *sorted((_HERE.parent / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
    t0 = time.time()
    try:
        cfg = load_cfg(overrides=overrides)
        sim = DockingSim(scn, cfg, det_over)
        if stress:
            sim.cam.stress = dict(stress)
            sim.cam._srng = np.random.default_rng(int(stress.get("seed", 11)))
        r = sim.run(keep_log=False)
        # a run that ended DOCKED after retries is listed as passed=False by the judge (retries > 0): look at the failures
        ok_after_retry = (not r.failures) and sim.core.phase == "DOCKED"
        out = "RECOVERED" if ok_after_retry and not r.passed else classify(r.passed, r.failures, r.metrics)
        return (scn, out, r.failures, r.metrics, time.time() - t0, "")
    except Exception as exc:                                                    # noqa: BLE001
        import traceback
        return (scn, "CRASH", [f"crash: {exc}"], {}, time.time() - t0, traceback.format_exc())


def make_starts(grid: str) -> List[Tuple[float, float, float]]:
    g = GRIDS[grid]
    return [(r, e, float(h)) for r, e, h in itertools.product(g["ranges"], g["lats"], g["heads"])]


def run_sweep(starts, plants=("nominal",), overrides=None, det_over=None, workers=24, T=150.0, seed=1, stress=None, verbose=True, faults=None):
    jobs = [(Scenario(range_m=r, lateral_m=e, heading_deg=h, plant=p, vision="detector", seed=seed, T=T, odom_faults=faults), overrides or {}, det_over, stress)
            for (r, e, h) in starts for p in plants]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    if verbose:
        print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    return res


def summarise(res) -> str:
    lines = []
    c = Counter((view_class(x[0].range_m, x[0].lateral_m, x[0].heading_deg), x[1]) for x in res)
    lines.append(f"{'start view':10s} " + " ".join(f"{k:>10s}" for k in ("DOCKED", "RECOVERED", "NOT_SEEN", "COLLISION", "FAIL", "CRASH")) + "    total")
    for v in ("IN_VIEW", "PARTIAL", "HIDDEN"):
        n = sum(c[(v, k)] for k in ("DOCKED", "RECOVERED", "NOT_SEEN", "COLLISION", "FAIL", "CRASH"))
        if n:
            lines.append(f"{v:10s} " + " ".join(f"{c[(v, k)]:10d}" for k in ("DOCKED", "RECOVERED", "NOT_SEEN", "COLLISION", "FAIL", "CRASH")) + f"    {n:5d}")
    tot = Counter(x[1] for x in res)
    n = len(res)
    lines.append(f"ALL        " + " ".join(f"{tot[k]:10d}" for k in ("DOCKED", "RECOVERED", "NOT_SEEN", "COLLISION", "FAIL", "CRASH")) + f"    {n:5d}")
    lines.append(f"docked (first try or after recovery): {tot['DOCKED'] + tot['RECOVERED']}/{n}   collisions: {tot['COLLISION']}")
    return "\n".join(lines)


COLORS = {"DOCKED": "#2e9d5b", "RECOVERED": "#8bd17c", "NOT_SEEN": "#b0b0b0", "COLLISION": "#d63b3b", "FAIL": "#e0a526", "CRASH": "#000000"}


def save_outputs(res, out: Path, tag: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plant", "range", "lateral", "heading", "view", "outcome", "failures", "retries", "min_clearance", "t_end", "final_heading", "cross_lat", "cross_vert", "cross_speed", "cross_heading"])
        for s, o, fl, m, _, _ in res:
            w.writerow([s.plant, s.range_m, s.lateral_m, s.heading_deg, view_class(s.range_m, s.lateral_m, s.heading_deg), o, " | ".join(fl), m.get("retries"),
                        m.get("min_clearance_m"), m.get("t_end"), m.get("final_heading_deg"), m.get("cross_lat"), m.get("cross_vert"), m.get("cross_speed"), m.get("cross_heading_deg")])
    with open(out / "summary.txt", "w") as f:
        f.write(f"docking sweep '{tag}': {len(res)} runs\n\n{summarise(res)}\n\n")
        reasons = Counter()
        for s, o, fl, *_ in res:
            for x in fl:
                reasons[x.split(":")[0]] += 1
        f.write(f"failure reasons: {dict(reasons)}\n\n")
        for s, o, fl, m, _, tb in sorted(res, key=lambda r: (r[1], r[0].plant, r[0].range_m, r[0].lateral_m, r[0].heading_deg)):
            if o not in ("DOCKED",):
                f.write(f"{o:9s} {s.plant:10s} range {s.range_m:4.1f} lat {s.lateral_m:+5.1f} head {s.heading_deg:+6.1f}  {'; '.join(fl)[:150]}\n")
            if tb:
                f.write(tb + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:                                                           # noqa: BLE001
        return
    lats = sorted({s.lateral_m for s, *_ in res})
    heads = sorted({s.heading_deg for s, *_ in res})
    rngs = sorted({s.range_m for s, *_ in res})
    fig, axs = plt.subplots(1, len(lats), figsize=(3.6 * len(lats), 3.8), squeeze=False)
    for i, lat in enumerate(lats):
        ax = axs[0][i]
        for s, o, *_ in res:
            if s.lateral_m != lat or s.plant != res[0][0].plant:
                continue
            ax.add_patch(plt.Rectangle((heads.index(s.heading_deg) - 0.5, rngs.index(s.range_m) - 0.5), 1, 1, color=COLORS[o]))
        ax.set_xlim(-0.5, len(heads) - 0.5)
        ax.set_ylim(-0.5, len(rngs) - 0.5)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([f"{h:+.0f}" for h in heads], fontsize=6, rotation=60)
        ax.set_yticks(range(len(rngs)))
        ax.set_yticklabels([f"{r:g}" for r in rngs], fontsize=7)
        ax.set_title(f"lateral {lat:+.1f} m", fontsize=9)
        ax.set_xlabel("heading error [deg]", fontsize=8)
        if i == 0:
            ax.set_ylabel("range to the dock [m]", fontsize=8)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items() if k != "CRASH"], loc="lower center", ncol=5, fontsize=7)
    fig.suptitle(f"docking outcome by start pose ({tag}, plant {res[0][0].plant}, real detector + controller, synthetic world)", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out / "feasibility_map.png", dpi=120)
    plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="quick", choices=list(GRIDS))
    ap.add_argument("--plants", default="nominal")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--T", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="controller yaml override section.key=value")
    ap.add_argument("--hint", default="", help="rough dock position 'x,y,z' given to the search (mission-plan prompt); '' = blind search. Use 'near' for (10.6,-0.5,3.1): 0.8 m off")
    ap.add_argument("--rerun", type=Path, default=None, help="re-run only the starts of a previous results.csv whose outcome is in --outcomes (to test a change quickly)")
    ap.add_argument("--outcomes", default="COLLISION,FAIL,NOT_SEEN")
    ap.add_argument("--faults", default="", help='odometry position faults as JSON, e.g. {"pos_drift_mps":[0.05,0.03]} or {"pos_jump_m":[0.6,0.4],"pos_jump_t_s":20} or {"pos_offset_m":[3,-2]} or {"drop_twist":true}')
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out", type=Path, default=_HERE.parents[1] / "outputs" / "docking_sweeps")
    a = ap.parse_args(argv)
    from terminal_docking_eval import parse_overrides
    plants = list(PLANTS) if a.plants == "all" else a.plants.split(",")
    ov = parse_overrides(a.set)
    if a.hint:
        xyz = [10.6, -0.5, 3.1] if a.hint == "near" else [float(v) for v in a.hint.split(",")]
        ov.setdefault("recover", {})["dock_hint_ned"] = xyz
    starts = make_starts(a.grid)
    if a.rerun:
        want = set(a.outcomes.split(","))
        starts = [(float(r["range"]), float(r["lateral"]), float(r["heading"])) for r in csv.DictReader(open(a.rerun)) if r["outcome"] in want]
    import json as _json
    faults = _json.loads(a.faults) if a.faults else None
    if faults:
        faults = {k: (tuple(v) if isinstance(v, list) else v) for k, v in faults.items()}
    res = run_sweep(starts, plants, ov, None, a.workers, a.T, a.seed, faults=faults)
    save_outputs(res, a.out / a.tag, a.tag)
    print(summarise(res))
    print(f"-> {a.out / a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
