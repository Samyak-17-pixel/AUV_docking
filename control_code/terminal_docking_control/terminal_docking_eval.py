#!/usr/bin/env python3
"""Evaluate the docking controller over a grid of start poses and plants, in parallel; print per-cell results and the failure statistics.

  python3 terminal_docking_eval.py --grid wide --vision perfect          # fast (no rendering), for gain search
  python3 terminal_docking_eval.py --grid wide --vision detector         # the real detector on rendered frames
  python3 terminal_docking_eval.py --grid moderate --plants nominal
Options: --set section.key=value (override a yaml value, repeatable), --workers N, --csv FILE.
A start is only run if the dock is inside the camera's field of view at the beginning (dock bearing relative to the heading within +-25 deg).
Starts that are physically unreachable (the vehicle cannot turn tightly enough, `feasible()`) are run anyway but reported separately.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docking_sim import PLANTS, DockingSim, Scenario, load_cfg  # noqa: E402

TURN_RADIUS_M = 2.4          # minimum turning radius from the fin authority (|N| = 2.65 N*m at 1 m/s, quadratic yaw drag 15 N*m*s^2/rad^2): ASSUMED model
FOV_BEARING_DEG = 25.0       # the whole ring (+-11 deg at 5 m) must fit inside the +-37 deg horizontal half field of view


def bearing_deg(range_m: float, lateral_m: float, heading_deg: float) -> float:
    """Direction of the dock centre relative to the vehicle's heading (+ = to the right)."""
    to_dock = math.degrees(math.atan2(-lateral_m, range_m))            # world direction from the vehicle to the dock, relative to north
    return to_dock - heading_deg


def feasible(range_m: float, lateral_m: float, heading_deg: float, gate_s: float = 3.0, settle_m: float = 1.0) -> bool:
    """Rough reachability: can a car with TURN_RADIUS_M reach the axis, heading along it, `settle_m` before the gate?
    Uses the Dubins CSC length bound for the lateral offset; it is a screening rule, not a proof."""
    avail = range_m - gate_s - settle_m
    e = abs(lateral_m)
    h = math.radians(abs(heading_deg)) * (1 if lateral_m * heading_deg <= 0 else -1) if False else math.radians(heading_deg) * (1.0 if lateral_m > 0 else -1.0)
    # extra lateral displacement the initial heading already buys/costs (turning away costs more room)
    need = 0.0
    R = TURN_RADIUS_M
    if e > 1e-6:
        # S-curve with maximum heading deviation theta: e = 2R(1-cos theta) + L_straight*sin theta; shortest when theta is as large as allowed
        theta = math.radians(40.0)
        arc_e = 2 * R * (1 - math.cos(theta))
        straight = max(e - arc_e, 0.0) / math.sin(theta)
        need = 2 * R * math.sin(theta) + straight * math.cos(theta)
        if e < arc_e:                                                  # small offset: smaller deviation
            theta = math.acos(1 - e / (2 * R))
            need = 2 * R * math.sin(theta)
    # turning away from the axis first costs about an arc more
    if heading_deg * lateral_m > 0:
        need += R * abs(math.radians(heading_deg))
    return avail >= need


def make_grid(kind: str) -> List[Tuple[float, float, float]]:
    if kind == "wide":
        ranges, lats, heads = (5.0, 7.0, 9.0), (-2.0, -1.0, 0.0, 1.0, 2.0), (-30.0, -15.0, 0.0, 15.0, 30.0)
    elif kind == "moderate":
        ranges, lats, heads = (6.0, 8.0), (-1.0, 0.0, 1.0), (-20.0, 0.0, 20.0)
    elif kind == "smoke":
        ranges, lats, heads = (7.0,), (0.0, 1.0), (0.0, 15.0)
    else:
        raise ValueError(kind)
    out = [(r, e, h) for r, e, h in itertools.product(ranges, lats, heads) if abs(bearing_deg(r, e, h)) <= FOV_BEARING_DEG]
    if kind == "wide":                                               # a few random combinations on top
        rng = np.random.default_rng(5)
        while len(out) < 60 + 0:
            r, e, h = float(rng.uniform(5, 9)), float(rng.uniform(-2, 2)), float(rng.uniform(-30, 30))
            if abs(bearing_deg(r, e, h)) <= FOV_BEARING_DEG:
                out.append((round(r, 2), round(e, 2), round(h, 1)))
    return out


def _one(args):
    scn, overrides, det_over = args
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:                                                  # noqa: BLE001
        pass
    cfg = load_cfg(overrides=overrides)
    t0 = time.time()
    try:
        r = DockingSim(scn, cfg, det_over).run(keep_log=False)
        return (scn, r.passed, r.failures, r.metrics, time.time() - t0, "")
    except Exception as exc:                                           # noqa: BLE001
        import traceback
        return (scn, False, [f"crash: {exc}"], {}, time.time() - t0, traceback.format_exc())


def parse_overrides(items: List[str]) -> dict:
    out: dict = {}
    for it in items:
        key, val = it.split("=", 1)
        parts = key.split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml_value(val)
    return out


def yaml_value(v: str):
    import yaml
    return yaml.safe_load(v)


def evaluate(grid, plants, vision="perfect", overrides=None, workers=16, freeze=0.0, seeds=(1,), det_over=None, T=180.0, verbose=True):
    jobs = [(Scenario(range_m=r, lateral_m=e, heading_deg=h, plant=p, vision=vision, seed=sd, freeze_interval_s=freeze, T=T), overrides or {}, det_over)
            for (r, e, h) in grid for p in plants for sd in seeds]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    if verbose:
        print(f"{len(jobs)} runs in {time.time() - t0:.0f} s")
    return res


def report(res, show_pass=False) -> dict:
    n = len(res)
    feas = [x for x in res if feasible(x[0].range_m, x[0].lateral_m, x[0].heading_deg)]
    npass = sum(1 for x in res if x[1])
    nf = sum(1 for x in feas if x[1])
    print(f"PASS {npass}/{n} overall;  {nf}/{len(feas)} on starts the screening rule calls reachable")
    from collections import Counter
    c = Counter()
    for x in res:
        for f in x[2]:
            c[f.split(":")[0]] += 1
    if c:
        print("failure reasons:", dict(c))
    by_plant = {}
    for x in res:
        by_plant.setdefault(x[0].plant, [0, 0])
        by_plant[x[0].plant][0] += int(x[1])
        by_plant[x[0].plant][1] += 1
    print("by plant:", {k: f"{a}/{b}" for k, (a, b) in by_plant.items()})
    for x in sorted(res, key=lambda r: (r[0].plant, r[0].range_m, r[0].lateral_m, r[0].heading_deg)):
        if show_pass or not x[1]:
            s = x[0]
            fe = "reachable" if feasible(s.range_m, s.lateral_m, s.heading_deg) else "UNREACHABLE?"
            print(f"  {'PASS' if x[1] else 'FAIL'} {s.plant:11s} range {s.range_m:4.1f} lat {s.lateral_m:+5.2f} head {s.heading_deg:+6.1f} [{fe}] "
                  f"| {'; '.join(x[2]) if x[2] else ('retries=%d' % x[3].get('retries', 0))} | clr {x[3].get('min_clearance_m', float('nan')):.2f}")
            if x[5]:
                print(x[5])
    return {"n": n, "pass": npass, "reachable": len(feas), "reachable_pass": nf}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="moderate", choices=["wide", "moderate", "smoke"])
    ap.add_argument("--plants", default="all")
    ap.add_argument("--vision", default="perfect", choices=["perfect", "detector"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--freeze", type=float, default=0.0, help="mean seconds between odometry freezes (0 = none)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--show-pass", action="store_true")
    a = ap.parse_args(argv)
    plants = list(PLANTS) if a.plants == "all" else a.plants.split(",")
    res = evaluate(make_grid(a.grid), plants, a.vision, parse_overrides(a.set), a.workers, a.freeze, tuple(range(1, a.seeds + 1)))
    summary = report(res, a.show_pass)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["plant", "range", "lateral", "heading", "pass", "failures", "retries", "min_clearance", "t_end"])
            for s, ok, fl, met, _, _ in res:
                w.writerow([s.plant, s.range_m, s.lateral_m, s.heading_deg, int(ok), " | ".join(fl), met.get("retries"), met.get("min_clearance_m"), met.get("t_end")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
