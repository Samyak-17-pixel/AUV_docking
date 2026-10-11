#!/usr/bin/env python3
"""Parameter sweep for the dock_test heading-error recovery (standoff_sim.py harness, no ROS). Offline-model numbers only.

  python3 tune_standoff.py                      # the default grid
  python3 tune_standoff.py --set gains.yaw_kd=4 --set gains.yaw_nm_per_px=0.015 [--set ...]   # one candidate
Score per candidate = runs (3 plants x yaw starts) that spend >= 80% of the last 20 s in mode 'standoff' (square, in frame, stopped) AND never get closer than 2.0 m, out of N.
The printed 'mean late' column is the mean % of the last 20 s NOT at rest.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import standoff_sim as S  # noqa: E402

YAWS = (10.0, 20.0, 35.0, -25.0)
T = 100.0


RANGES = {                                         # sampled uniformly (tuple) or from a list. Second round: narrowed around the best of a first, wider round.
    "gains.yaw_kd": (4.0, 12.0),
    "gains.yaw_nm_per_px": (0.004, 0.025),
    "gains.yaw_nm_per_lateral_px": [0.01, 0.03],
    "speed.settle_yaw_rate_rad_s": (0.03, 0.17),
    "speed.realign_max_back_m": (1.0, 2.2),
    "speed.realign_mps": (0.45, 0.65),
    "speed.creep_mps": [0.45, 0.6],
    "deadband.hold_extra_px": (25.0, 55.0),
    "deadband.error_x_px": [20.0],
}


def _apply(cfg, sets):
    cfg = copy.deepcopy(cfg)
    for k, v in sets.items():
        sect, key = k.split(".")
        cfg[sect][key] = v
    return cfg


def _one(args):
    sets, plant, yaw0, seed = args
    cfg = _apply(S.load_cfg(), sets)
    r = S.run(cfg=cfg, yaw0_deg=yaw0, T=T, seed=seed, **S.PLANTS[plant])
    late = [row for row in r["log"] if row[0] > T - 20.0]
    yaw_late = max(abs(row[1]) for row in late)
    rest = sum(1 for row in late if row[5] == "standoff") / len(late)         # the controller itself says: square, in frame, stopped
    ok = rest >= 0.8 and r["min_dist_m"] > 2.0
    return ok, 100.0 * (1.0 - rest), r["min_dist_m"]


def score(sets, pool=None, seeds=(1,)):
    jobs = [(sets, p, y, s) for p in S.PLANTS for y in YAWS for s in seeds]
    res = (pool.map(_one, jobs) if pool else list(map(_one, jobs)))
    n_ok = sum(r[0] for r in res)
    return n_ok, len(res), float(np.mean([r[1] for r in res])), min(r[2] for r in res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--random", type=int, default=0, help="N random candidates from RANGES (seeded), best first")
    a = ap.parse_args()
    with Pool() as pool:
        if a.random:
            rng = np.random.default_rng(5)
            rows = []
            for i in range(a.random):
                sets = {k: (float(rng.choice(v)) if isinstance(v, list) else float(rng.uniform(*v))) for k, v in RANGES.items()}
                n_ok, n, nr, dmin = score(sets, pool, seeds=(1, 2))
                rows.append((n_ok, -nr, sets, nr, dmin))
                print(f"{i:3d} {n_ok:2d}/{n}  not-at-rest {nr:5.1f} %  closest {dmin:.2f} m  {sets}", flush=True)
            rows.sort(key=lambda r: (r[4] > 2.0, r[0], r[1]), reverse=True)
            print("BEST 5:")
            for r in rows[:5]:
                print(r[0], f"{r[3]:.1f}%", f"{r[4]:.2f} m", r[2])
            return
        if a.set:
            sets = {k: float(v) for k, v in (s.split("=") for s in a.set)}
            print(sets, score(sets, pool, seeds=(1, 2)))
            return
        grid = {
            "gains.yaw_kd": [2.0, 6.0],
            "gains.yaw_nm_per_px": [0.03, 0.01],
            "gains.yaw_nm_per_lateral_px": [0.03, 0.0],
            "speed.settle_yaw_rate_rad_s": [0.03, 0.08],
            "speed.realign_max_back_m": [2.5, 4.0],
        }
        keys = list(grid)
        rows = []
        for vals in itertools.product(*grid.values()):
            sets = dict(zip(keys, vals))
            n_ok, n, yaw_mean, dmin = score(sets, pool)
            rows.append((n_ok, -yaw_mean, sets, yaw_mean, dmin))
            print(f"{n_ok:2d}/{n}  mean not-at-rest {yaw_mean:5.1f} %  closest {dmin:.2f} m  {sets}", flush=True)
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        print("BEST:", rows[0][2])


if __name__ == "__main__":
    main()
