#!/usr/bin/env python3
"""Parameter search for terminal_docking.yaml against the docking grid (in-process closed loop, parallel). No ROS, no GUI.

  python3 tune_docking.py --grid wide --vision perfect --rounds 4 --params guidance.lookahead_base_m,speed.cruise_mps,...   (see DEFAULT_PARAMS)
The cost of one parameter set is the sum over all runs (grid x plants) of: 100 per failed run, plus a small margin term for passing runs (entry error,
heading, clearance, time) so that among sets that pass everything the one with the most margin wins. Coordinate descent with shrinking multiplicative steps.
The result is only printed (and optionally written to a yaml with --out): the shipped terminal_docking.yaml is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "terminal_docking_control"))
from docking_sim import PLANTS, load_cfg  # noqa: E402
from terminal_docking_eval import evaluate, feasible, make_grid  # noqa: E402

DEFAULT_PARAMS = [
    "guidance.lookahead_base_m", "guidance.lookahead_per_m", "guidance.max_deviation_deg", "guidance.terminal_lookahead_m", "guidance.gate_s_m",
    "speed.cruise_mps", "speed.terminal_mps", "speed.slow_zone_m", "speed.decel_mps2",
    "gains.yaw.kp", "gains.yaw.kd", "gains.yaw.max",
]


def _get(cfg: dict, path: str):
    d = cfg
    for k in path.split("."):
        d = d[k]
    return d


def _set(cfg: dict, path: str, v: float) -> None:
    ks = path.split(".")
    d = cfg
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = v


def overrides_from(base: dict, values: Dict[str, float]) -> dict:
    out: dict = {}
    for path, v in values.items():
        ks = path.split(".")
        d = out
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = float(v)
    return out


def run_cost(res) -> float:
    c = 0.0
    for scn, ok, fails, met, _, _ in res:
        if not ok:
            c += 100.0 + 5.0 * len(fails)
            continue
        c += (5.0 * max(abs(met.get("cross_lat", 0.0)), abs(met.get("cross_vert", 0.0))) / 0.15 + abs(met.get("cross_heading_deg", 0.0)) / 5.0
              + (2.0 if met.get("min_clearance_m", 1.0) < 0.1 else 0.0) + met.get("t_end", 0.0) / 180.0)
    return c


def search(base_cfg: dict, params: List[str], grid, plants, vision="perfect", rounds=4, workers=24, freeze=0.0, log=print, extra_sets=None) -> Dict[str, float]:
    """extra_sets: [(grid, plants, vision)] evaluated in addition (costs add): e.g. the cheap perfect-vision wide grid AND a small grid with the real detector,
    so the settings do not overfit one of them."""
    values = {p: float(_get(base_cfg, p)) for p in params}
    best = None
    step = 0.4
    sets = [(grid, plants, vision)] + list(extra_sets or [])

    def evalv(vals):
        cost, nf, allres = 0.0, 0, []
        for g, pl, vis in sets:
            res = evaluate(g, pl, vis, overrides_from(base_cfg, vals), workers, freeze, verbose=False)
            cost += run_cost(res)
            nf += sum(1 for r in res if not r[1])
            allres += res
        return cost, nf, allres

    best, nfail, _ = evalv(values)
    log(f"start: cost {best:.1f}  failures {nfail}")
    for r in range(rounds):
        improved = False
        for p in params:
            for f in (1 + step, 1 / (1 + step)):
                v2 = dict(values)
                v2[p] = values[p] * f
                c, nf, _ = evalv(v2)
                if c < best - 1e-6:
                    best, values, nfail, improved = c, v2, nf, True
                    log(f"  round {r} {p} -> {values[p]:.4g}: cost {best:.1f} failures {nfail}")
        if not improved:
            step *= 0.5
        if nfail == 0 and r >= 1:
            pass
    return values


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="wide", choices=["wide", "moderate", "smoke"])
    ap.add_argument("--plants", default="all")
    ap.add_argument("--vision", default="perfect", choices=["perfect", "detector"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--params", default=",".join(DEFAULT_PARAMS))
    ap.add_argument("--also-detector-moderate", action="store_true", help="also score the moderate grid with the real detector (nominal, heavy+late, light+fast): costs add")
    ap.add_argument("--only-reachable", action="store_true", help="drop starts the screening rule calls physically unreachable")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    base = load_cfg()
    plants = list(PLANTS) if a.plants == "all" else a.plants.split(",")
    grid = make_grid(a.grid)
    if a.only_reachable:
        grid = [g for g in grid if feasible(*g)]
    extra = [(make_grid("moderate"), ["nominal", "heavy+late", "light+fast"], "detector")] if a.also_detector_moderate else None
    vals = search(base, [p for p in a.params.split(",") if p], grid, plants, a.vision, a.rounds, extra_sets=extra)
    print(yaml.safe_dump(vals, sort_keys=False))
    if a.out:
        a.out.write_text(yaml.safe_dump(vals, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
