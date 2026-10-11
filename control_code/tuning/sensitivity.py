#!/usr/bin/env python3
"""One-at-a-time sensitivity of the docking result to every number in terminal_docking.yaml: how many runs of the grid pass when the value is changed by x0.5 ... x1.5.
These are the measured numbers quoted in the yaml comments ('effect size'). No ROS, no GUI.

  python3 sensitivity.py --grid wide --params guidance.gate_s_m,speed.cruise_mps --out sens.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "terminal_docking_control"))
from docking_sim import PLANTS, load_cfg  # noqa: E402
from terminal_docking_eval import evaluate, feasible, make_grid  # noqa: E402
from tune_docking import _get, overrides_from, run_cost  # noqa: E402

FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT = [
    "guidance.gate_s_m", "guidance.gate_lateral_m", "guidance.gate_heading_deg", "guidance.lookahead_base_m", "guidance.lookahead_per_m", "guidance.max_deviation_deg",
    "guidance.terminal_lookahead_m", "guidance.stop_nose_s_m", "guidance.docked_speed_mps", "speed.cruise_mps", "speed.align_cruise_mps", "speed.terminal_mps", "speed.slow_zone_m",
    "speed.decel_mps2", "speed.kp_n_per_mps", "speed.stop_gain_per_s", "gains.yaw.kp", "gains.yaw.kd", "gains.yaw.ki", "gains.yaw.max", "gains.heave.kp", "gains.heave.kd",
    "gains.pitch.kp", "gains.pitch.kd", "estimator.pixel_sigma_px", "estimator.process_pos_m_per_sqrt_s",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", default="wide")
    ap.add_argument("--params", default=",".join(DEFAULT))
    ap.add_argument("--vision", default="perfect")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    base = load_cfg()
    grid = make_grid(a.grid)
    plants = list(PLANTS)
    out: Dict[str, dict] = {}
    for p in [x for x in a.params.split(",") if x]:
        v0 = float(_get(base, p))
        row = {"value": v0, "passes": {}}
        for f in FACTORS:
            res = evaluate(grid, plants, a.vision, overrides_from(base, {p: v0 * f}), 24, 0.0, verbose=False)
            row["passes"][str(f)] = [sum(1 for r in res if r[1]), len(res), run_cost(res)]
        out[p] = row
        print(f"{p:40s} {v0:9.4g}  " + "  ".join(f"x{f}: {row['passes'][str(f)][0]:3d}/{row['passes'][str(f)][1]}" for f in FACTORS), flush=True)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
