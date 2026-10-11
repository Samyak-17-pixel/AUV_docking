#!/usr/bin/env python3
"""Run the wide docking grid and write a compact RESULTS summary (outputs/docking_sweeps/docking_results.txt), used in the yaml header and the docs.

  python3 make_docking_results.py [--vision perfect|detector]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "terminal_docking_control"))
from docking_sim import PLANTS  # noqa: E402
from terminal_docking_eval import evaluate, feasible, make_grid  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs" / "docking_sweeps" / "docking_results.txt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", default="detector")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args(argv)
    grid = make_grid("wide")
    res = evaluate(grid, list(PLANTS), a.vision, {}, a.workers, 0.0, verbose=False)
    n = len(res)
    ok = [r for r in res if r[1]]
    lines = [f"wide grid, {a.vision} vision, {len(grid)} starts x 4 plants = {n} runs: {len(ok)} first-try passes ({100 * len(ok) / n:.0f}%)."]
    by = defaultdict(lambda: [0, 0])
    for s, passed, *_ in res:
        by[s.plant][0] += int(passed)
        by[s.plant][1] += 1
    lines.append("by plant: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by.items()))
    reach = [r for r in res if feasible(r[0].range_m, r[0].lateral_m, r[0].heading_deg)]
    lines.append(f"starts the screening rule calls physically reachable (enough room to line up): {sum(1 for r in reach if r[1])}/{len(reach)} pass.")
    for lo, hi in ((5.0, 6.5), (6.5, 8.0), (8.0, 10.0)):
        sub = [r for r in res if lo <= r[0].range_m < hi]
        lines.append(f"range {lo:g}-{hi:g} m: {sum(1 for r in sub if r[1])}/{len(sub)} first-try passes")
    cause = Counter()
    for r in res:
        for f in r[2]:
            cause[f.split(':')[0]] += 1
    lines.append("failure reasons (a run can have several): " + (", ".join(f"{k} {v}" for k, v in cause.most_common()) or "none"))
    m = [r[3] for r in ok]
    if m:
        lines.append("margins over the passing runs: min wall clearance %.2f m (mean %.2f), entry speed mean %.2f max %.2f m/s (limit 0.5), entry lateral/vertical error max %.0f / %.0f cm (limit 15), "
                     "entry heading max %.1f deg (limit 5), time to dock mean %.0f s" % (
                         min(x["min_clearance_m"] for x in m), np.mean([x["min_clearance_m"] for x in m]), np.mean([x["cross_speed"] for x in m]), max(x["cross_speed"] for x in m),
                         100 * max(abs(x["cross_lat"]) for x in m), 100 * max(abs(x["cross_vert"]) for x in m), max(abs(x["cross_heading_deg"]) for x in m), np.mean([x["t_end"] for x in m])))
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
