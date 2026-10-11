#!/usr/bin/env python3
"""Compare two recorded runs (e.g. before and after a gain change): overlay plots and a table of differences. No ROS, no Qt.

  python3 plots/compare_runs.py A.csv B.csv --plot ab.png
Both runs start at t = 0. B is interpolated onto A's time grid for the numbers. Works on any file replay.load_run reads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import Run, UnsupportedLog, load_run  # noqa: E402

CHANNELS = [("depth [m]", lambda r: r.pos[:, 2]), ("pitch [deg]", lambda r: np.degrees(r.eul[:, 1])), ("yaw [deg]", lambda r: np.degrees(r.eul[:, 2])),
            ("surge u [m/s]", lambda r: r.nu[:, 0]), ("north x [m]", lambda r: r.pos[:, 0]), ("east y [m]", lambda r: r.pos[:, 1])]


def _on_grid(a: Run, b: Run, f) -> np.ndarray:
    return np.interp(a.t, b.t, f(b))


def difference_table(a: Run, b: Run) -> List[dict]:
    rows = []
    T = min(a.duration, b.duration)
    m = a.t <= T
    for name, f in CHANNELS:
        va, vb = f(a)[m], _on_grid(a, b, f)[m]
        d = vb - va
        rows.append({"channel": name, "rms_diff": float(np.sqrt(np.mean(d ** 2))), "max_abs_diff": float(np.max(np.abs(d))), "final_a": float(f(a)[-1]), "final_b": float(f(b)[-1])})
    return rows


def format_table(rows: List[dict]) -> str:
    out = [f"{'channel':16s} {'RMS diff':>10s} {'max |diff|':>11s} {'final A':>10s} {'final B':>10s}"]
    for r in rows:
        out.append(f"{r['channel']:16s} {r['rms_diff']:10.4f} {r['max_abs_diff']:11.4f} {r['final_a']:10.3f} {r['final_b']:10.3f}")
    return "\n".join(out)


def plot_two(a: Run, b: Run, path: Path, label_a: str = "A", label_b: str = "B") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    for ax, (name, f) in zip(axes.ravel(), CHANNELS):
        ax.plot(a.t, f(a), lw=1.3, label=label_a)
        ax.plot(b.t, f(b), lw=1.2, ls="--", label=label_b)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
    for ax, key in zip(axes.ravel()[len(CHANNELS):], ("th_01", "th_02")):
        for r, lab, ls in ((a, label_a, "-"), (b, label_b, "--")):
            if key in r.cmd:
                ax.plot(r.t, r.cmd[key], lw=1.1, ls=ls, label=lab)
        ax.set_title(f"{key} command [RPM]", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args(argv)
    try:
        a, b = load_run(args.a), load_run(args.b)
    except UnsupportedLog as exc:
        print(f"compare_runs: {exc}", file=sys.stderr)
        return 2
    print(format_table(difference_table(a, b)))
    if args.plot:
        plot_two(a, b, args.plot, args.a.name, args.b.name)
        print("wrote", args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
