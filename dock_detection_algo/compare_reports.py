#!/usr/bin/env python3
"""Compare two reliability_study.py reports (results.csv of each): good / bad-lock / miss per axis value, side by side.

  python3 compare_reports.py outputs/detection_reports/baseline outputs/detection_reports/after [--out outputs/detection_reports/comparison.txt]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(d: Path):
    acc = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(d / "results.csv")):
        acc[(r["axis"], r["value"])][r["kind"]] += 1
    return acc


def frac(c, k):
    n = c["good"] + c["bad"] + c["miss"]
    return None if n == 0 else c[k] / n


def main(argv):
    a, b = Path(argv[1]), Path(argv[2])
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    A, B = load(a), load(b)
    lines = [f"reliability: {a.name} -> {b.name}   (good% / bad-lock% of the frames with the dock in view; a dash = the dock was never fully in view)", ""]
    axis = None
    for key in B:
        if key[0] != axis:
            axis = key[0]
            lines += ["", f"{axis}", f"  {'value':>9s}   {'good before':>11s} {'after':>6s}   {'bad before':>10s} {'after':>6s}"]
        ca, cb = A.get(key, defaultdict(int)), B[key]
        f = lambda x: "   -  " if x is None else f"{100 * x:6.1f}"
        lines.append(f"  {key[1]:>9s}   {f(frac(ca, 'good')):>11s} {f(frac(cb, 'good')):>6s}   {f(frac(ca, 'bad')):>10s} {f(frac(cb, 'bad')):>6s}")
    text = "\n".join(lines)
    print(text)
    if out:
        out.write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv)
