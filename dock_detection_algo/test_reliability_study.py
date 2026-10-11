"""The reliability study harness (reliability_study.py): the axes exist, one tiny run produces a table and files, and the shipped detector meets the stated thresholds
on the base case. Needs the workspace sourced (DockAlign)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("interfaces.msg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reliability_study as rs  # noqa: E402


def test_every_axis_builds_cases_with_the_right_values():
    for ax, (group, values, setter, label) in rs.AXES.items():
        cases = rs.make_cases(ax, 2, 1)
        assert len(cases) == 2 * len(values) and group in ("geometry", "camera", "stress") and label
    assert {c["range"] for _, c in rs.make_cases("range", 3, 1) if _ == 5.0} == {5.0}


def test_a_small_study_writes_a_report_and_meets_the_base_thresholds(tmp_path):
    res = rs.run(["range", "view_deg", "distractors", "occlude"], 8, 21, rs.DEFAULT_CONFIG, {}, 4, log=lambda *_: None)
    rs.save_report(res, tmp_path, "t")
    assert (tmp_path / "summary.txt").exists() and (tmp_path / "results.csv").stat().st_size > 500
    by = {ax: dict((str(v), s) for v, s in res[ax]) for ax in ("range", "view_deg", "distractors", "occlude")}
    assert by["range"]["5.0"]["good"] == 1.0 and by["range"]["8.0"]["good"] >= 0.85                   # in clear water 5-8 m is reliable
    assert by["view_deg"]["0"]["good"] == 1.0 and by["view_deg"]["30"]["good"] >= 0.85
    assert by["distractors"]["2"]["bad"] <= 0.15                                                       # ring check: false lights rarely win
    assert all(by["occlude"][k]["bad"] == 0.0 for k in ("top", "bottom", "left", "right"))             # a hidden light is never a false lock
    assert by["range"]["1.5"]["out"] == 1.0                                                             # 1.5 m: the ring does not fit the picture (not counted as a miss)
