"""run_summary (no Qt): path error against a plan, coverage, and the figure."""
import numpy as np
import pytest

import run_summary as RS
from replay import Run


def make_run(xy, z=3.0, dt=0.5, cmd=None):
    xy = np.asarray(xy, float)
    n = len(xy)
    return Run(t=np.arange(n) * dt, pos=np.column_stack([xy, np.full(n, z)]), eul=np.zeros((n, 3)), nu=np.zeros((n, 6)), cmd=cmd or {})


PLAN = [(0.0, 0.0, 3.0, "a"), (0.0, 10.0, 3.0, "a"), (6.0, 10.0, 3.0, "b")]


def test_distance_to_polyline():
    d, seg = RS.dist_to_polyline(np.array([[1.0, 5.0], [3.0, 12.0], [-2.0, -1.0]]), np.array([[0.0, 0.0], [0.0, 10.0], [6.0, 10.0]]))
    assert d == pytest.approx([1.0, 2.0, np.hypot(2.0, 1.0)]) and list(seg) == [0, 1, 0]


def test_a_run_on_the_plan_has_a_tiny_error_and_full_coverage():
    on = [(0.0, y) for y in np.linspace(0, 10, 21)] + [(x, 10.0) for x in np.linspace(0, 6, 13)[1:]]
    s = RS.summarize(make_run(on), PLAN, swath_m=3.0)
    assert s["path_error_max_m"] < 1e-6 and s["coverage_pct"] > 99.0 and s["depth_error_rms_m"] < 1e-9
    assert s["path_length_m"] == pytest.approx(16.0, abs=0.01) and s["mean_speed_mps"] == pytest.approx(1.0, abs=0.01)


def test_negative_control_a_run_that_drifted_has_a_big_error_and_less_coverage():
    drift = [(0.0 + 0.35 * i, y) for i, y in enumerate(np.linspace(0, 10, 21))]
    s = RS.summarize(make_run(drift, z=4.0), PLAN, swath_m=3.0)
    assert s["path_error_max_m"] > 3.5 and s["coverage_pct"] < 60.0 and s["depth_error_rms_m"] == pytest.approx(1.0, abs=0.01)


def test_figure_is_written(tmp_path):
    on = [(0.0, y) for y in np.linspace(0, 10, 21)]
    out = tmp_path / "s.png"
    s = RS.save_figure(make_run(on, cmd={"th_01": np.ones(21) * 500, "cs_04": np.ones(21) * 5}), PLAN, out, 3.0, "unit test")
    assert out.exists() and out.stat().st_size > 5000 and "path_error_rms_m" in s
    assert "duration" in RS.format_summary(s)
