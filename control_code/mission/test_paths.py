"""Path generators (no ROS, no vehicle): geometry, the skip pattern, reachable turns, the keep-out / geofence checks."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402


def test_lane_order_skips():
    assert P.lane_order(5, 1) == [0, 1, 2, 3, 4]
    assert P.lane_order(5, 2) == [0, 2, 4, 1, 3]
    assert P.lane_order(6, 3) == [0, 3, 1, 4, 2, 5]


def test_lawnmower_geometry_heading_east():
    pts, info = P.lawnmower((0.0, 5.0), 90.0, 10.0, 16.0, 8.0, 3.0)
    assert info["lanes"] == 3 and info["skip"] == 1 and info["warnings"] == []
    xs = sorted({round(p[0], 6) for p in pts if abs(p[1] - 5.0) < 1e-6 or abs(p[1] - 15.0) < 1e-6})
    assert xs == [-16.0, -8.0, 0.0]                                   # lanes go to the RIGHT of the heading: east -> south (decreasing x)
    assert pts[0][:2] == pytest.approx((0.0, 5.0)) and pts[1][:2] == pytest.approx((0.0, 15.0))
    assert all(p[2] == 3.0 for p in pts)


def test_lawnmower_u_turns_bulge_past_the_lane_end_and_are_flyable():
    pts, info = P.lawnmower((0.0, 0.0), 0.0, 10.0, 16.0, 8.0, 3.0)
    assert max(p[0] for p in pts) > 10.0                              # the arc bulges beyond the lane end (heading north)
    assert P.check_path(pts, 2.5, (0.0, 0.0)) == []


def test_tight_spacing_uses_a_skip_and_says_so():
    pts, info = P.lawnmower((0.0, 0.0), 0.0, 10.0, 24.0, 3.0, 3.0)    # 9 lanes, 3 m apart (< 5 m): skip 2
    assert info["skip"] == 2 and info["order"][:3] == [0, 2, 4] and any("skip" in w for w in info["warnings"])
    assert P.check_path(pts, 2.5, (0.0, 0.0)) == []                    # every sideways step is >= 6 m


def test_too_few_lanes_for_a_skip_warns_about_the_one_lane_step():
    _, info = P.lawnmower((0.0, 0.0), 0.0, 10.0, 6.0, 3.0, 3.0)        # 3 lanes: order 0, 2, 1 -> the last step is a single 3 m lane
    assert any("cannot be flown" in w for w in info["warnings"])


def test_negative_control_a_plain_tight_lawnmower_is_flagged_by_check_path():
    plain = [(0, 0, 3), (10, 0, 3), (10, 3, 3), (0, 3, 3), (0, 6, 3), (10, 6, 3)]
    assert P.check_path(plain, 2.5, (0.0, -2.0))                       # 3 m between lanes with square corners: the vehicle would orbit them


def test_orbit_points_lie_on_the_circle_and_close():
    pts, _ = P.orbit((5.0, 5.0), 4.0, 3.0, revs=1.0, points_per_rev=12)
    assert len(pts) == 13
    assert all(math.hypot(p[0] - 5.0, p[1] - 5.0) == pytest.approx(4.0) for p in pts)
    assert pts[0][:2] == pytest.approx(pts[-1][:2])


def test_orbit_smaller_than_the_turning_circle_is_refused():
    with pytest.raises(ValueError):
        P.orbit((0.0, 0.0), 2.0, 3.0)


def test_orbit_direction():
    cw, _ = P.orbit((0.0, 0.0), 4.0, 3.0, clockwise=True, start_deg=0.0)
    ccw, _ = P.orbit((0.0, 0.0), 4.0, 3.0, clockwise=False, start_deg=0.0)
    assert cw[1][1] > 0 > ccw[1][1]                                    # from north of the centre: clockwise goes east first


def test_spiral_radius_grows_by_the_pitch_per_revolution():
    pts, info = P.spiral((0.0, 0.0), 4.0, 10.0, 3.0, pitch_m=3.0)
    r = [math.hypot(p[0], p[1]) for p in pts]
    assert r[0] == pytest.approx(4.0) and r[-1] == pytest.approx(10.0) and info["revs"] == pytest.approx(2.0)
    assert all(b >= a - 1e-9 for a, b in zip(r, r[1:]))
    with pytest.raises(ValueError):
        P.spiral((0.0, 0.0), 1.0, 5.0, 3.0)


def test_yoyo_alternates_depth_and_ends_at_the_end_point():
    pts, _ = P.yoyo((0.0, 0.0), (0.0, 12.0), 2.0, 4.0, cycles=2)
    assert [p[2] for p in pts] == [4.0, 2.0, 4.0, 2.0] and pts[-1][:2] == pytest.approx((0.0, 12.0))
    with pytest.raises(ValueError):
        P.yoyo((0.0, 0.0), (0.0, 12.0), 4.0, 2.0)


def test_check_path_geofence_and_dock_keepout():
    gf = {"x_min": -5.0, "x_max": 8.0}
    msgs = P.check_path([(0, 0, 3), (9, 0, 3)], 2.5, None, False, gf, 3.0)
    assert any("geofence" in m for m in msgs) and any("dock" in m for m in msgs)
    assert P.check_path([(0, 0, 3), (4, 0, 3)], 2.5, None, False, gf, 3.0) == []


def test_swing_warning_for_a_reversal_next_to_the_fence():
    msgs = P.swing_warnings([(0.0, 0.0, 3), (7.0, 0.0, 3), (0.0, 1.0, 3)], 2.5, None, {"x_max": 8.0})
    assert msgs and "sweeps" in msgs[0]
