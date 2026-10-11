"""WaypointTracker (no ROS): closed loop against the offline vehicle, the corner-feasibility check, and a negative control that reproduces the old failure."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "common", _HERE.parent / "sim_offline"):
    sys.path.insert(0, str(p))

from state import State  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402
from waypoint_tracking_core import WaypointTracker, parse_waypoints, unreachable_corners  # noqa: E402

SHIPPED = yaml.safe_load((_HERE / "waypoint_tracking.yaml").read_text())
CFG = copy.deepcopy(SHIPPED)                                                    # the unit tests use the small 5x8 m rectangle at 3 m depth, whatever the shipped list is
CFG["waypoints"] = [{"x": 5.0, "y": 0.0, "z": 3.0}, {"x": 5.0, "y": 8.0, "z": 3.0}, {"x": 0.0, "y": 8.0, "z": 3.0}, {"x": 0.0, "y": 0.0, "z": 3.0}]
CFG["speed"]["cruise_mps"] = 1.5


def _fly(cfg, T=200.0, dt=0.05):
    wt = WaypointTracker(cfg)
    m = VehicleModel(pos=(0.0, 0.0, 3.0))
    log = []
    t = 0.0
    while t < T and not wt.done:
        out, s = wt.update(State.from_model(m), dt)
        m.set_command(out)
        for _ in range(int(round(dt / 0.01))):
            m.step(0.01)
        t += dt
        log.append((t, m.pos[0], m.pos[1], m.pos[2], s.get("wp", -1), math.degrees(m.eul[2])))
    return wt, np.array(log), t


def test_shipped_mission_completes_offline_and_keeps_its_depth():
    wt, log, t = _fly(CFG)
    assert wt.done, f"not finished after {t:.0f} s at waypoint {wt.idx}"
    assert np.abs(log[:, 3] - 3.0).max() < 0.3
    assert log[:, 1].max() < 9.0                                  # never gets near the dock at x = 10 (the 2.5 m turning circle at the first corner reaches ~7.5)


def test_every_waypoint_is_visited_in_order():
    wt, log, _ = _fly(CFG)
    order = [int(w) for w in log[:, 4] if w >= 0]
    assert order == sorted(order) and set(order) == {0, 1, 2, 3}


def test_negative_control_the_old_fin_sign_makes_the_heading_loop_positive_feedback():
    """The old waypoint_tracking.yaml mixed the yaw command into the fins as [1, 1, -1, -1] (cs_04, cs_06, cs_07, cs_08) with its own PID on fin degrees. The allocator's
    mix for a positive (to starboard) yaw moment is the opposite. Driving the allocator's fins with the OLD sign must NOT turn the vehicle the right way."""
    wt = WaypointTracker(CFG)
    m = VehicleModel(pos=(0.0, 0.0, 3.0))
    m.nu[0] = 1.0                                                 # moving at 1 m/s so the fins have flow
    yaw_rate_good, yaw_rate_old = [], []
    for sign_flip in (False, True):
        mm = VehicleModel(pos=(0.0, 0.0, 3.0))
        mm.nu[0] = 1.0
        for _ in range(60):
            out = wt.alloc.fins_from_moments(0.0, 0.0, 1.0, 1.0)  # +1 N*m yaw to starboard at 1 m/s
            if sign_flip:
                old = dict(zip(wt.alloc.fin_ids, [1.0, 1.0, -1.0, -1.0]))
                out = {k: old[k] * abs(v) for k, v in out.items()}      # the old mix at the same magnitude
            mm.set_command(out)
            for _ in range(10):
                mm.step(0.01)
        (yaw_rate_old if sign_flip else yaw_rate_good).append(mm.nu[5])
    assert yaw_rate_good[0] > 0.0                                 # the allocator's mix turns to starboard
    assert yaw_rate_old[0] < 0.0                                  # the old signs turn the other way: wrong, with the current fin mapping


def test_unreachable_corners_are_found():
    old_default = [(5, 0, 3), (5, 3, 3), (0, 3, 3), (0, 0, 3)]
    bad = unreachable_corners(old_default, 2.5, start_xy=(0.0, 0.0))
    assert len(bad) == 2 and "0 -> 1" in bad[0] and "2 -> 3" in bad[1]
    assert unreachable_corners([(5, 0, 3), (5, 8, 3), (0, 8, 3), (0, 0, 3)], 2.5, start_xy=(0.0, 0.0)) == []


def test_an_unreachable_corner_really_orbits():
    cfg = copy.deepcopy(CFG)
    cfg["waypoints"] = [{"x": 5.0, "y": 0.0, "z": 3.0}, {"x": 5.0, "y": 3.0, "z": 3.0}]
    cfg["mission"]["pass_radius_m"] = 0.0                        # the pass-by rule would end the orbit: switched off to show the physics
    wt, log, t = _fly(cfg, T=120.0)
    assert not wt.done and wt.idx == 1                           # still on the second waypoint after 120 s: circling it


def test_target_speed_keeps_flow_and_stops_only_on_the_last_waypoint():
    wt = WaypointTracker(CFG)
    flow = CFG["speed"]["flow_mps"]
    assert wt.target_speed(10.0, 0.0, last=False) == pytest.approx(CFG["speed"]["cruise_mps"])
    assert wt.target_speed(0.6, 0.0, last=False) >= flow - 1e-9                 # intermediate waypoint: never below the flow speed
    assert wt.target_speed(0.5, 0.0, last=True) < 0.1                            # on the last one inside the acceptance radius it stops
    assert wt.target_speed(10.0, math.radians(90), last=False) == pytest.approx(flow)   # far off heading: turn at the flow speed, do not stop


def test_parse_waypoints_accepts_dicts_and_lists_and_fills_the_hold_depth():
    assert parse_waypoints([{"x": 1, "y": 2}, [3, 4], [5, 6, 7]], 5.0) == [(1.0, 2.0, 5.0), (3.0, 4.0, 5.0), (5.0, 6.0, 7.0)]
    with pytest.raises(ValueError):
        parse_waypoints([42], 5.0)


def test_unreachable_corners_forgives_the_acceptance_radius_when_asked():
    tight = [(0, 0, 3), (5, 0, 3), (7.2, 2.5, 3)]                     # 2.2 m from the turn centre: inside the 2.5 m circle by 0.3 m only
    assert unreachable_corners(tight, 2.5, start_xy=(0.0, 0.0))        # exact: flagged
    assert unreachable_corners(tight, 2.5, start_xy=(0.0, 0.0), tol_m=0.5) == []        # forgiving the acceptance radius: not flagged


def test_set_path_swaps_the_path_without_touching_the_loops():
    wt = WaypointTracker(CFG)
    wt.loops.heave(State.from_model(VehicleModel(pos=(0.0, 0.0, 3.0))), 3.0, 0.05)
    lp_before = dict(wt.loops._lp)
    wt.set_path([(3.0, 0.0, 3.0)], prev=(0.0, 0.0), acc=1.0, cruise=0.7)
    assert wt.waypoints == [(3.0, 0.0, 3.0)] and wt.idx == 0 and not wt.done and wt.acc == 1.0 and wt.cruise == 0.7
    assert wt.loops._lp == lp_before                                    # filters keep their state: no jump at a leg change
    wt.set_path([])
    assert wt.done


def test_line_mode_holds_the_line_under_a_sideways_push_and_bearing_mode_does_not():
    cfg = copy.deepcopy(CFG)
    cfg["gains"]["cross_track"] = {"kp": 0.05, "ki": 0.01, "kd": 0.0, "i_max": 0.2, "max_heading_deg": 15.0}
    cfg["speed"]["lookahead_m"] = 4.0
    cfg["speed"]["cruise_mps"] = 0.8                              # slow enough that the 24 m leg lasts beyond the 20 s transient

    def fly_line(line):
        wt = WaypointTracker(cfg, allow_empty=True)
        wt.set_path([(0.0, 24.0, 3.0)], prev=(0.0, 0.0), line=line)
        m = VehicleModel(pos=(0.0, 0.0, 3.0), eul_deg=(0.0, 0.0, 90.0))
        worst, t = 0.0, 0.0
        while t < 60.0 and not wt.done:
            out, _ = wt.update(State.from_model(m), 0.05)
            m.set_command(out)
            m.ext = np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0])           # a steady sideways push
            for _ in range(5):
                m.step(0.01)
            t += 0.05
            if t > 20.0:
                worst = max(worst, abs(m.pos[0]))
        return worst

    on, off = fly_line(True), fly_line(False)
    assert on < 1.0 and off > 2.0 * on


def test_pass_by_counts_a_waypoint_as_reached_when_the_vehicle_passes_it_close_by():
    wt = WaypointTracker(CFG, allow_empty=True)
    wt.set_path([(0.0, 10.0, 3.0), (0.0, 20.0, 3.0)], prev=(0.0, 0.0), acc=0.5, pass_by=True)
    m = VehicleModel(pos=(1.4, 10.5, 3.0))                              # 1.4 m beside the line, 0.5 m past the first waypoint
    assert wt.reached(State.from_model(m))
    wt.set_path([(0.0, 10.0, 3.0)], prev=(0.0, 0.0), acc=0.5, pass_by=False)
    assert not wt.reached(State.from_model(m))                          # without the rule it would circle the point


def test_shipped_waypoint_list_with_depth_steps_completes_and_the_old_depth_gate_orbits():
    """The user's spin: the shipped 8-point list has 4-8 m depth steps between waypoints. With the old rule (the depth must also be within 0.25 m) the vehicle flew past a
    point with the gate shut and ORBITED it for ever; now a waypoint is reached on the horizontal distance (or a close pass) and the depth settles on the way."""
    wt, log, t = _fly(copy.deepcopy(SHIPPED), T=450.0)
    assert wt.done, f"shipped list not finished after {t:.0f} s at waypoint {wt.idx}"
    old = copy.deepcopy(SHIPPED)
    old["mission"].update(depth_gate=True, pass_radius_m=0.0)                   # negative control: the old rule ...
    old["speed"]["cruise_mps"] = 10.0                                            # ... with the old cruise setting (the user's file)
    wt2, log2, t2 = _fly(old, T=450.0)
    assert not wt2.done                                                          # it circles one waypoint and never finishes


def test_close_pass_counts_as_reached_but_a_far_miss_does_not():
    wt = WaypointTracker(CFG, allow_empty=True)
    wt.set_path([(0.0, 10.0, 3.0)], acc=0.5)
    wt.pass_radius_m = 3.0
    east = math.pi / 2                                                          # heading east: +y is forward
    st = lambda x, y, yaw=east: State.from_model(VehicleModel(pos=(x, y, 3.0), eul_deg=(0.0, 0.0, math.degrees(yaw))))
    assert not wt.reached(st(1.5, 8.0))                                         # still approaching: the waypoint is ahead
    assert not wt.reached(st(1.5, 10.1))                                        # abeam (0.1 m behind): the 0.3 m margin keeps estimator jitter from ending the leg early
    assert wt.reached(st(1.5, 11.5)) and wt.warnings                            # 1.5 m past it and 1.5 m beside: flew past
    assert not wt.reached(st(6.0, 14.0))                                        # a miss by 7 m is NOT accepted
