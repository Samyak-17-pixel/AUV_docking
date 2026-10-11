"""MissionRunner (no ROS): every leg type flown closed loop against the offline vehicle, cross-track following, the geofence / odometry-loss failsafe, and negative controls."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "common", _HERE.parent / "sim_offline", _HERE.parent / "waypoint_tracking"):
    sys.path.insert(0, str(p))

import paths as P  # noqa: E402
from mission_core import MissionRunner, expand_legs, plan_polyline, validate_mission  # noqa: E402
from state import State  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

CFG = yaml.safe_load((_HERE / "mission.yaml").read_text())


def with_legs(legs, **over):
    c = copy.deepcopy(CFG)
    c["legs"] = legs
    for k, v in over.items():
        c["safety"][k] = v
    return c


def fly(cfg, T=300.0, dt=0.05, pos=(0.0, 0.0, 3.0), eul=(0.0, 0.0, 0.0), push=None, hook=None):
    mr = MissionRunner(cfg)
    m = VehicleModel(pos=pos, eul_deg=eul)
    log = []
    t = 0.0
    while t < T and not mr.done:
        if push is not None:
            m.ext = np.array(push(t, m), dtype=float)
        if hook is not None:
            hook(t, mr, m)
        out, s = mr.update(State.from_model(m), dt)
        m.set_command(out)
        for _ in range(int(round(dt / 0.01))):
            m.step(0.01)
        t += dt
        log.append((t, m.pos[0], m.pos[1], m.pos[2], s["segment"], s.get("cross_track_m", 0.0)))
    return mr, log, t


def test_shipped_mission_completes_all_legs_and_stays_clear_of_the_dock():
    assert validate_mission(CFG) == []
    mr, log, t = fly(CFG)
    assert mr.done and mr.aborted is None, f"stopped at t={t:.0f} s segment {mr.seg_idx}, aborted={mr.aborted}"
    segs = [r[4] for r in log]
    assert set(range(len(mr.segments))) <= set(segs)
    L = np.array(log)
    assert np.hypot(L[:, 1] - 10.0, L[:, 2]).min() > 5.0              # never near the dock at (10, 0) (the keep-out is 6 m)
    assert math.hypot(L[-1, 1], L[-1, 2]) < 1.0                       # ends at home


def test_lawnmower_keeps_to_its_lane_with_cross_track_and_drifts_without_it_under_a_push():
    legs = [{"type": "lawnmower", "origin": [0.0, 5.0], "heading_deg": 90.0, "length_m": 24.0, "width_m": 0.0, "spacing_m": 8.0, "z": 3.0}]
    push = lambda t, m: [0.0, 1.5, 0.0, 0.0, 0.0, 0.0]               # a steady sideways push (a current), 1.5 N on a 20 kg vehicle

    def lane_err(cfg):
        mr, log, _ = fly(cfg, T=100.0, push=push)
        L = np.array(log)
        on = L[(L[:, 2] > 17.0) & (L[:, 2] < 28.0)]                   # the second half of the lane (after the run-in and the settling)
        return np.abs(on[:, 1]).max() if len(on) else 99.0

    with_ct = lane_err(with_legs(legs))
    off = with_legs(legs)
    off["cross_track"]["enabled"] = False
    without_ct = lane_err(off)
    assert with_ct < 1.0, f"lane error {with_ct:.2f} m"
    assert without_ct > 2.0 * with_ct                                  # negative control: aiming at the next point lets the push carry the vehicle off the lane


@pytest.mark.parametrize("leg", [
    {"type": "goto", "x": 3.0, "y": 0.0, "z": 3.0},
    {"type": "orbit", "centre": [0.0, 8.0], "radius_m": 4.0, "revs": 1.0, "clockwise": True, "start_deg": 180.0, "z": 3.0},
    {"type": "spiral", "centre": [0.0, 8.0], "r_start_m": 4.0, "r_end_m": 7.0, "pitch_m": 3.0, "start_deg": 180.0, "z": 3.0},
    {"type": "yoyo", "from": [0.0, 0.0], "to": [0.0, 14.0], "z_top": 2.0, "z_bottom": 4.0, "cycles": 2},
    {"type": "hold", "seconds": 5.0},
])
def test_each_leg_type_completes(leg):
    # curved legs start at a point the vehicle can enter along the circle (south of the centre, heading west); the yo-yo line runs east from the start
    pos, eul = ((-4.0, 14.0, 3.0), (0.0, 0.0, -90.0)) if leg["type"] in ("orbit", "spiral") else ((0.0, 0.0, 3.0), (0.0, 0.0, 90.0) if leg["type"] == "yoyo" else (0.0, 0.0, 0.0))
    mr, log, t = fly(with_legs([leg]), T=250.0, pos=pos, eul=eul)
    assert mr.done and mr.aborted is None, f"{leg['type']}: not done after {t:.0f} s (segment {mr.seg_idx}, aborted={mr.aborted})"


def test_orbit_stays_near_its_circle():
    leg = {"type": "orbit", "centre": [0.0, 8.0], "radius_m": 4.0, "revs": 1.0, "clockwise": True, "start_deg": 180.0, "z": 3.0}
    mr, log, _ = fly(with_legs([leg]), T=200.0, pos=(-4.0, 14.0, 3.0), eul=(0.0, 0.0, -90.0))
    L = np.array(log)
    r = np.hypot(L[:, 1], L[:, 2] - 8.0)
    late = r[len(r) // 3:]
    assert mr.done and abs(late.mean() - 4.0) < 1.0 and late.max() < 6.0


def test_stepped_lawnmower_flies_each_depth():
    leg = {"type": "lawnmower", "origin": [0.0, 5.0], "heading_deg": 90.0, "length_m": 10.0, "width_m": 0.0, "spacing_m": 8.0, "depths": [3.0, 5.0]}
    mr, log, _ = fly(with_legs([leg]), T=250.0)
    L = np.array(log)
    assert mr.done and L[:, 3].max() > 4.7 and len(mr.segments) == 2


def test_geofence_breach_aborts_to_return_home_and_stays_there():
    legs = [{"type": "goto", "x": 12.0, "y": 0.0, "z": 3.0}]
    cfg = with_legs(legs, geofence={"x_min": -10.0, "x_max": 5.0}, dock_keepout_m=0.0)
    mr, log, t = fly(cfg, T=200.0)
    assert mr.aborted and "geofence" in mr.aborted
    L = np.array(log)
    assert L[:, 1].max() < 9.0                                          # turned back well before the dock at x = 10
    assert math.hypot(L[-1, 1], L[-1, 2]) < 2.0                         # and went home


def test_negative_control_without_a_geofence_the_same_mission_flies_on():
    legs = [{"type": "goto", "x": 8.0, "y": 0.0, "z": 3.0}]
    mr, log, _ = fly(with_legs(legs, geofence={}, dock_keepout_m=0.0), T=100.0)
    assert mr.done and mr.aborted is None and np.array(log)[:, 1].max() > 7.0


def test_dock_keepout_aborts():
    legs = [{"type": "goto", "x": 9.0, "y": 0.0, "z": 3.0}]
    mr, log, _ = fly(with_legs(legs, geofence={}, dock_keepout_m=6.0), T=150.0)
    assert mr.aborted and "keep-out" in mr.aborted
    assert np.array(log)[:, 1].max() < 9.0                              # the vehicle needs ~2 turning radii to turn back: the keep-out must be that much bigger than the closest you accept


def test_odometry_loss_aborts_the_mission_when_it_comes_back():
    legs = [{"type": "goto", "x": 3.0, "y": 0.0, "z": 3.0}, {"type": "goto", "x": 3.0, "y": 8.0, "z": 3.0}]

    def hook(t, mr, m):
        if abs(t - 4.0) < 0.03:
            mr.report_odom_loss(40.0)                                   # the node tells the runner the odometry was gone for 40 s

    mr, log, _ = fly(with_legs(legs), T=200.0, hook=hook)
    assert mr.aborted and "odometry" in mr.aborted
    assert math.hypot(log[-1][1], log[-1][2]) < 2.0


def test_short_odometry_loss_does_not_abort():
    legs = [{"type": "goto", "x": 3.0, "y": 0.0, "z": 3.0}]
    mr, _, _ = fly(with_legs(legs), T=100.0, hook=lambda t, mr, m: mr.report_odom_loss(3.0) if abs(t - 4.0) < 0.03 else None)
    assert mr.done and mr.aborted is None


def test_progress_and_eta_report_sensible_numbers():
    mr, log, _ = fly(with_legs([{"type": "goto", "x": 3.0, "y": 0.0, "z": 3.0}]), T=100.0)
    assert mr.done and mr.progress() == 1.0
    mr2 = MissionRunner(with_legs([{"type": "goto", "x": 3.0, "y": 0.0, "z": 3.0}]))
    mr2.update(State.from_model(VehicleModel(pos=(0, 0, 3))), 0.05)
    assert 0.0 <= mr2.progress() < 0.1 and mr2.eta_s() > 2.5 and mr2.debug()["leg"] == "goto"


def test_unreachable_leg_is_reported_by_validation():
    legs = [{"type": "goto", "x": 5.0, "y": 0.0, "z": 3.0}, {"type": "goto", "x": 5.0, "y": 1.0, "z": 3.0}]
    assert any("turning circle" in m for m in validate_mission(with_legs(legs)))
    bad = [{"type": "lawnmower", "origin": [0.0, 5.0]}]
    assert any("missing parameter" in m for m in validate_mission(with_legs(bad)))
    assert any("unknown leg type" in m for m in validate_mission(with_legs([{"type": "dance"}])))


def test_plan_polyline_lists_every_leg():
    pl = plan_polyline(CFG)
    labels = {p[3] for p in pl}
    assert any(l.startswith("lawnmower") for l in labels) and "orbit" in labels and "yoyo" in labels and "return_home" in labels
