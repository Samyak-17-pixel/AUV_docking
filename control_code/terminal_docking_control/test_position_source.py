"""estimator.position_source: odometry vs dead_reckoning (2026-10-11). A constant odometry offset never matters (the dock is triangulated in the same frame); drift and jumps of the
odometry POSITION hurt the odometry mode but not dead reckoning; with no speed information neither can finish (negative control). Perfect vision keeps these runs fast."""
import math

import numpy as np
import pytest

from docking_sim import DockingSim, Scenario, load_cfg
from state import State
from terminal_docking_core import TerminalDockingCore


def run(mode, faults=None, start=(7.0, 0.5, 5.0), T=100.0):
    cfg = load_cfg(overrides={"estimator": {"position_source": mode}})
    sim = DockingSim(Scenario(range_m=start[0], lateral_m=start[1], heading_deg=start[2], vision="perfect", T=T, odom_faults=faults), cfg)
    r = sim.run(keep_log=False)
    return r, sim


def test_dead_reckoner_integrates_a_known_path():
    cfg = load_cfg(overrides={"estimator": {"position_source": "dead_reckoning"}})
    core = TerminalDockingCore(cfg)
    assert core.dr_on
    t = 0.0
    for _ in range(200):                                           # 10 s at 0.5 m/s heading east (yaw 90 deg)
        t += 0.05
        v = core._dr_view(State(pos=np.array([123.0, -45.0, 3.0]), eul=np.array([0.0, 0.0, math.pi / 2]), nu=np.array([0.5, 0, 0, 0, 0, 0])), t)
    assert abs(v.pos[0]) < 0.05 and abs(v.pos[1] - 5.0) < 0.1 and v.pos[2] == 3.0       # the absolute odometry position (123, -45) is never used
    assert np.allclose(core._dr_xy_at(t - 1.0), [0.0, 4.5], atol=0.15)                  # the position at an earlier (image) time


def test_a_constant_odometry_offset_changes_nothing_in_either_mode():
    for mode in ("odometry", "dead_reckoning"):
        a, _ = run(mode)
        b, _ = run(mode, {"pos_offset_m": (3.0, -2.0)})
        assert a.passed and b.passed, (mode, a.failures, b.failures)
        assert abs(a.metrics["cross_lat"] - b.metrics["cross_lat"]) < 0.02


def test_odometry_drift_costs_the_odometry_mode_its_first_try_but_not_dead_reckoning():
    drift = {"pos_drift_mps": (0.15, 0.09)}                         # 0.17 m/s: the published position runs away 10 m per minute
    odo, _ = run("odometry", drift, T=200.0)
    dr, sim = run("dead_reckoning", drift, T=200.0)
    assert not odo.passed, "negative control: with a drifting odometry position the odometry mode needs retries or fails"
    assert dr.passed, dr.failures
    assert sim.core.pos_src == "dead_reckoning"
    far, _ = run("odometry", {"pos_drift_mps": (0.3, 0.18)}, T=200.0)
    assert far.failures, "at 0.35 m/s the odometry mode no longer docks at all"


def test_without_any_speed_information_dead_reckoning_cannot_finish():
    r, _ = run("dead_reckoning", {"drop_twist": True})
    assert not r.passed and r.failures                                   # the last 2 m (lights out of view) need the speed: dead reckoning is not magic
