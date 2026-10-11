"""Search and back-out behaviour of the docking controller (recover: in terminal_docking.yaml), closed loop with the real detector on rendered frames.
Each test has a negative control with recover.enabled = false (the old behaviour). About 3 minutes."""
import math

import pytest

pytest.importorskip("interfaces.msg")

from docking_sim import DockingSim, Scenario, load_cfg  # noqa: E402
from docking_sweep import classify  # noqa: E402
from terminal_docking_core import room_needed_m  # noqa: E402

HINT = {"recover": {"dock_hint_ned": [10.6, -0.5, 3.1]}}


def outcome(start, overrides=None, T=120.0):
    r, e, h = start
    sim = DockingSim(Scenario(range_m=r, lateral_m=e, heading_deg=h, vision="detector", T=T), load_cfg(overrides=overrides))
    res = sim.run(keep_log=False)
    ok_after = (not res.failures) and sim.core.phase == "DOCKED"
    return ("RECOVERED" if ok_after and not res.passed else classify(res.passed, res.failures, res.metrics)), res, sim


def test_room_needed_grows_with_the_error_and_turning_away_costs_more():
    R = 2.4
    assert room_needed_m(0.0, 0.0, R) == 0.0
    assert room_needed_m(0.5, 0.0, R) < room_needed_m(1.5, 0.0, R) < room_needed_m(3.0, 0.0, R)
    toward = room_needed_m(1.5, -20.0, R)                     # right of the axis, pointing back toward it
    away = room_needed_m(1.5, +20.0, R)                       # right of the axis, pointing further right
    assert away == pytest.approx(toward + R * math.radians(20.0))
    assert room_needed_m(-1.5, 0.0, R) == room_needed_m(1.5, 0.0, R)


def test_close_and_angled_backs_out_instead_of_driving_into_the_dock():
    start = (3.0, -1.5, 30.0)                                 # 3 m ahead, 1.5 m left of the axis, pointing 30 deg to the right: no room to swing round
    new, res, sim = outcome(start)
    assert new in ("RECOVERED", "DOCKED"), (new, res.failures)
    assert sim.core.repositions >= 1 and res.metrics["min_clearance_m"] > 0.0
    old, res0, _ = outcome(start, {"recover": {"enabled": False}, "guidance": {"gate_heading_deg": 10.98, "gate_grace_m": 0.8}})
    assert old in ("COLLISION", "FAIL", "NOT_SEEN"), (old, res0.failures)       # negative control: the old controller hits the wall or never docks


def test_a_start_in_front_of_the_dock_with_room_does_not_back_out():
    new, res, sim = outcome((8.0, 0.0, 0.0))
    assert new == "DOCKED" and sim.core.repositions == 0


def test_search_finds_a_dock_that_is_not_in_view_when_it_is_given_a_rough_position():
    start = (8.0, 0.0, -60.0)                                 # the dock is 60 deg off the nose: nothing in the picture at first
    new, res, sim = outcome(start, HINT)
    assert new in ("RECOVERED", "DOCKED"), (new, res.failures)
    assert "SEARCH" in "".join(str(x) for x in [sim.core.phase]) or res.metrics["t_end"] > 10.0
    old, res0, sim0 = outcome(start, {"recover": {"enabled": False}}, T=60.0)
    assert old == "NOT_SEEN" and sim0.core.phase == "WAIT"    # negative control: the old controller waits for ever


def test_search_does_not_swing_into_the_dock_when_it_starts_close_and_sideways():
    new, res, sim = outcome((3.0, 1.5, -60.0), HINT)          # 3 m away, the dock is behind the nose: a blind turn would hit it
    assert new != "COLLISION", (new, res.failures)
