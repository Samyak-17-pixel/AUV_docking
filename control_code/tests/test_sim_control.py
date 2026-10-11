"""SimControl (pause / step / reset / push / time scale) and VehicleModel.reset. No ROS."""
import json

import numpy as np
import pytest

from sim_control import MAX_SCALE, MIN_SCALE, SimControl
from vehicle_model import VehicleModel


def test_time_scale_is_clipped_and_scales_the_step():
    c = SimControl()
    assert c.sim_dt(0.01) == pytest.approx(0.01)
    assert c.handle({"cmd": "time_scale", "value": 2.0}) and c.sim_dt(0.01) == pytest.approx(0.02)
    c.handle({"cmd": "time_scale", "value": 1000}); assert c.time_scale == MAX_SCALE
    c.handle({"cmd": "time_scale", "value": 0.0}); assert c.time_scale == MIN_SCALE


def test_pause_stops_time_and_resume_restarts_it():
    c = SimControl()
    c.handle({"cmd": "pause", "value": True})
    assert c.paused and c.sim_dt(0.01) == 0.0
    c.handle({"cmd": "pause", "value": False})
    assert c.sim_dt(0.01) > 0.0


def test_step_advances_exactly_the_requested_time_then_stops():
    c = SimControl()
    c.handle({"cmd": "pause"})
    c.handle({"cmd": "step", "seconds": 0.5})
    total = sum(c.sim_dt(0.01) for _ in range(200))
    assert total == pytest.approx(0.5, abs=1e-9) and c.sim_dt(0.01) == 0.0


def test_push_lasts_its_duration_in_simulated_time_and_is_cleared_by_reset():
    c = SimControl()
    c.handle({"cmd": "push", "wrench": [3, 0, 0, 0, 0, 0.5], "duration": 0.05})
    applied = [c.external_wrench(0.01) for _ in range(8)]
    assert sum(1 for w in applied if w[0] == 3.0) == 5 and applied[-1][0] == 0.0
    c.handle({"cmd": "push", "wrench": [1, 1, 1, 0, 0, 0], "duration": 10.0})
    c.handle({"cmd": "reset"})
    assert not c.external_wrench(0.01).any()


def test_reset_defaults_to_the_start_pose_and_accepts_a_new_one():
    c = SimControl(start_pos=(1, 2, 3), start_eul_deg=(0, 0, 90))
    c.handle({"cmd": "reset"})
    assert c.take_reset() == {"pos": [1.0, 2.0, 3.0], "eul_deg": [0.0, 0.0, 90.0]} and c.take_reset() is None
    c.handle({"cmd": "reset", "pos": [5, 0, 4], "eul_deg": [0, 5, 0]})
    assert c.take_reset()["pos"] == [5.0, 0.0, 4.0]


@pytest.mark.parametrize("bad", ["not json", '{"cmd": "dance"}', '{"cmd": "push", "wrench": [1, 2]}', '{"cmd": "reset", "pos": [1]}', '{}'])
def test_bad_commands_are_rejected_without_crashing(bad):
    c = SimControl()
    assert c.handle(bad) is False and c.last_error
    assert c.handle(json.dumps({"cmd": "pause", "value": False}))        # still usable, error cleared
    assert c.last_error == ""


def test_status_reports_the_state():
    c = SimControl()
    c.handle({"cmd": "pause"}); c.handle({"cmd": "time_scale", "value": 4})
    s = c.status()
    assert s["paused"] is True and s["time_scale"] == 4.0


def test_vehicle_model_reset_clears_motion_and_actuators():
    m = VehicleModel(pos=(0, 0, 3))
    m.set_command({"th_01": 800.0})
    for _ in range(300):
        m.step(0.01)
    assert m.nu[0] > 0.1 and m.pos[0] > 0.1
    m.ext[0] = 5.0
    m.reset([7.0, 0.5, 3.2], [0, 4, 10])
    assert m.pos == pytest.approx([7.0, 0.5, 3.2]) and np.degrees(m.eul) == pytest.approx([0, 4, 10])
    assert not m.nu.any() and not m.ext.any() and all(v == 0.0 for v in m.rpm.values())
