import time

import numpy as np
import pytest

from telemetry import Telemetry
from sim_viewer_math import body_to_ned, pose_matrix, rotation_between


def test_window_forward_fills_actuators_and_aligns():
    t = Telemetry()
    t.push_odom([0, 0, 3], [0, 0, 0], [0] * 6, t=0.0)
    t.push_odom([1, 0, 3], [0, 0, 0], [0] * 6, t=1.0)
    t.push_cmd(["th_01", "th_02"], [100.0, -200.0], t=0.2)
    t.push_cmd(["th_01"], [300.0], t=0.8)                      # th_02 absent: keeps its last value
    t.push_align({"error_x_px": 5.0, "valid": 1.0}, t=0.5)
    w = t.window(10.0)
    assert w["t"][-1] == 0.0 and w["t"][0] == pytest.approx(-1.0)
    assert list(w["cmd_th_01"]) == [100.0, 300.0] and list(w["cmd_th_02"]) == [-200.0, -200.0]
    assert w["align_error_x_px"][0] == 5.0 and w["align_t"][0] == pytest.approx(-0.5)


def test_window_is_empty_without_data():
    assert Telemetry().window(5.0) == {}


def test_old_samples_are_trimmed():
    t = Telemetry(keep_s=5.0)
    for i in range(20):
        t.push_odom([i, 0, 3], [0, 0, 0], [0] * 6, t=float(i))
    w = t.window(100.0)
    assert len(w["t"]) <= 7                                      # only the last ~5 s are kept


def test_pose_now_moves_forward_with_body_velocity_but_not_beyond_the_limit():
    t = Telemetry()
    t.push_odom([1.0, 2.0, 3.0], [0, 0, np.pi / 2], [0.5, 0, 0, 0, 0, 0])      # heading East at 0.5 m/s
    time.sleep(0.2)
    pos, _, _ = t.pose_now(extrapolate_s=0.4)
    assert pos[0] == pytest.approx(1.0, abs=0.02) and pos[1] > 2.05          # moved East, not North
    time.sleep(0.6)
    pos2, _, _ = t.pose_now(extrapolate_s=0.4)
    assert pos2[1] <= 2.0 + 0.5 * 0.4 + 1e-6                                   # capped at the extrapolation limit
    assert t.pose_now(extrapolate_s=0.0)[0][1] == 2.0


def test_latest_image_cmd_and_align_roundtrip():
    t = Telemetry()
    assert t.image() is None and t.latest_cmd() == {} and t.latest_align() == (0.0, {})
    t.set_image(b"\xff\xd8abc")
    t.push_cmd(["th_01"], [5.0])
    assert t.image() == b"\xff\xd8abc" and t.latest_cmd() == {"th_01": 5.0}


def test_pose_matrix_and_rotation_helpers():
    R = body_to_ned([0, 0, np.pi / 2])
    assert R @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-9)                # yaw +90: body forward -> East
    M = pose_matrix([1, 2, 3], [0, 0, 0])
    assert M[:3, 3].tolist() == [1, 2, 3] and np.allclose(M[:3, :3], np.eye(3))
    for a, b in (([1, 0, 0], [0, 0, -1]), ([1, 0, 0], [-1, 0, 0]), ([0, 0, 1], [0, 0, 1])):
        a, b = np.array(a, float), np.array(b, float)
        assert rotation_between(a, b) @ a == pytest.approx(b, abs=1e-9)


def test_controller_setpoints_are_forward_filled_and_text_fields_are_kept():
    t = Telemetry()
    t.push_odom([0, 0, 3], [0, 0, 0], [0] * 6, t=0.0)
    t.push_odom([0, 0, 3], [0, 0, 0], [0] * 6, t=4.0)
    t.push_ctrl({"ctrl": "station_keeping", "mode": "hold", "depth_sp": 3.0, "pitch_sp_deg": 0.0}, t=1.0)
    t.push_ctrl({"ctrl": "station_keeping", "mode": "hold", "depth_sp": 4.0}, t=3.0)               # pitch_sp absent: keeps its last value
    w = t.window(10.0)
    assert list(w["ctrl_depth_sp"]) == [3.0, 4.0] and list(w["ctrl_pitch_sp_deg"]) == [0.0, 0.0]
    assert "ctrl_mode" not in w and "ctrl_ctrl" not in w                                          # text is not plotted
    assert w["ctrl_t"][0] == pytest.approx(-3.0) and t.latest_ctrl()[1]["mode"] == "hold"
