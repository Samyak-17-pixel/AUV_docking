"""IMU / elevation convention checks. No ROS node needed (pytest, or run as a script).

The IMU roll/pitch (dock_align_msg.quat_to_roll_pitch) and the odometry attitude (control_code/common/state.quat_to_euler) must be the same function,
and the elevation formula must agree with the synthetic camera that is built from the vessel file: a dock at the vehicle's own depth must give ~0
elevation at any pitch, a dock below must give a positive one, and the pitch sign must be the nose-up one. If the REAL IMU is in another convention this
cannot be seen offline: see check_imu_convention.py.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geometry_msgs")        # dock_align_msg imports ROS message types; without ROS these tests are skipped

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_d) for _d in [_HERE.parent / "control_code" / "sim_viewer", *sorted((_HERE.parent / "control_code" / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # sim_viewer and its sub-folders
for p in (_HERE, _HERE.parent / "control_code" / "common", _HERE.parent / "control_code" / "sim_viewer"):
    sys.path.insert(0, str(p))

from check_imu_convention import verdict  # noqa: E402
from dock_align_msg import elevation_down_rad, focal_length_px, quat_to_roll_pitch  # noqa: E402
from state import quat_to_euler  # noqa: E402
from synthetic_camera import SyntheticCamera  # noqa: E402


def _quat_from_euler(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)   # x, y, z, w


def test_imu_and_odometry_conversions_are_the_same():
    rng = random.Random(3)
    for _ in range(200):
        r, p, y = rng.uniform(-1.2, 1.2), rng.uniform(-1.3, 1.3), rng.uniform(-math.pi, math.pi)
        q = _quat_from_euler(r, p, y)
        ra, pa = quat_to_roll_pitch(*q)
        rb, pb, _ = quat_to_euler(*q)
        assert abs(ra - rb) < 1e-12 and abs(pa - pb) < 1e-12
        assert abs(ra - r) < 1e-9 and abs(pa - p) < 1e-9          # and it really is ZYX roll / pitch: nose-up pitch is positive


CAM_X = 0.575          # the nose camera sits 0.575 m ahead of the vehicle origin: a pitched vehicle lifts / lowers the camera


def _true_elevation(pos, pitch_deg, dock):
    """Angle of the dock centre below the horizontal, seen from the camera position (NED: z down)."""
    cam_pos = np.array(pos, float) + CAM_X * np.array([math.cos(math.radians(pitch_deg)), 0.0, -math.sin(math.radians(pitch_deg))])
    d = np.array(dock, float) - cam_pos
    return math.atan2(d[2], math.hypot(d[0], d[1]))


def test_elevation_matches_the_geometric_angle_below_the_horizon_at_any_pitch():
    cam = SyntheticCamera()
    f = focal_length_px(cam.H, 60.0)
    for pitch_deg in (-12.0, -5.0, 0.0, 6.0, 15.0):
        for dock_z in (3.0, 4.0):
            uvz = cam.project([4.0, 0.0, 3.0], [0.0, pitch_deg, 0.0], [10.0, 0.0, dock_z])
            assert uvz is not None
            elev = elevation_down_rad(uvz[1], cam.H / 2.0, f, math.radians(pitch_deg))
            assert abs(elev - _true_elevation([4.0, 0.0, 3.0], pitch_deg, [10.0, 0.0, dock_z])) < 2e-3, (pitch_deg, dock_z, elev)


def test_elevation_is_positive_for_a_dock_deeper_than_the_vehicle():
    cam = SyntheticCamera()
    f = focal_length_px(cam.H, 60.0)
    uvz = cam.project([4.0, 0.0, 3.0], [0.0, 0.0, 0.0], [10.0, 0.0, 4.0])      # 1 m deeper (NED z down), 6 m ahead
    elev = elevation_down_rad(uvz[1], cam.H / 2.0, f, 0.0)
    assert elev > 0.0
    # NEGATIVE CONTROL: with the pitch sign flipped (an ENU / FLU IMU) the same pitched view gives an elevation that is wrong by twice the pitch
    uvz = cam.project([4.0, 0.0, 3.0], [0.0, 10.0, 0.0], [10.0, 0.0, 3.0])
    good = elevation_down_rad(uvz[1], cam.H / 2.0, f, math.radians(10.0))
    bad = elevation_down_rad(uvz[1], cam.H / 2.0, f, math.radians(-10.0))
    assert abs(good - _true_elevation([4.0, 0.0, 3.0], 10.0, [10.0, 0.0, 3.0])) < 2e-3
    assert abs(bad - good) > math.radians(19.0)


def test_verdict_tool_recognises_same_flipped_and_undecided():
    t = np.linspace(0, 6.28, 80)
    odom = 0.1 * np.sin(t)
    assert verdict(list(zip(odom + 0.02, odom)))[0] == "SAME"
    assert verdict(list(zip(-odom, odom)))[0] == "FLIPPED"
    assert verdict(list(zip(odom * 0.0, odom * 0.0)))[0] == "UNDECIDED"          # level vehicle: nothing to compare
    assert verdict(list(zip(0.001 * np.sin(t), 0.001 * np.sin(t))))[0] == "UNDECIDED"   # amplitude under 1.5 deg
