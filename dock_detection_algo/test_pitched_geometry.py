"""Project the known 1 m dock circle and check pitched-view cues.

No ROS and no simulator. Run from this folder:

    python3 test_pitched_geometry.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dock_acquire import PartialAcquire
from dock_align_msg import elevation_down_rad, focal_length_px
from dock_geometry import evaluate_dock_geometry, unrotate_cores

# Dock body, z down. Top and bottom are the vertical diameter. Side lights are
# both above the center, at ±45° from the top, on a 1 m radius circle.
_S = math.sqrt(2.0) / 2.0
LIGHTS = {
    "T": np.array([0.0, 0.0, -1.0]),
    "B": np.array([0.0, 0.0, 1.0]),
    "L": np.array([0.0, -_S, -_S]),
    "R": np.array([0.0, _S, -_S]),
}

W, H = 640, 480
HFOV = math.radians(60.0)
FX = (W / 2.0) / math.tan(HFOV / 2.0)


def _camera_axes(yaw_deg: float, pitch_deg: float, roll_deg: float):
    """Camera axes in the dock frame. +pitch is nose up, +yaw looks toward +Y."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    forward = np.array(
        [-math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)]
    )
    right0 = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    down0 = np.cross(right0, forward)
    c, s = math.cos(roll), math.sin(roll)
    right = c * right0 + s * down0
    down = -s * right0 + c * down0
    return right, down, forward


def project(cam, yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0):
    right, down, forward = _camera_axes(yaw_deg, pitch_deg, roll_deg)
    out = []
    order = ("T", "B", "L", "R")
    for name in order:
        v = LIGHTS[name] - cam
        zc = float(np.dot(v, forward))
        if zc <= 0.05:
            raise AssertionError(f"{name} is behind the camera")
        u = FX * float(np.dot(v, right)) / zc + W / 2.0
        img_v = FX * float(np.dot(v, down)) / zc + H / 2.0
        out.append((u, img_v))
    return out


def _geo(cam, yaw=0.0, pitch=0.0, roll=0.0):
    cores = project(np.asarray(cam, dtype=float), yaw, pitch, roll)
    if abs(roll) > 1e-9:
        cores = unrotate_cores(cores, math.radians(roll), W / 2.0, H / 2.0)
    geo = evaluate_dock_geometry(cores)
    assert geo.ok, geo.message
    return geo


def test_level_centered_is_a_circle():
    geo = _geo([8.0, 0.0, 0.0])
    assert geo.center_exact
    assert abs(geo.lateral_px) < 1.0
    assert abs(geo.obliqueness) < 0.01
    assert geo.spread < 2.0
    assert abs(geo.center[0] - W / 2.0) < 2.0
    assert abs(geo.center[1] - H / 2.0) < 2.0


def test_level_but_deep_keeps_zero_obliqueness():
    geo = _geo([8.0, 0.0, 2.0])
    assert geo.center_exact
    assert abs(geo.obliqueness) < 0.01
    assert geo.spread < 2.0
    # Dock is above the camera, so the center is above the image center (smaller y).
    assert geo.center[1] < H / 2.0 - 20.0
    fy = focal_length_px(W, 60.0)
    elev = elevation_down_rad(geo.center[1], H / 2.0, fy, 0.0)
    assert elev < 0.0


def test_pitched_at_center_moves_off_the_diameter_midpoint():
    pitch = math.degrees(math.atan2(2.0, 2.0))
    geo = _geo([2.0, 0.0, 2.0], pitch=pitch)
    assert geo.center_exact
    assert abs(geo.lateral_px) < 1.5
    assert geo.obliqueness < -0.02
    assert geo.spread > 8.0
    # Cross-ratio center stays near the principal point; the diameter midpoint does not.
    assert abs(geo.center[1] - H / 2.0) < 8.0
    assert abs(geo.diameter_mid[1] - H / 2.0) > 15.0


def test_level_offset_shows_up_as_image_shift():
    # Heading still perpendicular: the side midpoint stays on the diameter.
    # The whole pattern shifts sideways. That pixel shift is the cross-track cue,
    # because a pure sway does not exist on this vehicle.
    geo = _geo([8.0, 1.5, 0.0])
    assert abs(geo.lateral_px) < 1.0
    assert geo.center_exact
    assert abs(geo.center[0] - W / 2.0) > 40.0
    assert geo.spread < 2.0


def test_yaw_pulls_the_side_midpoint_off_the_diameter():
    # A level yaw keeps the top–bottom line vertical. It does not show up as
    # diameter_angle. Close in, the nearer side light pulls the side midpoint
    # off that line, and the dock center leaves the middle of the image.
    geo = _geo([3.0, 0.0, 0.0], yaw=15.0)
    dx = geo.bottom[0] - geo.top[0]
    dy = geo.bottom[1] - geo.top[1]
    angle = math.degrees(math.atan2(dx, dy))
    assert abs(angle) < 2.0
    assert abs(geo.lateral_px) > 8.0
    assert abs(geo.center[0] - W / 2.0) > 40.0


def test_roll_is_removed_before_the_diameter_angle():
    cores = project(np.array([8.0, 0.0, 0.0]), 0.0, 0.0, 12.0)
    raw = evaluate_dock_geometry(cores)
    assert raw.ok
    dx = raw.bottom[0] - raw.top[0]
    dy = raw.bottom[1] - raw.top[1]
    assert abs(math.degrees(math.atan2(dx, dy))) > 5.0
    leveled = unrotate_cores(cores, math.radians(12.0), W / 2.0, H / 2.0)
    geo = evaluate_dock_geometry(leveled)
    assert geo.ok and geo.center_exact
    dx = geo.bottom[0] - geo.top[0]
    dy = geo.bottom[1] - geo.top[1]
    assert abs(math.degrees(math.atan2(dx, dy))) < 1.5


def test_edge_search_yaws_toward_the_light():
    acq = PartialAcquire()
    cmd = acq.update([(20.0, 240.0)], W, H, t_s=0.0)
    assert cmd.status == "search_edge"
    assert cmd.yaw_norm < 0.0
    assert cmd.surge_norm > 0.0
    assert cmd.pitch_norm == 0.0


def test_interior_pair_nods_without_heave():
    acq = PartialAcquire(nod_period_s=8.0)
    cmd = acq.update([(300.0, 200.0), (340.0, 260.0)], W, H, t_s=2.0)
    assert cmd.status == "search_nod"
    assert cmd.surge_norm > 0.0
    assert abs(cmd.pitch_norm) > 0.05 or abs(cmd.yaw_norm) > 0.05


def test_lights_on_top_and_bottom_edges_reverse():
    acq = PartialAcquire(edge_frac=0.10)
    cmd = acq.update([(320.0, 10.0), (300.0, 470.0), (360.0, 240.0)], W, H, t_s=1.0)
    assert cmd.status == "backup_taller_than_frame"
    assert cmd.surge_norm < 0.0


def test_close_cluster_reverses_after_a_failed_nod():
    acq = PartialAcquire(backup_radius_frac=0.22, backup_after_s=8.0, nod_period_s=8.0)
    first = acq.update([(200.0, 120.0), (440.0, 360.0)], W, H, t_s=0.0)
    assert first.status == "search_nod"
    later = acq.update([(200.0, 120.0), (440.0, 360.0)], W, H, t_s=9.0)
    assert later.status == "backup_close"
    assert later.surge_norm < 0.0


def main() -> None:
    tests = [
        test_level_centered_is_a_circle,
        test_level_but_deep_keeps_zero_obliqueness,
        test_pitched_at_center_moves_off_the_diameter_midpoint,
        test_level_offset_shows_up_as_image_shift,
        test_yaw_pulls_the_side_midpoint_off_the_diameter,
        test_roll_is_removed_before_the_diameter_angle,
        test_edge_search_yaws_toward_the_light,
        test_interior_pair_nods_without_heave,
        test_lights_on_top_and_bottom_edges_reverse,
        test_close_cluster_reverses_after_a_failed_nod,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — report every failing case
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
