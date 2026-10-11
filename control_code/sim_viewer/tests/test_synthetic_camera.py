"""Synthetic camera: geometry against analytic cases, and the REAL detector on its frames. No ROS."""
import math

import cv2
import numpy as np
import pytest

from dock_detection_config import detector_kwargs, load_config as load_detector_config
from dock_geometry import evaluate_dock_geometry
from dock_light_mask import bloom_mask_and_cores
from synthetic_camera import SyntheticCamera

CAM_X = 0.575          # camera ahead of the vehicle origin
DOCK_X = 10.0


@pytest.fixture(scope="module")
def cam():
    return SyntheticCamera()


def _pos(range_m, y=0.0, z=3.0):
    """Vehicle position with the camera range_m from the dock plane."""
    return [DOCK_X - CAM_X - range_m, y, z]


def _detect(cam, pos, eul=(0.0, 0.0, 0.0)):
    img = cam.render(pos, list(eul))
    img = cv2.imdecode(np.frombuffer(cam.encode_jpeg(img), np.uint8), cv2.IMREAD_COLOR)
    _mask, cores = bloom_mask_and_cores(img, **detector_kwargs(load_detector_config()))
    return cores, evaluate_dock_geometry(cores)


def test_mounting_convention_matches_the_sim(cam):
    """sensor_orientation [-90,0,90]: optical axis = body +x, image right = body +y (starboard), image up = body -z."""
    assert cam.R_c @ np.array([0.0, 0.0, -1.0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert cam.R_c @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert cam.R_c @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_focal_length_uses_the_vertical_fov(cam):
    assert cam.f == pytest.approx(240.0 / math.tan(math.radians(30.0)), rel=1e-9)   # 415.7 px, not 554


@pytest.mark.parametrize("rng", [2.5, 4.0, 8.0])
def test_projection_of_the_ring_matches_the_pinhole_formula(cam, rng):
    t = cam.truth(_pos(rng), [0, 0, 0])
    for name, sign in (("top", -1.0), ("bottom", +1.0)):
        assert t[name]["u"] == pytest.approx(320.0, abs=1e-6)
        assert t[name]["v"] == pytest.approx(240.0 + sign * cam.f * 1.0 / rng, rel=1e-9)
    sy = cam.f * 0.707 / rng
    assert t["right"]["u"] == pytest.approx(320.0 - sy, rel=1e-6)    # the dock's own 'right' is on the camera's LEFT: it faces us
    assert t["left"]["u"] == pytest.approx(320.0 + sy, rel=1e-6)
    assert t["left"]["v"] == pytest.approx(240.0 - cam.f * 0.707 / rng, rel=1e-6)   # side lights are above the centre


def test_pitch_nose_up_moves_the_dock_down_the_image(cam):
    level = cam.truth(_pos(5.0), [0, 0, 0])["top"]["v"]
    up10 = cam.truth(_pos(5.0), [0, 10, 0])["top"]["v"]
    assert up10 > level + 30.0                                       # nose up -> lights appear lower


def test_yaw_right_moves_the_dock_left_in_the_image(cam):
    centre = cam.truth(_pos(5.0), [0, 0, 0])["top"]["u"]
    right10 = cam.truth(_pos(5.0), [0, 0, 10])["top"]["u"]
    assert right10 < centre - 30.0


def test_sideways_offset_shifts_the_dock_the_right_way(cam):
    """Vehicle 1 m to starboard (+y) of the dock axis: the dock is to its LEFT."""
    assert cam.truth(_pos(5.0, y=1.0), [0, 0, 0])["top"]["u"] < 320.0 - 50.0


@pytest.mark.parametrize("rng", [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
def test_real_detector_finds_all_four_lights_and_the_right_geometry(cam, rng):
    cores, geo = _detect(cam, _pos(rng))
    assert len(cores) == 4 and geo.ok
    assert math.hypot(geo.center[0] - 320.0, geo.center[1] - 240.0) < 3.0
    assert geo.radius_tb == pytest.approx(cam.f * 1.0 / rng, rel=0.03)


def test_dim_side_lights_are_lost_at_long_range(cam):
    """Characterisation of the ASSUMED look (sim_viewer.yaml look.core_gain): only the two bright lights survive at 12 m."""
    cores, geo = _detect(cam, _pos(12.0))
    assert len(cores) <= 3 and not geo.ok


def test_off_centre_pose_is_detected_with_the_right_centre(cam):
    pos = _pos(4.0, y=0.5)
    cores, geo = _detect(cam, pos)
    assert geo.ok
    truth = cam.truth(pos, [0, 0, 0])
    u_exp = 0.5 * (truth["top"]["u"] + truth["bottom"]["u"])
    assert abs(geo.center[0] - u_exp) < 4.0


def test_lights_fade_out_when_seen_from_behind_the_dock(cam):
    behind = [DOCK_X + 3.0, 0.0, 3.0]                                # beyond the funnel, looking the other way is not needed: beam is the point
    t = cam.truth(behind, [0, 0, 180])
    assert all(v["brightness"] < 1e-3 for v in t.values())


def test_beam_pattern_can_be_switched_off():
    cfg = SyntheticCamera().cfg
    cfg["dock"]["use_beam_pattern"] = False
    cam2 = SyntheticCamera(cfg)
    t = cam2.truth([DOCK_X + 3.0, 0.0, 3.0], [0, 0, 180])
    assert all(v["brightness"] > 1e-3 for v in t.values())


def test_render_is_deterministic_with_a_fixed_seed():
    a = SyntheticCamera().render(_pos(4.0), [0, 0, 0])
    b = SyntheticCamera().render(_pos(4.0), [0, 0, 0])
    assert np.array_equal(a, b) and a.shape == (480, 640, 3) and a.dtype == np.uint8


def test_scene_floods_are_optional_extra_lights():
    cfg = SyntheticCamera().cfg
    n0 = len(SyntheticCamera(cfg).lights)
    cfg["dock"]["draw_scene_floods"] = True
    assert len(SyntheticCamera(cfg).lights) == n0 + 2


def test_vehicle_behind_or_beside_the_lights_does_not_crash(cam):
    for pos, eul in (([20.0, 0, 3], [0, 0, 0]), ([10.0, 0, 3], [0, 0, 0]), ([5.0, 0, 3], [0, 0, 180])):
        img = cam.render(pos, eul)
        assert img.shape == (480, 640, 3)
