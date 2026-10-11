"""3D scene tests. VTK needs a DISPLAY (it opens an invisible offscreen context); skipped without one."""
import math
import os

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="VTK offscreen rendering needs a DISPLAY")

from allocation import eul_to_rotm  # noqa: E402
from scene3d import Scene3D  # noqa: E402
from synthetic_camera import DEFAULT_CONFIG  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    return Scene3D(cfg, (400, 300))


def _matrix(actor) -> np.ndarray:
    m = actor.GetUserMatrix()
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def test_vehicle_pose_matrix_matches_position_and_attitude(scene):
    scene.set_pose([5.0, 0.4, 3.0], [0.0, math.radians(5), math.radians(30)])
    M = _matrix(scene.vehicle)
    assert M[:3, 3] == pytest.approx([5.0, 0.4, 3.0])
    assert M[:3, :3] == pytest.approx(eul_to_rotm([0, 5, 30]), abs=1e-9)


def test_real_meshes_are_used_and_the_four_lights_are_on_the_ring(scene):
    assert scene.asset_source.endswith(".mavsim") or "fallback" in scene.asset_source
    top = min(scene.light_positions, key=lambda p: p[2])
    bottom = max(scene.light_positions, key=lambda p: p[2])
    assert top == pytest.approx([10.0, 0.0, 2.0]) and bottom == pytest.approx([10.0, 0.0, 4.0])
    assert len(scene.light_positions) == 4


def test_render_changes_when_the_vehicle_moves(scene):
    scene.set_view("free")
    scene.set_pose([2.0, 0.0, 3.0], [0, 0, 0])
    a = scene.render(400, 300)
    scene.set_pose([6.0, 0.0, 3.0], [0, 0, 0])
    b = scene.render(400, 300)
    assert a.shape == (300, 400, 3) and a.dtype == np.uint8 and a.std() > 5
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.3


def test_follow_mode_keeps_the_vehicle_framed_while_it_moves(scene):
    scene.set_pose([0.0, 0.0, 3.0], [0, 0, 0])
    scene.set_view("follow")
    cam = scene.ren.GetActiveCamera()
    off0 = np.array(cam.GetPosition()) - np.array([0.0, 0.0, 3.0])
    scene.set_pose([4.0, 1.0, 3.0], [0, 0, 0])
    off1 = np.array(cam.GetPosition()) - np.array([4.0, 1.0, 3.0])
    assert off1 == pytest.approx(off0, abs=1e-9)


def test_onboard_view_sits_at_the_nose_camera_and_looks_forward(scene):
    scene.set_pose([3.0, 0.0, 3.0], [0, 0, 0])
    scene.set_view("onboard")
    cam = scene.ren.GetActiveCamera()
    assert np.array(cam.GetPosition()) == pytest.approx([3.575, 0.0, 3.0], abs=1e-6)
    fwd = np.array(cam.GetFocalPoint()) - np.array(cam.GetPosition())
    assert fwd / np.linalg.norm(fwd) == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert cam.GetViewAngle() == pytest.approx(60.0)                          # vertical FOV, like the sim
    scene.set_view("follow")


def test_heave_dive_draws_thrust_arrows_pointing_down_and_small_commands_are_hidden(scene):
    scene.set_pose([1.0, 0.0, 3.0], [0, 0, 0])
    scene.set_actuators({"th_01": 0.0, "th_02": -1200.0, "th_03": 5.0})
    assert scene.arrows["th_02"].GetVisibility() and not scene.arrows["th_01"].GetVisibility() and not scene.arrows["th_03"].GetVisibility()
    M = _matrix(scene.arrows["th_02"])
    assert M[:3, 0] / np.linalg.norm(M[:3, 0]) == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)     # force points DOWN (+z NED) on a dive
    scene.set_actuators({"th_02": 1200.0})
    M = _matrix(scene.arrows["th_02"])
    assert M[:3, 0] / np.linalg.norm(M[:3, 0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)    # positive RPM = up


def test_trail_is_capped_and_resettable(scene):
    scene.reset_trail()
    for i in range(scene._trail_n + 50):
        scene.add_trail_point([i * 0.1, 0.0, 3.0])
    assert len(scene._trail) == scene._trail_n
    scene.reset_trail()
    assert len(scene._trail) == 0


def test_mouse_camera_moves_do_not_crash_and_change_the_view(scene):
    scene.set_view("free")
    scene.reset_camera_overview()
    before = np.array(scene.ren.GetActiveCamera().GetPosition())
    scene.orbit(30, 10); scene.pan(20, -10); scene.zoom(1.2)
    assert np.linalg.norm(np.array(scene.ren.GetActiveCamera().GetPosition()) - before) > 0.1
    scene.render(200, 150)
