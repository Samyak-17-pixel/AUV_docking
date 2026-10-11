"""Underwater environment (terrain, waves, particles, shadow), the HUD, minimap and detection overlay."""
import math
import os

import numpy as np
import pytest
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

import hud
from synthetic_camera import DEFAULT_CONFIG
from telemetry import Sample

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="VTK offscreen rendering needs a DISPLAY")


def _cfg(**look):
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    cfg["viewer"]["look3d"].update(look)
    return cfg


@needs_display
def test_environment_builds_a_sea_floor_below_the_vehicle_with_relief_and_is_reproducible():
    from scene3d import Scene3D
    def cfg_t(terrain_seed):                                   # the floor is the SHARED terrain (terrain.seed), the same one the side-scan sonar maps
        cfg = _cfg(seed=3)
        cfg["terrain"]["seed"] = terrain_seed
        return cfg
    a, b = Scene3D(cfg_t(5), (320, 240)), Scene3D(cfg_t(5), (320, 240))
    c = Scene3D(cfg_t(9), (320, 240))
    assert a.env is not None and {"floor", "surface", "shafts", "cones", "particles", "shadow"} <= set(a.env.actors)
    h = [a.env.floor_height(x, y) for x in np.linspace(-10, 25, 12) for y in np.linspace(-15, 15, 9)]
    assert min(h) > 8.0 and max(h) < 13.0 and np.ptp(h) > 0.3                  # below the vehicle (3 m), with real relief
    assert a.env.floor_height(5.0, 1.0) == b.env.floor_height(5.0, 1.0)         # same seed, same terrain
    assert a.env.floor_height(-8.0, 12.0) != c.env.floor_height(-8.0, 12.0)     # other seed, other terrain


@needs_display
def test_waves_move_particles_follow_the_vehicle_and_shadow_sits_under_it():
    from scene3d import Scene3D
    from vtk.util import numpy_support as ns
    sc = Scene3D(_cfg(), (320, 240))
    sc.set_pose([4.0, 1.0, 3.0], [0, 0, 0.5])
    z0 = ns.vtk_to_numpy(sc.env._surf_poly.GetPoints().GetData())[:, 2].copy()
    sc.update_environment(0.5)
    z1 = ns.vtk_to_numpy(sc.env._surf_poly.GetPoints().GetData())[:, 2]
    assert np.abs(z1 - z0).max() > 0.02 and np.abs(z1).max() < 0.5              # the surface moves, gently
    for _ in range(40):                                                          # the vehicle moves 40 m: the particle cloud must stay around it
        sc.set_pose([4.0 + _, 1.0, 3.0], [0, 0, 0.0])
        sc.update_environment(0.2)
    p = ns.vtk_to_numpy(sc.env._pts.GetData())
    assert np.abs(p[:, 0] - sc._pos[0]).max() <= sc.env._pbox[0] / 2 + 1e-3
    m = sc.env.actors["shadow"].GetUserTransform().GetMatrix()
    assert (m.GetElement(0, 3), m.GetElement(1, 3)) == pytest.approx((sc._pos[0], sc._pos[1]), abs=1e-3) and m.GetElement(2, 3) > 8.0


@needs_display
def test_look3d_off_restores_the_plain_scene_and_changes_the_picture():
    from scene3d import Scene3D
    on, off = Scene3D(_cfg(), (400, 300)), Scene3D(_cfg(enabled=False), (400, 300))
    assert off.env is None and off.grid.GetVisibility()
    for s in (on, off):
        s.set_pose([4.0, 0.0, 3.0], [0, 0, 0])
        s.set_view("side")
    a, b = on.render().astype(int), off.render().astype(int)
    assert np.abs(a - b).mean() > 5.0
    on.env.set_visible("floor", False)
    c = on.render().astype(int)
    assert np.abs(a - c).mean() > 2.0                                            # hiding the floor is visible


def test_hud_info_numbers():
    s = Sample(1.0, np.array([6.0, 0.5, 2.5]), np.array([0.0, math.radians(-3.0), math.radians(10.0)]), np.array([0.4, 0, 0, 0, 0, 0]))
    info = hud.hud_info(s, np.array([10.0, 0.0, 3.0]), {"mode": "APPROACH", "ctrl": "terminal_docking", "retries": 1}, {"num_lights": 4, "valid": 1}, {})
    assert info["depth"] == pytest.approx(2.5) and info["heading"] == pytest.approx(10.0) and info["pitch"] == pytest.approx(-3.0)
    assert info["dock_range"] == pytest.approx(math.hypot(4.0, 0.5)) and info["dock_bearing"] == pytest.approx(math.degrees(math.atan2(-0.5, 4.0)) - 10.0, abs=1e-6)
    assert info["lights"] == 4 and info["valid"] and info["mode"] == "APPROACH" and info["retries"] == 1
    assert hud.hud_info(None, np.zeros(3), {}, {}, {}) is None


def _painted(draw, size=(300, 220)):
    img = QtGui.QImage(size[0], size[1], QtGui.QImage.Format_RGB888)
    img.fill(QtGui.QColor(10, 30, 60))
    before = np.frombuffer(img.constBits().asstring(img.byteCount()), np.uint8).copy()
    p = QtGui.QPainter(img)
    draw(p, QtCore.QRect(0, 0, *size))
    p.end()
    after = np.frombuffer(img.constBits().asstring(img.byteCount()), np.uint8)
    return int(np.count_nonzero(after != before))


def test_hud_minimap_and_detection_overlay_draw_something_and_survive_empty_data():
    info = {"depth": 3.0, "heading": 5.0, "pitch": 0.0, "roll": 0.0, "speed": 0.4, "dock_range": 6.0, "dock_bearing": 1.0, "dock_dz": 0.0, "mode": "TERMINAL",
            "ctrl": "terminal_docking", "retries": 0, "lights": 4, "valid": True}
    assert _painted(lambda p, r: hud.draw_hud(p, r, info)) > 2000
    assert _painted(lambda p, r: hud.draw_hud(p, r, None)) == 0
    trail = [np.array([float(i), 0.1 * i, 3.0]) for i in range(30)]
    n = _painted(lambda p, r: hud.draw_minimap(p, QtCore.QRect(10, 10, 150, 150), trail, np.array([5.0, 0.5, 3.0]), 0.1, np.array([10.0, 0.0, 3.0]), ghost_pos=np.array([5.2, 0.4, 3.0])))
    assert n > 3000
    al = {"valid": 1, "num_lights": 4, "radius_px": 80.0, "top_x": 320, "top_y": 200, "bottom_x": 320, "bottom_y": 360, "right_x": 355, "right_y": 250, "left_x": 285, "left_y": 250,
          "center_x": 320, "center_y": 280}
    assert _painted(lambda p, r: hud.draw_detection_overlay(p, r, 640, 480, al)) > 500
    assert _painted(lambda p, r: hud.draw_detection_overlay(p, r, 640, 480, {})) == 0
    # a half-filled dict (e.g. an old replay without pixel positions) must not raise
    _painted(lambda p, r: hud.draw_detection_overlay(p, r, 640, 480, {"valid": 0, "num_lights": 2}))
