"""The viewer's Detection tab: it draws the detector's view from the viewer's telemetry (live or replay), updates, saves pictures, and the dock_hud converter is right."""
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt5 import QtWidgets

from telemetry import Telemetry

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _push_scene(tel, valid=True, ex=30.0, ey=-12.0, radius=70.0, roll=0.0):
    from synthetic_camera import SyntheticCamera
    cam = SyntheticCamera()
    tel.push_odom(np.array([4.0, 0.0, 3.0]), np.array([roll, 0.0, 0.0]), np.zeros(6))
    tel.set_image(cam.encode_jpeg(cam.render([4.0, 0.0, 3.0], [math.degrees(roll), 0.0, 0.0])))
    a = {"valid": 1.0 if valid else 0.0, "num_lights": 4.0 if valid else 0.0, "error_x_px": ex, "error_y_px": ey, "radius_px": radius, "elevation_deg": 1.5, "confidence": 0.9, "aligned": 1.0}
    if valid:
        cx, cy = 320 + ex, 240 + ey
        a.update({"top_x": cx, "top_y": cy - radius, "bottom_x": cx, "bottom_y": cy + radius, "left_x": cx - 0.7071 * radius, "left_y": cy - 0.7071 * radius,
                  "right_x": cx + 0.7071 * radius, "right_y": cy - 0.7071 * radius, "center_x": cx, "center_y": cy})
    tel.push_align(a)


def test_converter_matches_the_detector_conventions():
    sys_path = Path(__file__).resolve().parents[3] / "dock_detection_algo"
    import sys
    sys.path.insert(0, str(sys_path))
    import dock_hud as H
    info = H.info_from_align_dict({"valid": 1.0, "num_lights": 4.0, "error_x_px": 40.0, "error_y_px": 0.0, "radius_px": 83.14, "top_x": 360, "top_y": 157, "bottom_x": 360, "bottom_y": 323,
                                   "left_x": 301, "left_y": 181, "right_x": 419, "right_y": 181, "center_x": 360, "center_y": 240, "confidence": 0.8})
    assert info["valid"] and info["num"] == 4 and info["pts"]["top"] == (360.0, 157.0) and info["center"] == (360.0, 240.0)
    assert abs(H.range_m(info) - 5.0) < 0.01 and abs(H.bearing_deg(info) - math.degrees(math.atan(40 / 415.69))) < 0.01
    lost = H.info_from_align_dict({"valid": 0.0, "num_lights": 2.0})
    assert not lost["valid"] and lost["pts"]["top"] is None and lost["center"] is None


def test_panel_draws_live_telemetry_and_saves_pictures(tmp_path):
    from detection_panel import DetectionPanel
    tel = Telemetry()
    panel = DetectionPanel(tel)
    panel.resize(1000, 420)
    panel.show()
    app.processEvents()
    _push_scene(tel, True, ex=30.0)
    panel.refresh()
    app.processEvents()
    assert panel.view.pixmap() is not None and not panel.view.pixmap().isNull() and panel._frames >= 1
    big = cv2.imread(str(panel.render_to_png(tmp_path / "big.png", big=True)))
    small = cv2.imread(str(panel.render_to_png(tmp_path / "small.png")))
    assert big.shape == (736, 1120, 3) and small.shape == (400, 1000, 3)
    # LOCKED banner dot (green) when valid, red when no lights: the panel follows the telemetry
    import sys
    import dock_hud as H
    assert tuple(small[26, 26]) == H.GOOD or tuple(small[24, 28]) == H.GOOD or (small[10:40, 10:60] == np.array(H.GOOD)).all(axis=2).any()
    _push_scene(tel, False)
    lost = cv2.imread(str(panel.render_to_png(tmp_path / "lost.png")))
    assert (lost[10:40, 10:60] == np.array(H.BAD)).all(axis=2).any()
    panel.timer.stop()


def test_steering_lamps_follow_the_error_sign(tmp_path):
    from detection_panel import DetectionPanel
    import dock_hud as H
    tel = Telemetry()
    panel = DetectionPanel(tel)
    _push_scene(tel, True, ex=60.0, ey=-40.0)
    img = cv2.imread(str(panel.render_to_png(tmp_path / "r.png")))
    warn = (img == np.array(H.WARN)).all(axis=2)
    h, w = warn.shape
    right_half_top = warn[50:90, int(w * 0.72):]
    assert right_half_top.any()                       # a lit lamp exists in the lamp block
    panel.timer.stop()


def test_tab_exists_in_the_viewer_window_and_shows_replay_detection():
    pytest.importorskip("interfaces.msg")
    from replay import ReplaySource, load_run
    from recording import CsvRecorder
    import tempfile
    root = Path(__file__).resolve().parents[3] / "outputs" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    p = root / "det_tab_run.csv"
    rec = CsvRecorder(p, {"controller": "test"})
    for k in range(60):
        rec.add_state(k * 0.1, [2.0 + 0.4 * k * 0.1, 0.0, 3.0], [0.0, 0.0, 0.0], [0.4, 0, 0, 0, 0, 0])
    rec.close()
    from detection_panel import DetectionPanel
    tel = Telemetry()
    src = ReplaySource(tel, load_run(p), speed=3.0)
    src.start()
    src.ready.wait(5.0)
    panel = DetectionPanel(tel)
    t0 = time.time()
    while time.time() - t0 < 4.0 and not tel.latest_align()[1].get("valid"):
        time.sleep(0.05)
    src.stop()
    src.join(3.0)
    bgr, info = panel.snapshot()
    assert info["valid"] and info["num"] == 4 and info["confidence"] > 0.3 and bgr is not None      # the replay's detector output carries the extra fields now
    panel.timer.stop()
