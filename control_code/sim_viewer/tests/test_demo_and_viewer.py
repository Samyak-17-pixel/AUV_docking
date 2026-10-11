"""Demo data source and a headless run of the whole window (Qt offscreen platform + VTK offscreen on DISPLAY)."""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from demo_source import DemoSource
from synthetic_camera import DEFAULT_CONFIG
from telemetry import Telemetry

VIEWER = Path(__file__).resolve().parents[1] / "app" / "viewer_app.py"


def test_demo_source_flies_towards_the_dock_and_feeds_everything():
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    tel = Telemetry()
    src = DemoSource(tel, cfg, time_scale=6.0)      # 6x: reaches ~5 m in 1.5 s, well before the 7.4 m restart
    src.start()
    src.ready.wait(5.0)
    time.sleep(1.5)
    src.stop()
    src.join(3.0)
    assert src.error is None
    s = tel.latest()
    assert s is not None and s.pos[0] > 0.5 and s.nu[0] > 0.3                # moved forward, with speed
    assert tel.image() is not None and tel.latest_cmd().get("th_01") is not None
    assert "DEMO" in tel.source_name
    assert len(tel.sonar_rows("port")) > 0 and len(tel.sonar_rows("starboard")) > 0      # the demo feeds the Sonar tab too


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="VTK offscreen rendering needs a DISPLAY")
def test_whole_window_runs_headless_and_draws_something(tmp_path):
    shot = tmp_path / "w.png"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, str(VIEWER), "--demo", "--run-seconds", "5", "--time-scale", "4", "--screenshot", str(shot)],
                       capture_output=True, text=True, timeout=120, env=env)
    noise = ("Axes3D", "warnings.warn", "propagateSizeHints")
    err = "\n".join(l for l in r.stderr.splitlines() if not any(n in l for n in noise))
    assert r.returncode == 0, f"viewer exited with code {r.returncode}\n{err[-1500:]}"
    assert shot.exists() and shot.stat().st_size > 30_000
    import cv2
    img = cv2.imread(str(shot))
    assert img is not None and img.shape[0] > 400 and img.shape[1] > 600 and img.std() > 20
    # the vehicle is orange: expect a visible cluster of orange-ish pixels (BGR ~ (25,140,240))
    orange = ((img[:, :, 2] > 180) & (img[:, :, 1] > 80) & (img[:, :, 1] < 200) & (img[:, :, 0] < 90)).sum()
    assert orange > 100
    assert img.shape[1] <= 1500 and img.shape[0] <= 1100                    # the window keeps its configured width (camera pane + status text must not widen it)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a real X display")
def test_window_starts_on_the_real_xcb_platform_not_only_offscreen(tmp_path):
    """Regression: importing cv2 points Qt at opencv's own plugin folder, which cannot start the desktop. Every other test uses 'offscreen' and missed it."""
    shot = tmp_path / "x.png"
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    r = subprocess.run([sys.executable, str(VIEWER), "--demo", "--run-seconds", "3", "--screenshot", str(shot)], capture_output=True, text=True, timeout=120, env=env)
    assert "Could not load the Qt platform plugin" not in r.stderr, r.stderr[-800:]
    assert r.returncode == 0 and shot.exists() and shot.stat().st_size > 30_000


def test_camera_pane_is_fixed_size_hidden_by_default_and_the_camera_lives_in_the_detection_tab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets, QtGui
    src = open(Path(VIEWER)).read()
    assert "self.cam_label.setFixedSize(320, 240)" in src and "self.cam_label.setVisible(False)" in src      # a small optional preview, never sized by the picture
    assert 'self.bottom.addTab(self.detection, "Detection")' in src                                          # the camera view is in the Detection tab
    # negative control: a label that is not pinned DOES grow with a big pixmap
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    free = QtWidgets.QLabel(); free.setPixmap(QtGui.QPixmap(1600, 1200))
    assert free.sizeHint().width() >= 1600
    pinned = QtWidgets.QLabel(); pinned.setFixedSize(320, 240); pinned.setPixmap(QtGui.QPixmap(1600, 1200))
    assert (pinned.minimumSize().width(), pinned.maximumSize().width(), pinned.maximumSize().height()) == (320, 320, 240)
