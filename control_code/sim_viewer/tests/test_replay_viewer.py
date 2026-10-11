"""Replay and overlay in the viewer: telemetry model series, model lines in the plots, ghost vehicle, Replay tab, and the whole window started with --replay."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PyQt5 import QtWidgets

from compare import simulate
from plots import LivePlots
from replay import ReplaySource, load_run
from replay_panel import ReplayPanel
from runs_helper import write_model_run
from synthetic_camera import DEFAULT_CONFIG
from telemetry import Telemetry

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
VIEWER = Path(__file__).resolve().parents[1] / "app" / "viewer_app.py"
needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="VTK offscreen rendering needs a DISPLAY")


def test_telemetry_keeps_model_series_separate_and_clears_them():
    t = Telemetry()
    assert t.pose_now(model=True) is None and t.latest_model() is None
    for i in range(10):
        t.push_odom([i, 0, 3.0], [0, 0, 0], [1.0, 0, 0, 0, 0, 0], t=float(i))
        t.push_model([i + 0.5, 0, 3.2], [0, 0.1, 0], [1.0, 0, 0, 0, 0, 0], t=float(i))
    w = t.window(100.0)
    assert len(w["m_t"]) == 10 and w["m_depth"][0] == pytest.approx(3.2) and w["m_pitch"][0] == pytest.approx(np.degrees(0.1))
    assert t.latest_model().pos[0] == 9.5
    t.clear()
    assert t.latest_model() is None and t.window(10.0) == {}


def test_plots_draw_model_lines_dashed():
    t = Telemetry()
    for i in range(20):
        t.push_odom([i, 0, 3.0], [0, 0, 0], [1, 0, 0, 0, 0, 0], t=float(i))
        t.push_model([i, 0, 3.1], [0, 0.05, 0], [0.9, 0, 0, 0, 0, 0], t=float(i))
    p = LivePlots(window_s=60.0)
    p.update_plots(t.window(60.0))
    for panel, key in ((p.p_depth, "m_depth"), (p.p_att, "m_pitch"), (p.p_speed, "m_u")):
        ln = panel.lines[key]
        assert ln.get_linestyle() == "--" and len(ln.get_xdata()) == 20
    assert len(p.xy_model.get_xdata()) == 20
    p.update_plots(Telemetry().window(10.0))          # no model data and no data at all must not raise
    t2 = Telemetry()
    t2.push_odom([0, 0, 3], [0, 0, 0], [0] * 6, t=0.0)
    p.update_plots(t2.window(10.0))
    assert len(p.p_depth.lines["m_depth"].get_xdata()) == 0


@needs_display
def test_ghost_is_hidden_by_default_follows_set_ghost_and_changes_the_picture():
    from scene3d import Scene3D
    sc = Scene3D(yaml.safe_load(open(DEFAULT_CONFIG)), (400, 300))
    assert not sc.ghost.GetVisibility()
    sc.set_pose([3.0, 0.0, 3.0], [0, 0, 0])
    sc.set_view("free")
    sc.reset_camera_overview()
    a = sc.render().astype(int)
    sc.set_ghost([3.0, 0.6, 3.0], [0, 0, 0])
    assert sc.ghost.GetVisibility()
    m = sc.ghost.GetUserMatrix()
    assert (m.GetElement(0, 3), m.GetElement(1, 3), m.GetElement(2, 3)) == pytest.approx((3.0, 0.6, 3.0))
    b = sc.render().astype(int)
    assert np.abs(a - b).sum() > 20000                 # a second, translucent vehicle is visible
    sc.hide_ghost()
    assert not sc.ghost.GetVisibility()


def test_replay_panel_controls_drive_the_source(tmp_path):
    run = load_run(write_model_run(tmp_path / "r.csv", T=10.0))
    src = ReplaySource(Telemetry(), run, speed=4.0)            # thread not started: the panel only talks to attributes and methods
    panel = ReplayPanel(src)
    assert panel.speed_box.currentText() == "4" and panel.slider.maximum() == 1000
    panel.play_btn.click()
    assert not src.playing and "paused" in panel.time_label.text()
    panel.play_btn.click()
    assert src.playing
    panel.speed_box.setCurrentText("8")
    assert src.speed == 8.0
    panel.loop_box.setChecked(True)
    assert src.loop is True
    seen = []
    panel2 = ReplayPanel(src, on_seek=lambda: seen.append(1))
    panel2.slider.setValue(500)
    panel2._released()
    assert src._seek_to == pytest.approx(5.0) and seen == [1]
    panel2.restart_btn.click()
    assert src._seek_to == 0.0 and len(seen) == 2


@needs_display
def test_whole_window_in_replay_overlay_mode(tmp_path):
    csv = write_model_run(tmp_path / "w.csv", {"net_up_n": 0.8, "drag_quad_X": 9.0}, T=20.0)
    shot = tmp_path / "w.png"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, str(VIEWER), "--replay", str(csv), "--overlay", "--overlay-mode", "free", "--speed", "6", "--run-seconds", "5", "--screenshot", str(shot)],
                       capture_output=True, text=True, timeout=180, env=env)
    noise = ("Axes3D", "warnings.warn", "propagateSizeHints", "XDG_RUNTIME_DIR")
    err = "\n".join(l for l in r.stderr.splitlines() if not any(n in l for n in noise))
    assert r.returncode == 0, f"viewer exited with {r.returncode}\n{err[-1500:]}"
    assert err.strip() == "", err
    import cv2
    img = cv2.imread(str(shot))
    assert img is not None and img.shape[1] > 600 and img.std() > 20


def test_viewer_refuses_unusable_replay_file_with_a_clear_message(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n3,4\n5,6\n")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, str(VIEWER), "--replay", str(bad), "--run-seconds", "1"], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 2 and "not a recorder" in r.stderr
