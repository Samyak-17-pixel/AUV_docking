"""A replayed run has no camera frames in its CSV, so the replay source RE-RENDERS the nose-camera picture from the recorded pose and runs the real detector on it
(before 2026-10-11 the picture pane stayed empty in a replay)."""
import time
from pathlib import Path

import numpy as np
import pytest

from replay import ReplaySource, load_run
from telemetry import Telemetry

RUN = Path(__file__).resolve().parents[3] / "outputs" / "sim_viewer_runs" / "showcase_straight_in_r8_lp0_hp0.csv"


def _synthetic_run(tmp_path):
    """A short recorded approach written with the recorder format: 8 m out, driving at the dock, no align columns filled."""
    from recording import CsvRecorder
    p = tmp_path / "run.csv"
    rec = CsvRecorder(p, {"controller": "test"})
    for k in range(80):
        rec.add_state(k * 0.1, [2.0 + 0.4 * k * 0.1, 0.0, 3.0], [0.0, 0.0, 0.0], [0.4, 0, 0, 0, 0, 0])
    rec.close()
    return p


def _play(path, seconds=2.0, **kw):
    tel = Telemetry()
    src = ReplaySource(tel, load_run(path), speed=4.0, **kw)
    src.start()
    src.ready.wait(5.0)
    t0 = time.time()
    while time.time() - t0 < seconds and tel.image() is None:
        time.sleep(0.05)
    time.sleep(0.5)
    src.stop()
    src.join(3.0)
    return tel, src


def test_replay_shows_a_camera_picture_with_the_dock_lights(tmp_path):
    tel, src = _play(_synthetic_run(tmp_path))
    assert src.error is None and src.counts["image"] > 0
    import cv2
    img = cv2.imdecode(np.frombuffer(tel.image(), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == (480, 640) and img.max() > 200      # the lights are drawn


def test_replay_runs_the_real_detector_on_that_picture(tmp_path):
    pytest.importorskip("interfaces.msg")
    tel, src = _play(_synthetic_run(tmp_path), seconds=4.0)
    t, al = tel.latest_align()
    assert al and al["valid"] == 1.0 and al["num_lights"] == 4.0 and al["radius_px"] > 20.0    # 8 m from the dock: all four lights found


def test_the_camera_can_be_switched_off(tmp_path):
    tel, src = _play(_synthetic_run(tmp_path), seconds=1.0, camera=False)
    assert tel.image() is None and src.counts["image"] == 0
