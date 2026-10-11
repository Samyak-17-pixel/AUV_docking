"""The viewer's Sonar tab: waterfall and mosaic from telemetry, the command box, saved pictures (into a temp folder)."""
import numpy as np
import pytest
from PyQt5 import QtWidgets

import sonar_panel
from synthetic_sidescan import SideScanConfig, SideScanSim
from telemetry import Telemetry
from terrain import Terrain

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
FLAT = {"size_m": 120.0, "dx_m": 0.25, "boulders": 0, "trenches": 0, "relief_m": 0.0, "ripples_m": 0.0, "center_m": [0.0, 0.0], "dock_xy_m": [0.0, 0.0], "calm_radius_m": 1000.0,
        "dock_floor_depth_m": 10.0, "calm_slope": [0.0, 0.0], "objects": False}


class FakeSource:
    def __init__(self):
        self.sent = []

    def send_sonar_command(self, d):
        self.sent.append(d)
        return True


def fill(tel, n=120):
    sim = SideScanSim(Terrain(FLAT), SideScanConfig(gain_index=3))
    for k in range(n):
        pos = [-20.0 + 0.1 * k, 0.0, 3.0]
        for side in ("port", "starboard"):
            r = sim.ping(pos, [0.0, 0.0, 0.0], side)
            tel.push_sonar(side, r.intensity, {"start_range_m": 0.0, "range_m": 30.0, "gain_db": 0.0, "gain_index": 3, "altitude_m": r.altitude_m, "pos": pos, "eul": [0.0, 0.0, 0.0]})


def test_waterfall_and_mosaic_draw_from_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(sonar_panel, "out_dir", lambda name: tmp_path / name)
    tel = Telemetry()
    src = FakeSource()
    panel = sonar_panel.SonarPanel(tel, src, (-60.0, 60.0, -60.0, 60.0))
    panel.resize(900, 420)
    panel.show()
    app.processEvents()
    fill(tel)
    panel.tabs.setCurrentWidget(panel.wf)
    panel.refresh()
    assert panel.wf.pixmap() is not None and not panel.wf.pixmap().isNull()
    panel.tabs.setCurrentWidget(panel.mo)
    panel.refresh()
    assert panel.mosaic.pings >= 200 and panel.mo.pixmap() is not None and "mosaic coverage" in panel.note.text()
    panel.tabs.setCurrentWidget(panel.split)                                    # the default Overview: a narrow waterfall and a mosaic that gets the larger share of the width
    panel.refresh()
    assert not panel.wf2.pixmap().isNull() and not panel.mo2.pixmap().isNull()
    assert panel.split.sizes()[1] > 2 * panel.split.sizes()[0]
    panel.pop_out()
    panel.refresh()
    assert panel.big.isVisible() and not panel.big_label.pixmap().isNull()      # the pop-out window shows the live mosaic too
    panel.big.close()
    got = []
    panel.enlarge.connect(got.append)
    panel.enlarge_btn.setChecked(True)
    assert got == [True]
    panel.save_pictures()
    assert list((tmp_path / "sim_viewer_shots").glob("waterfall_*.png")) and list((tmp_path / "sim_viewer_shots").glob("mosaic_*.png"))
    panel.timer.stop()


def test_range_and_gain_go_to_the_sonar_and_a_replay_cannot_command_it():
    tel = Telemetry()
    src = FakeSource()
    panel = sonar_panel.SonarPanel(tel, src)
    panel.range_box.setValue(45.0)
    panel.gain_box.setCurrentText("5")
    panel.bins_box.setValue(800)
    panel.apply()
    assert src.sent == [{"range_m": 45.0, "gain": 5, "bins": 800}]
    panel.gain_box.setCurrentText("auto")
    panel.apply()
    assert src.sent[-1]["gain"] == -1
    p2 = sonar_panel.SonarPanel(tel, object())                  # a source without send_sonar_command (a replay)
    p2.apply()
    assert "cannot command" in p2.note.text()
    panel.timer.stop()
    p2.timer.stop()
