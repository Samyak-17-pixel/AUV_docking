"""Edit a mission in the Config tab, start it from the Controls tab, and watch it fly over REAL ROS (private domain 79): fake vehicle + the mission node + RosLink.
Needs rclpy and the built interfaces package; about 40 s."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from PyQt5 import QtCore, QtWidgets

pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

import controls_panel
from config_panel import ConfigPanel
from controls_panel import ControlsPanel
from process_manager import ProcessManager

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ROOT = Path(__file__).resolve().parents[2]


def pump(seconds, cond=lambda: False):
    t0 = time.time()
    while time.time() - t0 < seconds and not cond():
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.02)


def test_edited_mission_flies_with_the_edited_values(tmp_path, monkeypatch):
    env = dict(os.environ, ROS_DOMAIN_ID="79", ROS_LOCALHOST_ONLY="1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "79")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    cfg_file = tmp_path / "mission.yaml"
    cfg_file.write_text((ROOT / "mission" / "mission.yaml").read_text())
    monkeypatch.setitem(controls_panel.CONTROLLERS["mission"], "config", cfg_file)
    from ros_link import RosLink
    from synthetic_camera import DEFAULT_CONFIG
    from telemetry import Telemetry

    fake = subprocess.Popen([sys.executable, str(ROOT / "sim_offline" / "fake_vehicle.py"), "--realistic", "--odom-freeze-interval", "0", "--z", "3"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vcfg = yaml.safe_load(open(DEFAULT_CONFIG))
    vcfg["node"]["ros_domain_id"] = 79
    tel = Telemetry()
    link = RosLink(tel, vcfg)
    procs = ProcessManager()
    ctl = ControlsPanel(link, procs, confirm=lambda t, m: True)
    ctl._ext_finder = lambda: {}
    panel = ConfigPanel(ctl, tel, confirm=lambda t, m: True)
    try:
        link.start()
        assert link.ready.wait(10.0) and link.error is None
        pump(3.0)
        panel.ctrl.setCurrentText("mission")
        doc = panel.doc
        doc.data["legs"] = [{"type": "goto", "x": 3.0, "y": 0.0, "z": 3.0}, {"type": "hold", "seconds": 2.0}]      # a short mission, edited in the UI model
        doc.data["speed"]["cruise_mps"] = 0.7
        panel._refresh_all()
        panel._after_change()                                                  # what every edit in the tab ends with
        assert "mission" in ctl._edited
        ctl.ctrl.setCurrentText("mission")
        ctl._on_start()
        seen = []

        def done():
            _, c = tel.latest_ctrl()
            if c.get("ctrl") == "mission":
                seen.append(dict(c))
            panel.update_live()
            return not procs.is_running("mission") and bool(seen)

        pump(90.0, done)
        assert seen, "no mission ctrl_debug reached the viewer"
        assert max(float(c.get("u_sp", 0.0)) for c in seen) <= 0.71          # the EDITED cruise speed was flown (the file says 1.0)
        assert max(float(c.get("progress", 0.0)) for c in seen) > 0.5 and seen[-1].get("leg")
        s = tel.latest()
        assert abs(s.pos[0] - 3.0) < 1.0 and abs(s.pos[1]) < 1.0              # ended near the goto point
        assert yaml.safe_load(cfg_file.read_text())["speed"]["cruise_mps"] == 1.0                      # the file was not touched
        assert panel.map.trail and panel.map.vehicle is not None             # the map followed the vehicle
    finally:
        procs.stop_all()
        pump(2.0)
        link.stop()
        link.join(3.0)
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()                                                  # the next ROS test calls rclpy.init() itself
        fake.send_signal(2)
        try:
            fake.wait(8)
        except subprocess.TimeoutExpired:
            fake.kill()
