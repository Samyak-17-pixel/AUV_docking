"""station_keeping publishes its setpoints on /ctrl_debug and RosLink delivers them to Telemetry. Needs rclpy and the built interfaces package."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

ROOT = Path(__file__).resolve().parents[2]


def test_station_keeping_setpoints_reach_the_viewer():
    env = dict(os.environ, ROS_DOMAIN_ID="76", ROS_LOCALHOST_ONLY="1")
    os.environ.update(ROS_DOMAIN_ID="76", ROS_LOCALHOST_ONLY="1")
    from ros_link import RosLink
    from synthetic_camera import DEFAULT_CONFIG
    from telemetry import Telemetry

    procs = [subprocess.Popen([sys.executable, str(ROOT / "sim_offline" / "fake_vehicle.py"), "--z", "3.4"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    cfg["node"]["ros_domain_id"] = 76
    tel = Telemetry()
    link = RosLink(tel, cfg)
    try:
        link.start()
        assert link.ready.wait(10.0) and link.error is None
        time.sleep(2.0)
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "station_keeping" / "station_keeping.py")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        deadline = time.time() + 25.0
        while time.time() < deadline and "depth_sp" not in tel.latest_ctrl()[1]:
            time.sleep(0.3)
        _, ctl = tel.latest_ctrl()
        assert ctl.get("ctrl") == "station_keeping" and ctl.get("mode") == "hold"
        assert ctl["depth_sp"] == pytest.approx(3.4, abs=0.1)                       # it captured where the vehicle was
        assert "ctrl_depth_sp" in tel.window(60.0)
    finally:
        link.stop()
        link.join(3.0)
        for p in reversed(procs):
            p.send_signal(2)
        for p in procs:
            try:
                p.wait(8)
            except subprocess.TimeoutExpired:
                p.kill()
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
