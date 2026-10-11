"""The simulated side-scan sonars over REAL ROS on a private domain (75): fake vehicle -> sidescan_node -> RosLink -> Telemetry; run-time range / gain / on-off commands through
/Mako_01/sonar/cmd. Needs rclpy and the built interfaces package (with SideScan); about 30 s."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")
try:
    from interfaces.msg import SideScan  # noqa: F401
except ImportError:
    pytest.skip("interfaces was built without SideScan.msg", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
SV = ROOT / "sim_viewer"


def _stop(p):
    if p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def wait_for(cond, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds and not cond():
        time.sleep(0.1)
    return cond()


def test_pings_flow_commands_change_them_and_off_stops_them(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "75")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    env = dict(os.environ, ROS_DOMAIN_ID="75", ROS_LOCALHOST_ONLY="1")
    from ros_link import RosLink
    from synthetic_camera import DEFAULT_CONFIG
    from telemetry import Telemetry

    procs = [subprocess.Popen([sys.executable, str(ROOT / "sim_offline" / "fake_vehicle.py"), "--realistic", "--z", "3"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable, str(SV / "sonar" / "sidescan_node.py"), "--config", str(DEFAULT_CONFIG)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    cfg["node"]["ros_domain_id"] = 75
    tel = Telemetry()
    link = RosLink(tel, cfg)
    try:
        link.start()
        assert link.ready.wait(10.0) and link.error is None
        assert wait_for(lambda: len(tel.sonar_rows("port", 50)) > 10 and len(tel.sonar_rows("starboard", 50)) > 10, 40.0), f"no pings (counts {link.counts})"
        t, inten, meta = tel.sonar_rows("starboard", 1)[-1]
        assert inten.dtype == np.uint16 and len(inten) == 600 and meta["range_m"] == 30.0 and meta["gain_index"] == -1
        assert meta["altitude_m"] > 3.0 and abs(meta["pos"][2] - 3.0) < 0.3                  # the floor is deeper than 6 m under the vehicle at 3 m depth
        assert inten.max() > 20000 and (inten > 20000).sum() > 20                            # there is an echo
        assert wait_for(lambda: tel.sonar_status.get("ping_hz", 0) > 5, 5.0)                 # the node reports its ping rate
        # commands: range 15 m, gain 5, 800 bins
        link.send_sonar_command({"range_m": 15.0, "gain": 5, "bins": 800})
        assert wait_for(lambda: tel.sonar_rows("port", 1)[-1][2]["range_m"] == 15.0 and tel.sonar_rows("port", 1)[-1][2]["gain_index"] == 5, 10.0), "the command did not change the pings"
        assert len(tel.sonar_rows("port", 1)[-1][1]) == 800
        # off: the ping count stops growing (negative control: it was growing before)
        n0 = len(tel.sonar_rows("port", 4000))
        time.sleep(1.0)
        assert len(tel.sonar_rows("port", 4000)) > n0
        link.send_sonar_command({"enabled": False})
        time.sleep(1.5)
        n1 = tel.sonar_rows("port", 1)[-1][2]["ping_number"]
        time.sleep(1.5)
        assert tel.sonar_rows("port", 1)[-1][2]["ping_number"] == n1
    finally:
        link.stop()
        link.join(3.0)
        for p in procs:
            _stop(p)
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
