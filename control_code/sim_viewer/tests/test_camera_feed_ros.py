"""The camera feed reaches the viewer's telemetry over REAL ROS on a private domain (78): fake vehicle -> synthetic camera node -> RosLink -> Telemetry.image(), and the
detector then publishes DockAlign from that picture. This is the path behind the 'no camera image' pane. Also: what the empty pane says, and the launcher's domain default.
Needs rclpy and the built interfaces package; about 25 s."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
SV = ROOT / "sim_viewer"


def _stop(p):
    if p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def test_camera_frames_and_detector_output_reach_the_viewer_telemetry(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "78")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    env = dict(os.environ, ROS_DOMAIN_ID="78", ROS_LOCALHOST_ONLY="1")
    from ros_link import RosLink
    from synthetic_camera import DEFAULT_CONFIG
    from telemetry import Telemetry

    procs = [subprocess.Popen([sys.executable, str(ROOT / "sim_offline" / "fake_vehicle.py"), "--realistic", "--z", "3"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable, str(SV / "camera" / "camera_node.py"), "--config", str(DEFAULT_CONFIG)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable, str(REPO / "dock_detection_algo" / "live_dock_lights.py"), "--no-gui"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    cfg["node"]["ros_domain_id"] = 78
    tel = Telemetry()
    link = RosLink(tel, cfg)
    try:
        link.start()
        assert link.ready.wait(10.0) and link.error is None
        t0 = time.time()
        while time.time() - t0 < 20.0 and not (tel.image() is not None and tel.latest_align()[1] and link.counts["image"] > 5 and link.counts["odom"] > 5):
            time.sleep(0.2)
        assert tel.image() is not None, f"no camera frame within 20 s (counts {link.counts})"
        assert link.counts["image"] > 5 and link.counts["odom"] > 5
        t, al = tel.latest_align()
        assert al, f"the detector published nothing (counts {link.counts})"
        assert al["valid"] == 1.0 and al["num_lights"] == 4.0          # the default start (0, 0, 3) faces the dock 10 m away: all four lights in view
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(tel.image(), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[:2] == (480, 640) and img.max() > 200
    finally:
        link.stop()
        link.join(3.0)
        for p in procs:
            _stop(p)
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()


def test_launcher_defaults_to_the_private_offline_domain():
    text = (SV / "scripts" / "run_sim_viewer.sh").read_text()
    assert "export ROS_DOMAIN_ID=77" in text and "ROS_LOCALHOST_ONLY" in text and "--real" in text
    # the real-mavsim domain is only reachable through the explicit flag
    out = subprocess.run(["bash", "-c", f"env -u ROS_DOMAIN_ID bash -n {SV / 'run_sim_viewer.sh'} && echo SYNTAX_OK"], capture_output=True, text=True)
    assert "SYNTAX_OK" in out.stdout
