"""Frozen odometry must not make the controllers run away: station_keeping goes neutral while the data is frozen. Fake vehicle with realistic odometry
(freezes about every 8 s, 5-25 s long) over ROS; the GROUND TRUTH comes from the fake vehicle's status. Skipped without rclpy / interfaces."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

CTRL = Path(__file__).resolve().parents[2]


def test_station_keeping_does_not_run_away_when_the_odometry_freezes():
    os.environ["ROS_DOMAIN_ID"] = "68"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    env = dict(os.environ)
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fake = subprocess.Popen([sys.executable, str(CTRL / "sim_offline" / "fake_vehicle.py"), "--realistic", "--odom-freeze-interval", "8", "--odom-seed", "5", "--z", "3"], env=env, **quiet)
    sk = None
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init()
    try:
        node = Node("freeze_probe")
        truth = []
        node.create_subscription(String, "/Mako_01/sim/status", lambda m: truth.append(json.loads(m.data).get("truth")), 10)
        time.sleep(3.0)
        sk = subprocess.Popen([sys.executable, str(CTRL / "station_keeping" / "station_keeping.py"), "--config", str(CTRL / "station_keeping" / "station_keeping.yaml")], env=env, **quiet)
        t0 = time.time()
        while time.time() - t0 < 50.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
    finally:
        rclpy.shutdown()
        for p in (sk, fake):
            if p is not None:
                p.send_signal(2)
        for p in (sk, fake):
            if p is not None:
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
    tr = np.array([t for t in truth if t])
    assert len(tr) > 100
    depth_dev = np.abs(tr[:, 2] - 3.0).max()
    pitch_dev = np.degrees(np.abs(tr[:, 4]).max())
    assert depth_dev < 0.5 and pitch_dev < 12.0, f"depth deviation {depth_dev:.2f} m, pitch {pitch_dev:.1f} deg"
