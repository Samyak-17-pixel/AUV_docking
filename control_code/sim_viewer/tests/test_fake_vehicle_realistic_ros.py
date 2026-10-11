"""fake_vehicle --realistic over ROS: odometry rate, irregular timing, no perfect 100 Hz stream. Skipped without rclpy / interfaces."""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

FAKE = Path(__file__).resolve().parents[2] / "sim_offline" / "fake_vehicle.py"


def _collect(args, seconds=12.0, domain="74"):
    os.environ["ROS_DOMAIN_ID"] = domain
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    proc = subprocess.Popen([sys.executable, str(FAKE), *args], env=dict(os.environ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rclpy.init()
    node = Node("odo_rate_test")
    got = []
    node.create_subscription(Odometry, "/Mako_01/odometry_sim", lambda m: got.append((time.monotonic(), m.pose.pose.position.z)), 50)
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(node, timeout_sec=0.02)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        proc.send_signal(2)
        proc.wait(8)
    return np.array(got)


def test_realistic_odometry_is_slow_irregular_and_noisy():
    g = _collect(["--realistic", "--odom-freeze-interval", "0"], seconds=12.0)
    rate = len(g) / 12.0
    assert 3.3 < rate < 5.6, rate
    d = np.diff(g[:, 0])
    assert d.std() > 0.004
    assert g[:, 1].std() > 0.003                                           # noise on depth (vehicle is at rest at z = 3)


def test_default_odometry_is_still_perfect_and_fast():
    g = _collect(["--rate", "50"], seconds=5.0, domain="73")
    assert len(g) / 5.0 > 35 and g[:, 1].std() < 1e-6
