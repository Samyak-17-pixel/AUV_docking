"""A real fake_vehicle process controlled over ROS: pause, step, push, reset, time scale. Skipped without rclpy / the built interfaces package."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

FAKE = Path(__file__).resolve().parents[2] / "sim_offline" / "fake_vehicle.py"


@pytest.fixture(scope="module")
def link():
    os.environ["ROS_DOMAIN_ID"] = "78"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    from interfaces.msg import Actuator
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import String

    proc = subprocess.Popen([sys.executable, str(FAKE), "--x", "1", "--y", "0", "--z", "3"], env=dict(os.environ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rclpy.init()
    node = Node("sim_control_test")
    state = {"pos": None, "status": {}}
    node.create_subscription(Odometry, "/Mako_01/odometry_sim", lambda m: state.update(pos=(m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)), 10)
    node.create_subscription(String, "/Mako_01/sim/status", lambda m: state.update(status=json.loads(m.data)), 10)
    cmd_pub = node.create_publisher(String, "/Mako_01/sim/cmd", 10)
    act_pub = node.create_publisher(Actuator, "/Mako_01/actuator_cmd", 10)

    def spin(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(node, timeout_sec=0.02)

    def send(d):
        cmd_pub.publish(String(data=json.dumps(d)))
        spin(0.25)

    def thrust(rpm, sec):
        a = Actuator(); a.actuator_names = ["th_01"]; a.actuator_values = [float(rpm)]; a.covariance = [0.0]
        t0 = time.time()
        while time.time() - t0 < sec:
            act_pub.publish(a)
            spin(0.05)

    spin(2.0)
    assert state["pos"] is not None, "no odometry from the fake vehicle"
    yield state, send, spin, thrust
    node.destroy_node()
    rclpy.shutdown()
    proc.terminate()
    proc.wait(5)


def test_pause_freezes_the_vehicle_resume_moves_it(link):
    state, send, spin, thrust = link
    send({"cmd": "reset", "pos": [1, 0, 3], "eul_deg": [0, 0, 0]})
    send({"cmd": "pause", "value": True})
    assert state["status"]["paused"] is True
    p0 = state["pos"]
    thrust(800, 1.0)
    assert math.dist(p0, state["pos"]) < 1e-6                         # thrusting but paused: nothing moves
    send({"cmd": "pause", "value": False})
    thrust(800, 1.0)
    assert state["pos"][0] > p0[0] + 0.05


def test_step_advances_a_little_then_holds(link):
    state, send, spin, thrust = link
    send({"cmd": "reset", "pos": [1, 0, 3], "eul_deg": [0, 0, 0]})
    send({"cmd": "pause", "value": True})
    thrust(1500, 0.3)
    p0 = state["pos"]
    send({"cmd": "step", "seconds": 1.0})
    spin(1.0)
    moved = state["pos"][0] - p0[0]
    assert moved > 0.02
    p1 = state["pos"]
    spin(0.5)
    assert math.dist(p1, state["pos"]) < 1e-6
    send({"cmd": "pause", "value": False})


def test_push_displaces_the_vehicle_and_reset_brings_it_back(link):
    state, send, spin, thrust = link
    send({"cmd": "reset", "pos": [1, 0, 3], "eul_deg": [0, 0, 0]})
    send({"cmd": "push", "wrench": [0, 6, 0, 0, 0, 0], "duration": 1.5})     # a sideways shove
    spin(2.5)
    assert abs(state["pos"][1]) > 0.1
    send({"cmd": "reset", "pos": [2.0, 0.5, 3.5], "eul_deg": [0, 0, 0]})
    spin(0.3)
    assert state["pos"] == pytest.approx((2.0, 0.5, 3.5), abs=0.05)


def test_time_scale_speeds_the_simulation_up(link):
    state, send, spin, thrust = link
    send({"cmd": "reset", "pos": [1, 0, 3], "eul_deg": [0, 0, 0]})
    thrust(800, 1.5)
    slow = state["pos"][0] - 1.0
    send({"cmd": "reset", "pos": [1, 0, 3], "eul_deg": [0, 0, 0]})
    send({"cmd": "time_scale", "value": 3.0})
    thrust(800, 1.5)
    fast = state["pos"][0] - 1.0
    send({"cmd": "time_scale", "value": 1.0})
    assert fast > 3.0 * slow                                          # distance grows roughly as t^2 under constant thrust
