"""live_gains.attach over real ROS, and RosLink.send_gain reaching a node. Skipped without rclpy."""
import os
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

from loops import HoldLoops  # noqa: E402
import live_gains  # noqa: E402


def test_gain_message_over_ros_changes_a_running_loop_and_respects_the_addressee():
    os.environ["ROS_DOMAIN_ID"] = "72"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init()
    try:
        loops = HoldLoops({"heave": {"kp": 10.0, "ki": 0.0, "kd": 20.0, "max": 25.0}})
        node, sender = Node("receiver"), Node("sender")
        logged = []
        live_gains.attach(node, "Mako_01", "terminal_docking", loops, logged.append)
        pub = sender.create_publisher(String, "/Mako_01/ctrl_gains", 10)
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(sender)
        t0 = time.time()
        sent_other = sent_mine = False
        while time.time() - t0 < 5.0 and loops.pids["heave"].kp != 33.0:
            if not sent_other and time.time() - t0 > 0.8:
                pub.publish(String(data=live_gains.make_message("station_keeping", "heave", "kp", 99.0)))      # another controller's slider
                sent_other = True
            if time.time() - t0 > 1.2 and not sent_mine:
                pub.publish(String(data=live_gains.make_message("terminal_docking", "heave", "kp", 33.0)))
                sent_mine = True
            ex.spin_once(timeout_sec=0.05)
        assert loops.pids["heave"].kp == 33.0 and loops.gains["heave"]["kp"] == 33.0
        assert any("heave.kp" in l for l in logged) and not any("99" in l for l in logged)
    finally:
        rclpy.shutdown()
