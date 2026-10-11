"""RosLink feeds Telemetry from real ROS topics. Skipped when rclpy or the built interfaces package are not available."""
import os
import time

import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

from synthetic_camera import DEFAULT_CONFIG  # noqa: E402
from telemetry import Telemetry  # noqa: E402


def test_ros_link_receives_odometry_actuator_and_dockalign():
    os.environ["ROS_DOMAIN_ID"] = "79"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    from interfaces.msg import Actuator, DockAlign
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from ros_link import RosLink

    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    cfg["node"]["ros_domain_id"] = 79
    tel = Telemetry()
    link = RosLink(tel, cfg)
    link.start()
    assert link.ready.wait(10.0) and link.error is None
    pub_node = Node("test_pub")
    odo = pub_node.create_publisher(Odometry, "/Mako_01/odometry_sim", 10)
    act = pub_node.create_publisher(Actuator, "/Mako_01/actuator_cmd", 10)
    ali = pub_node.create_publisher(DockAlign, "/Mako_01/dock_align", 10)
    time.sleep(1.0)
    for _ in range(10):
        o = Odometry(); o.pose.pose.position.x = 4.2; o.pose.pose.position.z = 3.0; o.pose.pose.orientation.w = 1.0; o.twist.twist.linear.x = 0.3
        a = Actuator(); a.actuator_names = ["th_01", "th_02"]; a.actuator_values = [123.0, -456.0]
        d = DockAlign(); d.valid = True; d.num_lights = 4; d.error_x_px = -7.0; d.radius_px = 90.0; d.elevation_valid = True; d.elevation_rad = 0.1
        odo.publish(o); act.publish(a); ali.publish(d)      # publishers need no spinning; only the RosLink thread may spin the global executor
        time.sleep(0.05)
    time.sleep(0.5)
    link.stop()
    link.join(3.0)
    pub_node.destroy_node()
    rclpy.shutdown()
    s = tel.latest()
    assert s is not None and s.pos[0] == pytest.approx(4.2) and s.nu[0] == pytest.approx(0.3)
    assert tel.latest_cmd() == {"th_01": 123.0, "th_02": -456.0}
    t, al = tel.latest_align()
    assert al["valid"] == 1.0 and al["num_lights"] == 4.0 and al["error_x_px"] == -7.0 and al["elevation_deg"] == pytest.approx(5.73, abs=0.01)
