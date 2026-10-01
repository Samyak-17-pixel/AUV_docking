#!/usr/bin/env python3
"""ROS 2 wrapper: fake Mako_01. Subscribes /Mako_01/actuator_cmd, publishes /Mako_01/odometry_sim.

  source /opt/ros/humble/setup.bash && source control_code/ws/install/setup.bash
  python3 control_code/sim_offline/fake_vehicle.py [--z 3 --yaw 0 --x 0 --y 0]

Odometry: pose in NED (z down), orientation quaternion (ZYX), twist.linear/angular = BODY-frame
velocities (ASSUMPTION: confirm the real sim's twist frame on first connection).
Commands older than --cmd-timeout are treated as zero (like the real bridge's cmd timeout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vehicle_model import VehicleModel  # noqa: E402

from interfaces.msg import Actuator  # noqa: E402


class FakeVehicle(Node):
    def __init__(self, model: VehicleModel, vessel: str, rate_hz: float, cmd_timeout: float) -> None:
        super().__init__("fake_vehicle")
        self.m = model
        self.dt = 1.0 / rate_hz
        self.cmd_timeout = cmd_timeout
        self._last_cmd_t = None
        self.create_subscription(Actuator, f"/{vessel}/actuator_cmd", self._on_cmd, 10)
        self.pub = self.create_publisher(Odometry, f"/{vessel}/odometry_sim", 10)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"fake_vehicle up @ {rate_hz:.0f} Hz, start pos={model.pos}")

    def _on_cmd(self, msg: Actuator) -> None:
        self.m.set_command({n: float(v) for n, v in zip(msg.actuator_names, msg.actuator_values) if n})
        self._last_cmd_t = self.get_clock().now()

    def _tick(self) -> None:
        if self._last_cmd_t is not None:
            age = (self.get_clock().now() - self._last_cmd_t).nanoseconds * 1e-9
            if age > self.cmd_timeout:
                self.m.set_command({})
        self.m.step(self.dt)
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id = "ned"
        o.child_frame_id = "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, self.m.pos)
        qx, qy, qz, qw = self.m.quaternion()
        o.pose.pose.orientation.x, o.pose.pose.orientation.y = qx, qy
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = qz, qw
        lv, av = self.m.nu[:3], self.m.nu[3:]
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = map(float, lv)
        o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = map(float, av)
        self.pub.publish(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel", default="Mako_01")
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--cmd-timeout", type=float, default=1.0)
    a = ap.parse_args()
    rclpy.init()
    node = FakeVehicle(
        VehicleModel(pos=(a.x, a.y, a.z), eul_deg=(a.roll, a.pitch, a.yaw)), a.vessel, a.rate, a.cmd_timeout
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
