#!/usr/bin/env python3
"""ROS 2 wrapper: fake Mako_01. Subscribes /Mako_01/actuator_cmd, publishes /Mako_01/odometry_sim.

  source /opt/ros/humble/setup.bash && source control_code/ws/install/setup.bash
  python3 control_code/sim_offline/fake_vehicle.py [--z 3 --yaw 0 --x 0 --y 0]
  Also listens on /<vessel>/sim/cmd (std_msgs/String, JSON; see sim_control.py) for pause / step / reset / push / time_scale and publishes
  /<vessel>/sim/status (JSON, 5 Hz): this is how the desktop viewer controls the simulation.

Odometry: pose in NED (z down), orientation quaternion (ZYX), twist.linear/angular = BODY-frame
velocities (ASSUMPTION: confirm the real sim's twist frame on first connection).
Commands older than --cmd-timeout are treated as zero (like the real bridge's cmd timeout).
--realistic makes the odometry look like the real sim's: ~4.5 Hz with jitter, 0.25 s latency, noise and occasional freezes (sensors.py);
without it the odometry is perfect and published at --rate.
"""

from __future__ import annotations

import argparse
import numpy as np
import sys
from pathlib import Path
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensors import OdometryConfig, OdometrySensor, quat_from_eul  # noqa: E402
from sim_control import SimControl  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

from interfaces.msg import Actuator  # noqa: E402
from std_msgs.msg import String  # noqa: E402
import json  # noqa: E402


class FakeVehicle(Node):
    def __init__(self, model: VehicleModel, vessel: str, rate_hz: float, cmd_timeout: float, sensor: Optional[OdometrySensor] = None) -> None:
        super().__init__("fake_vehicle")
        self.m = model
        self.sensor = sensor
        self._paused_pub_t = 0.0
        self.dt = 1.0 / rate_hz
        self.cmd_timeout = cmd_timeout
        self._last_cmd_t = None
        self.create_subscription(Actuator, f"/{vessel}/actuator_cmd", self._on_cmd, 10)
        self.pub = self.create_publisher(Odometry, f"/{vessel}/odometry_sim", 10)
        self.ctl = SimControl(start_pos=model.pos, start_eul_deg=np.degrees(model.eul))
        self.create_subscription(String, f"/{vessel}/sim/cmd", self._on_sim_cmd, 10)
        self.status_pub = self.create_publisher(String, f"/{vessel}/sim/status", 10)
        self.create_timer(self.dt, self._tick)
        self.create_timer(0.2, self._publish_status)
        self.get_logger().info(f"fake_vehicle up @ {rate_hz:.0f} Hz, start pos={model.pos}")

    def _on_cmd(self, msg: Actuator) -> None:
        self.m.set_command({n: float(v) for n, v in zip(msg.actuator_names, msg.actuator_values) if n})
        self._last_cmd_t = self.get_clock().now()

    def _on_sim_cmd(self, msg: String) -> None:
        if not self.ctl.handle(msg.data):
            self.get_logger().warning(f"bad sim command {msg.data!r}: {self.ctl.last_error}")

    def _publish_status(self) -> None:
        d = self.ctl.status()
        d["truth"] = [float(v) for v in (*self.m.pos, *self.m.eul, self.m.nu[0])]       # ground truth (x y z roll pitch yaw u): the viewer's scenario judge uses it, controllers do not
        self.status_pub.publish(String(data=json.dumps(d)))

    def _tick(self) -> None:
        if self._last_cmd_t is not None:
            age = (self.get_clock().now() - self._last_cmd_t).nanoseconds * 1e-9
            if age > self.cmd_timeout:
                self.m.set_command({})
        reset = self.ctl.take_reset()
        if reset is not None:
            self.m.reset(reset["pos"], reset["eul_deg"])
        sim_dt = self.ctl.sim_dt(self.dt)
        n = int(np.ceil(sim_dt / 0.01 - 1e-9)) if sim_dt > 0 else 0
        for _ in range(n):
            h = sim_dt / n
            self.m.ext[:] = self.ctl.external_wrench(h)
            self.m.step(h)
        if self.sensor is None:
            self._publish(self.m.pos, self.m.eul, self.m.nu)
            return
        outs = self.sensor.update(self.m.t, self.m.pos, self.m.eul, self.m.nu) if sim_dt > 0 else []
        for smp in outs:
            self._publish(smp.pos, smp.eul, smp.nu)
        if sim_dt <= 0:                                                  # paused: the real topic keeps publishing the same sample
            now = self.get_clock().now().nanoseconds * 1e-9
            last = self.sensor.repeat_last()
            if last is not None and now - self._paused_pub_t >= 1.0 / self.sensor.cfg.rate_hz:
                self._paused_pub_t = now
                self._publish(last.pos, last.eul, last.nu)

    def _publish(self, pos, eul, nu) -> None:
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id = "ned"
        o.child_frame_id = "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, pos)
        qx, qy, qz, qw = quat_from_eul(eul)
        o.pose.pose.orientation.x, o.pose.pose.orientation.y = qx, qy
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = qz, qw
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = map(float, nu[:3])
        o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = map(float, nu[3:])
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
    ap.add_argument("--realistic", action="store_true", help="odometry like the real sim: ~4.5 Hz, latency, noise, freezes (sensors.py)")
    ap.add_argument("--odom-rate", type=float, default=4.5)
    ap.add_argument("--odom-latency", type=float, default=0.25)
    ap.add_argument("--odom-freeze-interval", type=float, default=60.0, help="mean seconds between freezes, 0 = never")
    ap.add_argument("--odom-seed", type=int, default=1)
    a = ap.parse_args()
    rclpy.init()
    node = FakeVehicle(
        VehicleModel(pos=(a.x, a.y, a.z), eul_deg=(a.roll, a.pitch, a.yaw)), a.vessel, a.rate, a.cmd_timeout,
        OdometrySensor(OdometryConfig(rate_hz=a.odom_rate, latency_s=a.odom_latency, freeze_mean_interval_s=a.odom_freeze_interval, seed=a.odom_seed)) if a.realistic else None,
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
