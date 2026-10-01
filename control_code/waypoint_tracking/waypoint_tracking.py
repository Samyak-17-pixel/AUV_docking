#!/usr/bin/env python3
"""Waypoint tracking controller for mavsim Mako (ROS 2).

Subscribes to odometry, steers through a YAML waypoint list (NED), and publishes
interfaces/Actuator on the actuator_cmd topic. Units are what the sim expects
(the bridge forwards them raw): th_XX = RPM, cs_XX = degrees, 0 = stop/centred.

  - Heading: PID → X-fin yaw mix (degrees)
  - Surge: axial thruster RPM when heading is aligned
  - Depth: PID → heave thruster RPM

  export ROS_DOMAIN_ID=42
  source /opt/ros/humble/setup.bash
  source control_code/ws/install/setup.bash
  ./control_code/waypoint_tracking/run_waypoint_tracking.sh
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from interfaces.msg import Actuator

DEFAULT_CONFIG = Path(__file__).resolve().parent / "waypoint_tracking.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Yaw (heading) from ROS quaternion, NED / aerospace ZYX."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class Pid:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        i_max: float,
        d_filter_tau_s: float = 0.05,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_max = abs(i_max)
        self.d_filter_tau_s = max(0.0, d_filter_tau_s)
        self.integral = 0.0
        self._prev_error: Optional[float] = None
        self._d_filtered = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self._prev_error = None
        self._d_filtered = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return self.kp * error

        self.integral += error * dt
        if self.ki > 1e-12:
            self.integral = clamp(
                self.integral, -self.i_max / self.ki, self.i_max / self.ki
            )
        else:
            self.integral = 0.0

        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        if self.d_filter_tau_s > 0.0:
            alpha = dt / (self.d_filter_tau_s + dt)
            self._d_filtered += alpha * (derivative - self._d_filtered)
            derivative = self._d_filtered

        return self.kp * error + self.ki * self.integral + self.kd * derivative


class WaypointTracking(Node):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        node_cfg = cfg.get("node", {})
        super().__init__(str(node_cfg.get("name", "waypoint_tracking")))

        self.cfg = cfg
        self.topics = cfg["topics"]
        self.actuators = cfg["actuators"]
        self.mission = cfg["mission"]
        self.surge_cfg = cfg["surge"]
        self.limits = cfg["limits"]
        self.logging_cfg = cfg.get("logging", {})

        lim = cfg["limits"]
        self.rpm_max = float(lim["rpm_max"])
        self.rpm_cap = float(lim["rpm_cap"])
        self.fin_deg_max = float(lim["fin_deg_max"])
        self.fin_deg_cap = float(lim["fin_deg_cap"])
        if not (0.0 < self.rpm_cap <= self.rpm_max):
            raise ValueError("limits: require 0 < rpm_cap <= rpm_max")
        if not (0.0 < self.fin_deg_cap <= self.fin_deg_max):
            raise ValueError("limits: require 0 < fin_deg_cap <= fin_deg_max")

        self.waypoints = self._parse_waypoints(cfg.get("waypoints", []))
        if not self.waypoints:
            raise ValueError("waypoint_tracking.yaml: waypoints list is empty")

        h = cfg["heading"]
        self.heading_pid = Pid(
            kp=float(h["kp"]),
            ki=float(h["ki"]),
            kd=float(h["kd"]),
            i_max=float(h.get("i_max", 80.0)),
            d_filter_tau_s=float(h.get("d_filter_tau_s", 0.05)),
        )
        self.max_fin_delta = float(h["max_fin_delta_deg"])
        self.align_deg = float(h["align_deg"])

        d = cfg["depth_pid"]
        self.depth_pid = Pid(
            kp=float(d["kp"]),
            ki=float(d["ki"]),
            kd=float(d["kd"]),
            i_max=float(d.get("i_max", 200.0)),
            d_filter_tau_s=float(d.get("d_filter_tau_s", 0.05)),
        )

        self.actuator_names: List[str] = list(self.actuators["names"])
        self.heave_fwd = str(self.actuators["heave_fwd"])
        self.heave_aft = str(self.actuators["heave_aft"])
        self.surge_name = str(self.actuators["surge"])
        self.fins: List[str] = list(self.actuators["fins"])
        self.fin_yaw_signs = [
            float(s) for s in self.actuators.get("fin_yaw_signs", [1, 1, -1, -1])
        ]
        if len(self.fin_yaw_signs) != len(self.fins):
            raise ValueError("fin_yaw_signs length must match fins")

        self.heave_sign = float(self.limits.get("heave_sign", 1.0))

        self.rate_hz = float(node_cfg.get("rate_hz", 20.0))
        self.dt = 1.0 / self.rate_hz

        self.wp_idx = 0
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.z: Optional[float] = None
        self.yaw: Optional[float] = None
        self._have_odom = False
        self._done = False

        self.pub = self.create_publisher(
            Actuator, str(self.topics["actuator_cmd"]), 10
        )
        self.create_subscription(
            Odometry, str(self.topics["odometry"]), self._on_odom, 10
        )

        vessel_state = str(self.topics.get("vessel_state") or "").strip()
        if vessel_state:
            self.create_subscription(
                Float64MultiArray, vessel_state, self._on_vessel_state, 10
            )

        self._status_period = float(self.logging_cfg.get("status_period_s", 2.0))
        self._status_elapsed = 0.0

        self.timer = self.create_timer(self.dt, self._tick)
        wp0 = self.waypoints[0]
        self.get_logger().info(
            f"Waypoint tracking: {len(self.waypoints)} wps | "
            f"first=({wp0[0]:.1f},{wp0[1]:.1f},{wp0[2]:.1f}) | "
            f"caps ±{self.rpm_cap:.0f} RPM / ±{self.fin_deg_cap:.0f}°"
        )

    def _parse_waypoints(
        self, items: List[Any]
    ) -> List[Tuple[float, float, float]]:
        hold_z = float(self.mission.get("hold_depth_m", 5.0))
        out: List[Tuple[float, float, float]] = []
        for item in items:
            if isinstance(item, dict):
                x = float(item["x"])
                y = float(item["y"])
                z = float(item["z"]) if item.get("z") is not None else hold_z
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x = float(item[0])
                y = float(item[1])
                z = float(item[2]) if len(item) >= 3 else hold_z
            else:
                raise ValueError(f"Bad waypoint entry: {item!r}")
            out.append((x, y, z))
        return out

    def _cap_rpm(self, value: float) -> float:
        return clamp(value, -self.rpm_cap, self.rpm_cap)

    def _cap_fin(self, value: float) -> float:
        return clamp(value, -self.fin_deg_cap, self.fin_deg_cap)

    def _on_odom(self, msg: Odometry) -> None:
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.z = float(msg.pose.pose.position.z)
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._have_odom = True

    def _on_vessel_state(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 13:
            return
        self.x = float(data[7])
        self.y = float(data[8])
        self.z = float(data[9])
        self.yaw = float(data[12])  # psi (euler mode)
        self._have_odom = True

    def _publish(
        self, heave_rpm: float, surge_rpm: float, fin_yaw_deg: float
    ) -> None:
        values = {name: 0.0 for name in self.actuator_names}
        values[self.heave_fwd] = heave_rpm
        values[self.heave_aft] = heave_rpm
        values[self.surge_name] = surge_rpm
        for name, sign in zip(self.fins, self.fin_yaw_signs):
            values[name] = self._cap_fin(sign * fin_yaw_deg)

        msg = Actuator()
        msg.actuator_names = list(self.actuator_names)
        msg.actuator_values = [float(values[n]) for n in self.actuator_names]
        msg.covariance = [0.0] * len(self.actuator_names)
        self.pub.publish(msg)

    def publish_neutral(self) -> None:
        self._publish(0.0, 0.0, 0.0)

    def _advance_if_reached(self, dx: float, dy: float, dz: float) -> bool:
        r_xy = math.hypot(dx, dy)
        acc_xy = float(self.mission["acceptance_radius_m"])
        acc_z = float(self.mission["depth_acceptance_m"])
        if r_xy <= acc_xy and abs(dz) <= acc_z:
            self.get_logger().info(
                f"Reached WP{self.wp_idx} "
                f"(r_xy={r_xy:.2f} m, |dz|={abs(dz):.2f} m)"
            )
            self.wp_idx += 1
            self.heading_pid.reset()
            if self.wp_idx >= len(self.waypoints):
                if bool(self.mission.get("loop", False)):
                    self.wp_idx = 0
                    self.get_logger().info("Looping mission → WP0")
                else:
                    return True
        return False

    def _tick(self) -> None:
        if self._done:
            return

        wait = bool(self.mission.get("wait_for_odometry", True))
        if wait and not self._have_odom:
            self.publish_neutral()
            return

        if None in (self.x, self.y, self.z, self.yaw):
            self.publish_neutral()
            return

        wx, wy, wz = self.waypoints[self.wp_idx]
        dx = wx - self.x
        dy = wy - self.y
        dz = wz - self.z

        if self._advance_if_reached(dx, dy, dz):
            self._done = True
            self.publish_neutral()
            self.get_logger().info("Mission complete — actuators at neutral.")
            raise SystemExit(0)

        wx, wy, wz = self.waypoints[self.wp_idx]
        dx = wx - self.x
        dy = wy - self.y
        depth_error = wz - self.z

        desired_yaw = math.atan2(dy, dx)
        heading_error = wrap_pi(desired_yaw - self.yaw)
        heading_error_deg = math.degrees(heading_error)

        fin_cmd = self.heading_pid.update(heading_error_deg, self.dt)
        fin_yaw_delta = clamp(fin_cmd, -self.max_fin_delta, self.max_fin_delta)

        cruise = float(self.surge_cfg["cruise_rpm"])
        creep = float(self.surge_cfg.get("creep_rpm", 0.0))
        if abs(heading_error_deg) <= self.align_deg:
            if bool(self.surge_cfg.get("scale_with_heading", True)):
                # Blend from zero toward cruise with heading alignment
                surge_rpm = cruise * max(0.0, math.cos(heading_error))
            else:
                surge_rpm = cruise
        else:
            surge_rpm = creep
        surge_rpm = clamp(
            surge_rpm,
            float(self.surge_cfg.get("rpm_min", 0.0)),
            float(self.surge_cfg.get("rpm_max", self.rpm_cap)),
        )
        surge_rpm = self._cap_rpm(surge_rpm)

        heave_u = self.depth_pid.update(depth_error, self.dt)
        heave_rpm = self._cap_rpm(self.heave_sign * heave_u)

        self._publish(heave_rpm, surge_rpm, fin_yaw_delta)

        self._status_elapsed += self.dt
        if self._status_period > 0.0 and self._status_elapsed >= self._status_period:
            self._status_elapsed = 0.0
            r_xy = math.hypot(dx, dy)
            self.get_logger().info(
                f"WP{self.wp_idx}/{len(self.waypoints)-1} "
                f"pos=({self.x:.1f},{self.y:.1f},{self.z:.1f}) "
                f"r_xy={r_xy:.1f} hd_err={heading_error_deg:.1f}° "
                f"surge={surge_rpm:.0f} RPM heave={heave_rpm:.0f} RPM "
                f"fin={fin_yaw_delta:.1f}°"
            )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Waypoint tracker (mavsim)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to waypoint_tracking.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    domain = cfg.get("node", {}).get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    rclpy.init()
    node = WaypointTracking(cfg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if bool(cfg.get("mission", {}).get("zero_on_exit", True)):
            try:
                node.publish_neutral()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
