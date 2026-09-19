#!/usr/bin/env python3
"""Closed-loop depth PID for mavsim Mako (ROS 2).

Reads absolute NED depth setpoint from depth_control.yaml, drives heave
thrusters with a PID in ESC PWM (µs), holds within tolerance for
settle_time_s, then commands neutral and exits.

PWM: 1100–1900, neutral 1500, soft-capped to 1200–1800.

  export ROS_DOMAIN_ID=42
  source /opt/ros/humble/setup.bash
  source control_code/ws/install/setup.bash
  ./control_code/depth_control/run_depth_control.sh
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node

from interfaces.msg import Actuator

DEFAULT_CONFIG = Path(__file__).resolve().parent / "depth_control.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class Pid:
    """Basic PID with integral clamp and optional derivative low-pass."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        i_max: float,
        d_filter_tau_s: float,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_max = abs(i_max)
        self.d_filter_tau_s = max(0.0, d_filter_tau_s)
        self.integral = 0.0
        self._prev_error: Optional[float] = None
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


class DepthControl(Node):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        node_cfg = cfg.get("node", {})
        super().__init__(str(node_cfg.get("name", "depth_control")))

        self.topics = cfg["topics"]
        self.actuators = cfg["actuators"]
        self.depth_cfg = cfg["depth"]
        self.limits = cfg["limits"]
        self.behaviour = cfg.get("behaviour", {})
        self.logging_cfg = cfg.get("logging", {})

        pwm = cfg["pwm"]
        self.pwm_min = float(pwm["min"])
        self.pwm_max = float(pwm["max"])
        self.pwm_neutral = float(pwm["neutral"])
        self.pwm_cap_min = float(pwm["cap_min"])
        self.pwm_cap_max = float(pwm["cap_max"])
        if not (
            self.pwm_min
            <= self.pwm_cap_min
            <= self.pwm_neutral
            <= self.pwm_cap_max
            <= self.pwm_max
        ):
            raise ValueError(
                "pwm: require min <= cap_min <= neutral <= cap_max <= max"
            )

        pid_cfg = cfg["pid"]
        self.pid = Pid(
            kp=float(pid_cfg["kp"]),
            ki=float(pid_cfg["ki"]),
            kd=float(pid_cfg["kd"]),
            i_max=float(pid_cfg.get("i_max", 200.0)),
            d_filter_tau_s=float(pid_cfg.get("d_filter_tau_s", 0.05)),
        )

        self.setpoint = float(self.depth_cfg["setpoint_m"])

        self.actuator_names: List[str] = list(self.actuators["names"])
        self.heave_fwd = str(self.actuators["heave_fwd"])
        self.heave_aft = str(self.actuators["heave_aft"])
        self.surge = str(self.actuators["surge"])
        self.fins: List[str] = list(self.actuators.get("fins", []))

        self.heave_sign = float(self.limits.get("heave_sign", 1.0))
        self.surge_pwm = float(self.limits.get("surge_pwm", self.pwm_neutral))
        self.fin_pwm = float(self.limits.get("fin_pwm", self.pwm_neutral))

        self.settle_tol = float(self.behaviour["settle_tolerance_m"])
        self.settle_time_s = float(self.behaviour["settle_time_s"])
        if self.settle_tol <= 0.0 or self.settle_time_s <= 0.0:
            raise ValueError(
                "behaviour.settle_tolerance_m and settle_time_s must both be > 0"
            )

        self.rate_hz = float(node_cfg.get("rate_hz", 20.0))
        self.dt = 1.0 / self.rate_hz

        self.z: Optional[float] = None
        self._have_odom = False
        self._settle_elapsed = 0.0
        self._status_period = float(self.logging_cfg.get("status_period_s", 2.0))
        self._status_elapsed = 0.0

        self.pub = self.create_publisher(
            Actuator, str(self.topics["actuator_cmd"]), 10
        )
        self.create_subscription(
            Odometry, str(self.topics["odometry"]), self._on_odom, 10
        )

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"Depth PID → absolute z_d={self.setpoint:.2f} m | "
            f"PWM neutral={self.pwm_neutral:.0f} cap=[{self.pwm_cap_min:.0f},"
            f"{self.pwm_cap_max:.0f}] | settle |err|<{self.settle_tol:.2f} m "
            f"for {self.settle_time_s:.0f} s, then stop"
        )

    def _on_odom(self, msg: Odometry) -> None:
        self.z = float(msg.pose.pose.position.z)
        self._have_odom = True

    def _to_pwm(self, signed_delta: float) -> float:
        """Map signed effort (µs from neutral) to capped PWM."""
        return clamp(
            self.pwm_neutral + signed_delta,
            self.pwm_cap_min,
            self.pwm_cap_max,
        )

    def _value_map(self, heave_pwm: float) -> Dict[str, float]:
        values = {name: self.pwm_neutral for name in self.actuator_names}
        values[self.heave_fwd] = heave_pwm
        values[self.heave_aft] = heave_pwm
        values[self.surge] = clamp(
            self.surge_pwm, self.pwm_cap_min, self.pwm_cap_max
        )
        for fin in self.fins:
            values[fin] = clamp(self.fin_pwm, self.pwm_cap_min, self.pwm_cap_max)
        return values

    def _publish(self, heave_pwm: float) -> None:
        values = self._value_map(heave_pwm)
        msg = Actuator()
        msg.actuator_names = list(self.actuator_names)
        msg.actuator_values = [float(values[n]) for n in self.actuator_names]
        msg.covariance = [0.0] * len(self.actuator_names)
        self.pub.publish(msg)

    def publish_neutral(self) -> None:
        self._publish(self.pwm_neutral)

    def _tick(self) -> None:
        wait = bool(self.behaviour.get("wait_for_odometry", True))
        if wait and not self._have_odom:
            self.publish_neutral()
            return

        if self.z is None:
            self.publish_neutral()
            return

        # NED: positive error → too shallow → PWM above neutral (dive)
        error = self.setpoint - self.z
        u = self.pid.update(error, self.dt)
        heave_pwm = self._to_pwm(self.heave_sign * u)
        self._publish(heave_pwm)

        if abs(error) < self.settle_tol:
            self._settle_elapsed += self.dt
            if self._settle_elapsed >= self.settle_time_s:
                self.get_logger().info(
                    f"Settled at z={self.z:.2f} m (err={error:.3f}) for "
                    f"{self.settle_time_s:.0f} s. Neutral PWM and exiting."
                )
                self.publish_neutral()
                raise SystemExit(0)
        else:
            self._settle_elapsed = 0.0

        self._status_elapsed += self.dt
        if self._status_period > 0.0 and self._status_elapsed >= self._status_period:
            self._status_elapsed = 0.0
            self.get_logger().info(
                f"z={self.z:.2f} m  z_d={self.setpoint:.2f}  "
                f"err={error:.3f}  heave={heave_pwm:.0f} µs  "
                f"settle={self._settle_elapsed:.1f}/{self.settle_time_s:.0f} s"
            )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Depth PID controller (mavsim)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to depth_control.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    domain = cfg.get("node", {}).get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    rclpy.init()
    node = DepthControl(cfg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if bool(cfg.get("behaviour", {}).get("zero_on_exit", True)):
            try:
                node.publish_neutral()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
