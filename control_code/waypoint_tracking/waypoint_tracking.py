#!/usr/bin/env python3
"""Waypoint tracking controller for mavsim Mako (ROS 2). Config: waypoint_tracking.yaml. Engine: waypoint_tracking_core.WaypointTracker (pure Python).

Subscribes to odometry, steers through a YAML waypoint list (NED), and publishes interfaces/Actuator on the actuator_cmd topic. Units are what the sim expects
(the bridge forwards them raw): th_XX = RPM, cs_XX = degrees, 0 = stop/centred.

  - Heading: yaw loop (N*m) -> X-fins through common/allocation.py (speed-scheduled, sign mix from the vessel geometry)
  - Surge:   speed loop (keeps flow over the fins, slows towards the waypoint)
  - Depth:   heave loop (N) -> heave thrusters through the allocator
  (rebuilt on common/ on 2026-10-10; the first version spun after the first waypoint offline, see waypoint_tracking_core.py)

  export ROS_DOMAIN_ID=42
  source /opt/ros/humble/setup.bash
  source control_code/ws/install/setup.bash
  ./control_code/waypoint_tracking/run_waypoint_tracking.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "common"))

from pose_filter import PoseFilterConfig, PoseTracker  # noqa: E402
from state import State  # noqa: E402
from waypoint_tracking_core import WaypointTracker, unreachable_corners  # noqa: E402

DEFAULT_CONFIG = _HERE / "waypoint_tracking.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Waypoint tracker (mavsim)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to waypoint_tracking.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    domain = cfg.get("node", {}).get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import String
    from interfaces.msg import Actuator

    class WaypointNode(Node):
        def __init__(self) -> None:
            super().__init__(str(cfg.get("node", {}).get("name", "waypoint_tracking")))
            self.wt = WaypointTracker(cfg)
            self.names = self.wt.alloc.th_ids + self.wt.alloc.fin_ids
            est = cfg.get("estimator", {})
            self.use_filter = bool(est.get("filter", True))
            self.tracker = PoseTracker(float(est.get("odom_latency_s", 0.25)), float(cfg["safety"]["odom_timeout_s"]),
                                       PoseFilterConfig(**{k: v for k, v in est.items() if k not in ("filter", "odom_latency_s")}))
            self._t0 = time.monotonic()
            self._last_odom = 0.0
            self.st: Optional[State] = None
            self.dt = 1.0 / float(cfg["node"].get("rate_hz", 20.0))
            self.tripped: Optional[str] = None
            self._cmd_active = True
            self._status_period = float(cfg.get("logging", {}).get("status_period_s", 2.0))
            self._since_status = 0.0
            self.pub = self.create_publisher(Actuator, cfg["topics"]["actuator_cmd"], 10)
            self.dbg_pub = self.create_publisher(String, f"/{cfg['node'].get('vessel', 'Mako_01')}/ctrl_debug", 10)
            self._dbg_t = 0.0
            self.create_subscription(Odometry, cfg["topics"]["odometry"], self._on_odom, 10)
            self.create_timer(self.dt, self._tick)
            self._checked_start = False
            wp0 = self.wt.waypoints[0]
            self.get_logger().info(f"Waypoint tracking: {len(self.wt.waypoints)} wps | first=({wp0[0]:.1f},{wp0[1]:.1f},{wp0[2]:.1f}) | "
                                   f"cruise {cfg['speed']['cruise_mps']} m/s, flow {cfg['speed']['flow_mps']} m/s")

        def _on_odom(self, msg) -> None:
            raw = State.from_odom(msg)
            self._last_odom = time.monotonic()
            self.tracker.push(raw, self._last_odom - self._t0)
            if not self.use_filter:
                self.st = raw

        def publish(self, values: Dict[str, float]) -> None:
            m = Actuator()
            m.actuator_names = list(self.names)
            m.actuator_values = [float(values.get(n, 0.0)) for n in self.names]
            m.covariance = [0.0] * len(self.names)
            self._cmd_active = any(abs(float(values.get(n, 0.0))) > (50.0 if n.startswith("th_") else 1.0) for n in self.names)
            self.pub.publish(m)

        def _tick(self) -> None:
            if self.tripped or self.wt.done:
                self.publish({})
                raise SystemExit(1 if self.tripped else 0)
            if self.use_filter:
                now = time.monotonic() - self._t0
                f = self.tracker.at(now)
                if f is not None:
                    self.st = f
                    if self.tracker.frozen(now, expect_motion=self._cmd_active):
                        self.publish({})                          # frozen odometry (the same sample repeating): neutral and forget the integrals
                        self.wt.loops.reset()
                        self.get_logger().warning("odometry frozen -> neutral", throttle_duration_sec=2.0)
                        return
            if self.st is None:
                self.publish({})
                return
            if time.monotonic() - self._last_odom > float(cfg["safety"]["odom_timeout_s"]):
                self.publish({})
                self.get_logger().warning("odometry stale -> neutral", throttle_duration_sec=2.0)
                return
            why = self.wt.safety_reason(self.st)
            if why:
                self.tripped = why
                self.get_logger().error(f"SAFETY TRIP: {why} -> neutral and exit")
                self.publish({})
                raise SystemExit(1)
            if not self._checked_start:                    # once, at the first state: corners the vehicle cannot make (it would orbit the waypoint)
                self._checked_start = True
                for msg in unreachable_corners(self.wt.waypoints, float(cfg["speed"]["turn_radius_m"]), bool(cfg["mission"].get("loop", False)),
                                               (float(self.st.pos[0]), float(self.st.pos[1]))):
                    self.get_logger().warning(msg)
            out, s = self.wt.update(self.st, self.dt)
            self.publish(out)
            if self.wt.done:
                self.get_logger().info("Mission complete - actuators at neutral.")
                raise SystemExit(0)
            self._dbg_t += self.dt
            if self._dbg_t >= 0.5:
                self._dbg_t = 0.0
                self.dbg_pub.publish(String(data=json.dumps(self.wt.debug())))
            self._since_status += self.dt
            if self._since_status >= self._status_period:
                self._since_status = 0.0
                self.get_logger().info(
                    f"WP{s['wp']}/{len(self.wt.waypoints) - 1} pos=({self.st.pos[0]:.1f},{self.st.pos[1]:.1f},{self.st.depth:.1f}) r_xy={s['r_xy']:.1f} "
                    f"hd_err={s['heading_err_deg']:+.1f}deg u={s['u']:.2f}/{s['u_sp']:.2f} m/s depth_err={s['depth_err_m']:+.2f} fin_authority={s['fin_authority']:.2f}")

    rclpy.init()
    node = WaypointNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if bool(cfg.get("mission", {}).get("zero_on_exit", True)):
            try:
                node.publish({})
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
