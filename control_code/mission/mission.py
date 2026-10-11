#!/usr/bin/env python3
"""Mission controller for mavsim Mako (ROS 2). Config: mission.yaml. Engine: mission_core.MissionRunner (pure Python) on waypoint_tracking_core.WaypointTracker.

Subscribes to odometry, flies the `legs:` of the YAML (goto, lawnmower, orbit, spiral, yoyo, hold, return_home), and publishes interfaces/Actuator on the actuator_cmd
topic (th_XX = RPM, cs_XX = degrees, 0 = stop/centred). Neutral on frozen or stale odometry; a geofence / dock keep-out breach or a long odometry loss aborts to
return-home; a depth or attitude limit trip is a hard stop. Publishes leg, progress and ETA on /<vessel>/ctrl_debug (the viewer shows them).

  export ROS_DOMAIN_ID=42        # a private domain (e.g. 77) with the offline fake vehicle
  source /opt/ros/humble/setup.bash
  source control_code/ws/install/setup.bash
  ./control_code/mission/run_mission.sh
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
sys.path.insert(0, str(_HERE.parent / "waypoint_tracking"))

from pose_filter import PoseFilterConfig, PoseTracker  # noqa: E402
from state import State  # noqa: E402
from mission_core import MissionRunner  # noqa: E402

DEFAULT_CONFIG = _HERE / "mission.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Mission controller (mavsim)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to mission.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    domain = cfg.get("node", {}).get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Range
    from rclpy.node import Node
    from std_msgs.msg import String
    from interfaces.msg import Actuator

    class MissionNode(Node):
        def __init__(self) -> None:
            super().__init__(str(cfg.get("node", {}).get("name", "mission")))
            self.mr = MissionRunner(cfg)
            self.wt = self.mr.wt
            self.names = self.wt.alloc.th_ids + self.wt.alloc.fin_ids
            est = cfg.get("estimator", {})
            self.use_filter = bool(est.get("filter", True))
            self.tracker = PoseTracker(float(est.get("odom_latency_s", 0.25)), float(cfg["safety"]["odom_timeout_s"]),
                                       PoseFilterConfig(**{k: v for k, v in est.items() if k not in ("filter", "odom_latency_s")}))
            self._t0 = time.monotonic()
            self._last_odom = 0.0
            self._frozen_since: Optional[float] = None
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
            self.altitude = None
            self._alt_t = 0.0
            self.create_subscription(Range, cfg["topics"].get("altimeter", "/Mako_01/altimeter/range"), self._on_alt, 10)   # only used when terrain_follow.enabled
            self.create_timer(self.dt, self._tick)
            self._checked_start = False
            self.get_logger().info(f"Mission: {len(cfg.get('legs', []))} legs | cruise {cfg['speed']['cruise_mps']} m/s, flow {cfg['speed']['flow_mps']} m/s | "
                                   f"geofence {cfg['safety'].get('geofence')} | dock keep-out {cfg['safety'].get('dock_keepout_m')} m")

        def _on_alt(self, msg) -> None:
            self.altitude = float(msg.range)
            self._alt_t = time.monotonic()

        def _on_odom(self, msg) -> None:
            raw = State.from_odom(msg)
            now = time.monotonic()
            if self._last_odom > 0.0 and now - self._last_odom > 2.0:
                self.mr.report_odom_loss(now - self._last_odom)          # a long gap: the runner aborts to return-home if it exceeds safety.failsafe.odom_loss_s
            self._last_odom = now
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
            if self.tripped or self.mr.done:
                self.publish({})
                raise SystemExit(1 if self.tripped else 0)
            if self.use_filter:
                now = time.monotonic() - self._t0
                f = self.tracker.at(now)
                if f is not None:
                    self.st = f
                    if self.tracker.frozen(now, expect_motion=self._cmd_active):
                        if self._frozen_since is None:
                            self._frozen_since = time.monotonic()
                        self.publish({})                          # frozen odometry (the same sample repeating): neutral and forget the integrals
                        self.wt.loops.reset()
                        self.get_logger().warning("odometry frozen -> neutral", throttle_duration_sec=2.0)
                        return
                    if self._frozen_since is not None:             # the freeze ended: a long one counts as an odometry loss (safety.failsafe.odom_loss_s -> return home)
                        self.mr.report_odom_loss(time.monotonic() - self._frozen_since)
                        self._frozen_since = None
            if self.st is None:
                self.publish({})
                return
            if time.monotonic() - self._last_odom > float(cfg["safety"]["odom_timeout_s"]):
                self.publish({})
                self.get_logger().warning("odometry stale -> neutral", throttle_duration_sec=2.0)
                return
            why = self.mr.safety_reason(self.st)
            if why:
                self.tripped = why
                self.get_logger().error(f"SAFETY TRIP: {why} -> neutral and exit")
                self.publish({})
                raise SystemExit(1)
            if not self._checked_start:                    # once, at the first state: the whole mission is checked (turning circle, geofence, dock keep-out)
                self._checked_start = True
                self.mr.start(self.st)
                for msg in self.mr.warnings:
                    self.get_logger().warning(msg)
                self.get_logger().info(f"Mission plan: {len(self.mr.segments)} segments, {self.mr.total_length:.0f} m, ETA {self.mr.eta_s():.0f} s at cruise speed")
            was_aborted = self.mr.aborted
            alt = self.altitude if (time.monotonic() - self._alt_t) < 1.0 else None          # an old altimeter reading is no reading
            out, s = self.mr.update(self.st, self.dt, alt)
            if self.mr.aborted and not was_aborted:
                self.get_logger().error(f"MISSION ABORTED: {self.mr.aborted} -> returning home")
            self.publish(out)
            if self.mr.done:
                self.get_logger().info("Mission complete - actuators at neutral.")
                raise SystemExit(0)
            self._dbg_t += self.dt
            if self._dbg_t >= 0.5:
                self._dbg_t = 0.0
                self.dbg_pub.publish(String(data=json.dumps(self.mr.debug())))
            self._since_status += self.dt
            if self._since_status >= self._status_period:
                self._since_status = 0.0
                self.get_logger().info(
                    f"{s['leg']} wp {s['wp']}/{s['wps']} {100 * s['progress']:.0f}% eta {s['eta_s']:.0f}s pos=({self.st.pos[0]:.1f},{self.st.pos[1]:.1f},{self.st.depth:.1f}) "
                    f"hd_err={s.get('heading_err_deg', 0.0):+.1f}deg ct={s.get('cross_track_m', 0.0):+.2f}m u={s.get('u', 0.0):.2f}/{s.get('u_sp', 0.0):.2f} m/s depth_err={s.get('depth_err_m', 0.0):+.2f}")

    rclpy.init()
    node = MissionNode()
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
