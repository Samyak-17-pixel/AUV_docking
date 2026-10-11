#!/usr/bin/env python3
"""Station keeping (hover) for Mako_01 (ROS 2). Config: station_keeping.yaml.

  ./run_station_keeping.sh                 # capture current pose, then hold it
  ./run_station_keeping.sh --config my.yaml

Engine: station_keeping_core.StationKeeper (pure Python). Ctrl-C or a safety trip publishes neutral.
"""

from __future__ import annotations

import argparse
import csv
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
from outdirs import out_dir  # noqa: E402

from live_gains import attach as attach_live_gains  # noqa: E402
from pose_filter import PoseFilterConfig, PoseTracker  # noqa: E402
from state import State  # noqa: E402
from station_keeping_core import StationKeeper  # noqa: E402

DEFAULT_CONFIG = _HERE / "station_keeping.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Station keeping (mavsim)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    domain = cfg["node"].get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from interfaces.msg import Actuator
    from std_msgs.msg import String
    import json

    class StationKeepingNode(Node):
        def __init__(self) -> None:
            super().__init__(str(cfg["node"].get("name", "station_keeping")))
            self.sk = StationKeeper(cfg)
            self.names = self.sk.alloc.th_ids + self.sk.alloc.fin_ids
            self.st: Optional[State] = None
            est = cfg.get("estimator", {})
            self.use_filter = bool(est.get("filter", True))
            self.tracker = PoseTracker(float(est.get("odom_latency_s", 0.25)), float(cfg["safety"]["odom_timeout_s"]),
                                       PoseFilterConfig(**{k: v for k, v in est.items() if k not in ("filter", "odom_latency_s")}))
            self._t0 = time.monotonic()
            self._last_odom = 0.0
            self.dt = 1.0 / float(cfg["node"]["rate_hz"])
            self._since_status = 0.0
            self.tripped: Optional[str] = None
            self.rows: List[dict] = []
            self.pub = self.create_publisher(Actuator, cfg["topics"]["actuator_cmd"], 10)
            self.dbg_pub = self.create_publisher(String, f"/{cfg['node'].get('vessel', 'Mako_01')}/ctrl_debug", 10)   # setpoints for the viewer's plots
            self._dbg_t = 0.0
            self.create_subscription(Odometry, cfg["topics"]["odometry"], self._on_odom, 10)
            attach_live_gains(self, cfg["node"].get("vessel", "Mako_01"), "station_keeping", self.sk.loops, self.get_logger().info)
            self.create_timer(self.dt, self._tick)
            self.get_logger().info(
                f"station_keeping: capture={cfg['capture']['mode']} "
                f"loops={[k for k, v in cfg['enable'].items() if v]}; waiting for odometry"
            )

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
            if self.tripped:
                self.publish({})
                raise SystemExit(1)
            if self.use_filter:                                   # smooth, up-to-date state from the slow, late, noisy odometry (common/pose_filter.py)
                now = time.monotonic() - self._t0
                f = self.tracker.at(now)
                if f is not None:
                    self.st = f
                    if self.tracker.frozen(now, expect_motion=getattr(self, "_cmd_active", True)):
                        # frozen odometry (the same sample repeating, as the real sim does for 5-25 s): controlling on it winds the integrators up and, with no righting moment,
                        # lets the pitch run away. Neutral, forget the integrals, wait.
                        self.publish({})
                        self.sk.loops.reset()
                        self.get_logger().warning("odometry frozen -> neutral", throttle_duration_sec=2.0)
                        return
            if self.st is None:
                self.publish({})
                return
            if time.monotonic() - self._last_odom > float(cfg["safety"]["odom_timeout_s"]):
                self.publish({})
                self.get_logger().warning("odometry stale -> neutral", throttle_duration_sec=2.0)
                return
            if not self.sk.feed_capture(self.st, self.dt):
                self.publish(self.sk.capture_command())   # hold the buoyancy trim while averaging the hold point
                return
            why = self.sk.safety_reason(self.st)
            if why:
                self.tripped = why
                self.get_logger().error(f"SAFETY TRIP: {why} -> neutral and exit")
                self.publish({})
                raise SystemExit(1)
            out, s = self.sk.update(self.st, self.dt)
            self._dbg_t += self.dt
            if self._dbg_t >= 0.5:
                self._dbg_t = 0.0
                self.dbg_pub.publish(String(data=json.dumps(self.sk.debug())))
            self.publish(out)
            self.rows.append({
                "t": round(self.sk.t, 2), **{k: v for k, v in s.items() if k != "wrench"},
                "depth": self.st.depth, **{k: round(v, 2) for k, v in out.items()},
                # full state, so the viewer can replay and compare this log (x, y, z=depth, attitude, body velocities)
                "x": float(self.st.pos[0]), "y": float(self.st.pos[1]), "z": self.st.depth,
                "roll_deg": math.degrees(self.st.roll), "pitch_deg": math.degrees(self.st.pitch), "yaw_deg": math.degrees(self.st.yaw),
                "v": float(self.st.nu[1]), "w": float(self.st.nu[2]), "p": float(self.st.nu[3]), "q": float(self.st.nu[4]), "r": float(self.st.nu[5]),
            })
            self.sk.t += self.dt
            self._since_status += self.dt
            if self._since_status >= float(cfg["logging"]["status_period_s"]):
                self._since_status = 0.0
                warn = ""
                if s["fin_authority"] < 0.5 and abs(s["heading_err_deg"]) > 5.0:
                    warn = "  [heading NOT controllable: no flow over fins]"
                self.get_logger().info(
                    f"depth_err={s['depth_err_m']:+.3f} m  pitch_err={s['pitch_err_deg']:+.2f} deg  "
                    f"ahead_err={s['ahead_err_m']:+.3f} m  lateral_drift={s['lateral_drift_m']:+.2f} m  "
                    f"hdg_err={s['heading_err_deg']:+.1f} deg  u={s['u']:.2f} m/s{warn}"
                )

    rclpy.init()
    node = StationKeepingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.publish({})
        except Exception:
            pass
        if node.rows:
            d = out_dir(cfg["logging"]["dir"])
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"station_keeping_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            fields: List[str] = []
            for r in node.rows:
                fields += [k for k in r if k not in fields]
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, restval=0.0)
                w.writeheader()
                w.writerows(node.rows)
            print(f"log: {p}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
