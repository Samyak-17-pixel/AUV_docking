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

    class StationKeepingNode(Node):
        def __init__(self) -> None:
            super().__init__(str(cfg["node"].get("name", "station_keeping")))
            self.sk = StationKeeper(cfg)
            self.names = self.sk.alloc.th_ids + self.sk.alloc.fin_ids
            self.st: Optional[State] = None
            self._last_odom = 0.0
            self.dt = 1.0 / float(cfg["node"]["rate_hz"])
            self._since_status = 0.0
            self.tripped: Optional[str] = None
            self.rows: List[dict] = []
            self.pub = self.create_publisher(Actuator, cfg["topics"]["actuator_cmd"], 10)
            self.create_subscription(Odometry, cfg["topics"]["odometry"], self._on_odom, 10)
            self.create_timer(self.dt, self._tick)
            self.get_logger().info(
                f"station_keeping: capture={cfg['capture']['mode']} "
                f"loops={[k for k, v in cfg['enable'].items() if v]}; waiting for odometry"
            )

        def _on_odom(self, msg) -> None:
            self.st = State.from_odom(msg)
            self._last_odom = time.monotonic()

        def publish(self, values: Dict[str, float]) -> None:
            m = Actuator()
            m.actuator_names = list(self.names)
            m.actuator_values = [float(values.get(n, 0.0)) for n in self.names]
            m.covariance = [0.0] * len(self.names)
            self.pub.publish(m)

        def _tick(self) -> None:
            if self.tripped:
                self.publish({})
                raise SystemExit(1)
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
            self.publish(out)
            self.rows.append({
                "t": round(self.sk.t, 2), **{k: v for k, v in s.items() if k != "wrench"},
                "depth": self.st.depth, **{k: round(v, 2) for k, v in out.items()},
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
            d = Path(os.path.expanduser(cfg["logging"]["dir"]))
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
