#!/usr/bin/env python3
"""Standoff dock test controller (ROS 2). Config: dock_test.yaml.

  ./run_dock_test.sh

Reads DockAlign and odometry, publishes actuator commands. Ctrl-C or a safety
trip publishes neutral. Run dock_detection_algo/run_live.sh in another terminal,
and do not run teleop or another controller at the same time.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "common"))

from allocation import Allocator  # noqa: E402
from dock_test_core import DockTestCore, DockView, VehicleSnap  # noqa: E402
from state import State  # noqa: E402

DEFAULT_CONFIG = _HERE / "dock_test.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Dock standoff test controller")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    domain = cfg["node"].get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from interfaces.msg import Actuator, DockAlign
    from std_msgs.msg import String
    import json
    from nav_msgs.msg import Odometry
    from rclpy.node import Node

    class DockTestNode(Node):
        def __init__(self) -> None:
            super().__init__(str(cfg["node"].get("name", "dock_test")))
            lim = cfg["limits"]
            self.core = DockTestCore(cfg)
            self.alloc = Allocator(
                rpm_cap=lim["rpm_cap"],
                fin_deg_cap=lim["fin_deg_cap"],
                u_fin_min=lim["u_fin_min_mps"],
                u_fin_off=lim["u_fin_off_mps"],
                u_fin_full=lim["u_fin_full_mps"],
            )
            self.names = self.alloc.th_ids + self.alloc.fin_ids
            self.st: Optional[State] = None
            self.view = DockView()
            self._last_odom = 0.0
            self._last_dock = 0.0
            self.dt = 1.0 / float(cfg["node"]["rate_hz"])
            self._since_status = 0.0
            self.tripped: Optional[str] = None
            self.pub = self.create_publisher(Actuator, cfg["topics"]["actuator_cmd"], 10)
            self.dbg_pub = self.create_publisher(String, f"/{cfg['node'].get('vessel', 'Mako_01')}/ctrl_debug", 10)   # mode and distance for the viewer
            self._dbg_t = 0.0
            self.create_subscription(Odometry, cfg["topics"]["odometry"], self._on_odom, 10)
            self.create_subscription(DockAlign, cfg["topics"]["dock_align"], self._on_dock, 10)
            self.create_timer(self.dt, self._tick)
            self.get_logger().info(
                f"dock_test: align={cfg['topics']['dock_align']} "
                f"cmd={cfg['topics']['actuator_cmd']}"
            )

        def _on_odom(self, msg: Odometry) -> None:
            self.st = State.from_odom(msg)
            self._last_odom = time.monotonic()

        def _on_dock(self, msg: DockAlign) -> None:
            self.view = DockView(
                fresh=True,
                valid=bool(msg.valid),
                elevation_valid=bool(msg.elevation_valid),
                elevation_rad=float(msg.elevation_rad),
                error_x_px=float(msg.error_x_px),
                error_y_px=float(msg.error_y_px),
                lateral_px=float(msg.lateral_px),
                radius_px=float(msg.radius_px),
                search_yaw_norm=float(msg.search_yaw_norm),
                search_pitch_norm=float(msg.search_pitch_norm),
                search_surge_norm=float(msg.search_surge_norm),
            )
            self._last_dock = time.monotonic()

        def publish(self, values: Dict[str, float]) -> None:
            m = Actuator()
            m.actuator_names = list(self.names)
            m.actuator_values = [float(values.get(n, 0.0)) for n in self.names]
            m.covariance = [0.0] * len(self.names)
            self.pub.publish(m)

        def _safety_reason(self, st: State) -> Optional[str]:
            s = cfg["safety"]
            if st.depth < s["min_depth_m"]:
                return f"too shallow ({st.depth:.2f} m)"
            if st.depth > s["max_depth_m"]:
                return f"too deep ({st.depth:.2f} m)"
            if abs(math.degrees(st.pitch)) > s["max_pitch_deg"]:
                return f"pitch {math.degrees(st.pitch):.1f} deg"
            if abs(math.degrees(st.roll)) > s["max_roll_deg"]:
                return f"roll {math.degrees(st.roll):.1f} deg"
            return None

        def _tick(self) -> None:
            if self.tripped:
                self.publish({})
                raise SystemExit(1)
            now = time.monotonic()
            if self.st is None or now - self._last_odom > float(cfg["safety"]["odom_timeout_s"]):
                self.publish({})
                self.get_logger().warning("odometry stale -> neutral", throttle_duration_sec=2.0)
                return
            why = self._safety_reason(self.st)
            if why:
                self.tripped = why
                self.get_logger().error(f"SAFETY TRIP: {why} -> neutral and exit")
                self.publish({})
                raise SystemExit(1)
            view = self.view
            if self._last_dock == 0.0 or now - self._last_dock > float(cfg["safety"]["dock_timeout_s"]):
                view = DockView(fresh=False)
                self.get_logger().warning("dock_align stale -> neutral", throttle_duration_sec=2.0)
            snap = VehicleSnap(
                speed_u=self.st.speed_u,
                roll=self.st.roll,
                roll_rate=float(self.st.nu[3]),
                pitch_rate=float(self.st.nu[4]),
                yaw_rate=float(self.st.nu[5]),
                heave_rate=float(self.st.vel_ned()[2]),
                dt=self.dt,
            )
            wrench, status = self.core.update(view, snap)
            self._dbg_t += self.dt
            if self._dbg_t >= 0.5:
                self._dbg_t = 0.0
                d = status.get("d_m", float("nan"))
                self.dbg_pub.publish(String(data=json.dumps({"ctrl": "dock_test", "mode": status["mode"], "distance_m": None if d != d else float(d)})))
            out = self.alloc.allocate(wrench, snap.speed_u)
            self.publish(out)
            self._since_status += self.dt
            if self._since_status >= float(cfg["logging"]["status_period_s"]):
                self._since_status = 0.0
                self.get_logger().info(
                    f"mode={status['mode']}  X={status['x_n']:+.1f} N  Z={status['z_n']:+.1f} N  "
                    f"M={status['m_nm']:+.2f}  N={status['n_nm']:+.2f}  u={status['u']:.2f} m/s"
                )

    rclpy.init()
    node = DockTestNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.publish({})
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
