#!/usr/bin/env python3
"""Terminal docking controller (ROS 2): aligns with the dock and enters the funnel. Config: terminal_docking.yaml.

  ./run_terminal_docking.sh

Reads /Mako_01/dock_align (the light pixels from the detector) and /Mako_01/odometry_sim, publishes /Mako_01/actuator_cmd and /Mako_01/ctrl_debug.
Start it once the dock is in the camera's view (run dock_detection_algo/run_live.sh, or the viewer's offline stack). Ctrl-C and every safety trip
publish neutral. Do not run teleop or another controller at the same time. The logic lives in terminal_docking_core.py (no ROS).
"""

from __future__ import annotations

import argparse
import csv
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
from outdirs import out_dir  # noqa: E402

from allocation import Allocator  # noqa: E402
from state import State  # noqa: E402
from live_gains import attach as attach_live_gains  # noqa: E402
from terminal_docking_core import DockObs, TerminalDockingCore  # noqa: E402

DEFAULT_CONFIG = _HERE / "terminal_docking.yaml"


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Terminal docking controller")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    domain = cfg["node"].get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from interfaces.msg import Actuator, DockAlign
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import String

    class TerminalDockingNode(Node):
        def __init__(self) -> None:
            super().__init__("terminal_docking")
            lim = cfg["limits"]
            self.core = TerminalDockingCore(cfg)
            self.alloc = Allocator(rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"], u_fin_min=lim["u_fin_min_mps"], u_fin_off=lim["u_fin_off_mps"],
                                   u_fin_full=lim["u_fin_full_mps"], small_force_n=lim.get("small_force_n", 0.0))
            self.names = self.alloc.th_ids + self.alloc.fin_ids
            self.obs: Optional[DockObs] = None
            self.dt = 1.0 / float(cfg["node"]["rate_hz"])
            self.t0 = time.monotonic()
            self.tripped: Optional[str] = None
            self.rows: List[dict] = []
            self._since_status = 0.0
            self._dbg_t = 0.0
            vessel = cfg["node"].get("vessel", "Mako_01")
            self.pub = self.create_publisher(Actuator, cfg["topics"]["actuator_cmd"], 10)
            self.dbg_pub = self.create_publisher(String, cfg["topics"].get("ctrl_debug", f"/{vessel}/ctrl_debug"), 10)
            self.create_subscription(Odometry, cfg["topics"]["odometry"], self._on_odom, 10)
            self.create_subscription(DockAlign, cfg["topics"]["dock_align"], self._on_dock, 10)
            attach_live_gains(self, vessel, "terminal_docking", self.core.loops, self.get_logger().info)
            self.create_timer(self.dt, self._tick)
            self.get_logger().info(f"terminal_docking: align={cfg['topics']['dock_align']} odom={cfg['topics']['odometry']} cmd={cfg['topics']['actuator_cmd']}")

        def now(self) -> float:
            return time.monotonic() - self.t0

        def _on_odom(self, msg: Odometry) -> None:
            self.core.pose.push(State.from_odom(msg), self.now())

        def _on_dock(self, msg: DockAlign) -> None:
            self.obs = DockObs.from_msg(msg, fresh=True)

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
            tilt = math.degrees(max(abs(st.roll), abs(st.pitch)))
            if tilt > s["max_tilt_deg"]:
                return f"tilt {tilt:.1f} deg"
            return None

        def _tick(self) -> None:
            if self.tripped:
                self.publish({})
                raise SystemExit(1)
            t = self.now()
            st_now = self.core.pose.at(t)
            if st_now is None:
                self.publish({})
                self.get_logger().warning("waiting for odometry -> neutral", throttle_duration_sec=2.0)
                return
            why = self._safety_reason(st_now)
            if why:
                self.tripped = why
                self.get_logger().error(f"SAFETY TRIP: {why} -> neutral and exit")
                self.publish({})
                raise SystemExit(1)
            obs, self.obs = (self.obs or DockObs(fresh=False)), None            # each detection is used once
            wrench, stat = self.core.update(obs, st_now, self.dt, t)
            out = self.alloc.allocate(wrench, st_now.speed_u)
            self.publish(out)
            self._dbg_t += self.dt
            if self._dbg_t >= 0.5:
                self._dbg_t = 0.0
                d = {"ctrl": "terminal_docking", "mode": stat.get("phase"), "distance_m": stat.get("s_nose"), "dock_axis_err_deg": stat.get("sig_axis_deg"),
                     "depth_sp": stat.get("z_sp"), "yaw_sp_deg": stat.get("psi_des_deg"), "cross_track_m": stat.get("e"), "retries": stat.get("retries"), "repositions": stat.get("repositions")}
                self.dbg_pub.publish(String(data=json.dumps({k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()})))
            self.rows.append({"t": round(t, 3), "phase": stat.get("phase"), **{k: stat.get(k) for k in ("s", "e", "chi_deg", "dz", "s_nose", "u_target", "psi_des_deg", "sig_pos", "sig_axis_deg")},
                              "x": st_now.pos[0], "y": st_now.pos[1], "z": st_now.pos[2], "roll_deg": math.degrees(st_now.roll), "pitch_deg": math.degrees(st_now.pitch),
                              "yaw_deg": math.degrees(st_now.yaw), "u": st_now.nu[0], "v": st_now.nu[1], "w": st_now.nu[2], "p": st_now.nu[3], "q": st_now.nu[4], "r": st_now.nu[5],
                              **{k: round(v, 2) for k, v in out.items()}})
            self._since_status += self.dt
            if self._since_status >= float(cfg["logging"].get("status_period_s", 2.0)):
                self._since_status = 0.0
                self.get_logger().info(
                    f"{stat.get('phase')}  s={stat.get('s', float('nan')):.2f} m  e={stat.get('e', float('nan')):+.2f} m  chi={stat.get('chi_deg', float('nan')):+.1f} deg  "
                    f"dz={stat.get('dz', float('nan')):+.2f} m  u={st_now.speed_u:.2f} m/s  retries={stat.get('retries', 0)}")

    rclpy.init()
    node = TerminalDockingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.publish({})
        except Exception:                                                  # noqa: BLE001
            pass
        if node.rows:
            d = out_dir(cfg["logging"]["dir"])
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"terminal_docking_{time.strftime('%Y%m%d_%H%M%S')}.csv"
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
