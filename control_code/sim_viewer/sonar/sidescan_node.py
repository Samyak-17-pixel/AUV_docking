#!/usr/bin/env python3
"""Simulated pair of Omniscan 450 SS side-scan sonars as a ROS 2 node (offline simulator only).

  /Mako_01/odometry_sim  ->  /Mako_01/sonar_01/ping   (port,      interfaces/SideScan)
                             /Mako_01/sonar_02/ping   (starboard, interfaces/SideScan)
                             /Mako_01/altimeter/range (sensor_msgs/Range, straight down, with noise)
                             /Mako_01/sonar/status    (std_msgs/String JSON: range, gain, bins, ping rate, counts; 2 Hz)
  /Mako_01/sonar/cmd     <-  std_msgs/String JSON, e.g. {"range_m": 40, "gain": 5} ("gain": -1 = auto), {"bins": 800}, {"enabled": false}
The pings come from the physics in synthetic_sidescan.py over the terrain in terrain.py (same seed as the 3D view). The vehicle pose is the odometry moved forward with its own velocity
for the time since it arrived (so the pings are smooth even with the realistic 4.5 Hz odometry). Do NOT run it together with the real mavsim bridge (it would publish fake sonar topics there).
Config: sim_viewer.yaml, blocks `sonar:` and `terrain:`.   ./run_sidescan_sim.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HERE)] + [str(_d) for _d in sorted(_HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(_HERE.parent / "common"))

from state import State  # noqa: E402
from synthetic_camera import DEFAULT_CONFIG, load_config  # noqa: E402
from synthetic_sidescan import SideScanConfig, SideScanSim  # noqa: E402
from terrain import get_terrain  # noqa: E402


def sonar_config_from_yaml(cfg: dict) -> SideScanConfig:
    sc = dict(cfg.get("sonar", {}).get("sensor", {}) or {})
    fields = SideScanConfig.__dataclass_fields__
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sc.items() if k in fields}
    return SideScanConfig(**kw)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Simulated Omniscan 450 SS side-scan sonars (offline simulator)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    domain = cfg["node"].get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from interfaces.msg import SideScan
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Range
    from std_msgs.msg import String

    class SideScanNode(Node):
        def __init__(self) -> None:
            super().__init__("synthetic_sidescan")
            sc = cfg.get("sonar", {})
            self.sim = SideScanSim(get_terrain(cfg.get("terrain")), sonar_config_from_yaml(cfg))
            vessel = cfg.get("vessel", "Mako_01")
            t = sc.get("topics", {})
            self.pub = {"port": self.create_publisher(SideScan, t.get("port", f"/{vessel}/sonar_01/ping"), 10), "starboard": self.create_publisher(SideScan, t.get("starboard", f"/{vessel}/sonar_02/ping"), 10)}
            self.alt_pub = self.create_publisher(Range, t.get("altimeter", f"/{vessel}/altimeter/range"), 10)
            self.status_pub = self.create_publisher(String, t.get("status", f"/{vessel}/sonar/status"), 5)
            self.create_subscription(Odometry, cfg["topics"]["odometry"], self._on_odom, 10)
            self.create_subscription(String, t.get("cmd", f"/{vessel}/sonar/cmd"), self._on_cmd, 10)
            self.st: Optional[State] = None
            self.t_odom = 0.0
            self.enabled = True
            self.next = {"port": 0.0, "starboard": 0.0}
            self.count = {"port": 0, "starboard": 0}
            self.compute_ms = 0.0
            self._alt_next = 0.0
            self.create_timer(0.01, self._tick)
            self.create_timer(0.5, self._status)
            self.get_logger().info(f"synthetic side-scan: {cfg['topics']['odometry']} -> sonar_01/02 pings, range {self.sim.cfg.range_m} m, {self.sim.cfg.num_bins} bins, gain {self.sim.cfg.gain_index}")

        def _on_odom(self, msg: Odometry) -> None:
            self.st = State.from_odom(msg)
            self.t_odom = time.monotonic()

        def _on_cmd(self, msg: String) -> None:
            try:
                d = json.loads(msg.data)
            except ValueError:
                return
            if "range_m" in d:
                self.sim.set_range(float(d["range_m"]))
            if "gain" in d:
                self.sim.set_gain(int(d["gain"]))
            if "bins" in d:
                self.sim.set_bins(int(d["bins"]))
            if "enabled" in d:
                self.enabled = bool(d["enabled"])
            self.get_logger().info(f"sonar command {d} -> range {self.sim.cfg.range_m} m, gain {self.sim.cfg.gain_index}, bins {self.sim.cfg.num_bins}, enabled {self.enabled}")

        def _pose_now(self):
            st = self.st
            dt = min(max(time.monotonic() - self.t_odom, 0.0), 0.4)
            return st.pos + st.vel_ned() * dt, st.eul

        def _tick(self) -> None:
            if self.st is None or not self.enabled:
                return
            now = time.monotonic()
            pos, eul = self._pose_now()
            for side in ("port", "starboard"):
                if now >= self.next[side]:
                    t0 = time.perf_counter()
                    r = self.sim.ping(pos, eul, side)
                    self.compute_ms = 0.9 * self.compute_ms + 0.1 * 1000.0 * (time.perf_counter() - t0)
                    m = SideScan()
                    m.header.stamp = self.get_clock().now().to_msg()
                    m.header.frame_id = "sonar_01" if side == "port" else "sonar_02"
                    m.side = side
                    self.count[side] += 1
                    m.ping_number = self.count[side]
                    m.start_range_m, m.range_m = float(r.start_range_m), float(r.range_m)
                    m.sound_speed_mps, m.gain_index, m.gain_db, m.ping_hz = float(r.sound_speed_mps), int(r.gain_index), float(r.gain_db), float(r.ping_hz)
                    m.altitude_m = float(r.altitude_m)
                    m.vehicle_pos_ned = [float(v) for v in pos]
                    m.vehicle_eul_rad = [float(v) for v in eul]
                    m.num_bins = int(len(r.intensity))
                    m.intensity = r.intensity.tolist()
                    self.pub[side].publish(m)
                    self.next[side] = max(self.next[side] + 1.0 / r.ping_hz, now - 0.5 / r.ping_hz)
            if now >= self._alt_next:
                self._alt_next = now + 0.1
                a = self.sim.altimeter(pos, eul)
                if a is not None:
                    rg = Range()
                    rg.header.stamp = self.get_clock().now().to_msg()
                    rg.header.frame_id = "altimeter"
                    rg.radiation_type = Range.ULTRASOUND
                    rg.field_of_view = 0.1
                    rg.min_range, rg.max_range, rg.range = 0.3, 60.0, float(a)
                    self.alt_pub.publish(rg)

        def _status(self) -> None:
            c = self.sim.cfg
            self.status_pub.publish(String(data=json.dumps({"range_m": c.range_m, "gain": c.gain_index, "bins": c.num_bins, "ping_hz": self.sim.ping_hz(), "enabled": self.enabled,
                                                               "pings": dict(self.count), "compute_ms": round(self.compute_ms, 2)})))

    rclpy.init()
    node = SideScanNode()
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
