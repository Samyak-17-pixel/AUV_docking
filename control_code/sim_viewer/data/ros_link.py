"""Live ROS 2 data source: odometry, actuator commands, DockAlign and the camera image -> Telemetry. A plain thread (no Qt)."""

from __future__ import annotations

import math
import os
import sys
import json

import numpy as np
import queue
import threading
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HERE)] + [str(_d) for _d in sorted(_HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(_HERE.parent / "common"))

from state import State  # noqa: E402
from telemetry import Telemetry  # noqa: E402


class RosLink(threading.Thread):
    def __init__(self, telemetry: Telemetry, cfg: dict) -> None:
        super().__init__(daemon=True, name="ros_link")
        self.tel = telemetry
        self.cfg = cfg
        self._stop_evt = threading.Event()
        self.error: Optional[str] = None
        self.counts = {"odom": 0, "cmd": 0, "align": 0, "image": 0}
        self.ready = threading.Event()
        self.sim_status: dict = {}
        self._sim_seen = 0.0
        self._outbox: "queue.Queue[dict]" = queue.Queue()
        self._zero_req = threading.Event()
        self._gain_outbox: "queue.Queue[str]" = queue.Queue()
        self._sonar_outbox: "queue.Queue[dict]" = queue.Queue()

    @property
    def sim_controls_available(self) -> bool:
        """True while a fake_vehicle (which publishes /sim/status) is answering. The real mavsim never does, so its controls stay disabled."""
        return time.monotonic() - self._sim_seen < 3.0

    def send_sim_command(self, d: dict) -> bool:
        self._outbox.put(dict(d))
        return True

    def send_sonar_command(self, d: dict) -> bool:
        """Queue a side-scan command ({"range_m": 40, "gain": 5, "bins": 600, "enabled": true}) for /<vessel>/sonar/cmd."""
        self._sonar_outbox.put(dict(d))
        return True

    def send_gain(self, message_json: str) -> bool:
        """Queue a live-gain message (common/live_gains.py format) for /<vessel>/ctrl_gains. Returns False if the ROS thread is not running."""
        if not self.is_alive() or self.error:
            return False
        self._gain_outbox.put(message_json)
        return True

    def publish_zero(self) -> None:
        """Ask the ROS thread to publish neutral actuator commands (emergency stop; the thread is the only one that may touch rclpy)."""
        self._zero_req.set()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:                                                   # noqa: C901 (one flat wiring function)
        try:
            domain = self.cfg.get("node", {}).get("ros_domain_id")
            if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
                os.environ["ROS_DOMAIN_ID"] = str(domain)
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from sensor_msgs.msg import CompressedImage
            from std_msgs.msg import String
            from interfaces.msg import Actuator, DockAlign

            if not rclpy.ok():
                rclpy.init()
            node = Node("sim_viewer")
            t = self.cfg["topics"]
            vessel = self.cfg["node"].get("vessel", "Mako_01")

            def on_odom(m: Odometry) -> None:
                s = State.from_odom(m)
                self.tel.push_odom(s.pos, s.eul, s.nu)
                self.counts["odom"] += 1

            def on_cmd(m: Actuator) -> None:
                self.tel.push_cmd(m.actuator_names, m.actuator_values)
                self.counts["cmd"] += 1

            def on_align(m: DockAlign) -> None:
                self.tel.push_align({
                    "error_x_px": m.error_x_px, "error_y_px": m.error_y_px, "lateral_px": m.lateral_px,
                    "elevation_deg": math.degrees(m.elevation_rad) if m.elevation_valid else 0.0,
                    "radius_px": m.radius_px, "num_lights": float(m.num_lights), "valid": 1.0 if m.valid else 0.0,
                    "top_x": m.top.x, "top_y": m.top.y, "bottom_x": m.bottom.x, "bottom_y": m.bottom.y, "right_x": m.right.x, "right_y": m.right.y,
                    "left_x": m.left.x, "left_y": m.left.y, "center_x": m.center.x, "center_y": m.center.y,
                    "confidence": m.confidence, "spread_px": m.spread_px, "obliqueness": m.obliqueness, "aligned": 1.0 if m.aligned else 0.0, "center_exact": 1.0 if m.center_exact else 0.0,
                    "error_x_norm": m.error_x_norm, "error_y_norm": m.error_y_norm,
                    "search_yaw_norm": m.search_yaw_norm, "search_pitch_norm": m.search_pitch_norm, "search_surge_norm": m.search_surge_norm,
                })
                self.counts["align"] += 1

            def on_image(m: CompressedImage) -> None:
                self.tel.set_image(bytes(m.data))
                self.counts["image"] += 1

            node.create_subscription(Odometry, t["odometry"], on_odom, 10)
            node.create_subscription(Actuator, f"/{vessel}/actuator_cmd", on_cmd, 10)
            node.create_subscription(DockAlign, f"/{vessel}/dock_align", on_align, 10)
            node.create_subscription(CompressedImage, t["image"], on_image, 2)
            try:                                                    # the simulated side-scan sonars (sidescan_node.py); absent on the real bridge
                from interfaces.msg import SideScan

                def make_on_ping(side):
                    def on_ping(m) -> None:
                        self.tel.push_sonar(side, np.asarray(m.intensity, np.uint16), {"start_range_m": m.start_range_m, "range_m": m.range_m, "gain_db": m.gain_db, "gain_index": m.gain_index,
                                                                                     "altitude_m": m.altitude_m, "pos": list(m.vehicle_pos_ned), "eul": list(m.vehicle_eul_rad), "ping_number": m.ping_number,
                                                                                     "ping_hz": m.ping_hz, "sound_speed_mps": m.sound_speed_mps})
                        self.counts["sonar_" + side] = self.counts.get("sonar_" + side, 0) + 1
                    return on_ping

                node.create_subscription(SideScan, t.get("sonar_port", f"/{vessel}/sonar_01/ping"), make_on_ping("port"), 20)
                node.create_subscription(SideScan, t.get("sonar_starboard", f"/{vessel}/sonar_02/ping"), make_on_ping("starboard"), 20)

                def on_sonar_status(m: String) -> None:
                    try:
                        self.tel.sonar_status = json.loads(m.data)
                    except ValueError:
                        pass
                node.create_subscription(String, f"/{vessel}/sonar/status", on_sonar_status, 5)
                sonar_pub = node.create_publisher(String, f"/{vessel}/sonar/cmd", 10)
            except ImportError:
                sonar_pub = None

            def on_sim_status(m: String) -> None:
                try:
                    self.sim_status = json.loads(m.data)
                    self._sim_seen = time.monotonic()
                except ValueError:
                    pass

            def on_ctrl(m: String) -> None:
                try:
                    d = json.loads(m.data)
                    if isinstance(d, dict):
                        self.tel.push_ctrl(d)
                        self.counts["ctrl"] = self.counts.get("ctrl", 0) + 1
                except ValueError:
                    pass

            node.create_subscription(String, f"/{vessel}/ctrl_debug", on_ctrl, 10)
            node.create_subscription(String, f"/{vessel}/sim/status", on_sim_status, 5)
            sim_pub = node.create_publisher(String, f"/{vessel}/sim/cmd", 10)
            act_pub = node.create_publisher(Actuator, f"/{vessel}/actuator_cmd", 10)
            gain_pub = node.create_publisher(String, f"/{vessel}/ctrl_gains", 10)
            self.tel.source_name = f"ROS domain {os.environ.get('ROS_DOMAIN_ID', '0')}"
            self.ready.set()
            while not self._stop_evt.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                while True:
                    try:
                        gain_pub.publish(String(data=self._gain_outbox.get_nowait()))
                    except queue.Empty:
                        break
                if self._zero_req.is_set():                                  # emergency stop asked for from the GUI: send neutral a few times
                    self._zero_req.clear()
                    names = ["th_01", "th_02", "th_03", "cs_04", "cs_06", "cs_07", "cs_08"]
                    for _ in range(3):
                        z = Actuator()
                        z.actuator_names, z.actuator_values, z.covariance = names, [0.0] * len(names), [0.0] * len(names)
                        act_pub.publish(z)
                        time.sleep(0.02)
                while True:
                    try:
                        d = self._sonar_outbox.get_nowait()
                        if sonar_pub is not None:
                            sonar_pub.publish(String(data=json.dumps(d)))
                    except queue.Empty:
                        break
                while True:
                    try:
                        sim_pub.publish(String(data=json.dumps(self._outbox.get_nowait())))
                    except queue.Empty:
                        break
            node.destroy_node()
        except Exception as exc:                                             # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
