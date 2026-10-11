#!/usr/bin/env python3
"""Synthetic nose camera as a ROS 2 node, so the real dock detector and dock_test can run without mavsim.

  odometry_sim  ->  /Mako_01/camera_03/image/compressed   (what the nose camera would see of the dock lights)
                    /Mako_01/imu_01/data                   (orientation from the same odometry; live_dock_lights needs roll and pitch)

Pair it with sim_offline/fake_vehicle.py. Do NOT run it together with the real mavsim bridge: both would publish the camera topic.
Config: sim_viewer.yaml.   ./run_camera_sim.sh
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HERE)] + [str(_d) for _d in sorted(_HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(_HERE.parent / "common"))

from synthetic_camera import DEFAULT_CONFIG, SyntheticCamera, load_config  # noqa: E402
from state import State  # noqa: E402


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Synthetic dock camera (offline simulator)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    domain = cfg["node"].get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Imu

    class CameraNode(Node):
        def __init__(self) -> None:
            super().__init__("synthetic_camera")
            self.cam = SyntheticCamera(cfg)
            self.st: Optional[State] = None
            self.quat = None
            self.omega = (0.0, 0.0, 0.0)
            t = cfg["topics"]
            self.img_pub = self.create_publisher(CompressedImage, t["image"], 5)
            self.imu_pub = self.create_publisher(Imu, t["imu"], 10)
            self.create_subscription(Odometry, t["odometry"], self._on_odom, 10)
            self.create_timer(1.0 / float(cfg["node"]["rate_hz"]), self._tick)
            self._n = 0
            self.get_logger().info(f"synthetic camera: {t['odometry']} -> {t['image']} + {t['imu']} @ {cfg['node']['rate_hz']} Hz")

        def _on_odom(self, msg: Odometry) -> None:
            self.st = State.from_odom(msg)
            q = msg.pose.pose.orientation
            self.quat = (q.x, q.y, q.z, q.w)
            a = msg.twist.twist.angular
            self.omega = (a.x, a.y, a.z)

        def _tick(self) -> None:
            if self.st is None:
                return
            stamp = self.get_clock().now().to_msg()
            eul_deg = np.degrees(self.st.eul)
            frame = self.cam.render(self.st.pos, eul_deg)
            img = CompressedImage()
            img.header.stamp = stamp
            img.header.frame_id = "camera_03"
            img.format = "jpeg"
            img.data = self.cam.encode_jpeg(frame)
            self.img_pub.publish(img)
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = "imu_01"
            imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = self.quat
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = self.omega
            self.imu_pub.publish(imu)
            self._n += 1

    rclpy.init()
    node = CameraNode()
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
