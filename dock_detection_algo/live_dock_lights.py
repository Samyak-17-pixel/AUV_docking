#!/usr/bin/env python3
"""Live dock-light viewer + /<vessel>/dock_align publisher.

Windows (main thread, required by OpenCV):
  - Dock camera : the picture with labelled lights, range / bearing / elevation / view-angle gauges, a top-down mini map and a history strip (dock_hud.py)
  - Bloom mask  : the binary mask in colour with the chosen lights marked (+ the trackbars)
  - Dock align  : (optional, --align-window) steering arrows, target, numbers and rolling plots of the DockAlign message (dock_align_view.py can also run on its own)

Publishes interfaces/DockAlign on /<vessel>/dock_align (from camera namespace).

  ./run_live.sh
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import Header

from dock_align_msg import align_topic_from_camera, quat_to_roll_pitch
from dock_detection_config import DEFAULT_CONFIG, detector_kwargs, load_config
from dock_detector import DockDetector
import dock_hud

DEFAULT_TOPIC = "/Mako_01/camera_03/image/compressed"
WIN_CAM = "Dock camera"
WIN_MASK = "Bloom mask"
dock_hud_WIN_ALIGN = "Dock align"


_EMPTY_INFO = {"valid": False, "num": 0, "status": "waiting", "confidence": 0.0, "err_x_px": 0.0, "err_y_px": 0.0, "err_x_norm": 0.0, "err_y_norm": 0.0, "radius_px": 0.0,
               "spread_px": 0.0, "lateral_px": 0.0, "obliqueness": 0.0, "elevation_deg": 0.0, "elevation_valid": False, "aligned": False, "center_exact": False,
               "w": 640, "h": 480, "pts": {"top": None, "bottom": None, "left": None, "right": None}, "center": None, "search": (0.0, 0.0, 0.0), "t": 0.0, "vfov": 60.0,
               "ring_resid": float("nan")}


def _placeholder(text: str, w: int = 640, h: int = 360) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (20, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


class DockLightsLive(Node):
    """Camera subscriber + DockAlign publisher. GUI stays on the main thread."""

    def __init__(self, camera_topic: str, align_topic: str, imu_topic: str, roll_sign: float = 1.0, pitch_sign: float = 1.0) -> None:
        super().__init__("dock_lights_live")
        self._roll_sign = -1.0 if roll_sign < 0 else 1.0
        self._pitch_sign = -1.0 if pitch_sign < 0 else 1.0
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._msg_count = 0
        self._align_topic = align_topic
        self._roll_rad = 0.0
        self._pitch_rad: float | None = None

        from interfaces.msg import DockAlign

        self.create_subscription(CompressedImage, camera_topic, self._on_image, 10)
        self.create_subscription(Imu, imu_topic, self._on_imu, 10)
        self._align_pub = self.create_publisher(DockAlign, align_topic, 10)
        self.get_logger().info(f"Camera: {camera_topic}")
        self.get_logger().info(f"IMU: {imu_topic}")
        self.get_logger().info(f"Publishing DockAlign on {align_topic}")

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        roll, pitch = quat_to_roll_pitch(q.x, q.y, q.z, q.w)
        roll, pitch = roll * self._roll_sign, pitch * self._pitch_sign
        with self._lock:
            self._roll_rad = roll
            self._pitch_rad = pitch

    def take_attitude(self) -> tuple[float, float | None]:
        with self._lock:
            return self._roll_rad, self._pitch_rad

    def _on_image(self, msg: CompressedImage) -> None:
        with self._lock:
            self._latest_jpeg = bytes(msg.data)
            self._msg_count += 1

    def take_jpeg(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest_jpeg, self._msg_count

    def publish_align(self, align_msg) -> None:
        self._align_pub.publish(align_msg)


def _open_windows(cfg: dict) -> None:
    m, p = cfg["mask"], cfg["peaks"]
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAM, 1000, 657)
    cv2.resizeWindow(WIN_MASK, 800, 500)
    cv2.createTrackbar("V_thresh", WIN_MASK, int(m.get("v_thresh", 180)), 255, lambda _x: None)
    cv2.createTrackbar("MinArea", WIN_MASK, int(m.get("min_area", 20)), 2000, lambda _x: None)
    cv2.createTrackbar("Open", WIN_MASK, int(m.get("open_k", 3)), 21, lambda _x: None)
    cv2.createTrackbar("Close", WIN_MASK, int(m.get("close_k", 7)), 31, lambda _x: None)
    cv2.createTrackbar("CyanAssist", WIN_MASK, int(bool(m.get("use_cyan_assist", True))), 1, lambda _x: None)
    cv2.createTrackbar("MaxBlobs", WIN_MASK, int(m.get("max_blobs", 4)), 12, lambda _x: None)
    cv2.createTrackbar("PeakMode", WIN_MASK, int(bool(p.get("peak_mode", True))), 1, lambda _x: None)
    cv2.createTrackbar("PeakSep", WIN_MASK, int(p.get("peak_sep", 28)), 120, lambda _x: None)
    cv2.createTrackbar("CorePct", WIN_MASK, int(p.get("core_pct", 92)), 99, lambda _x: None)

    cv2.imshow(WIN_CAM, dock_hud.draw_camera_view(None, _EMPTY_INFO))
    cv2.imshow(WIN_MASK, dock_hud.draw_mask_view(None, _EMPTY_INFO))
    cv2.waitKey(1)


def imu_topic_from_camera(camera_topic: str) -> str:
    """'/Mako_01/camera_03/image/compressed' → '/Mako_01/imu_01/data'."""
    parts = camera_topic.strip("/").split("/")
    if parts:
        return f"/{parts[0]}/imu_01/data"
    return "/imu_01/data"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = DEFAULT_CONFIG
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            config_path = Path(argv[i + 1])
    align_win_req = "--align-window" in argv
    headless = "--no-gui" in argv   # publish DockAlign without OpenCV windows (tests, closed-loop sim); trackbars use the YAML values
    align_win = align_win_req and not headless
    cfg = load_config(config_path)
    print(f"Detection config: {config_path}" + ("  (no GUI)" if headless else ""), flush=True)
    topic = cfg["camera"].get("topic") or DEFAULT_TOPIC
    align_topic = cfg["camera"].get("align_topic") or ""
    if "--topic" in argv:
        i = argv.index("--topic")
        if i + 1 < len(argv):
            topic = argv[i + 1]
    if "--align-topic" in argv:
        i = argv.index("--align-topic")
        if i + 1 < len(argv):
            align_topic = argv[i + 1]
    if not align_topic:
        align_topic = align_topic_from_camera(topic)
    imu_topic = cfg["camera"].get("imu_topic") or imu_topic_from_camera(topic)

    try:
        from interfaces.msg import DockAlign  # noqa: F401
    except ImportError:
        print(
            "ERROR: interfaces.msg.DockAlign not found.\n"
            "  cd ~/Docking/control_code/ws && "
            "source /opt/ros/humble/setup.bash && "
            "colcon build --packages-select interfaces && "
            "source install/setup.bash\n"
            "Or just use ./run_live.sh which sources the workspace.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not headless and not (hasattr(cv2, "imshow") and hasattr(cv2, "namedWindow")):
        print(
            "ERROR: This OpenCV build has no GUI (imshow). "
            "Install: pip install opencv-python",
            file=sys.stderr,
        )
        sys.exit(1)

    print(("Headless, " if headless else "Opening GUI windows...  ") + f"camera={topic}", flush=True)
    print(f"DockAlign topic: {align_topic}", flush=True)
    if not headless:
        _open_windows(cfg)
    base_kw = detector_kwargs(cfg)
    _m, _p = cfg["mask"], cfg["peaks"]
    _defaults = {
        "V_thresh": int(_m.get("v_thresh", 180)), "MinArea": int(_m.get("min_area", 20)),
        "Open": int(_m.get("open_k", 3)), "Close": int(_m.get("close_k", 7)),
        "CyanAssist": int(bool(_m.get("use_cyan_assist", True))), "MaxBlobs": int(_m.get("max_blobs", 4)),
        "PeakMode": int(bool(_p.get("peak_mode", True))), "PeakSep": int(_p.get("peak_sep", 28)),
        "CorePct": int(_p.get("core_pct", 92)),
    }

    def tb(name: str) -> int:
        """Trackbar value (GUI) or the YAML value (headless)."""
        return _defaults[name] if headless else cv2.getTrackbarPos(name, WIN_MASK)

    align_cfg, hud_cfg = cfg["alignment"], cfg["hud"]
    frac = float(align_cfg.get("spread_align_frac", 0.08))
    min_px = float(align_cfg.get("spread_align_min_px", 8.0))
    conf_base = float(align_cfg.get("confidence_base", 0.4))
    lateral_exact_px = float(align_cfg.get("lateral_exact_px", 8.0))
    lateral_exact_frac = float(align_cfg.get("lateral_exact_frac", 0.06))
    vfov_deg = float(cfg["camera"].get("vfov_deg", 60.0))
    print(
        "Windows: 'Dock camera' + 'Bloom mask'" + (" + 'Dock align'" if align_win else "") + ". "
        "error_x>0 => dock RIGHT of center. p = print tuned values, q/Esc to quit.",
        flush=True,
    )

    detector = DockDetector(cfg=cfg)
    hist = dock_hud.History(float(hud_cfg.get("history_s", 12.0)))
    if align_win:
        cv2.namedWindow(dock_hud_WIN_ALIGN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(dock_hud_WIN_ALIGN, 900, 612)
    rclpy.init()
    node = DockLightsLive(topic, align_topic, imu_topic, float(cfg["camera"].get("imu_roll_sign", 1)), float(cfg["camera"].get("imu_pitch_sign", 1)))
    last_count = 0
    last_status_t = time.time()
    saw_frame = False

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)

            jpeg, count = node.take_jpeg()
            if jpeg is not None and count != last_count:
                last_count = count
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    node.get_logger().warning("Failed to decode JPEG")
                else:
                    kw = dict(
                        v_thresh=tb("V_thresh"), use_cyan_assist=tb("CyanAssist") == 1, open_k=tb("Open"), close_k=tb("Close"),
                        min_area=float(max(1, tb("MinArea"))), max_blobs=tb("MaxBlobs"), peak_mode=tb("PeakMode") == 1,
                        peak_sep=max(5, tb("PeakSep")), core_pct=max(80, tb("CorePct")),
                    )
                    roll_rad, pitch_rad = node.take_attitude()
                    h, w = bgr.shape[:2]
                    header = Header()
                    header.stamp = node.get_clock().now().to_msg()
                    header.frame_id = "camera"
                    res = detector.process(bgr, roll_rad, pitch_rad, time.time(), header, kw)
                    mask, leveled, geo, align_msg = res.mask, res.cores, res.geo, res.msg
                    cores = leveled
                    node.publish_align(align_msg)

                    # Show the roll-compensated frame so T/B/L/R markers sit on the lights.
                    # OpenCV's positive angle is the opposite of our y-down unrotation.
                    info = None
                    if not headless:
                        view_angle = -math.degrees(roll_rad)
                        view_m = cv2.getRotationMatrix2D((0.5 * w, 0.5 * h), view_angle, 1.0)
                        view = cv2.warpAffine(bgr, view_m, (w, h))
                        view_mask = cv2.warpAffine(mask, view_m, (w, h), flags=cv2.INTER_NEAREST)
                        info = dock_hud.info_from_result(res, vfov_deg, time.time())
                        hist.add(info)
                        cam_vis = dock_hud.draw_camera_view(view, info, hist)
                        mask_vis = dock_hud.draw_mask_view(view_mask, info)

                    if not headless:
                        cv2.imshow(WIN_CAM, cam_vis)
                        cv2.imshow(WIN_MASK, mask_vis)
                        if align_win:
                            cv2.imshow(dock_hud_WIN_ALIGN, dock_hud.draw_align_view(info, hist, 0.0))
                    if not saw_frame:
                        saw_frame = True
                        print(
                            f"First frame {bgr.shape[1]}x{bgr.shape[0]}, "
                            f"cores={len(cores)}, geo_ok={geo.ok}, "
                            f"publishing {align_topic}",
                            flush=True,
                        )

            now = time.time()
            if not saw_frame and now - last_status_t > 2.0:
                last_status_t = now
                print(
                    f"Still waiting for messages on {topic} "
                    f"(count={count}). Is the sim camera publishing?",
                    flush=True,
                )
                if not headless:
                    w_info = dict(_EMPTY_INFO, status=f"waiting for {topic}"[:34])
                    cv2.imshow(WIN_CAM, dock_hud.draw_camera_view(None, w_info))

            key = 0 if headless else cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                print(
                    "# current trackbar values -> paste into dock_detection.yaml\n"
                    f"mask:  {{v_thresh: {cv2.getTrackbarPos('V_thresh', WIN_MASK)}, "
                    f"min_area: {cv2.getTrackbarPos('MinArea', WIN_MASK)}, "
                    f"open_k: {cv2.getTrackbarPos('Open', WIN_MASK)}, "
                    f"close_k: {cv2.getTrackbarPos('Close', WIN_MASK)}, "
                    f"use_cyan_assist: {str(cv2.getTrackbarPos('CyanAssist', WIN_MASK) == 1).lower()}, "
                    f"max_blobs: {cv2.getTrackbarPos('MaxBlobs', WIN_MASK)}}}\n"
                    f"peaks: {{peak_mode: {str(cv2.getTrackbarPos('PeakMode', WIN_MASK) == 1).lower()}, "
                    f"peak_sep: {cv2.getTrackbarPos('PeakSep', WIN_MASK)}, "
                    f"core_pct: {cv2.getTrackbarPos('CorePct', WIN_MASK)}}}",
                    flush=True,
                )
            if key in (27, ord("q")):
                print("Quit.", flush=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if not headless:
            cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
