#!/usr/bin/env python3
"""Live dock-light viewer + /<vessel>/dock_align publisher.

Windows (main thread, required by OpenCV):
  - Dock camera : RGB + T/B/L/R, diameter, center, distances, align errors
  - Bloom mask  : binary mask (+ T/B/L/R markers)

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

from dock_acquire import PartialAcquire
from dock_align_msg import align_topic_from_camera, build_dock_align_msg, quat_to_roll_pitch
from dock_detection_config import DEFAULT_CONFIG, detector_kwargs, load_config
from dock_geometry import draw_dock_geometry, draw_mask_debug, evaluate_dock_geometry, unrotate_cores
from dock_light_mask import bloom_mask_and_cores

DEFAULT_TOPIC = "/Mako_01/camera_03/image/compressed"
WIN_CAM = "Dock camera"
WIN_MASK = "Bloom mask"


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


def _draw_align_hud(
    vis: np.ndarray, align_msg, deadband_px: float = 2.0, ok_px: float = 10.0
) -> np.ndarray:
    """Append pixel / norm errors on the camera overlay."""
    y = vis.shape[0] - 58
    if align_msg.valid:
        text = (
            f"err_x={align_msg.error_x_px:+.1f}px ({align_msg.error_x_norm:+.3f})  "
            f"err_y={align_msg.error_y_px:+.1f}px ({align_msg.error_y_norm:+.3f})"
        )
        hint = "RIGHT" if align_msg.error_x_px > deadband_px else (
            "LEFT" if align_msg.error_x_px < -deadband_px else "X-OK"
        )
        hint_y = "DOWN" if align_msg.error_y_px > deadband_px else (
            "UP" if align_msg.error_y_px < -deadband_px else "Y-OK"
        )
        lat = "LAT-R" if align_msg.lateral_px > deadband_px else (
            "LAT-L" if align_msg.lateral_px < -deadband_px else "LAT-OK"
        )
        if align_msg.elevation_valid:
            elev = f"elev={math.degrees(align_msg.elevation_rad):+.1f}deg"
        else:
            elev = "elev=no imu"
        color = (0, 255, 0) if abs(align_msg.error_x_px) < ok_px and abs(align_msg.error_y_px) < ok_px else (0, 200, 255)
        cv2.putText(vis, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"cmd hint: {hint} / {hint_y}  {lat}  {elev}  conf={align_msg.confidence:.2f}",
            (8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            f"dock_align invalid: {align_msg.status}",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 128, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"search yaw={align_msg.search_yaw_norm:+.2f}  "
            f"pitch={align_msg.search_pitch_norm:+.2f}  "
            f"surge={align_msg.search_surge_norm:+.2f}",
            (8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 128, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


class DockLightsLive(Node):
    """Camera subscriber + DockAlign publisher. GUI stays on the main thread."""

    def __init__(self, camera_topic: str, align_topic: str, imu_topic: str) -> None:
        super().__init__("dock_lights_live")
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
    cv2.resizeWindow(WIN_CAM, 960, 540)
    cv2.resizeWindow(WIN_MASK, 960, 540)
    cv2.createTrackbar("V_thresh", WIN_MASK, int(m.get("v_thresh", 180)), 255, lambda _x: None)
    cv2.createTrackbar("MinArea", WIN_MASK, int(m.get("min_area", 20)), 2000, lambda _x: None)
    cv2.createTrackbar("Open", WIN_MASK, int(m.get("open_k", 3)), 21, lambda _x: None)
    cv2.createTrackbar("Close", WIN_MASK, int(m.get("close_k", 7)), 31, lambda _x: None)
    cv2.createTrackbar("CyanAssist", WIN_MASK, int(bool(m.get("use_cyan_assist", True))), 1, lambda _x: None)
    cv2.createTrackbar("MaxBlobs", WIN_MASK, int(m.get("max_blobs", 4)), 12, lambda _x: None)
    cv2.createTrackbar("PeakMode", WIN_MASK, int(bool(p.get("peak_mode", True))), 1, lambda _x: None)
    cv2.createTrackbar("PeakSep", WIN_MASK, int(p.get("peak_sep", 28)), 120, lambda _x: None)
    cv2.createTrackbar("CorePct", WIN_MASK, int(p.get("core_pct", 92)), 99, lambda _x: None)

    cv2.imshow(WIN_CAM, _placeholder("Waiting for camera frames..."))
    cv2.imshow(WIN_MASK, np.zeros((360, 640), dtype=np.uint8))
    cv2.waitKey(1)


def imu_topic_from_camera(camera_topic: str) -> str:
    """'/Mako_01/camera_03/image/compressed' → '/Mako_01/imu_01/data'."""
    parts = camera_topic.strip("/").split("/")
    if parts:
        return f"/{parts[0]}/imu_01/data"
    return "/imu_01/data"


def _resplit_cores(bgr: np.ndarray, cores: list, kw: dict) -> list:
    """If peaks merged, try again at half the separation before moving the vehicle."""
    if len(cores) >= 4 or not kw.get("peak_mode", True):
        return cores
    tighter = dict(kw)
    tighter["peak_sep"] = max(5, int(kw.get("peak_sep", 28)) // 2)
    _mask, retry = bloom_mask_and_cores(bgr, **tighter)
    if len(retry) > len(cores):
        return retry
    return cores


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = DEFAULT_CONFIG
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            config_path = Path(argv[i + 1])
    cfg = load_config(config_path)
    print(f"Detection config: {config_path}", flush=True)
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

    if not (hasattr(cv2, "imshow") and hasattr(cv2, "namedWindow")):
        print(
            "ERROR: This OpenCV build has no GUI (imshow). "
            "Install: pip install opencv-python",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Opening GUI windows...  camera={topic}", flush=True)
    print(f"DockAlign topic: {align_topic}", flush=True)
    _open_windows(cfg)
    base_kw = detector_kwargs(cfg)
    align_cfg, hud_cfg = cfg["alignment"], cfg["hud"]
    frac = float(align_cfg.get("spread_align_frac", 0.08))
    min_px = float(align_cfg.get("spread_align_min_px", 8.0))
    conf_base = float(align_cfg.get("confidence_base", 0.4))
    lateral_exact_px = float(align_cfg.get("lateral_exact_px", 8.0))
    lateral_exact_frac = float(align_cfg.get("lateral_exact_frac", 0.06))
    hfov_deg = float(cfg["camera"].get("hfov_deg", 60.0))
    acq_cfg = cfg.get("acquire", {})
    acquire = PartialAcquire(
        edge_frac=float(acq_cfg.get("edge_frac", 0.10)),
        nod_pitch_deg=float(acq_cfg.get("nod_pitch_deg", 20.0)),
        yaw_wiggle_deg=float(acq_cfg.get("yaw_wiggle_deg", 8.0)),
        nod_period_s=float(acq_cfg.get("nod_period_s", 8.0)),
        backup_radius_frac=float(acq_cfg.get("backup_radius_frac", 0.22)),
        backup_after_s=float(acq_cfg.get("backup_after_s", 8.0)),
        hfov_deg=hfov_deg,
        flow_surge_norm=float(acq_cfg.get("flow_surge_norm", 0.35)),
    )
    print(
        "Windows: 'Dock camera' + 'Bloom mask'. "
        "error_x>0 => dock RIGHT of center. p = print tuned values, q/Esc to quit.",
        flush=True,
    )

    rclpy.init()
    node = DockLightsLive(topic, align_topic, imu_topic)
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
                    v_th = cv2.getTrackbarPos("V_thresh", WIN_MASK)
                    min_area = float(max(1, cv2.getTrackbarPos("MinArea", WIN_MASK)))
                    open_k = cv2.getTrackbarPos("Open", WIN_MASK)
                    close_k = cv2.getTrackbarPos("Close", WIN_MASK)
                    use_cyan = cv2.getTrackbarPos("CyanAssist", WIN_MASK) == 1
                    max_blobs = cv2.getTrackbarPos("MaxBlobs", WIN_MASK)
                    peak_mode = cv2.getTrackbarPos("PeakMode", WIN_MASK) == 1
                    peak_sep = max(5, cv2.getTrackbarPos("PeakSep", WIN_MASK))
                    core_pct = cv2.getTrackbarPos("CorePct", WIN_MASK)
                    if core_pct < 80:
                        core_pct = 80

                    kw = dict(base_kw)
                    kw.update(
                        v_thresh=v_th,
                        use_cyan_assist=use_cyan,
                        open_k=open_k,
                        close_k=close_k,
                        min_area=min_area,
                        max_blobs=max_blobs,
                        peak_mode=peak_mode,
                        peak_sep=peak_sep,
                        core_pct=core_pct,
                    )
                    mask, cores = bloom_mask_and_cores(bgr, **kw)
                    cores = _resplit_cores(bgr, cores, kw)
                    roll_rad, pitch_rad = node.take_attitude()
                    h, w = bgr.shape[:2]
                    leveled = unrotate_cores(cores, roll_rad, 0.5 * w, 0.5 * h)
                    geo = evaluate_dock_geometry(
                        leveled,
                        lateral_exact_px=lateral_exact_px,
                        lateral_exact_frac=lateral_exact_frac,
                    )
                    search = None
                    if not geo.ok and len(leveled) < 4:
                        search = acquire.update(leveled, w, h, time.time())
                    else:
                        acquire.reset()

                    header = Header()
                    header.stamp = node.get_clock().now().to_msg()
                    header.frame_id = "camera"
                    align_msg = build_dock_align_msg(
                        header=header,
                        geo=geo,
                        cores=leveled,
                        image_width=w,
                        image_height=h,
                        spread_align_frac=frac,
                        spread_align_min_px=min_px,
                        confidence_base=conf_base,
                        pitch_rad=pitch_rad,
                        hfov_deg=hfov_deg,
                        acquire=search,
                        flow_surge_norm=float(acq_cfg.get("flow_surge_norm", 0.35)),
                    )
                    node.publish_align(align_msg)

                    # Show the roll-compensated frame so T/B/L/R markers sit on the lights.
                    # OpenCV's positive angle is the opposite of our y-down unrotation.
                    view_angle = -math.degrees(roll_rad)
                    view_m = cv2.getRotationMatrix2D((0.5 * w, 0.5 * h), view_angle, 1.0)
                    view = cv2.warpAffine(bgr, view_m, (w, h))
                    view_mask = cv2.warpAffine(mask, view_m, (w, h), flags=cv2.INTER_NEAREST)
                    cam_vis = draw_dock_geometry(
                        view, geo, len(leveled), spread_align_frac=frac, spread_align_min_px=min_px
                    )
                    cam_vis = _draw_align_hud(
                        cam_vis,
                        align_msg,
                        float(hud_cfg.get("hint_deadband_px", 2.0)),
                        float(hud_cfg.get("ok_error_px", 10.0)),
                    )
                    mask_vis = draw_mask_debug(view_mask, geo)

                    cv2.imshow(WIN_CAM, cam_vis)
                    cv2.imshow(WIN_MASK, mask_vis)
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
                cv2.imshow(WIN_CAM, _placeholder(f"Waiting for {topic}"))

            key = cv2.waitKey(1) & 0xFF
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
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
