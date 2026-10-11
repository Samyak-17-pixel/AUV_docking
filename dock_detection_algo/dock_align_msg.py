"""Build interfaces/DockAlign messages from dock geometry."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from geometry_msgs.msg import Point
from std_msgs.msg import Header

from dock_acquire import AcquireCommand, track_surge_norm
from dock_geometry import DockGeometry

Core = Tuple[float, float]


def quat_to_roll_pitch(x: float, y: float, z: float, w: float) -> Tuple[float, float]:
    """ZYX roll and pitch [rad], same convention as control_code/common/state.py.

    Positive pitch is nose up. Positive roll is starboard down.
    """
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def focal_length_px(image_height: int, vfov_deg: float) -> float:
    """Pinhole fx=fy [px] from the VERTICAL field of view and the image height.

    The sim renders camera_03 with a three.js PerspectiveCamera(fov, width/height, ...), whose fov is the
    vertical one (mavsim-controller/core/visualizer_server.py, and its test_overlay_projection.py computes
    f = (height/2) / tan(fov/2)). For 640x480 and fov 60 this gives f = 415.7 px and a horizontal FOV of
    75.2 deg. Treating the 60 deg as horizontal (f = 554 px) makes elevations 25% too small.
    """
    half = max(math.radians(vfov_deg) * 0.5, 1e-3)
    return (0.5 * float(image_height)) / math.tan(half)


def hfov_from_vfov_deg(vfov_deg: float, image_width: int, image_height: int) -> float:
    """Horizontal field of view [deg] for square pixels."""
    half = math.tan(math.radians(vfov_deg) * 0.5) * float(image_width) / max(float(image_height), 1.0)
    return math.degrees(2.0 * math.atan(half))


def elevation_down_rad(dock_cy: float, cy_img: float, fy: float, pitch_rad: float) -> float:
    """Angle of the dock center below the horizon [rad].

    Positive means the dock is deeper than the camera, so heave should go down.
    Pixel y grows downward, and pitch_rad is nose-up, so a nose-up camera looking
    straight at the dock reports a negative elevation (heave up).
    """
    return math.atan((dock_cy - cy_img) / max(fy, 1.0)) - pitch_rad


def align_topic_from_camera(camera_topic: str) -> str:
    """'/Mako_01/camera_03/image/compressed' → '/Mako_01/dock_align'."""
    parts = camera_topic.strip("/").split("/")
    if parts:
        return f"/{parts[0]}/dock_align"
    return "/dock_align"


def _point(p: Optional[Core]) -> Point:
    msg = Point()
    if p is None:
        return msg
    msg.x = float(p[0])
    msg.y = float(p[1])
    msg.z = 0.0
    return msg


def build_dock_align_msg(
    *,
    header: Header,
    geo: DockGeometry,
    cores: List[Core],
    image_width: int,
    image_height: int,
    spread_align_frac: float = 0.08,
    spread_align_min_px: float = 8.0,
    confidence_base: float = 0.4,
    pitch_rad: Optional[float] = None,
    vfov_deg: float = 60.0,
    acquire: Optional[AcquireCommand] = None,
    flow_surge_norm: float = 0.35,
):
    """Fill a DockAlign message. Import interfaces.msg.DockAlign at call site after sourcing ws."""
    from interfaces.msg import DockAlign

    msg = DockAlign()
    msg.header = header
    msg.num_lights = int(len(cores))
    msg.image_width = int(image_width)
    msg.image_height = int(image_height)
    if acquire is not None:
        msg.search_yaw_norm = float(acquire.yaw_norm)
        msg.search_pitch_norm = float(acquire.pitch_norm)
        msg.search_surge_norm = float(acquire.surge_norm)

    cx_img = 0.5 * float(image_width)
    cy_img = 0.5 * float(image_height)
    half_w = max(cx_img, 1.0)
    half_h = max(cy_img, 1.0)

    if not geo.ok or geo.center is None:
        msg.valid = False
        if acquire is not None and acquire.status and acquire.status != "have_four":
            msg.status = acquire.status
        else:
            msg.status = geo.message or "invalid"
        msg.confidence = min(1.0, msg.num_lights / 4.0) * 0.25
        msg.aligned = False
        msg.elevation_valid = False
        return msg

    dock_cx, dock_cy = geo.center
    # +error_x => dock is to the RIGHT of image center
    # +error_y => dock is BELOW image center
    err_x = float(dock_cx - cx_img)
    err_y = float(dock_cy - cy_img)

    align_thr = max(spread_align_min_px, spread_align_frac * max(geo.radius_tb, 1.0))
    aligned = geo.spread < align_thr

    # Angle of top→bottom vs image +y (down). 0 = upright diameter.
    assert geo.top is not None and geo.bottom is not None
    dx = geo.bottom[0] - geo.top[0]
    dy = geo.bottom[1] - geo.top[1]
    diameter_angle_deg = math.degrees(math.atan2(dx, dy))

    # Confidence: full lights + low spread
    spread_term = 1.0 / (1.0 + geo.spread / max(geo.radius_tb, 1.0))
    confidence = (msg.num_lights / 4.0) * (confidence_base + (1.0 - confidence_base) * spread_term)

    msg.valid = True
    msg.error_x_px = err_x
    msg.error_y_px = err_y
    msg.error_x_norm = float(err_x / half_w)
    msg.error_y_norm = float(err_y / half_h)
    msg.radius_px = float(geo.radius_tb)
    msg.spread_px = float(geo.spread)
    msg.err_left_px = float(geo.err_left)
    msg.err_right_px = float(geo.err_right)
    msg.aligned = bool(aligned)
    msg.diameter_angle_deg = float(diameter_angle_deg)
    msg.center_exact = bool(geo.center_exact)
    msg.lateral_px = float(geo.lateral_px)
    msg.obliqueness = float(geo.obliqueness)
    if pitch_rad is None:
        msg.elevation_valid = False
        msg.elevation_rad = 0.0
    else:
        fy = focal_length_px(image_height, vfov_deg)
        msg.elevation_rad = float(elevation_down_rad(dock_cy, cy_img, fy, pitch_rad))
        msg.elevation_valid = True
    msg.search_yaw_norm = 0.0
    msg.search_pitch_norm = 0.0
    msg.search_surge_norm = float(
        track_surge_norm(
            geo.lateral_px,
            diameter_angle_deg,
            err_x,
            lateral_ok_px=max(spread_align_min_px, 8.0),
            error_x_ok_px=max(spread_align_min_px, 10.0),
            flow_surge_norm=flow_surge_norm,
        )
    )
    msg.center = _point(geo.center)
    msg.top = _point(geo.top)
    msg.bottom = _point(geo.bottom)
    msg.left = _point(geo.left)
    msg.right = _point(geo.right)
    msg.d_top = float(geo.d_top)
    msg.d_bottom = float(geo.d_bottom)
    msg.d_left = float(geo.d_left)
    msg.d_right = float(geo.d_right)
    msg.status = "ok" if aligned else "ok_not_aligned"
    msg.confidence = float(confidence)
    return msg
