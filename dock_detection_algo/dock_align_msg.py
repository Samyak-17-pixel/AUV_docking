"""Build interfaces/DockAlign messages from dock geometry."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from geometry_msgs.msg import Point
from std_msgs.msg import Header

from dock_geometry import DockGeometry

Core = Tuple[float, float]


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
):
    """Fill a DockAlign message. Import interfaces.msg.DockAlign at call site after sourcing ws."""
    from interfaces.msg import DockAlign

    msg = DockAlign()
    msg.header = header
    msg.num_lights = int(len(cores))
    msg.image_width = int(image_width)
    msg.image_height = int(image_height)

    cx_img = 0.5 * float(image_width)
    cy_img = 0.5 * float(image_height)
    half_w = max(cx_img, 1.0)
    half_h = max(cy_img, 1.0)

    if not geo.ok or geo.center is None:
        msg.valid = False
        msg.status = geo.message or "invalid"
        msg.confidence = min(1.0, msg.num_lights / 4.0) * 0.25
        msg.aligned = False
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
