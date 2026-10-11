"""Dock light geometry: label Top/Bottom/Left/Right and measure radii from center.

The four dock lights lie on a 1 m circle. Top and bottom are a vertical diameter.
The two side lights are both above the dock center, at ±45° from the top, so their
3D midpoint lies on that diameter. A level, square-on view projects the circle to a
circle and the diameter midpoint is the dock center. A pitched view projects an
ellipse: the diameter midpoint is pulled toward the nearer light, and the side-light
distances leave the top–bottom radius even when heading is already correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Core = Tuple[float, float]

# Side-light midpoint, as a fraction of the top→bottom segment. z is down:
# top z=-1, side midpoint z=-√2/2, bottom z=+1.
FRONTAL_FRACTION = (1.0 - math.sqrt(2.0) / 2.0) / 2.0
# Cross-ratio (top, side-midpoint; center, bottom) on that diameter.
_CENTER_CROSS_RATIO = (1.0 + math.sqrt(2.0)) / 2.0


@dataclass
class DockGeometry:
    """Per-frame dock light geometry in image pixels."""

    ok: bool
    top: Optional[Core] = None
    bottom: Optional[Core] = None
    left: Optional[Core] = None
    right: Optional[Core] = None
    # Guidance center: cross-ratio dock center when center_exact, else top–bottom midpoint.
    center: Optional[Core] = None
    # Midpoint of the top–bottom segment. Spread is always measured from here.
    diameter_mid: Optional[Core] = None
    # Distance from the diameter midpoint to each light [px]
    d_top: float = 0.0
    d_bottom: float = 0.0
    d_left: float = 0.0
    d_right: float = 0.0
    # Expected radius from top–bottom diameter
    radius_tb: float = 0.0
    # How unequal the four radii are (0 ≈ front-on / perpendicular, level camera)
    spread: float = 0.0
    # Side errors vs diameter radius
    err_left: float = 0.0
    err_right: float = 0.0
    # Signed px: side-light midpoint to the right of the directed top→bottom line.
    # Near zero when the heading is perpendicular to the dock, including a pure
    # sideways shift. A yaw, especially at close range, pulls it off the line.
    lateral_px: float = 0.0
    # True when the side midpoint lies on the diameter, so the cross-ratio center is valid.
    center_exact: bool = False
    # Fraction of side midpoint along top→bottom, minus FRONTAL_FRACTION.
    # Positive when the upper half is foreshortened (camera above the dock and pitched).
    obliqueness: float = 0.0
    message: str = ""


def _dist(a: Core, b: Core) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mid(a: Core, b: Core) -> Core:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def unrotate_cores(cores: List[Core], roll_rad: float, cx: float, cy: float) -> List[Core]:
    """Undo camera roll about the image center.

    Positive roll is starboard-down. In x-right, y-down pixels that tips the image
    right side downward, and rotating the measured cores by +roll puts the
    top–bottom diameter back on the image vertical.
    """
    if not cores or abs(roll_rad) < 1e-6:
        return list(cores)
    c = math.cos(roll_rad)
    s = math.sin(roll_rad)
    out: List[Core] = []
    for x, y in cores:
        dx, dy = x - cx, y - cy
        out.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    return out


def _line_direction(top: Core, bottom: Core) -> Optional[Tuple[np.ndarray, float]]:
    tb = np.array(bottom, dtype=float) - np.array(top, dtype=float)
    length = float(np.linalg.norm(tb))
    if length < 5.0:
        return None
    return tb / length, length


def _along(point: Core, top: Core, direction: np.ndarray) -> float:
    rel = np.array(point, dtype=float) - np.array(top, dtype=float)
    return float(np.dot(rel, direction))


def _lateral(point: Core, top: Core, direction: np.ndarray) -> float:
    """Signed pixels to image-right of the directed top→bottom line."""
    rel = np.array(point, dtype=float) - np.array(top, dtype=float)
    return float(rel[0] * direction[1] - rel[1] * direction[0])


def _cross_ratio_center(top: Core, direction: np.ndarray, length: float, along_m: float) -> Optional[Core]:
    """Image of the dock center from the known cross-ratio along the diameter."""
    denom = length - along_m
    if abs(denom) < 1e-3:
        return None
    k = _CENTER_CROSS_RATIO * length / denom
    if abs(1.0 - k) < 1e-6:
        return None
    along_o = (-k * along_m) / (1.0 - k)
    xy = np.array(top, dtype=float) + along_o * direction
    return (float(xy[0]), float(xy[1]))


def label_dock_lights(cores: List[Core]) -> Optional[Dict[str, Core]]:
    """Assign Top/Bottom/Left/Right from 4 image centroids.

    Top = min y, Bottom = max y; remaining two split by x into Left/Right.
    """
    if len(cores) < 4:
        return None
    # Prefer exactly 4; if more, take first 4 (caller should MaxBlobs=4)
    pts = list(cores[:4])
    by_y = sorted(pts, key=lambda p: p[1])
    top = by_y[0]
    bottom = by_y[-1]
    middle = [p for p in pts if p is not top and p is not bottom]
    # Fallback if duplicates somehow collapse identity
    if len(middle) != 2:
        rest = by_y[1:-1]
        if len(rest) < 2:
            return None
        middle = rest[:2]
    left, right = sorted(middle, key=lambda p: p[0])
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def evaluate_dock_geometry(
    cores: List[Core],
    *,
    lateral_exact_px: float = 8.0,
    lateral_exact_frac: float = 0.06,
) -> DockGeometry:
    """Label four lights and measure both the level-view spread and the pitched-view cues.

    Spread is the old square-on test, taken from the top–bottom midpoint. It is near
    zero only when the camera is level and the dock is seen straight on. While the
    camera is pitched, use lateral_px (cross-track) and, when that is near zero, the
    cross-ratio center instead of the diameter midpoint.
    """
    if len(cores) < 4:
        return DockGeometry(ok=False, message=f"need 4 lights, got {len(cores)}")

    labels = label_dock_lights(cores)
    if labels is None:
        return DockGeometry(ok=False, message="failed to label lights")

    top, bottom = labels["top"], labels["bottom"]
    left, right = labels["left"], labels["right"]
    framed = _line_direction(top, bottom)
    if framed is None:
        return DockGeometry(ok=False, message="top and bottom are too close")
    direction, length = framed

    along_l = _along(left, top, direction)
    along_r = _along(right, top, direction)
    margin = 0.12 * length
    if not (-margin <= along_l <= length + margin and -margin <= along_r <= length + margin):
        return DockGeometry(
            ok=False,
            message="side lights are not between top and bottom",
        )

    diameter_mid = _mid(top, bottom)
    d_top = _dist(top, diameter_mid)
    d_bottom = _dist(bottom, diameter_mid)
    d_left = _dist(left, diameter_mid)
    d_right = _dist(right, diameter_mid)
    radius_tb = 0.5 * length
    dists = [d_top, d_bottom, d_left, d_right]
    spread = float(max(dists) - min(dists))
    err_left = d_left - radius_tb
    err_right = d_right - radius_tb

    side_mid = _mid(left, right)
    lateral_px = _lateral(side_mid, top, direction)
    along_m = _along(side_mid, top, direction)
    exact_thr = max(float(lateral_exact_px), float(lateral_exact_frac) * max(radius_tb, 1.0))
    center_exact = abs(lateral_px) <= exact_thr
    guidance = diameter_mid
    obliqueness = 0.0
    if center_exact:
        recovered = _cross_ratio_center(top, direction, length, along_m)
        if recovered is not None:
            # Blend instead of switching: full cross-ratio centre at lateral 0, the plain midpoint at the gate. A hard switch made the
            # centre jump by ~11 px (it flickered when the lateral cue hovered at the gate), which the controller turned into thruster spikes.
            w = 1.0 - abs(lateral_px) / max(exact_thr, 1e-6)
            guidance = (diameter_mid[0] + w * (recovered[0] - diameter_mid[0]), diameter_mid[1] + w * (recovered[1] - diameter_mid[1]))
            obliqueness = along_m / length - FRONTAL_FRACTION
        else:
            center_exact = False

    return DockGeometry(
        ok=True,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        center=guidance,
        diameter_mid=diameter_mid,
        d_top=d_top,
        d_bottom=d_bottom,
        d_left=d_left,
        d_right=d_right,
        radius_tb=radius_tb,
        spread=spread,
        err_left=err_left,
        err_right=err_right,
        lateral_px=lateral_px,
        center_exact=center_exact,
        obliqueness=obliqueness,
        message="ok",
    )


def draw_dock_geometry(
    bgr: np.ndarray,
    geo: DockGeometry,
    n_cores: int,
    *,
    spread_align_frac: float = 0.08,
    spread_align_min_px: float = 8.0,
) -> np.ndarray:
    """Overlay diameter, center, radii, and distance readout on the camera image."""
    vis = bgr.copy()
    cv2.putText(
        vis,
        f"cores={n_cores}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not geo.ok or geo.center is None:
        cv2.putText(
            vis,
            geo.message or "geometry unavailable",
            (8, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def ipt(p: Core) -> Tuple[int, int]:
        return (int(round(p[0])), int(round(p[1])))

    c = ipt(geo.center)
    t, b = ipt(geo.top), ipt(geo.bottom)  # type: ignore[arg-type]
    l, r = ipt(geo.left), ipt(geo.right)  # type: ignore[arg-type]

    # Vertical diameter (top–bottom)
    cv2.line(vis, t, b, (255, 200, 0), 2, cv2.LINE_AA)
    # Rays from center to each light
    for pt, color in (
        (t, (0, 255, 0)),
        (b, (0, 255, 0)),
        (l, (255, 128, 0)),
        (r, (255, 128, 0)),
    ):
        cv2.line(vis, c, pt, color, 1, cv2.LINE_AA)

    # Center
    cv2.drawMarker(vis, c, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(vis, c, 6, (0, 255, 255), 2)

    labels = (
        (t, "T", geo.d_top, (0, 255, 0)),
        (b, "B", geo.d_bottom, (0, 255, 0)),
        (l, "L", geo.d_left, (255, 128, 0)),
        (r, "R", geo.d_right, (255, 128, 0)),
    )
    for pt, name, dist, color in labels:
        cv2.circle(vis, pt, 8, (0, 0, 255), 2)
        cv2.drawMarker(vis, pt, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            vis,
            f"{name}",
            (pt[0] + 10, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"{dist:.1f}px",
            (pt[0] + 10, pt[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if geo.left is not None and geo.right is not None and geo.top is not None and geo.bottom is not None:
        side_mid = _mid(geo.left, geo.right)
        sm = ipt(side_mid)
        cv2.drawMarker(vis, sm, (255, 128, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.line(vis, sm, c, (255, 128, 0), 1, cv2.LINE_AA)

    exact = "exact" if geo.center_exact else "approx"
    y0 = 52
    lines = [
        f"center=({geo.center[0]:.0f},{geo.center[1]:.0f}) {exact}",
        f"r_TB={geo.radius_tb:.1f}  dT={geo.d_top:.1f}  dB={geo.d_bottom:.1f}",
        f"dL={geo.d_left:.1f}  dR={geo.d_right:.1f}",
        f"errL={geo.err_left:+.1f}  errR={geo.err_right:+.1f}  spread={geo.spread:.1f}",
        f"lateral={geo.lateral_px:+.1f}px  obliqueness={geo.obliqueness:+.3f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (8, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Compact alignment hint
    aligned = geo.spread < max(spread_align_min_px, spread_align_frac * geo.radius_tb)
    hint = "ALIGN OK" if aligned else "ALIGN OFF"
    hint_color = (0, 255, 0) if aligned else (0, 128, 255)
    cv2.putText(
        vis,
        hint,
        (8, y0 + len(lines) * 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        hint_color,
        2,
        cv2.LINE_AA,
    )
    return vis


def draw_mask_debug(mask: np.ndarray, geo: DockGeometry) -> np.ndarray:
    """BGR view of bloom mask with optional T/B/L/R markers."""
    if mask.ndim == 2:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        vis = mask.copy()
    if geo.ok and geo.center is not None:
        for p, name, color in (
            (geo.top, "T", (0, 255, 0)),
            (geo.bottom, "B", (0, 255, 0)),
            (geo.left, "L", (255, 128, 0)),
            (geo.right, "R", (255, 128, 0)),
        ):
            if p is None:
                continue
            pt = (int(round(p[0])), int(round(p[1])))
            cv2.circle(vis, pt, 6, color, 2)
            cv2.putText(
                vis,
                name,
                (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        c = (int(round(geo.center[0])), int(round(geo.center[1])))
        cv2.drawMarker(vis, c, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    return vis
