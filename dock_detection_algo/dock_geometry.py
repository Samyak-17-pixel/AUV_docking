"""Dock light geometry: label Top/Bottom/Left/Right and measure radii from center."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Core = Tuple[float, float]


@dataclass
class DockGeometry:
    """Per-frame dock light geometry in image pixels."""

    ok: bool
    top: Optional[Core] = None
    bottom: Optional[Core] = None
    left: Optional[Core] = None
    right: Optional[Core] = None
    center: Optional[Core] = None
    # Distance from vertical-diameter center to each light [px]
    d_top: float = 0.0
    d_bottom: float = 0.0
    d_left: float = 0.0
    d_right: float = 0.0
    # Expected radius from top–bottom diameter
    radius_tb: float = 0.0
    # How unequal the four radii are (0 ≈ front-on / perpendicular)
    spread: float = 0.0
    # Side errors vs diameter radius
    err_left: float = 0.0
    err_right: float = 0.0
    message: str = ""


def _dist(a: Core, b: Core) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mid(a: Core, b: Core) -> Core:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


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


def evaluate_dock_geometry(cores: List[Core]) -> DockGeometry:
    """Center = midpoint(top, bottom); distances to all lights updated every call."""
    if len(cores) < 4:
        return DockGeometry(ok=False, message=f"need 4 lights, got {len(cores)}")

    labels = label_dock_lights(cores)
    if labels is None:
        return DockGeometry(ok=False, message="failed to label lights")

    top, bottom = labels["top"], labels["bottom"]
    left, right = labels["left"], labels["right"]
    center = _mid(top, bottom)

    d_top = _dist(top, center)
    d_bottom = _dist(bottom, center)
    d_left = _dist(left, center)
    d_right = _dist(right, center)
    radius_tb = 0.5 * _dist(top, bottom)
    dists = [d_top, d_bottom, d_left, d_right]
    spread = float(max(dists) - min(dists))
    err_left = d_left - radius_tb
    err_right = d_right - radius_tb

    return DockGeometry(
        ok=True,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        center=center,
        d_top=d_top,
        d_bottom=d_bottom,
        d_left=d_left,
        d_right=d_right,
        radius_tb=radius_tb,
        spread=spread,
        err_left=err_left,
        err_right=err_right,
        message="ok",
    )


def draw_dock_geometry(bgr: np.ndarray, geo: DockGeometry, n_cores: int) -> np.ndarray:
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

    y0 = 52
    lines = [
        f"center=({geo.center[0]:.0f},{geo.center[1]:.0f})",
        f"r_TB={geo.radius_tb:.1f}  dT={geo.d_top:.1f}  dB={geo.d_bottom:.1f}",
        f"dL={geo.d_left:.1f}  dR={geo.d_right:.1f}",
        f"errL={geo.err_left:+.1f}  errR={geo.err_right:+.1f}  spread={geo.spread:.1f}",
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
    aligned = geo.spread < max(8.0, 0.08 * geo.radius_tb)
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
