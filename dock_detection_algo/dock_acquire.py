"""Search commands when fewer than four dock lights are in frame.

The vehicle has no sway thruster. Pitch comes from the heave thrusters and works
while stopped. Yaw comes from the X-fins and needs axial flow, so a yaw search
also asks for a little forward thrust. Heave is never commanded from a partial
detection: the lamps are the same color, and a wrong label would change depth
the wrong way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Core = Tuple[float, float]


@dataclass
class AcquireCommand:
    """Normalized search bias. +yaw = image right, +pitch = image down, +surge = forward."""

    yaw_norm: float = 0.0
    pitch_norm: float = 0.0
    surge_norm: float = 0.0
    status: str = ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PartialAcquire:
    """Turn 0–3 light cores into a yaw/pitch/surge search that keeps seen lights in frame."""

    def __init__(
        self,
        *,
        edge_frac: float = 0.10,
        nod_pitch_deg: float = 20.0,
        yaw_wiggle_deg: float = 8.0,
        nod_period_s: float = 8.0,
        backup_radius_frac: float = 0.22,
        backup_after_s: float = 8.0,
        hfov_deg: float = 60.0,
        flow_surge_norm: float = 0.35,
    ) -> None:
        self.edge_frac = float(edge_frac)
        self.nod_pitch_deg = float(nod_pitch_deg)
        self.yaw_wiggle_deg = float(yaw_wiggle_deg)
        self.nod_period_s = max(float(nod_period_s), 0.5)
        self.backup_radius_frac = float(backup_radius_frac)
        self.backup_after_s = float(backup_after_s)
        self.hfov_deg = float(hfov_deg)
        self.flow_surge_norm = float(flow_surge_norm)
        self._interior_since: Optional[float] = None

    def reset(self) -> None:
        self._interior_since = None

    def update(
        self,
        cores: Sequence[Core],
        width: int,
        height: int,
        t_s: float,
    ) -> AcquireCommand:
        if width < 2 or height < 2:
            return AcquireCommand(status="bad_image")
        if len(cores) >= 4:
            self._interior_since = None
            return AcquireCommand(status="have_four")
        if len(cores) == 0:
            self._interior_since = None
            yaw = math.sin(2.0 * math.pi * t_s / self.nod_period_s)
            return AcquireCommand(
                yaw_norm=0.5 * yaw,
                surge_norm=self.flow_surge_norm,
                status="search_no_lights",
            )

        cx = sum(p[0] for p in cores) / len(cores)
        cy = sum(p[1] for p in cores) / len(cores)
        half_w = 0.5 * width
        half_h = 0.5 * height
        margin_x = self.edge_frac * width
        margin_y = self.edge_frac * height
        near_left = any(p[0] < margin_x for p in cores)
        near_right = any(p[0] > width - margin_x for p in cores)
        near_top = any(p[1] < margin_y for p in cores)
        near_bottom = any(p[1] > height - margin_y for p in cores)

        if near_top and near_bottom:
            self._interior_since = None
            return AcquireCommand(
                yaw_norm=_clamp((cx - half_w) / half_w, -1.0, 1.0),
                surge_norm=-1.0,
                status="backup_taller_than_frame",
            )

        if near_left or near_right or near_top or near_bottom:
            self._interior_since = None
            yaw = 0.0
            pitch = 0.0
            if near_left and not near_right:
                yaw = -1.0
            elif near_right and not near_left:
                yaw = 1.0
            if near_top and not near_bottom:
                pitch = -1.0
            elif near_bottom and not near_top:
                pitch = 1.0
            surge = self.flow_surge_norm if abs(yaw) > 1e-6 else 0.0
            return AcquireCommand(yaw, pitch, surge, status="search_edge")

        if self._interior_since is None:
            self._interior_since = t_s
        radius = max(math.hypot(p[0] - cx, p[1] - cy) for p in cores)
        elapsed = t_s - self._interior_since
        if radius > self.backup_radius_frac * height and elapsed >= self.backup_after_s:
            return AcquireCommand(
                yaw_norm=_clamp((cx - half_w) / half_w, -1.0, 1.0),
                pitch_norm=_clamp((cy - half_h) / half_h, -1.0, 1.0),
                surge_norm=-1.0,
                status="backup_close",
            )

        half_hfov = max(self.hfov_deg * 0.5, 1.0)
        vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(half_hfov)) * (height / float(width))))
        half_vfov = max(vfov_deg * 0.5, 1.0)
        phase = 2.0 * math.pi * t_s / self.nod_period_s
        nod = math.sin(phase) * (self.nod_pitch_deg / half_vfov)
        wiggle = math.sin(phase + 0.5 * math.pi) * (self.yaw_wiggle_deg / half_hfov)
        yaw = _clamp((cx - half_w) / half_w + wiggle, -1.0, 1.0)
        pitch = _clamp((cy - half_h) / half_h + nod, -1.0, 1.0)
        return AcquireCommand(yaw, pitch, self.flow_surge_norm, status="search_nod")


def track_surge_norm(
    lateral_px: float,
    diameter_angle_deg: float,
    error_x_px: float = 0.0,
    *,
    lateral_ok_px: float = 8.0,
    angle_ok_deg: float = 3.0,
    error_x_ok_px: float = 10.0,
    flow_surge_norm: float = 0.35,
) -> float:
    """Forward thrust so the X-fins can yaw while the approach is not yet square.

    error_x is the sideways miss of the dock center. lateral_px grows when the
    heading is yawed relative to the dock face. Both need fin flow to correct,
    and the fins do nothing at zero speed.
    """
    if (
        abs(lateral_px) > lateral_ok_px
        or abs(diameter_angle_deg) > angle_ok_deg
        or abs(error_x_px) > error_x_ok_px
    ):
        return flow_surge_norm
    return 0.0
