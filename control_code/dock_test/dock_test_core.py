"""Standoff docking wrench from DockAlign cues. No ROS.

The vehicle has no sway thruster. Heave and pitch come from the two vertical
thrusters and work at zero speed. Yaw and roll come from the X-fins and need
forward flow, so a yaw correction also asks for a small axial force. Once the
dock is at the same depth, square, and in frame, surge is zero. This core never
commands a closing run into the funnel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

Wrench = np.ndarray  # [X, Y, Z, K, M, N]


def _clamp(value: float, limit: float) -> float:
    return max(-abs(limit), min(abs(limit), value))


def _outside(value: float, deadband: float) -> float:
    """Zero inside the deadband. Outside, return the original value."""
    if abs(value) <= deadband:
        return 0.0
    return value


@dataclass
class DockView:
    """The DockAlign fields this controller uses, plus whether the message is fresh."""

    fresh: bool = False
    valid: bool = False
    elevation_valid: bool = False
    elevation_rad: float = 0.0
    error_x_px: float = 0.0
    error_y_px: float = 0.0
    lateral_px: float = 0.0
    radius_px: float = 0.0
    search_yaw_norm: float = 0.0
    search_pitch_norm: float = 0.0
    search_surge_norm: float = 0.0


@dataclass
class VehicleSnap:
    """Body speed and the rates the dampers need. Angles in rad, rates in rad/s."""

    speed_u: float = 0.0
    roll: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0


class DockTestCore:
    """Map one DockAlign sample to a body wrench."""

    def __init__(self, cfg: dict) -> None:
        g = cfg["gains"]
        d = cfg["deadband"]
        lim = cfg["limits"]
        self.heave_ff_n = float(cfg["feedforward"]["heave_n"])
        self.kp_elev = float(g["elevation_n_per_rad"])
        self.z_max = float(g["heave_max_n"])
        self.kp_y = float(g["pitch_nm_per_px"])
        self.kd_pitch = float(g["pitch_kd"])
        self.m_max = float(g["pitch_max_nm"])
        self.kp_x = float(g["yaw_nm_per_px"])
        self.kp_lat = float(g["yaw_nm_per_lateral_px"])
        self.kd_yaw = float(g["yaw_kd"])
        self.n_max = float(g["yaw_max_nm"])
        self.kp_roll = float(g["roll_nm_per_rad"])
        self.kd_roll = float(g["roll_kd"])
        self.k_max = float(g["roll_max_nm"])
        self.flow_surge_n = float(lim["flow_surge_n"])
        self.elev_db = float(d["elevation_rad"])
        self.y_db = float(d["error_y_px"])
        self.x_db = float(d["error_x_px"])
        self.lat_db = float(d["lateral_px"])
        self.too_close_radius_px = float(lim["too_close_radius_px"])

    def update(self, view: DockView, snap: VehicleSnap) -> Tuple[Wrench, Dict[str, float]]:
        wrench = np.zeros(6)
        status = {
            "mode": "stale",
            "x_n": 0.0,
            "z_n": 0.0,
            "k_nm": 0.0,
            "m_nm": 0.0,
            "n_nm": 0.0,
            "u": float(snap.speed_u),
        }
        if not view.fresh:
            return wrench, status

        k = self._roll_moment(snap)
        if not view.valid:
            x = _clamp(view.search_surge_norm * self.flow_surge_n, self.flow_surge_n)
            # search_pitch_norm > 0 means pitch down. M > 0 is nose up.
            m = _clamp(-view.search_pitch_norm * self.m_max, self.m_max)
            n = _clamp(view.search_yaw_norm * self.n_max, self.n_max)
            z = self.heave_ff_n
            wrench[:] = (x, 0.0, z, k, m, n)
            status.update(mode="search", x_n=x, z_n=z, k_nm=k, m_nm=m, n_nm=n)
            return wrench, status

        elev = _outside(view.elevation_rad, self.elev_db) if view.elevation_valid else 0.0
        err_y = _outside(view.error_y_px, self.y_db)
        err_x = _outside(view.error_x_px, self.x_db)
        lateral = _outside(view.lateral_px, self.lat_db)

        z = _clamp(self.heave_ff_n + self.kp_elev * elev, self.z_max)
        # Dock below the image (error_y > 0) -> nose down.
        m = _clamp(-self.kp_y * err_y - self.kd_pitch * snap.pitch_rate, self.m_max)
        # Dock to the right, or the side midpoint to the right of the diameter -> yaw right.
        n = _clamp(
            self.kp_x * err_x + self.kp_lat * lateral - self.kd_yaw * snap.yaw_rate,
            self.n_max,
        )

        if view.radius_px >= self.too_close_radius_px:
            x = -self.flow_surge_n
            mode = "backup"
        elif err_x != 0.0 or lateral != 0.0:
            x = self.flow_surge_n
            mode = "creep"
        else:
            x = 0.0
            mode = "standoff"

        wrench[:] = (x, 0.0, z, k, m, n)
        status.update(mode=mode, x_n=x, z_n=z, k_nm=k, m_nm=m, n_nm=n)
        return wrench, status

    def _roll_moment(self, snap: VehicleSnap) -> float:
        """Hold roll at zero. The allocator drops this when there is no fin flow."""
        return _clamp(
            -self.kp_roll * snap.roll - self.kd_roll * snap.roll_rate,
            self.k_max,
        )
