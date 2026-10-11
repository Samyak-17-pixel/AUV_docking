"""Standoff docking wrench from DockAlign cues. No ROS.

The vehicle has no sway thruster. Heave and pitch come from the two vertical
thrusters and work at zero speed. Yaw and roll come from the X-fins and need
forward flow, so a yaw correction also asks for a small forward SPEED. Surge is
a speed loop, not a fixed force: the vehicle has almost no drag, so a fixed push
keeps accelerating and a fixed small reverse push cannot stop it. The allowed
speed also falls with the estimated distance to the dock (a stopping-distance
envelope), reaching zero before the dock is too close, and the axial thruster
brakes actively when the vehicle is faster than the allowed speed. Once the dock
is at the same depth, square, and in frame, the speed target is zero. This core
never commands a closing run into the funnel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

Wrench = np.ndarray  # [X, Y, Z, K, M, N]


def _clamp(value: float, limit: float) -> float:
    return max(-abs(limit), min(abs(limit), value))


def _outside(value: float, deadband: float) -> float:
    """Zero inside the deadband; outside, the part BEYOND the band (continuous at the edge).

    A hard gate (return the whole value once outside) made a measurement flickering across the edge produce a step of ~0.9 N, which the
    allocator's square-root force-to-RPM map turned into +-350 RPM thruster spikes on a vehicle that was standing still.
    """
    if abs(value) <= deadband:
        return 0.0
    return math.copysign(abs(value) - deadband, value)


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
    heave_rate: float = 0.0     # vertical speed in NED [m/s], positive = moving DOWN (damps the elevation loop)
    dt: float = 0.05            # seconds since the previous update (dead-reckons the remembered dock distance)


class DockTestCore:
    """Map one DockAlign sample to a body wrench."""

    def __init__(self, cfg: dict) -> None:
        g = cfg["gains"]
        d = cfg["deadband"]
        lim = cfg["limits"]
        self.heave_ff_n = float(cfg["feedforward"]["heave_n"])
        self.kp_elev = float(g["elevation_n_per_rad"])
        self.kd_elev = float(g.get("elevation_kd_n_per_mps", 0.0))
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
        sp = cfg["speed"]
        cam = cfg.get("camera", {})
        dock = cfg.get("dock", {})
        self.creep_mps = float(sp["creep_mps"])
        self.backup_mps = float(sp["backup_mps"])
        self.kp_speed = float(sp["kp_n_per_mps"])
        self.fwd_max_n = float(sp["max_forward_n"])
        self.brake_max_n = float(sp["max_brake_n"])
        self.decel = max(float(sp["decel_mps2"]), 1e-3)
        self.mass_kg = float(sp["mass_kg"])
        self.stop_margin_m = float(sp["stop_margin_m"])
        self.search_mps = float(sp["search_mps"])
        self.memory_trust_s = float(sp.get("memory_trust_s", 20.0))
        self.reject_tol_m = float(sp.get("reject_tolerance_m", 1.0))
        self.reject_tol_frac = float(sp.get("reject_tolerance_frac", 0.5))
        self.drift_allow_mps = float(sp.get("drift_allowance_mps", 0.05))
        self.lost_margin_m = float(sp.get("lost_close_margin_m", 0.3))
        self.min_radius_px = float(sp.get("min_radius_px", 8.0))
        self.realign_mps = float(sp.get("realign_mps", 0.5))
        self.flow_min_mps = float(sp.get("flow_min_mps", 0.3))
        self.realign_max_back_m = float(sp.get("realign_max_back_m", 2.5))
        self.settle_yaw_rate = float(sp.get("settle_yaw_rate_rad_s", 0.03))
        self.blind_max_fwd_m = float(sp.get("blind_max_forward_m", -1.0))          # < 0 = no limit (the old behaviour)
        self.blind_back_m = float(sp.get("blind_back_m", 1.0))
        vfov = math.radians(float(cam.get("vfov_deg", 60.0)))
        self.fy = 0.5 * float(cam.get("image_height_px", 480)) / math.tan(0.5 * vfov)
        self.dock_radius_m = float(dock.get("radius_m", 1.0))
        self.elev_db = float(d["elevation_rad"])
        self.y_db = float(d["error_y_px"])
        self.x_db = float(d["error_x_px"])
        self.lat_db = float(d["lateral_px"])
        self.hold_extra_px = float(d.get("hold_extra_px", 0.0))
        self.too_close_radius_px = float(lim["too_close_radius_px"])
        # Distance at which the approach speed must be zero (a margin outside the too-close distance).
        self.too_close_distance_m = self.fy * self.dock_radius_m / self.too_close_radius_px
        self.hold_distance_m = self.too_close_distance_m + self.stop_margin_m
        # Remembered distance to the dock: set from every trusted detection, then dead-reckoned with the measured speed.
        self._d_est = None
        self._since_valid = 1e9
        self._realign = False          # latched: backing away to get fin flow for a heading correction
        self._blind_travel = 0.0       # net forward travel [m] since the last trusted detection, while no distance is known (search without a memory)
        self._blind_back = False       # latched: the blind forward budget is used up, back away until the whole dock fits in the picture

    def distance_m(self, radius_px: float):
        """Distance to the dock plane from the apparent radius of the light circle (None if too small to trust)."""
        if radius_px < self.min_radius_px:
            return None
        return self.fy * self.dock_radius_m / radius_px

    def allowed_speed(self, distance_m) -> float:
        """Largest closing speed from which the planned deceleration still stops at hold_distance_m."""
        if distance_m is None:
            return self.creep_mps
        room = max(distance_m - self.hold_distance_m, 0.0)
        return min(self.creep_mps, math.sqrt(2.0 * self.decel * room))

    def speed_force(self, u: float, u_target: float, brake_ff: bool = False) -> float:
        """Axial force [N]: positive pushes forward, negative brakes or reverses.

        brake_ff adds the force that produces the planned deceleration (mass * decel). Without it a P-only loop
        lags the falling speed envelope and overshoots the stopping point by about half a metre.
        """
        f = self.kp_speed * (u_target - u)
        if brake_ff and u > 0.05:
            f -= self.mass_kg * self.decel
        return max(-self.brake_max_n, min(self.fwd_max_n, f))

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
            "u_target": 0.0,
            "d_m": float("nan"),
        }
        if not view.fresh:
            return wrench, status

        # Dead-reckon the remembered distance (positive speed closes the distance); forget it when it gets too old.
        if self._d_est is not None:
            self._d_est -= snap.speed_u * max(snap.dt, 0.0)
            self._since_valid += max(snap.dt, 0.0)
            if self._since_valid > self.memory_trust_s:
                self._d_est = None
        valid = view.valid
        d_meas = self.distance_m(view.radius_px) if valid else None
        if valid and d_meas is not None:
            tol = max(self.reject_tol_m, self.reject_tol_frac * self._d_est) + self.drift_allow_mps * min(self._since_valid, self.memory_trust_s) \
                if self._d_est is not None else 0.0
            if self._d_est is not None and abs(d_meas - self._d_est) > tol:
                valid = False        # disagrees with the dead-reckoned distance (e.g. blobs from a glow-swamped image): do not trust it
            else:
                self._d_est = d_meas
                self._since_valid = 0.0

        k = self._roll_moment(snap)
        if valid:
            self._blind_travel = 0.0
            self._blind_back = False
        if not valid:
            # Search hint: +norm -> forward flow for the fins, -norm -> back away, ~0 -> stop.
            if view.search_surge_norm > 0.05:
                u_target = self.search_mps
            elif view.search_surge_norm < -0.05:
                u_target = -self.backup_mps
            else:
                u_target = 0.0
            mode = "search"
            if self._d_est is None and self.blind_max_fwd_m >= 0.0:
                # No distance is known (the dock was never seen whole, or the memory expired): the picture says "move forward" because a light is cut off by
                # the frame, but forward is also where the dock is. Spend only a limited forward budget, then back away (the dock shrinks and fits) until
                # it is seen whole. This is what drove the vehicle into the dock from a close start with the dock partly out of frame.
                self._blind_travel += snap.speed_u * max(snap.dt, 0.0)
                if self._blind_travel >= self.blind_max_fwd_m:
                    self._blind_back = True
                elif self._blind_travel <= self.blind_max_fwd_m - self.blind_back_m:
                    self._blind_back = False
                if self._blind_back:
                    u_target = -self.backup_mps
                    mode = "search_back"
            # Never surge blindly towards a dock that was seen close a moment ago: obey the stopping envelope on the
            # remembered distance, and back away if the lights were lost inside the too-close distance.
            if self._d_est is not None:
                if self._d_est < self.too_close_distance_m + self.lost_margin_m:
                    u_target = -self.backup_mps
                    mode = "lost_close"
                elif u_target > 0.0:
                    u_target = min(u_target, self.allowed_speed(self._d_est))
            x = self.speed_force(snap.speed_u, u_target)
            # search_pitch_norm > 0 means pitch down. M > 0 is nose up.
            m = _clamp(-view.search_pitch_norm * self.m_max, self.m_max)
            n = _clamp(view.search_yaw_norm * self.n_max, self.n_max)
            z = self.heave_ff_n
            wrench[:] = (x, 0.0, z, k, m, n)
            status.update(
                mode=mode, x_n=x, z_n=z, k_nm=k, m_nm=m, n_nm=n, u_target=u_target,
                d_m=float("nan") if self._d_est is None else float(self._d_est),
            )
            return wrench, status

        elev = _outside(view.elevation_rad, self.elev_db) if view.elevation_valid else 0.0
        err_y = _outside(view.error_y_px, self.y_db)
        err_x = _outside(view.error_x_px, self.x_db)
        lateral = _outside(view.lateral_px, self.lat_db)

        # Moving down (heave_rate > 0) reduces the downward force: the damping a 20 kg vehicle with no drag does not have by itself.
        z = _clamp(self.heave_ff_n + self.kp_elev * elev - self.kd_elev * snap.heave_rate, self.z_max)
        # Dock below the image (error_y > 0) -> nose down.
        m = _clamp(-self.kp_y * err_y - self.kd_pitch * snap.pitch_rate, self.m_max)
        # Dock to the right, or the side midpoint to the right of the diameter -> yaw right.
        n = _clamp(
            self.kp_x * err_x + self.kp_lat * lateral - self.kd_yaw * snap.yaw_rate,
            self.n_max,
        )

        d = self.distance_m(view.radius_px)
        if view.radius_px >= self.too_close_radius_px:
            u_target = -self.backup_mps
            mode = "backup"
        else:
            squared = (_outside(view.error_x_px, self.x_db + self.hold_extra_px) == 0.0 and _outside(view.lateral_px, self.lat_db + self.hold_extra_px) == 0.0
                       and abs(snap.yaw_rate) <= self.settle_yaw_rate)
            if self._realign and d is not None and (squared or d >= self.hold_distance_m + self.realign_max_back_m):
                self._realign = False             # the heading is square and still (or there is no more room behind): stop backing away
            if self._realign:
                # Backing away at a constant speed keeps the fins in flow while the yaw loop turns the vehicle square. Reverse flow flips the
                # lift of the same deflection; the allocator handles that. The vehicle does NOT return forward until it is square, so it
                # arrives at the standoff aligned instead of arriving rotating (a forward leg that ends at zero speed loses its fins and the
                # leftover yaw rate is undamped: that gave a limit cycle of +-15 deg over ROS).
                u_target = -self.realign_mps
                mode = "realign"
            elif squared:
                u_target = 0.0                    # square and in frame: stop and hold the standoff
                mode = "standoff"
            else:
                u_target = self.allowed_speed(d)  # creep to get fin flow, but never faster than the stopping envelope
                mode = "creep"
                # At the hold distance the envelope allows no speed, so there is no fin flow and the heading can never be corrected: back away.
                if d is not None and u_target < self.flow_min_mps:
                    self._realign = True
                    u_target = -self.realign_mps
                    mode = "realign"
        # Even in "creep" the vehicle must brake if it is faster than the envelope allows (it may arrive fast).
        enveloped = False
        if d is not None and mode not in ("backup", "realign"):
            allowed = self.allowed_speed(d)
            if allowed < self.creep_mps - 1e-6 and (snap.speed_u > allowed - 0.05 or u_target >= allowed):
                enveloped = True                  # on the stopping envelope: hold the planned deceleration
            if snap.speed_u > allowed:
                u_target = min(u_target, allowed)
        x = self.speed_force(snap.speed_u, u_target, brake_ff=enveloped)

        wrench[:] = (x, 0.0, z, k, m, n)
        status.update(
            mode=mode, x_n=x, z_n=z, k_nm=k, m_nm=m, n_nm=n,
            u_target=u_target, d_m=float("nan") if d is None else float(d),
        )
        return wrench, status

    def _roll_moment(self, snap: VehicleSnap) -> float:
        """Hold roll at zero. The allocator drops this when there is no fin flow."""
        return _clamp(
            -self.kp_roll * snap.roll - self.kd_roll * snap.roll_rate,
            self.k_max,
        )
