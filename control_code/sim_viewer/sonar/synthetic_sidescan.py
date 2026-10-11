"""Physics of a pair of Cerulean Omniscan 450 SS side-scan sonars over the simulated sea floor (terrain.py). Pure numpy (+ numba), no ROS.

WHAT IS SIMULATED (every ping, every side)
  1. Geometry: each transducer sits at `location_m` on the hull, its beam axis tilted `mount_from_nadir_deg` (30 deg) away from straight down toward its side (port left, starboard
     right), a FAN of `beam_across_deg` (50 deg, -3 dB) across track and a thin `beam_along_deg` (0.5 deg) along track. The vehicle's roll, pitch and yaw move the beam.
  2. Ray tracing: a fan of rays (many across-track angles x a few along-track lines inside the 0.5 deg beam) is marched through the sea-floor heightmap; the FIRST surface hit gives
     slant range R, the surface normal and the surface backscatter. Parts of the floor hidden behind a ridge or boulder get no ray: ACOUSTIC SHADOWS appear by themselves; several
     surface points at one range add up (layover); straight below there is a blind gap (nadir) because the fan starts 5 deg from the vertical.
  3. Echo level: power per ray  P = G_across * G_along * mu * |n.r| * dOmega / R^2 * 10^(-2 alpha R / 10)  (area R^2 dOmega/|n.r|, Lambert backscatter mu |n.r|^2 with mu from the
     surface class in dB, two-way spreading, absorption alpha at 450 kHz), summed into range bins of width (range / num_bins); multiplicative speckle (gamma, `looks` looks), electronic
     noise floor, time-varied gain (spreading and absorption compensation like a real display), manual gain 0-7 (6 dB per step, ASSUMED) or auto gain, then quantised to uint16.
  4. Ping rate: min(20 Hz, 0.9 c / 2R). Sound speed c = 1500 m/s, absorption 0.10 dB/m (sea water 15 deg C, ASSUMED).
  5. Altimeter: a straight-down range to the floor with noise (for terrain following and for the slant-range correction of the mosaic).
ALL LEVELS (source level, noise floor, backscatter dB per class, gain steps) ARE ASSUMPTIONS from the datasheet (450 kHz, 0.5 x 50 deg, up to 150 m, 200-1200 bins, gain 0-7 / auto):
no real Omniscan recording was available to calibrate them.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE.parent / "common"))
from allocation import eul_to_rotm  # noqa: E402
from terrain import Terrain, _bilinear, njit  # noqa: E402


@dataclass
class SideScanConfig:
    freq_khz: float = 450.0
    beam_across_deg: float = 50.0        # -3 dB width across track (the fan)
    beam_along_deg: float = 0.5          # -3 dB width along track
    mount_from_nadir_deg: float = 30.0   # beam axis tilt away from straight down, toward the transducer's side
    location_m: Tuple[float, float, float] = (0.0, 0.0, 0.09)   # body frame x fwd, y stbd, z down: longitudinal centre, on the belly
    sound_speed_mps: float = 1500.0
    absorption_db_per_m: float = 0.10
    range_m: float = 30.0                # selectable 5 .. 150
    num_bins: int = 600                  # selectable 200 .. 1200
    gain_index: int = -1                 # -1 = auto, 0..7 manual
    gain_step_db: float = 6.0
    max_ping_hz: float = 20.0
    fan_rays: int = 1100                 # rays across track over +-fan_half_deg
    fan_half_deg: float = 32.0           # (beyond the -3 dB half width of 25 deg: the pattern tapers the weights)
    along_lines: int = 5
    looks: float = 3.0                   # speckle: number of independent looks (gamma shape)
    source_level_db: float = 111.0       # reference level (ASSUMED) that puts a sand bottom at about 60% of full scale at gain index 3 and 10 m range
    noise_floor_db: float = -27.0        # electronic noise level before the time-varied gain (same reference)
    dynamic_range_db: float = 80.0       # input window mapped onto 0..65535
    tvg: bool = True                     # time-varied gain: compensate 20 log R spreading and 2 alpha R absorption
    auto_target: float = 0.55            # auto gain aims the 90th percentile of the ping at this fraction of full scale
    seed: int = 1


@dataclass
class PingResult:
    side: str
    intensity: np.ndarray                # uint16 [num_bins]
    start_range_m: float
    range_m: float
    sound_speed_mps: float
    gain_index: int
    gain_db: float
    ping_hz: float
    altitude_m: float                    # straight-down range of the (noise-free) bottom under the vehicle
    origin_ned: np.ndarray
    hit_xyz: Optional[np.ndarray] = None  # (m, 3) world coordinates of the surface points that were hit (debug, mosaic truth)
    hit_range: Optional[np.ndarray] = None


@njit(cache=True)
def _march(z, x0, y0, inv_dx, ox, oy, oz, dirs, max_r, step, out_r):
    """For every ray: slant range of the first point where the ray goes below the floor (depth >= floor depth), or -1."""
    m = dirs.shape[0]
    for k in range(m):
        dx = dirs[k, 0]
        dy = dirs[k, 1]
        dz = dirs[k, 2]
        t = 0.2
        prev_t = t
        hit = -1.0
        while t <= max_r:
            px = ox + dx * t
            py = oy + dy * t
            pz = oz + dz * t
            zf = _bilinear(z, x0, y0, inv_dx, px, py)
            if pz >= zf:
                lo = prev_t
                hi = t
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    if oz + dz * mid >= _bilinear(z, x0, y0, inv_dx, ox + dx * mid, oy + dy * mid):
                        hi = mid
                    else:
                        lo = mid
                hit = 0.5 * (lo + hi)
                break
            prev_t = t
            t += step
        out_r[k] = hit


class SideScanSim:
    """Both sonars. `ping(pos, eul_rad, side)` -> PingResult; `altimeter(pos, eul_rad)` -> metres (with noise)."""

    def __init__(self, terrain: Terrain, cfg: Optional[SideScanConfig] = None) -> None:
        self.terrain = terrain
        self.cfg = cfg or SideScanConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._agc: Dict[str, float] = {"port": 0.0, "starboard": 0.0}
        self._agc_init: Dict[str, bool] = {"port": False, "starboard": False}

    # ------------------------------------------------------------------ settings that can change at run time
    def set_range(self, range_m: float) -> None:
        self.cfg.range_m = float(min(max(range_m, 2.0), 150.0))

    def set_gain(self, index: int) -> None:
        self.cfg.gain_index = int(min(max(index, -1), 7))

    def set_bins(self, n: int) -> None:
        self.cfg.num_bins = int(min(max(n, 200), 1200))

    def ping_hz(self) -> float:
        return float(min(self.cfg.max_ping_hz, 0.9 * self.cfg.sound_speed_mps / (2.0 * self.cfg.range_m)))

    # ------------------------------------------------------------------ geometry
    def _dirs(self, side: str):
        c = self.cfg
        s = 1.0 if side == "starboard" else -1.0
        half = math.radians(c.fan_half_deg)
        delta = np.linspace(-half, half, c.fan_rays)                              # across-track offset from the beam axis
        phi = math.radians(c.mount_from_nadir_deg) + delta                          # angle from the nadir toward the side
        sig_l = math.radians(c.beam_along_deg) / 2.3548
        thetas = np.linspace(-1.5 * 2.3548 * sig_l / 1.5, 1.5 * 2.3548 * sig_l / 1.5, c.along_lines) if c.along_lines > 1 else np.array([0.0])
        thetas = np.linspace(-1.2 * math.radians(c.beam_along_deg), 1.2 * math.radians(c.beam_along_deg), c.along_lines) if c.along_lines > 1 else np.array([0.0])
        PH, TH = np.meshgrid(phi, thetas, indexing="ij")
        cz = np.cos(PH)
        d = np.stack([cz * np.sin(TH), s * np.sin(PH), cz * np.cos(TH)], axis=-1)   # body frame (x fwd, y stbd, z down), rotated about body y by the along-track angle
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        g_across = np.exp(-2.7726 * (delta / math.radians(c.beam_across_deg)) ** 2)[:, None]
        g_along = np.exp(-2.7726 * (TH / math.radians(c.beam_along_deg)) ** 2)
        d_omega = (delta[1] - delta[0]) * (thetas[1] - thetas[0] if len(thetas) > 1 else math.radians(c.beam_along_deg))
        return d.reshape(-1, 3), (g_across * g_along).reshape(-1), d_omega

    # ------------------------------------------------------------------ one ping
    def ping(self, pos, eul_rad, side: str, keep_hits: bool = False) -> PingResult:
        c = self.cfg
        pos = np.asarray(pos, float)
        R_wb = eul_to_rotm(np.degrees(np.asarray(eul_rad, float)))
        origin = pos + R_wb @ np.asarray(c.location_m, float)
        d_b, w, d_omega = self._dirs(side)
        d_w = (R_wb @ d_b.T).T
        r = np.empty(d_w.shape[0])
        _march(self.terrain.z, self.terrain.x0, self.terrain.y0, self.terrain.inv_dx, float(origin[0]), float(origin[1]), float(origin[2]), np.ascontiguousarray(d_w), float(c.range_m), 0.12, r)
        ok = r > 0.0
        nb = c.num_bins
        P = np.zeros(nb)
        hits = None
        if ok.any():
            Rk = r[ok]
            pts = origin[None, :] + d_w[ok] * Rk[:, None]
            n = self.terrain.normal(pts[:, 0], pts[:, 1])
            cosi = np.abs(np.einsum("ij,ij->i", n, d_w[ok]))                          # |n.r| = sin(grazing angle)
            mu = 10.0 ** (self.terrain.reflectivity_db(pts[:, 0], pts[:, 1]) / 10.0)
            alpha = c.absorption_db_per_m
            p_ray = w[ok] * mu * cosi * d_omega / np.maximum(Rk, 0.3) ** 2 * 10.0 ** (-2.0 * alpha * Rk / 10.0)         # lambert (|n.r|^2) x area (1/|n.r|) x 1/R^2 (two-way R^4 x area R^2)
            idx = np.minimum((Rk / c.range_m * nb).astype(int), nb - 1)
            np.add.at(P, idx, p_ray)
            if keep_hits:
                hits = (pts, Rk)
        # noise, speckle, gain
        Rb = (np.arange(nb) + 0.5) * c.range_m / nb
        sig_db = np.full(nb, -300.0)
        pos_m = P > 0
        speck = self.rng.gamma(c.looks, 1.0 / c.looks, nb)
        Psp = P * speck
        pn = 10.0 ** (c.noise_floor_db / 10.0 - c.source_level_db / 10.0) * (1.0 + 0.25 * self.rng.standard_normal(nb) ** 2)   # noise in the same reference as P (after SL)
        level = 10.0 * np.log10(Psp + pn + 1e-30) + c.source_level_db
        tvg = (20.0 * np.log10(np.maximum(Rb, 1.0)) + 2.0 * c.absorption_db_per_m * Rb) if c.tvg else 0.0
        if c.gain_index < 0:                                                           # auto gain: aim the 90th percentile of the echo bins at the target
            lv = level + tvg
            ref = float(np.percentile(lv[pos_m], 90)) if pos_m.sum() > 5 else float(np.percentile(lv, 90))
            want = (c.auto_target * c.dynamic_range_db)
            g = want - (ref - 0.0)
            self._agc[side] = g if not self._agc_init[side] else 0.8 * self._agc[side] + 0.2 * g
            self._agc_init[side] = True
            gain_db = self._agc[side]
        else:
            gain_db = c.gain_step_db * (c.gain_index - 3)
        out = np.clip((level + tvg + gain_db) / c.dynamic_range_db, 0.0, 1.0)
        inten = np.round(out * 65535.0).astype(np.uint16)
        alt = self.terrain.altitude(pos[0], pos[1], pos[2])
        return PingResult(side, inten, 0.0, c.range_m, c.sound_speed_mps, int(c.gain_index), float(gain_db), self.ping_hz(), float(alt), origin,
                          hits[0] if hits else None, hits[1] if hits else None)

    def altimeter(self, pos, eul_rad, max_range_m: float = 60.0, noise_frac: float = 0.01, noise_m: float = 0.02) -> Optional[float]:
        """Straight-down range to the floor along the body z axis, with noise; None if beyond the range."""
        R_wb = eul_to_rotm(np.degrees(np.asarray(eul_rad, float)))
        d = (R_wb @ np.array([0.0, 0.0, 1.0]))[None, :]
        origin = np.asarray(pos, float)
        r = np.empty(1)
        _march(self.terrain.z, self.terrain.x0, self.terrain.y0, self.terrain.inv_dx, float(origin[0]), float(origin[1]), float(origin[2]), np.ascontiguousarray(d), float(max_range_m), 0.1, r)
        if r[0] <= 0:
            return None
        return float(r[0] * (1.0 + noise_frac * self.rng.standard_normal()) + noise_m * self.rng.standard_normal())


def flat_swath_geometry(altitude_m: float, cfg: Optional[SideScanConfig] = None) -> Dict[str, float]:
    """Analytic swath over a flat bottom: nearest and farthest ground range and slant range per side [m] for the mount and beam in `cfg`."""
    c = cfg or SideScanConfig()
    lo = math.radians(c.mount_from_nadir_deg - c.beam_across_deg / 2.0)
    hi = math.radians(c.mount_from_nadir_deg + c.beam_across_deg / 2.0)
    return {"ground_near_m": altitude_m * math.tan(lo), "ground_far_m": altitude_m * math.tan(hi), "slant_near_m": altitude_m / math.cos(lo), "slant_far_m": altitude_m / math.cos(hi)}
