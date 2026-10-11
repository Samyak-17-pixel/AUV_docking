"""Odometry sensor model: makes the offline fake vehicle's /Mako_01/odometry_sim look like the REAL sim's (observed 2026-10-01), so gains are tuned
against the data the controllers will actually get, not against a perfect 100 Hz signal. Pure Python (numpy), no ROS.

What the real topic showed: ~4.4 Hz (9 Hz idle), pose stamps 0, and intermittent FREEZES where every value stays exactly constant for 5-25 s.
Latency, noise and jitter are not measured: the defaults are plausible guesses (see `OdometryConfig`) and are the knobs to change.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class OdometryConfig:
    rate_hz: float = 4.5                 # publish rate (real sim: 4.4 Hz while commands arrive). Jitter below makes it irregular.
    jitter_frac: float = 0.25            # each period is rate^-1 * (1 +- jitter_frac): 0.25 -> 0.17..0.28 s at 4.5 Hz
    latency_s: float = 0.25              # the published state is this old when it is sent (ASSUMED; the real value is unknown)
    pos_noise_m: float = 0.015           # 1-sigma noise added to x, y, z (ASSUMED)
    att_noise_deg: float = 0.2           # roll, pitch, yaw (ASSUMED)
    vel_noise_mps: float = 0.01          # body linear velocity (ASSUMED)
    rate_noise_dps: float = 0.3          # body angular velocity (ASSUMED)
    freeze_mean_interval_s: float = 60.0 # a freeze starts on average this often (0 = never). The real sim froze for 5-25 s every few minutes.
    freeze_min_s: float = 5.0
    freeze_max_s: float = 25.0
    freeze_zero_twist: bool = True       # during a freeze the real topic repeated the last pose AND showed w = q = 0
    seed: Optional[int] = 1
    # --- position faults to PROVE what depends on the odometry position (all off by default; x and y only, depth stays honest)
    pos_offset_m: Tuple[float, float] = (0.0, 0.0)      # a constant offset of the published x, y (a different origin)
    pos_drift_mps: Tuple[float, float] = (0.0, 0.0)     # the published x, y drift away at this velocity (grows with time)
    pos_jump_m: Tuple[float, float] = (0.0, 0.0)        # a one-off jump of x, y ...
    pos_jump_t_s: float = -1.0                          # ... at this simulation time (-1 = never)
    drop_twist: bool = False                            # publish zero body velocity (no speed information at all: the negative control for dead reckoning)


def quat_from_eul(eul) -> Tuple[float, float, float, float]:
    """(x, y, z, w) of the ZYX Euler angles (same convention as VehicleModel.quaternion)."""
    phi, th, psi = (float(a) / 2.0 for a in eul)
    cr, sr, cp, sp, cy, sy = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th), math.cos(psi), math.sin(psi)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


@dataclass
class OdometrySample:
    t: float                              # simulation time of publication
    pos: np.ndarray
    eul: np.ndarray
    nu: np.ndarray
    frozen: bool = False


class OdometrySensor:
    """Feed it the true state every simulation step with `update`; it returns the samples (0 or 1 per call) that would be PUBLISHED."""

    def __init__(self, cfg: Optional[OdometryConfig] = None) -> None:
        self.cfg = cfg or OdometryConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._hist: Deque[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = deque()
        self._next_pub = 0.0
        self._freeze_until = -1.0
        self._next_freeze = self._draw_next_freeze(0.0)
        self.last: Optional[OdometrySample] = None
        self.published = 0
        self.freezes = 0

    def _draw_next_freeze(self, t: float) -> float:
        m = self.cfg.freeze_mean_interval_s
        return math.inf if m <= 0 else t + float(self.rng.exponential(m))

    def _period(self) -> float:
        j = self.cfg.jitter_frac
        return (1.0 / self.cfg.rate_hz) * float(1.0 + self.rng.uniform(-j, j))

    def _delayed(self, t: float):
        target = t - self.cfg.latency_s
        h = self._hist
        while len(h) > 2 and h[1][0] <= target:
            h.popleft()
        return h[0] if h else None

    def update(self, t: float, pos, eul, nu) -> List[OdometrySample]:
        self._hist.append((float(t), np.array(pos, float), np.array(eul, float), np.array(nu, float)))
        if t < self._next_pub:
            return []
        self._next_pub = t + self._period()
        if self._freeze_until < 0 and t >= self._next_freeze:                    # a freeze begins
            self._freeze_until = t + float(self.rng.uniform(self.cfg.freeze_min_s, self.cfg.freeze_max_s))
            self.freezes += 1
        if self._freeze_until >= 0:
            if t < self._freeze_until and self.last is not None:
                nu_f = self.last.nu.copy()
                if self.cfg.freeze_zero_twist:
                    nu_f[:] = 0.0
                s = OdometrySample(float(t), self.last.pos.copy(), self.last.eul.copy(), nu_f, True)
                self.published += 1
                return [s]
            self._freeze_until = -1.0                                           # freeze over
            self._next_freeze = self._draw_next_freeze(t)
        d = self._delayed(t)
        if d is None:
            return []
        c = self.cfg
        s = OdometrySample(float(t), d[1] + self.rng.normal(0, c.pos_noise_m, 3), d[2] + np.radians(self.rng.normal(0, c.att_noise_deg, 3)),
                           d[3] + np.concatenate([self.rng.normal(0, c.vel_noise_mps, 3), np.radians(self.rng.normal(0, c.rate_noise_dps, 3))]))
        s.pos = s.pos + self._pos_fault(t)
        if c.drop_twist:
            s.nu = s.nu.copy()
            s.nu[:3] = 0.0
        self.last = s
        self.published += 1
        return [s]

    def _pos_fault(self, t: float) -> np.ndarray:
        c = self.cfg
        f = np.zeros(3)
        f[:2] = np.asarray(c.pos_offset_m, float) + np.asarray(c.pos_drift_mps, float) * float(t)
        if c.pos_jump_t_s >= 0.0 and t >= c.pos_jump_t_s:
            f[:2] += np.asarray(c.pos_jump_m, float)
        return f

    def repeat_last(self) -> Optional[OdometrySample]:
        """For a paused simulation: the same sample again (the real topic keeps publishing while the sim is paused)."""
        return self.last
