"""Smooth pose and rate estimate from slow, late, noisy odometry. No ROS.

The real odometry arrives at ~4.5 Hz, ~0.25 s old, with noise, and the control loop runs at 20 Hz. Feeding the loops the raw samples (even extrapolated with the
sample's own velocity) turns every sample's noise into a step in the command: a +-1.5 cm depth error times a gain of 60 N/m is a 0.9 N flick, which the
thruster map turns into hundreds of RPM. This filter keeps the vehicle motion model in the loop instead: one small Kalman filter per axis with state
[value, rate], a random-acceleration process model, and two measurements per sample (the value, moved forward by the rate over the sample's age, and the
measured rate). Between samples it simply predicts, so the estimate is smooth at the control rate and already compensates the latency.

Axes: 0..2 = north, east, down [m]; 3..5 = roll, pitch, yaw [rad] (yaw unwrapped internally, wrapped on output). Rates for the angles come from the body rates
through the Euler kinematics. Pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from allocation import eul_to_rotm
from state import State, wrap_pi


@dataclass
class PoseFilterConfig:
    latency_s: float = 0.25                 # age of an odometry sample when it arrives
    accel_noise_lin: float = 0.25           # m/s^2   process noise of the position axes (how fast the vehicle can change speed: ~10 N / 20 kg = 0.5 m/s^2)
    accel_noise_ang: float = 0.8            # rad/s^2 process noise of the angle axes
    pos_sigma_m: float = 0.015              # measurement noise assumed for positions
    ang_sigma_rad: float = math.radians(0.25)
    vel_sigma_mps: float = 0.015
    rate_sigma_rps: float = math.radians(0.4)
    max_gap_s: float = 1.5                  # a gap longer than this (a freeze, a restart): re-initialise from the next sample


class PoseFilter:
    def __init__(self, cfg: Optional[PoseFilterConfig] = None) -> None:
        self.cfg = cfg or PoseFilterConfig()
        self.x = np.zeros((6, 2))            # [value, rate] per axis
        self.P = np.zeros((6, 2, 2))
        self.t = 0.0
        self.nu = np.zeros(6)                # last body velocities (the loops read u, q, r ... from the state)
        self.ready = False
        self._yaw_off = 0.0
        self._last_raw_yaw: Optional[float] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _rates(eul: np.ndarray, nu: np.ndarray) -> np.ndarray:
        """World-frame rates [dN, dE, dD, droll, dpitch, dyaw] from the body twist."""
        phi, th, _ = eul
        p, q, r = nu[3:]
        cth = math.cos(th)
        if abs(cth) < 1e-3:
            cth = math.copysign(1e-3, cth if cth != 0 else 1.0)
        eul_dot = np.array([p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(th), q * math.cos(phi) - r * math.sin(phi), (q * math.sin(phi) + r * math.cos(phi)) / cth])
        v = eul_to_rotm(np.degrees(eul)) @ nu[:3]
        return np.concatenate([v, eul_dot])

    def _q(self, i: int) -> float:
        return self.cfg.accel_noise_lin if i < 3 else self.cfg.accel_noise_ang

    def _predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        for i in range(6):
            q = self._q(i) ** 2
            Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
            self.x[i] = F @ self.x[i]
            self.P[i] = F @ self.P[i] @ F.T + Q
        self.t += dt

    def _meas_update(self, i: int, z_val: float, z_rate: float, s_val: float, s_rate: float) -> None:
        H = np.eye(2)
        R = np.diag([s_val ** 2, s_rate ** 2])
        z = np.array([z_val, z_rate])
        S = H @ self.P[i] @ H.T + R
        K = self.P[i] @ H.T @ np.linalg.inv(S)
        self.x[i] = self.x[i] + K @ (z - self.x[i])
        I = np.eye(2)
        self.P[i] = (I - K) @ self.P[i] @ (I - K).T + K @ R @ K.T

    # ------------------------------------------------------------------ interface
    def update(self, st: State, t_arrival: float) -> None:
        """A new odometry sample (the vehicle state `latency_s` ago) arrived at t_arrival."""
        c = self.cfg
        eul = st.eul.copy()
        if self._last_raw_yaw is not None:                                       # unwrap yaw continuously
            eul[2] = self._last_raw_yaw + wrap_pi(eul[2] - self._last_raw_yaw)
        self._last_raw_yaw = float(eul[2])
        rates = self._rates(eul, st.nu)
        vals = np.concatenate([st.pos, eul])
        # bring the (old) sample forward to the arrival time with its own rates
        vals_now = vals + rates * c.latency_s
        if (not self.ready) or (t_arrival - self.t > c.max_gap_s):
            self.x[:, 0], self.x[:, 1] = vals_now, rates
            for i in range(6):
                s = c.pos_sigma_m if i < 3 else c.ang_sigma_rad
                self.P[i] = np.diag([s ** 2 * 4, (c.vel_sigma_mps if i < 3 else c.rate_sigma_rps) ** 2 * 4])
            self.t, self.ready = t_arrival, True
            self.nu = st.nu.copy()
            return
        self._predict(t_arrival - self.t)
        for i in range(6):
            sv = c.pos_sigma_m if i < 3 else c.ang_sigma_rad
            sr = c.vel_sigma_mps if i < 3 else c.rate_sigma_rps
            # the velocity is also 0.25 s old; at constant-velocity this costs nothing, and the latency in the position is already moved forward above
            self._meas_update(i, vals_now[i], rates[i], sv * 1.5, sr * 1.5)       # x1.5: the moved-forward value carries the rate's error times the latency
        self.nu = st.nu.copy()

    def state_at(self, t: float) -> Optional[State]:
        """Smooth state at time t (>= last update): the filter predicted forward with its own rate estimate. Body rates come from the filtered angle rates."""
        if not self.ready:
            return None
        dt = max(-1.0, min(t - self.t, 2.0))                                      # forward (control) or a little backward (image time)
        val = self.x[:, 0] + self.x[:, 1] * dt
        rate = self.x[:, 1]
        pos, eul = val[:3].copy(), val[3:].copy()
        eul[2] = wrap_pi(eul[2])
        # body velocities: rotate the filtered world velocity into the body frame; angular rates back through the Euler kinematics
        R = eul_to_rotm(np.degrees(eul))
        v_body = R.T @ rate[:3]
        phi, th = eul[0], eul[1]
        dphi, dth, dpsi = rate[3:]
        p = dphi - math.sin(th) * dpsi
        q = math.cos(phi) * dth + math.sin(phi) * math.cos(th) * dpsi
        r = -math.sin(phi) * dth + math.cos(phi) * math.cos(th) * dpsi
        return State(pos=pos, eul=eul, nu=np.array([*v_body, p, q, r]))


class PoseTracker:
    """Wraps PoseFilter: raw odometry in (slow, late, noisy), smooth up-to-date state out. Also detects stale or frozen odometry."""

    def __init__(self, latency_s: float, stale_s: float, filt_cfg: Optional[PoseFilterConfig] = None) -> None:
        self.latency = float(latency_s)
        self.stale_s = float(stale_s)
        fc = filt_cfg or PoseFilterConfig()
        fc.latency_s = self.latency
        self.filt = PoseFilter(fc)
        self.state: Optional[State] = None              # the last RAW sample
        self.t_arrival = 0.0
        self._prev: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.identical = 0
        self.n_samples = 0
        self._latched = False                 # frozen was declared and no different sample has arrived since
        self._was_active = False              # the caller was commanding motion at the previous check

    def set_latency(self, latency_s: float) -> None:
        self.latency = float(latency_s)
        self.filt.cfg.latency_s = self.latency

    def push(self, st: State, t: float) -> None:
        cur = (st.pos.copy(), st.eul.copy(), st.nu.copy())
        if self._prev is not None and all(np.array_equal(a, b) for a, b in zip(cur, self._prev)):
            self.identical += 1                                      # the real sim froze: every value exactly repeated
        else:
            self.identical = 0
            self._latched = False                                    # a different sample: the data is alive again
        self._prev = cur
        self.state, self.t_arrival = st, t
        self.n_samples += 1
        if self.identical == 0:
            self.filt.update(st, t)                                  # a repeated sample carries no information (and would pull the rate estimate to zero)

    def age(self, t: float) -> float:
        return math.inf if self.state is None else t - self.t_arrival

    def frozen(self, t: float, odom_period_s: float = 0.25, expect_motion: bool = True) -> bool:
        """True if odometry stopped arriving OR keeps repeating the identical sample for longer than stale_s.

        A vehicle that is genuinely at rest in a noise-free simulation also repeats identical samples, and there is nothing to protect then: with expect_motion=False
        (the caller is commanding nothing) repetition alone does not count. Once frozen has been declared it stays declared until a DIFFERENT sample arrives, so the
        controller cannot flicker between 'neutral' and 'active' every stale_s while the data is still dead."""
        if self.state is None:
            return True
        if self.age(t) > self.stale_s:
            return True
        if self._latched:
            return True
        if expect_motion and not self._was_active:
            self.identical = 0                                       # commands only just started: count identical samples from NOW, not from the time the vehicle sat still
        self._was_active = expect_motion
        if self.identical * odom_period_s > self.stale_s and expect_motion:
            self._latched = True
            return True
        return False

    def at(self, t_query: float) -> Optional[State]:
        return self.filt.state_at(t_query)


