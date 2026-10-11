"""Single-axis hold loops (PID -> wrench component) shared by dof_testing and station_keeping.

Each loop returns a force [N] or moment [N*m] in the wrench convention of allocation.py
(Z positive DOWN, M positive nose-UP, N positive bow-to-starboard, K positive starboard-down roll).
Derivative terms use the MEASURED rate (depth rate, q, r, ...) = derivative-on-measurement, because
this vehicle has almost no natural damping (K_p = M_q = Z_w = 0 in the config).

gains dict layout (see dof_testing.yaml / station_keeping.yaml):
  <loop>: {kp, ki, kd, i_max, max, lpf_tau_s}   where max caps the loop output (N or N*m) and lpf_tau_s (optional, default 0 = off) is the time constant
  of a first-order low-pass on the loop OUTPUT (and sp_tau_s, default 0 = off, a first-order filter on the SETPOINT of heave/pitch/yaw/roll: a step in the
  setpoint becomes a smooth ramp, so a big move does not overshoot; it starts from the measurement, so the first call never jumps):
  the output filter's purpose: it keeps measurement noise from becoming thruster chatter, at the price of a little phase lag.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from pid import Pid, clamp
from state import State, wrap_pi

LOOPS = ("surge", "heave", "pitch", "yaw", "roll", "speed", "cross_track")


class HoldLoops:
    def __init__(self, gains: Dict[str, dict], heave_ff_n: float = 0.0) -> None:
        self.gains = gains
        self.heave_ff_n = float(heave_ff_n)
        self._lp: Dict[str, float] = {}
        self._sp_f: Dict[str, float] = {}
        self.pids = {}
        for name in LOOPS:
            g = gains.get(name)
            if g is None:
                continue
            self.pids[name] = Pid(
                kp=g["kp"], ki=g.get("ki", 0.0), kd=g.get("kd", 0.0),
                i_max=g.get("i_max", abs(g.get("max", 0.0))), d_filter_tau_s=g.get("d_filter_tau_s", 0.05),
            )

    def set_gain(self, loop: str, key: str, value: float) -> bool:
        """Change one number of one loop while it runs (live tuning). `key` is kp, ki, kd, i_max, max, lpf_tau_s, sp_tau_s or d_filter_tau_s. Returns False if unknown."""
        if loop not in self.gains or key not in ("kp", "ki", "kd", "i_max", "max", "lpf_tau_s", "sp_tau_s", "d_filter_tau_s"):
            return False
        v = float(value)
        if not math.isfinite(v) or (key != "max" and v < 0.0 and key in ("kp", "ki", "kd", "i_max", "lpf_tau_s", "sp_tau_s", "d_filter_tau_s")):
            return False
        self.gains[loop][key] = v
        pid = self.pids.get(loop)
        if pid is not None:
            if key in ("kp", "ki", "kd"):
                setattr(pid, key, v)
            elif key == "i_max":
                pid.i_max = abs(v)
            elif key == "d_filter_tau_s":
                pid.d_filter_tau_s = v
        return True

    def reset(self, name: str = None) -> None:
        for d in (self._lp, self._sp_f):
            for k in list(d):
                if name is None or k == name:
                    d.pop(k)
        for k, p in self.pids.items():
            if name is None or k == name:
                p.reset()

    def _sp(self, name: str, sp: float, meas: float, dt: float, angle: bool = False) -> float:
        tau = float(self.gains[name].get("sp_tau_s", 0.0))
        if tau <= 0.0 or dt <= 0.0:
            return sp
        r = self._sp_f.get(name)
        if r is None:
            r = meas
        err = wrap_pi(sp - r) if angle else sp - r
        r = r + (dt / (tau + dt)) * err
        self._sp_f[name] = wrap_pi(r) if angle else r
        return self._sp_f[name]

    def _cap(self, name: str, u: float, dt: float = 0.0) -> float:
        m = float(self.gains[name].get("max", 1e9))
        u = clamp(u, -m, m)
        tau = float(self.gains[name].get("lpf_tau_s", 0.0))
        if tau > 0.0 and dt > 0.0:
            y = self._lp.get(name, u)
            y += (dt / (tau + dt)) * (u - y)
            self._lp[name] = y
            return y
        return u

    # error sign convention: error = setpoint - measurement
    def heave(self, st: State, depth_sp: float, dt: float) -> float:
        """Depth hold -> Z force [N] (+ = down). Includes the buoyancy feed-forward."""
        e = self._sp("heave", depth_sp, st.depth, dt) - st.depth
        u = self.pids["heave"].update(e, dt, rate=float(st.vel_ned()[2]))
        return self._cap("heave", u + self.heave_ff_n, dt)

    def pitch(self, st: State, pitch_sp: float, dt: float) -> float:
        """Pitch hold -> M moment [N*m] (+ = nose up). Positive pitch error = nose too low."""
        e = self._sp("pitch", pitch_sp, st.pitch, dt) - st.pitch
        return self._cap("pitch", self.pids["pitch"].update(e, dt, rate=float(st.nu[4])), dt)

    def surge(self, st: State, ahead_err_m: float, axis_xy: np.ndarray, dt: float) -> float:
        """Position hold along a fixed horizontal axis -> X force [N]. ahead_err_m = sp - actual along axis."""
        v_axis = float(st.vel_ned()[:2] @ axis_xy)
        return self._cap("surge", self.pids["surge"].update(ahead_err_m, dt, rate=v_axis))

    def speed(self, st: State, u_sp: float, dt: float) -> float:
        """Forward speed hold -> X force [N] (used to give the fins flow)."""
        return self._cap("speed", self.pids["speed"].update(self._sp("speed", u_sp, st.speed_u, dt) - st.speed_u, dt), dt)

    def yaw(self, st: State, yaw_sp: float, dt: float) -> float:
        """Heading hold -> N moment [N*m]. Needs fin authority (forward speed)."""
        e = wrap_pi(self._sp("yaw", yaw_sp, st.yaw, dt, angle=True) - st.yaw)
        return self._cap("yaw", self.pids["yaw"].update(e, dt, rate=float(st.nu[5])), dt)

    def roll(self, st: State, roll_sp: float, dt: float) -> float:
        """Roll hold -> K moment [N*m]. Needs fin authority (forward speed)."""
        e = self._sp("roll", roll_sp, st.roll, dt) - st.roll
        return self._cap("roll", self.pids["roll"].update(e, dt, rate=float(st.nu[3])), dt)

    def cross_track(self, ct_err_m: float, dt: float) -> float:
        """Sway-by-yaw: lateral error [m] (+ = target is to the right of the line) -> heading offset [rad]."""
        g = self.gains["cross_track"]
        off = self.pids["cross_track"].update(ct_err_m, dt)
        return clamp(off, -math.radians(g.get("max_heading_deg", 30.0)), math.radians(g.get("max_heading_deg", 30.0)))
