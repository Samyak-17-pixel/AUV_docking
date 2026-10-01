"""Single-axis hold loops (PID -> wrench component) shared by dof_testing and station_keeping.

Each loop returns a force [N] or moment [N*m] in the wrench convention of allocation.py
(Z positive DOWN, M positive nose-UP, N positive bow-to-starboard, K positive starboard-down roll).
Derivative terms use the MEASURED rate (depth rate, q, r, ...) = derivative-on-measurement, because
this vehicle has almost no natural damping (K_p = M_q = Z_w = 0 in the config).

gains dict layout (see dof_testing.yaml / station_keeping.yaml):
  <loop>: {kp, ki, kd, i_max, max}   where max caps the loop output (N or N*m).
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
        self.pids = {}
        for name in LOOPS:
            g = gains.get(name)
            if g is None:
                continue
            self.pids[name] = Pid(
                kp=g["kp"], ki=g.get("ki", 0.0), kd=g.get("kd", 0.0),
                i_max=g.get("i_max", abs(g.get("max", 0.0))), d_filter_tau_s=g.get("d_filter_tau_s", 0.05),
            )

    def reset(self, name: str = None) -> None:
        for k, p in self.pids.items():
            if name is None or k == name:
                p.reset()

    def _cap(self, name: str, u: float) -> float:
        m = float(self.gains[name].get("max", 1e9))
        return clamp(u, -m, m)

    # error sign convention: error = setpoint - measurement
    def heave(self, st: State, depth_sp: float, dt: float) -> float:
        """Depth hold -> Z force [N] (+ = down). Includes the buoyancy feed-forward."""
        e = depth_sp - st.depth
        u = self.pids["heave"].update(e, dt, rate=float(st.vel_ned()[2]))
        return self._cap("heave", u + self.heave_ff_n)

    def pitch(self, st: State, pitch_sp: float, dt: float) -> float:
        """Pitch hold -> M moment [N*m] (+ = nose up). Positive pitch error = nose too low."""
        e = pitch_sp - st.pitch
        return self._cap("pitch", self.pids["pitch"].update(e, dt, rate=float(st.nu[4])))

    def surge(self, st: State, ahead_err_m: float, axis_xy: np.ndarray, dt: float) -> float:
        """Position hold along a fixed horizontal axis -> X force [N]. ahead_err_m = sp - actual along axis."""
        v_axis = float(st.vel_ned()[:2] @ axis_xy)
        return self._cap("surge", self.pids["surge"].update(ahead_err_m, dt, rate=v_axis))

    def speed(self, st: State, u_sp: float, dt: float) -> float:
        """Forward speed hold -> X force [N] (used to give the fins flow)."""
        return self._cap("speed", self.pids["speed"].update(u_sp - st.speed_u, dt))

    def yaw(self, st: State, yaw_sp: float, dt: float) -> float:
        """Heading hold -> N moment [N*m]. Needs fin authority (forward speed)."""
        e = wrap_pi(yaw_sp - st.yaw)
        return self._cap("yaw", self.pids["yaw"].update(e, dt, rate=float(st.nu[5])))

    def roll(self, st: State, roll_sp: float, dt: float) -> float:
        """Roll hold -> K moment [N*m]. Needs fin authority (forward speed)."""
        e = roll_sp - st.roll
        return self._cap("roll", self.pids["roll"].update(e, dt, rate=float(st.nu[3])))

    def cross_track(self, ct_err_m: float, dt: float) -> float:
        """Sway-by-yaw: lateral error [m] (+ = target is to the right of the line) -> heading offset [rad]."""
        g = self.gains["cross_track"]
        off = self.pids["cross_track"].update(ct_err_m, dt)
        return clamp(off, -math.radians(g.get("max_heading_deg", 30.0)), math.radians(g.get("max_heading_deg", 30.0)))
