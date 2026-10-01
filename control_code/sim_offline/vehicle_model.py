"""Simple 6-DOF rigid-body model of Mako_01 for OFFLINE testing (no mavsim needed).

Built from common/mako_geometry.yaml: mass, net buoyancy (+7.25 N), linear drag, thruster RPM^2 thrust,
fin lift ~ U^2. NOT the real sim: no Coriolis, simplified added mass, assumed fin/thruster units.
It catches sign / allocation / gain-structure bugs; it cannot prove behaviour in the real sim.
State: pos NED [m], euler [rad] (roll, pitch, yaw ZYX), body velocity nu=[u,v,w,p,q,r].
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from allocation import Allocator, eul_to_rotm, load_geometry  # noqa: E402


class VehicleModel:
    def __init__(
        self,
        geom: Optional[dict] = None,
        pos=(0.0, 0.0, 3.0),
        eul_deg=(0.0, 0.0, 0.0),
        thruster_tau_s: float = 0.2,
        fin_rate_deg_s: float = 57.0,
    ) -> None:
        self.g = geom or load_geometry()
        v = self.g["vehicle"]
        self.alloc = Allocator(self.g, rpm_cap=float(self.g["thrusters"]["th_01"]["n_max"]))
        m = float(v["mass_kg"])
        k = np.array(v["gyration_m"], dtype=float)
        af = np.array(v["added_mass_frac"], dtype=float)
        inertia = np.array([m * k[0] ** 2, m * k[1] ** 2, m * k[2] ** 2])
        # diag mass matrix [m(1+a_u), m(1+a_v), m(1+a_w), Ixx(1+..), Iyy(1+..), Izz(1+..)]
        self.M = np.concatenate([m * (1 + af[:3]), inertia * (1 + af[3:])])
        self.net_up_n = (float(v["buoyancy_mass_kg"]) - m) * float(v["gravity"])   # > 0 floats
        d = v["drag_lin"]
        self.D = np.array([d["X"], d["Y"], d["Z"], d["K"], d["M"], d["N"]], dtype=float)
        dq = v["drag_quad"]
        self.Dq = np.array([dq["X"], dq["Y"], dq["Z"], dq["K"], dq["M"], dq["N"]], dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.eul = np.radians(np.array(eul_deg, dtype=float))
        self.nu = np.zeros(6)
        self.tau_th = max(1e-3, thruster_tau_s)
        self.fin_rate = math.radians(fin_rate_deg_s)
        self.rpm = {k: 0.0 for k in self.alloc.th_ids}          # actual (lagged) RPM
        self.fin_deg = {k: 0.0 for k in self.alloc.fin_ids}     # actual fin angle
        self.cmd: Dict[str, float] = {}
        self.ext = np.zeros(6)          # external BODY-frame wrench [N, N*m]: current / bump / disturbance
        self.t = 0.0

    def set_command(self, cmd: Dict[str, float]) -> None:
        self.cmd = dict(cmd)

    def step(self, dt: float) -> None:
        # actuator dynamics
        a = dt / (self.tau_th + dt)
        for k in self.rpm:
            tgt = float(np.clip(self.cmd.get(k, 0.0), -2668.0, 2668.0))
            self.rpm[k] += a * (tgt - self.rpm[k])
        for k in self.fin_deg:
            tgt = float(np.clip(self.cmd.get(k, 0.0), -35.0, 35.0))
            step = math.degrees(self.fin_rate) * dt
            self.fin_deg[k] += float(np.clip(tgt - self.fin_deg[k], -step, step))

        u = self.nu[0]
        w = self.alloc.wrench_from_rpm(self.rpm) + self.alloc.wrench_from_fins(self.fin_deg, u)
        R = eul_to_rotm(np.degrees(self.eul))                       # body -> NED
        f_net_ned = np.array([0.0, 0.0, -self.net_up_n])            # NED z down: net up => negative z force
        w[:3] += R.T @ f_net_ned
        w += self.ext
        w -= self.D * self.nu + self.Dq * np.abs(self.nu) * self.nu
        # simple gyroscopic term for the rotation part is neglected (CG == CB, small rates)
        self.nu += dt * w / self.M

        phi, th, _ = self.eul
        p, q, r = self.nu[3:]
        cth = math.cos(th)
        if abs(cth) < 1e-3:                                         # avoid the pitch = +-90 singularity
            cth = math.copysign(1e-3, cth if cth != 0 else 1.0)
        eul_dot = np.array([
            p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(th),
            q * math.cos(phi) - r * math.sin(phi),
            (q * math.sin(phi) + r * math.cos(phi)) / cth,
        ])
        self.pos += dt * (R @ self.nu[:3])
        self.eul += dt * eul_dot
        self.eul[2] = math.atan2(math.sin(self.eul[2]), math.cos(self.eul[2]))
        self.t += dt

    def quaternion(self):
        phi, th, psi = self.eul / 2.0
        cr, sr, cp, sp, cy, sy = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th), math.cos(psi), math.sin(psi)
        return (
            sr * cp * cy - cr * sp * sy,   # x
            cr * sp * cy + sr * cp * sy,   # y
            cr * cp * sy - sr * sp * cy,   # z
            cr * cp * cy + sr * sp * sy,   # w
        )
