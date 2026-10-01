"""Actuator allocation for the Mako_01 AUV (1 axial thruster, 2 vertical heave thrusters, 4 X-fins).

Wrench convention (body frame, x fwd / y stbd / z DOWN), order [X, Y, Z, K, M, N]:
  X surge force [N], Y sway [N], Z heave [N] (+ = pushes DOWN), K roll [N*m],
  M pitch [N*m] (+ = nose UP), N yaw [N*m] (+ = bow to starboard).

Thrusters  -> X, Z, M  (work at zero speed). Y, K, N are not reachable by thrusters (2 vertical + 1 axial).
Fins       -> K, N (and optionally a share of M). Fin force ~ U^2, so authority is ~0 at zero speed:
              demand is divided by max(U, u_min)^2 and faded to zero below u_fin_off.
Sway has no actuator at all; it only happens via yaw + surge.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml

DEFAULT_GEOMETRY = Path(__file__).resolve().parent / "mako_geometry.yaml"
DOF_ORDER = ("X", "Y", "Z", "K", "M", "N")


def eul_to_rotm(eul_deg: Sequence[float]) -> np.ndarray:
    """Intrinsic ZYX rotation matrix from [roll, pitch, yaw] in deg (same as mavsim/teleop)."""
    phi, theta, psi = np.asarray(eul_deg, dtype=float) * math.pi / 180.0
    c1, s1, c2, s2, c3, s3 = (
        math.cos(phi), math.sin(phi), math.cos(theta), math.sin(theta), math.cos(psi), math.sin(psi),
    )
    return np.array([
        [c2 * c3, -c1 * s3 + s1 * s2 * c3, s1 * s3 + c1 * s2 * c3],
        [c2 * s3, c1 * c3 + s1 * s2 * s3, -s1 * c3 + c1 * s2 * s3],
        [-s2, s1 * c2, c1 * c2],
    ])


def load_geometry(path: Optional[Path] = None) -> dict:
    with open(path or DEFAULT_GEOMETRY, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Allocator:
    def __init__(
        self,
        geom: Optional[dict] = None,
        *,
        rpm_cap: Optional[float] = None,
        fin_deg_cap: Optional[float] = None,
        u_fin_min: float = 0.4,
        u_fin_off: float = 0.15,
        u_fin_full: float = 0.5,
        pitch_fin_share: float = 0.0,
    ) -> None:
        g = geom or load_geometry()
        self.g = g
        self.rho = float(g["vehicle"]["rho"])
        th = g["thrusters"]
        self.rpm_to_rps = float(th["rpm_to_rps"])
        self.kt = {"fwd": float(th["kt_fwd"]), "rev": float(th["kt_rev"])}

        self.th_ids = [k for k in th if k.startswith("th_")]
        self._th = {k: th[k] for k in self.th_ids}
        cols = []
        for k in self.th_ids:
            r = np.array(self._th[k]["location"], dtype=float)
            u = eul_to_rotm(self._th[k]["orientation"]) @ np.array([1.0, 0.0, 0.0])
            cols.append(np.concatenate([u, np.cross(r, u)]))
        self.B_th = np.stack(cols, axis=1)                      # 6 x n_thrusters
        self.B_th_pinv = np.linalg.pinv(self.B_th)
        self.rpm_cap = float(rpm_cap) if rpm_cap is not None else min(
            float(self._th[k]["n_max"]) for k in self.th_ids
        )

        fin = g["fins"]
        self.fin_ids = [k for k in fin if k.startswith("cs_")]
        self.fin_cap_deg = float(fin_deg_cap) if fin_deg_cap is not None else float(fin["delta_max_deg"])
        self.fin_delta_max = float(fin["delta_max_deg"])
        self.fin_k = 0.5 * self.rho * float(fin["area_m2"]) * float(fin["cl_alpha_per_rad"])  # N/rad/(m/s)^2
        fcols, fdirs = [], []
        for k in self.fin_ids:
            r = np.array(fin[k]["location"], dtype=float)
            lift = eul_to_rotm(fin[k]["orientation"]) @ np.array([0.0, 1.0, 0.0])
            fdirs.append(np.concatenate([lift, np.cross(r, lift)]))   # unit-lift 6-wrench per fin
            fcols.append(np.cross(r, lift))                           # [K, M, N] part
        self.B_fin6 = np.stack(fdirs, axis=1)                         # 6 x 4 (for the offline model)
        self.B_fin = np.stack(fcols, axis=1)                          # 3 x 4: [K, M, N] per unit lift
        self.B_fin_pinv = np.linalg.pinv(self.B_fin)                  # 4 x 3

        self.u_fin_min, self.u_fin_off, self.u_fin_full = float(u_fin_min), float(u_fin_off), float(u_fin_full)
        self.pitch_fin_share = float(np.clip(pitch_fin_share, 0.0, 1.0))

    # ---------------------------------------------------------------- thrusters
    def thrust_n(self, name: str, rpm: float) -> float:
        d = float(self._th[name]["D"])
        k = self.kt["fwd"] if rpm >= 0 else self.kt["rev"]
        return math.copysign(k * self.rho * d ** 4 * (abs(rpm) * self.rpm_to_rps) ** 2, rpm)

    def force_to_rpm(self, name: str, force_n: float) -> float:
        d = float(self._th[name]["D"])
        k = self.kt["fwd"] if force_n >= 0 else self.kt["rev"]
        n_rps = math.sqrt(abs(force_n) / (k * self.rho * d ** 4))
        rpm = math.copysign(n_rps / self.rpm_to_rps, force_n)
        cap = min(self.rpm_cap, float(self._th[name]["n_max"]))
        return float(np.clip(rpm, -cap, cap))

    def thrusters_from_wrench(self, wrench: Sequence[float]) -> Dict[str, float]:
        """Wrench [X,Y,Z,K,M,N] -> {th_XX: RPM}. Only X, Z, M are realisable; Y, K, N are ignored."""
        w = np.asarray(wrench, dtype=float).copy()
        w[[1, 3, 5]] = 0.0
        forces = self.B_th_pinv @ w
        return {k: self.force_to_rpm(k, float(f)) for k, f in zip(self.th_ids, forces)}

    def wrench_from_rpm(self, rpms: Dict[str, float]) -> np.ndarray:
        """Inverse: thruster RPMs -> body wrench (used by the offline vehicle model)."""
        f = np.array([self.thrust_n(k, rpms.get(k, 0.0)) for k in self.th_ids])
        return self.B_th @ f

    # --------------------------------------------------------------------- fins
    def fin_authority(self, u: float) -> float:
        """0..1 fade: 0 below u_fin_off (no flow, fins useless), 1 above u_fin_full."""
        if self.u_fin_full <= self.u_fin_off:
            return 1.0
        return float(np.clip((abs(u) - self.u_fin_off) / (self.u_fin_full - self.u_fin_off), 0.0, 1.0))

    def fins_from_moments(self, K: float, M: float, N: float, u: float) -> Dict[str, float]:
        """Roll/pitch/yaw moment demand [N*m] -> {cs_XX: deg}, gain-scheduled on speed u [m/s]."""
        m = np.array([K, self.pitch_fin_share * M, N], dtype=float)
        u_eff = max(abs(u), self.u_fin_min)
        lift = self.B_fin_pinv @ m                               # N of lift per fin
        delta_rad = lift / (self.fin_k * u_eff ** 2)
        delta_deg = np.degrees(delta_rad) * self.fin_authority(u)
        cap = min(self.fin_cap_deg, self.fin_delta_max)
        delta_deg = np.clip(delta_deg, -cap, cap)
        return {k: float(d) for k, d in zip(self.fin_ids, delta_deg)}

    def wrench_from_fins(self, deltas_deg: Dict[str, float], u: float) -> np.ndarray:
        """Inverse: fin deflections -> body wrench at speed u (offline model)."""
        d = np.radians([float(deltas_deg.get(k, 0.0)) for k in self.fin_ids])
        d = np.clip(d, -math.radians(self.fin_delta_max), math.radians(self.fin_delta_max))
        lift = self.fin_k * u * abs(u) * d
        return self.B_fin6 @ lift

    # ----------------------------------------------------------------- combined
    def allocate(self, wrench: Sequence[float], u: float) -> Dict[str, float]:
        """Full allocation: thrusters (X,Z,M[,share]) + fins (K,N[,M share])."""
        w = np.asarray(wrench, dtype=float).copy()
        w_th = w.copy()
        w_th[4] = (1.0 - self.pitch_fin_share) * w[4]
        out = self.thrusters_from_wrench(w_th)
        out.update(self.fins_from_moments(w[3], w[4], w[5], u))
        return out
