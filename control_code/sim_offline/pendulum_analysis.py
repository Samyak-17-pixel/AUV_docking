#!/usr/bin/env python3
"""What would explain the pitch pendulum seen on the live sim (2026-10-01, docs/project_notes.md section 2c item 1)? Offline analysis, no ROS.

Observed: uncommanded, the vehicle swung in pitch between roughly 7 and 85 deg with a period of ~11 s (nearly undamped), passing the Euler singularity.
The vessel file says CG == CB (they differ by ~1e-10 m), so there should be NO righting moment. A pendulum needs one. This script

  1. reads CG / CB from the .mavsim file and prints their difference;
  2. says how big a CG-CB offset the observation implies (a swing between 7 and 85 deg is symmetric about an equilibrium of 46 deg, amplitude 39 deg):
     fits the offset length d (and its direction) in the offline model so that the free period is 11 s;
  3. checks what the shipped pitch hold (kp, kd from dof_testing.yaml) does to such a pendulum: damping ratio of the closed loop.

  python3 pendulum_analysis.py [--mavsim "../../Mako (1).mavsim"] [--period 11] [--low 7] [--high 85]

The result is a HYPOTHESIS about the live sim (an offset of a few millimetres between the centre of gravity and the centre of buoyancy that the sim uses at
run time), not a measurement: the real sim could also differ in damping, added mass or buoyancy. A tilt-and-release experiment on the live sim would settle it
(see the printed experiment).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "common"):
    sys.path.insert(0, str(p))

from allocation import load_geometry  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402


def read_cg_cb(mavsim: Path):
    with zipfile.ZipFile(mavsim) as z:
        a = json.loads(z.read("config.json"))["agents"][0]
    g = a["geometry"]
    cg, cb = np.array(g["CG"]["position"], float), np.array(g["CB"]["position"], float)
    return cg, cb, a


def free_swing(d: float, beta_deg: float, theta0_deg: float, T: float = 60.0, dt: float = 0.01):
    """Free pitch swing (no commands, no damping in the model) with a CG-CB offset of length d at angle beta from the body z axis. -> (t, pitch_deg)."""
    geom = copy.deepcopy(load_geometry())
    beta = math.radians(beta_deg)
    geom["vehicle"]["cg_minus_cb_m"] = [d * math.sin(beta), 0.0, d * math.cos(beta)]    # [x, y, z down] of the CG relative to the CB
    for k in ("K", "M", "N", "X", "Y", "Z"):
        geom["vehicle"]["drag_quad"][k] = 0.0
        geom["vehicle"]["drag_lin"][k] = 0.0
    m = VehicleModel(geom=geom, pos=(0.0, 0.0, 3.0), eul_deg=(0.0, theta0_deg, 0.0))
    ts, th = [], []
    for i in range(int(T / dt)):
        m.step(dt)
        ts.append(m.t)
        th.append(math.degrees(m.eul[1]))
    return np.array(ts), np.array(th)


def period_and_range(ts, th):
    """Period from the up-crossings of the mean level; (period, min, max)."""
    mid = 0.5 * (th.max() + th.min())
    ups = [ts[i] for i in range(1, len(th)) if th[i - 1] < mid <= th[i]]
    per = float(np.mean(np.diff(ups))) if len(ups) >= 2 else math.nan
    return per, float(th.min()), float(th.max())


def main() -> None:
    ap = argparse.ArgumentParser(description="Pitch pendulum hypothesis for the live sim")
    ap.add_argument("--mavsim", type=Path, default=_HERE.parents[2] / "Mako (1).mavsim")
    ap.add_argument("--period", type=float, default=11.0)
    ap.add_argument("--low", type=float, default=7.0)
    ap.add_argument("--high", type=float, default=85.0)
    args = ap.parse_args()

    if args.mavsim.exists():
        cg, cb, a = read_cg_cb(args.mavsim)
        print(f"vessel file: CG {cg}  CB {cb}\n  CG - CB = {cg - cb}  (|.| = {np.linalg.norm(cg - cb):.2e} m)  -> no righting moment by the file")
    else:
        print(f"(vessel file {args.mavsim} not found; skipped)")

    eq = 0.5 * (args.low + args.high)
    amp = 0.5 * (args.high - args.low)
    beta = -eq                                           # moment ~ -d*W*sin(theta + beta): equilibrium at theta = -beta
    print(f"\nobserved swing {args.low:.0f}..{args.high:.0f} deg => equilibrium {eq:.0f} deg, amplitude {amp:.0f} deg, period {args.period:.1f} s")
    geom = load_geometry()["vehicle"]
    m = float(geom["mass_kg"])
    iyy = m * geom["gyration_m"][1] ** 2 * (1.0 + geom["added_mass_frac"][4])
    w0 = 2.0 * math.pi / args.period
    d_small = w0 ** 2 * iyy / (m * float(geom["gravity"]))
    print(f"model pitch inertia Iyy(1+added) = {iyy:.2f} kg*m^2 (ASSUMED added mass {geom['added_mass_frac'][4]:.2f})")
    print(f"small-angle estimate of the offset: d = w0^2 * I / (m g) = {1000 * d_small:.1f} mm")

    lo, hi = 0.2 * d_small, 3.0 * d_small                # bisection on d so that the simulated period (large amplitude) matches
    for _ in range(40):
        d = 0.5 * (lo + hi)
        ts, th = free_swing(d, beta, args.low, T=4 * args.period)
        per, mn, mx = period_and_range(ts, th)
        lo, hi = (d, hi) if per > args.period else (lo, d)       # larger d -> faster pendulum -> shorter period
    print(f"fitted in the model (start {args.low:.0f} deg at rest): d = {1000 * d:.1f} mm at {beta:.0f} deg from the body z axis "
          f"(CG {1000 * d * math.sin(math.radians(beta)):+.1f} mm in x, {1000 * d * math.cos(math.radians(beta)):+.1f} mm in z), period {per:.1f} s, swing {mn:.0f}..{mx:.0f} deg")
    print("  => a CG-CB mismatch of only a few millimetres (mesh-derived CB vs configured CG, say) is enough; the file's own CG/CB agree far better than that.")

    dof = yaml.safe_load(open(_HERE.parent / "dof_testing" / "dof_testing.yaml"))["gains"]["pitch"]
    k_rest = d * m * float(geom["gravity"])
    kp, kd = float(dof["kp"]), float(dof["kd"])
    wn = math.sqrt((kp + k_rest) / iyy)
    zeta = kd / (2.0 * math.sqrt((kp + k_rest) * iyy))
    print(f"\nshipped pitch hold (dof_testing.yaml: kp {kp}, kd {kd}) on this pendulum: stiffness {kp + k_rest:.2f} N*m/rad (the pendulum adds {k_rest:.2f}), "
          f"wn {wn:.2f} rad/s, damping ratio {zeta:.2f} (ideal rates; the real loop adds odometry lag)")
    print("  => a pendulum of this size is a SMALL disturbance for the hold: the hold stiffness is similar to the gravity stiffness. The hold needs the rate (kd) to be effective.")

    print("""
EXPERIMENT for the live sim (the only way to confirm): with all DOF active, thrusters commanded to zero, teleop stopped:
  1. set an initial pitch of ~20 deg (initial_conditions) or push it with a pitch step; release; record /Mako_01/odometry_sim for 60 s (record_run.py).
  2. read the period T and the equilibrium angle (the swing's mid-level) off the record; run this script with --period T --low <min> --high <max> to get d and its direction.
  3. a nonzero equilibrium pitch with the thrusters off = CG-CB offset along x; the period = its z component. Then put the numbers into
     mako_geometry.yaml (cg_minus_cb_m) so the offline model has the same pendulum, and keep the pitch hold running during every other test.""")


if __name__ == "__main__":
    main()
