#!/usr/bin/env python3
"""Closed-loop pitch + heave harness for dock_test (no ROS, no GUI).

Runs the REAL DockTestCore and the REAL allocator against the full offline vehicle model, with the delays the real system has: odometry (the
pitch rate and vertical speed) is sampled at ~4.5 Hz and held, the dock image arrives at 15 Hz with a latency, thrusters lag. The dock is 2.4 m ahead
at the vehicle's depth; the camera is the synthetic nose camera, so pitch and depth errors turn into pixel errors exactly as the detector would see them.
Only heave and pitch are applied (surge, yaw and roll are not under test here).

  python3 loop_sim.py                      # compare the shipped gains with the old ones on four plants
Use it before touching pitch_*, elevation_* gains in dock_test.yaml. The plant numbers (thruster lag, delays, inertia) are ASSUMPTIONS.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_d) for _d in [_HERE.parent / "sim_viewer", *sorted((_HERE.parent / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # sim_viewer and its sub-folders
for p in (_HERE, _HERE.parent / "common", _HERE.parent / "sim_offline", _HERE.parent / "sim_viewer"):
    sys.path.insert(0, str(p))

from allocation import Allocator, eul_to_rotm, load_geometry  # noqa: E402
from dock_test_core import DockTestCore, DockView, VehicleSnap  # noqa: E402
from synthetic_camera import SyntheticCamera  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

PLANTS: Dict[str, dict] = {
    "nominal": dict(tau=0.2, odom_hz=4.5, img_delay=0.1, inertia_scale=1.0),
    "heavy+late": dict(tau=0.3, odom_hz=4.5, img_delay=0.2, inertia_scale=1.3),
    "light+fast": dict(tau=0.1, odom_hz=9.0, img_delay=0.05, inertia_scale=0.7),
    "stress": dict(tau=0.45, odom_hz=3.0, img_delay=0.35, inertia_scale=1.8),
}

_CAM = SyntheticCamera()


def load_cfg() -> dict:
    return yaml.safe_load(open(_HERE / "dock_test.yaml"))


def run(gains: Optional[dict] = None, *, tau: float = 0.2, odom_hz: float = 4.5, img_delay: float = 0.1, inertia_scale: float = 1.0,
        T: float = 45.0, pitch0_deg: float = 8.0, depth0_m: float = 0.2, standoff_m: float = 2.4, dt: float = 0.02) -> dict:
    """-> dict with late-window oscillation amplitudes, the time to settle within (3 deg, 0.1 m), and the full log.

    3 deg because the controller rests anywhere inside its deadbands (image 10 px = 1.4 deg, elevation 0.02 rad = 1.15 deg), which is about 2-2.5 deg of pitch.
    """
    cfg = copy.deepcopy(load_cfg())
    for k, v in (gains or {}).items():
        cfg["gains"][k] = v
    core = DockTestCore(cfg)
    lim = cfg["limits"]
    alloc = Allocator(rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"])
    geom = copy.deepcopy(load_geometry())
    geom["vehicle"]["gyration_m"][1] *= math.sqrt(inertia_scale)
    x0 = 10.0 - 0.575 - standoff_m
    m = VehicleModel(geom=geom, pos=(x0, 0.0, 3.0 + depth0_m), eul_deg=(0.0, pitch0_deg, 0.0), thruster_tau_s=tau)
    n_delay = max(1, int(round(img_delay / dt)))
    buf: deque = deque(maxlen=n_delay + 1)
    held_q = held_u = held_wz = 0.0
    next_odom = next_img = 0.0
    view = DockView(fresh=False)
    log = []
    t = 0.0
    while t < T:
        if t >= next_odom:
            held_q, held_u = float(m.nu[4]), float(m.nu[0])
            held_wz = float((eul_to_rotm(np.degrees(m.eul)) @ m.nu[:3])[2])
            next_odom += 1.0 / odom_hz
        if t >= next_img:
            eul = np.degrees(m.eul)
            top, bot, mid = (_CAM.project(m.pos, eul, [10.0, 0.0, z]) for z in (2.0, 4.0, 3.0))
            if top and bot and mid:
                ey, ex = mid[1] - _CAM.H / 2, mid[0] - _CAM.W / 2
                buf.append(DockView(
                    fresh=True, valid=True, elevation_valid=True, elevation_rad=math.atan(ey / _CAM.f) - m.eul[1],
                    error_x_px=ex, error_y_px=ey, radius_px=0.5 * math.hypot(top[0] - bot[0], top[1] - bot[1]),
                ))
            next_img += 1.0 / 15.0
        if len(buf) > n_delay:
            view = buf[0]
        w, _ = core.update(view, VehicleSnap(speed_u=held_u, pitch_rate=held_q, heave_rate=held_wz, dt=dt))
        w[0] = w[3] = w[5] = 0.0
        cmd = alloc.allocate(w, 0.0)
        m.set_command({k: v for k, v in cmd.items() if k in ("th_02", "th_03")})
        for _ in range(int(round(dt / 0.01))):
            m.step(0.01)
        t += dt
        log.append((t, math.degrees(m.eul[1]), m.pos[2] - 3.0, cmd.get("th_02", 0.0), cmd.get("th_03", 0.0)))
    L = np.array(log)
    late = L[L[:, 0] > T - 20.0]
    ok = (np.abs(L[:, 1]) < 3.0) & (np.abs(L[:, 2]) < 0.1)
    settle = next((float(L[i, 0]) for i in range(len(L)) if ok[i:].all()), math.inf)
    return {
        "pitch_amp": float(np.ptp(late[:, 1]) / 2), "depth_amp": float(np.ptp(late[:, 2]) / 2),
        "rpm_swing": float(max(np.ptp(late[:, 3]), np.ptp(late[:, 4])) / 2), "max_pitch": float(np.abs(L[:, 1]).max()),
        "settle_s": settle, "log": L,
    }


OLD_GAINS = {"pitch_nm_per_px": 0.04, "pitch_kd": 4.0, "elevation_kd_n_per_mps": 0.0}


def main() -> None:
    cur = load_cfg()["gains"]
    print(f"{'plant':<12} | {'gains':<8} | pitch amp (deg) | depth amp (m) | rpm swing | settle (3 deg, 0.1 m)")
    for name, plant in PLANTS.items():
        for label, g in (("OLD", OLD_GAINS), ("shipped", {})):
            r = run(g, **plant)
            st = f"{r['settle_s']:.1f} s" if math.isfinite(r["settle_s"]) else "never"
            print(f"{name:<12} | {label:<8} | {r['pitch_amp']:15.2f} | {r['depth_amp']:13.3f} | {r['rpm_swing']:9.0f} | {st}")
    print(f"\nshipped: pitch_nm_per_px {cur['pitch_nm_per_px']}, pitch_kd {cur['pitch_kd']}, elevation_kd_n_per_mps {cur.get('elevation_kd_n_per_mps', 0.0)}")


if __name__ == "__main__":
    main()
