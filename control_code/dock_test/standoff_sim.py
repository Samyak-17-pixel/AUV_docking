#!/usr/bin/env python3
"""Closed-loop standoff harness for dock_test with a HEADING error (no ROS, no GUI).

The REAL DockTestCore + the REAL allocator against the full offline vehicle model, with held ~4.5 Hz odometry, image latency, and the synthetic camera
projecting the four lights (perfect detection + 0.5 px noise, then the real dock_geometry cues: error_x/y, lateral_px, radius). The vehicle starts near
the standoff distance with a yaw error; the question is whether it ends up square and holding (mode 'standoff' = the controller itself says: dock centred, square,
stopped), and whether it ever gets closer than the hold distance. NOTE the final yaw is relative to the dock axis, but the controller aims at the dock CENTRE, so
with a sideways offset a correct rest has a few degrees of yaw: judge by 'at rest', not by the yaw.

  python3 standoff_sim.py            # a table of yaw errors x plants
The plant numbers (thruster lag, delays, inertia) are ASSUMPTIONS, as in loop_sim.py.
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
for p in (_HERE, _HERE.parent / "common", _HERE.parent / "sim_offline", _HERE.parent / "sim_viewer", _HERE.parent.parent / "dock_detection_algo"):
    sys.path.insert(0, str(p))

from allocation import Allocator, eul_to_rotm, load_geometry  # noqa: E402
from dock_geometry import evaluate_dock_geometry  # noqa: E402
from dock_test_core import DockTestCore, DockView, VehicleSnap  # noqa: E402
from sensors import OdometryConfig, OdometrySensor  # noqa: E402
from synthetic_camera import SyntheticCamera  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

PLANTS: Dict[str, dict] = {                       # odom_latency: age of the published odometry (the live sim's is unknown; the offline stack assumes 0.25 s)
    "nominal": dict(tau=0.2, odom_hz=4.5, odom_latency=0.25, img_delay=0.1, inertia_scale=1.0),
    "heavy+late": dict(tau=0.3, odom_hz=4.5, odom_latency=0.35, img_delay=0.2, inertia_scale=1.3),
    "light+fast": dict(tau=0.1, odom_hz=9.0, odom_latency=0.15, img_delay=0.05, inertia_scale=0.7),
}
NOSE_X = 0.66            # the camera sits at the nose; distances in the controller are to the dock plane from the camera
DOCK_X = 10.0

_CAM = SyntheticCamera()


def load_cfg() -> dict:
    return yaml.safe_load(open(_HERE / "dock_test.yaml"))


def _view(pos, eul, rng, px_noise: float = 0.5) -> Optional[DockView]:
    """DockView the detector + dock_align_msg would produce for a perfect detection (None when a light leaves the frame)."""
    eul_l = np.array(eul, dtype=float).copy()
    eul_l[0] = 0.0                                                      # the detector removes roll
    pts = []
    for name in ("top", "bottom", "right", "left"):
        L = next(l for l in _CAM.lights if l["name"] == name)
        uvz = _CAM.project(pos, np.degrees(eul_l), L["p"])
        if uvz is None or not (5 < uvz[0] < _CAM.W - 5 and 5 < uvz[1] < _CAM.H - 5):
            return None
        pts.append((uvz[0] + rng.normal(0, px_noise), uvz[1] + rng.normal(0, px_noise)))
    side = sorted(pts[2:], key=lambda p: p[0])
    geo = evaluate_dock_geometry([pts[0], pts[1], side[1], side[0]])
    if not geo.ok:
        return None
    ex, ey = geo.center[0] - _CAM.W / 2, geo.center[1] - _CAM.H / 2
    return DockView(fresh=True, valid=True, elevation_valid=True, elevation_rad=math.atan(ey / _CAM.f) - float(eul[1]),
                    error_x_px=ex, error_y_px=ey, lateral_px=geo.lateral_px, radius_px=geo.radius_tb)


def run(core_cls=DockTestCore, cfg: Optional[dict] = None, *, yaw0_deg: float = 20.0, standoff_m: float = 2.4, lateral_m: float = 0.0,
        tau: float = 0.2, odom_hz: float = 4.5, odom_latency: float = 0.25, img_delay: float = 0.1, inertia_scale: float = 1.0, px_noise: float = 1.0,
        T: float = 90.0, dt: float = 0.05, seed: int = 1, blind_surge: float = -1.0) -> dict:
    cfg = copy.deepcopy(cfg or load_cfg())
    core = core_cls(cfg)
    lim = cfg["limits"]
    alloc = Allocator(rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"], u_fin_min=lim["u_fin_min_mps"], u_fin_off=lim["u_fin_off_mps"],
                      u_fin_full=lim["u_fin_full_mps"], small_force_n=lim.get("small_force_n", 0.0))
    geom = copy.deepcopy(load_geometry())
    for i in (1, 2):
        geom["vehicle"]["gyration_m"][i] *= math.sqrt(inertia_scale)
    m = VehicleModel(geom=geom, pos=(DOCK_X - 0.575 - standoff_m, lateral_m, 3.0), eul_deg=(0.0, 0.0, yaw0_deg), thruster_tau_s=tau)
    rng = np.random.default_rng(seed)
    n_delay = max(1, int(round(img_delay / dt)))
    buf: deque = deque(maxlen=n_delay + 1)
    sensor = OdometrySensor(OdometryConfig(rate_hz=odom_hz, latency_s=odom_latency, freeze_mean_interval_s=0.0, seed=seed))
    snap_v = [0.0] * 6
    next_img = 0.0
    view = DockView(fresh=False)
    log = []
    t = 0.0
    while t < T:
        # (the controller sees the late, noisy, slow odometry, held between publications: updated inside the physics loop below)
        if t >= next_img:
            v = _view(m.pos, m.eul, rng, px_noise)
            buf.append(v if v is not None else DockView(fresh=True, valid=False, search_surge_norm=blind_surge))
            next_img += 1.0 / 15.0
        if len(buf) > n_delay:
            view = buf[0]
        snap = VehicleSnap(speed_u=snap_v[0], roll=snap_v[1], roll_rate=snap_v[2], pitch_rate=snap_v[3], yaw_rate=snap_v[4], heave_rate=snap_v[5], dt=dt)
        w, st = core.update(view, snap)
        cmd = alloc.allocate(w, snap.speed_u)
        m.set_command(cmd)
        for k in range(int(round(dt / 0.01))):
            m.step(0.01)
            for o in sensor.update(t + 0.01 * (k + 1), m.pos, m.eul, m.nu):
                snap_v = [float(o.nu[0]), float(o.eul[0]), float(o.nu[3]), float(o.nu[4]), float(o.nu[5]),
                          float((eul_to_rotm(np.degrees(o.eul)) @ o.nu[:3])[2])]
        t += dt
        dock_d = DOCK_X - (m.pos[0] + NOSE_X)                                    # nose to the dock plane (along x)
        log.append((t, math.degrees(m.eul[2]), dock_d, m.nu[0], m.pos[1], str(st.get("mode")), view.error_x_px, view.lateral_px, view.radius_px))
    yaw = np.array([r[1] for r in log])
    dist = np.array([r[2] for r in log])
    late = [r for r in log if r[0] > T - 15.0]
    return {
        "final_yaw_deg": float(yaw[-1]), "late_yaw_amp": float(np.ptp([r[1] for r in late]) / 2), "min_dist_m": float(dist.min()),
        "final_dist_m": float(dist[-1]), "final_y_m": float(log[-1][4]), "modes": sorted({r[5] for r in log}),
        "rest_frac": sum(1 for r in late if r[5] == "standoff") / max(len(late), 1),          # fraction of the last 15 s in mode 'standoff' (square, in frame, stopped)
        "log": log,
    }


def main() -> None:
    print(f"{'plant':<11} | yaw0 | final yaw | min dist | final dist | at rest (last 15 s) | modes")
    for name, plant in PLANTS.items():
        for yaw0 in (0.0, 10.0, 20.0, 35.0, -25.0):
            r = run(yaw0_deg=yaw0, **plant)
            print(f"{name:<11} | {yaw0:+4.0f} | {r['final_yaw_deg']:+9.1f} | {r['min_dist_m']:8.2f} | {r['final_dist_m']:10.2f} | {100 * r['rest_frac']:17.0f} % | {','.join(r['modes'])}")


if __name__ == "__main__":
    main()
