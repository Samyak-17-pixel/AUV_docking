"""Station-keeping engine (pure Python, no ROS): hover hold of depth, pitch, surge position and heading.

Loops (all from common/loops.py):
  depth   -> Z force on the two vertical heave thrusters (common mode)
  pitch   -> M moment on the heave thrusters (differential mode)
  surge   -> X force on the axial thruster (+/- RPM), position along the capture heading
  heading -> N moment on the X-fins; needs forward flow (force ~ U^2), so it fades out near zero speed.
Lateral (sway) position is NOT controllable (no sway actuator): it is reported as drift, not held.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from allocation import Allocator  # noqa: E402
from loops import HoldLoops  # noqa: E402
from state import State, wrap_pi  # noqa: E402


def shrink(err: float, deadband: float) -> float:
    """Dead-band: errors smaller than `deadband` count as zero; bigger ones are reduced by it (no jump)."""
    if deadband <= 0.0:
        return err
    return math.copysign(max(abs(err) - deadband, 0.0), err)


class StationKeeper:
    def __init__(self, cfg: Dict[str, Any], alloc: Optional[Allocator] = None) -> None:
        self.cfg = cfg
        lim = cfg["limits"]
        self.alloc = alloc or Allocator(
            rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"],
            u_fin_min=lim["u_fin_min_mps"], u_fin_off=lim["u_fin_off_mps"], u_fin_full=lim["u_fin_full_mps"],
        )
        g = {k: v for k, v in cfg["gains"].items()}
        self.loops = HoldLoops(g, heave_ff_n=cfg["feedforward"]["heave_n"])
        self.en = cfg["enable"]
        self.sp: Optional[Dict[str, float]] = None
        self._axis = np.array([1.0, 0.0])
        self._origin = np.zeros(2)
        self._prev_out: Dict[str, float] = {}
        self._cap_samples = []
        self.t = 0.0
        self.captured = False
        self.status: Dict[str, Any] = {}

    # ----------------------------------------------------------------- capture
    def capture(self, st: State) -> None:
        """Latch the setpoints: current pose (mode 'current') or the explicit values from the YAML."""
        c = self.cfg["capture"]
        cur = {
            "depth": st.depth, "pitch": st.pitch, "yaw": st.yaw,
            "north": float(st.pos[0]), "east": float(st.pos[1]),
        }
        if c["mode"] == "explicit":
            e = c["explicit"]
            cur = {
                "depth": float(e["depth_m"]), "pitch": math.radians(e["pitch_deg"]),
                "yaw": math.radians(e["heading_deg"]), "north": float(e["north_m"]), "east": float(e["east_m"]),
            }
        self.sp = cur
        self._axis = np.array([math.cos(cur["yaw"]), math.sin(cur["yaw"])])
        self._origin = np.array([cur["north"], cur["east"]])
        self.loops.reset()
        self.captured = True

    def feed_capture(self, st: State, dt: float) -> bool:
        """Average the first capture.average_s seconds of state, then latch. True when captured."""
        if self.captured:
            return True
        self._cap_samples.append(st)
        self.t += dt
        if self.t >= float(self.cfg["capture"]["average_s"]):
            n = len(self._cap_samples)
            avg = State(
                pos=np.mean([s.pos for s in self._cap_samples], axis=0),
                eul=np.array([
                    float(np.mean([s.roll for s in self._cap_samples])),
                    float(np.mean([s.pitch for s in self._cap_samples])),
                    math.atan2(np.mean([math.sin(s.yaw) for s in self._cap_samples]),
                               np.mean([math.cos(s.yaw) for s in self._cap_samples])),
                ]),
                nu=np.mean([s.nu for s in self._cap_samples], axis=0),
            )
            self.capture(avg)
        return self.captured

    # ------------------------------------------------------------------ update
    def safety_reason(self, st: State) -> Optional[str]:
        s = self.cfg["safety"]
        if st.depth < s["min_depth_m"]:
            return f"too shallow ({st.depth:.2f} m)"
        if st.depth > s["max_depth_m"]:
            return f"too deep ({st.depth:.2f} m)"
        if abs(math.degrees(st.pitch)) > s["max_pitch_deg"]:
            return f"pitch {math.degrees(st.pitch):.0f} deg"
        if abs(math.degrees(st.roll)) > s["max_roll_deg"]:
            return f"roll {math.degrees(st.roll):.0f} deg"
        if self.captured:
            d = float(np.linalg.norm(st.pos[:2] - self._origin))
            if d > s["max_drift_m"]:
                return f"drifted {d:.1f} m from the hold point (leash {s['max_drift_m']} m)"
        return None

    def update(self, st: State, dt: float):
        """-> (actuator dict {th_XX: RPM, cs_XX: deg}, status dict)."""
        assert self.sp is not None, "call capture()/feed_capture() first"
        db = self.cfg["deadband"]
        w = np.zeros(6)
        sp = self.sp
        d = st.pos[:2] - self._origin
        ahead = float(d @ self._axis)
        lateral = float(d @ np.array([-self._axis[1], self._axis[0]]))

        if self.en["depth"]:
            w[2] = self.loops.heave(st, st.depth + shrink(sp["depth"] - st.depth, db["depth_m"]), dt)
        elif self.cfg["feedforward"]["apply_when_depth_disabled"]:
            w[2] = self.cfg["feedforward"]["heave_n"]
        if self.en["pitch"]:
            e = shrink(sp["pitch"] - st.pitch, math.radians(db["pitch_deg"]))
            w[4] = self.loops.pitch(st, st.pitch + e, dt)
        if self.en["surge"]:
            w[0] = self.loops.surge(st, shrink(-ahead, db["surge_m"]), self._axis, dt)
        if self.en["heading"]:
            e = shrink(wrap_pi(sp["yaw"] - st.yaw), math.radians(db["heading_deg"]))
            w[5] = self.loops.yaw(st, st.yaw + e, dt)

        out = self.alloc.allocate(w, st.speed_u)
        out = self._slew(out, dt)
        self.status = {
            "depth_err_m": sp["depth"] - st.depth, "pitch_err_deg": math.degrees(sp["pitch"] - st.pitch),
            "ahead_err_m": -ahead, "lateral_drift_m": lateral,
            "heading_err_deg": math.degrees(wrap_pi(sp["yaw"] - st.yaw)),
            "u": st.speed_u, "fin_authority": self.alloc.fin_authority(st.speed_u),
            "wrench": w.copy(),
        }
        return out, self.status

    def _slew(self, out: Dict[str, float], dt: float) -> Dict[str, float]:
        sl = self.cfg["slew"]
        res = {}
        for k, v in out.items():
            rate = sl["rpm_per_s"] if k.startswith("th_") else sl["fin_deg_per_s"]
            prev = self._prev_out.get(k, 0.0)
            step = float(rate) * dt if rate and rate > 0 else 1e12
            res[k] = prev + float(np.clip(v - prev, -step, step))
        self._prev_out = res
        return res

    def capture_command(self) -> Dict[str, float]:
        """Command while the hold point is still being averaged: only the buoyancy trim, so the vehicle does
        not float up 7 N worth during the capture window (the hold point would then be above where you started)."""
        out = self.alloc.thrusters_from_wrench([0, 0, self.cfg["feedforward"]["heave_n"], 0, 0, 0])
        out.update({n: 0.0 for n in self.alloc.fin_ids})
        return out

    def neutral(self) -> Dict[str, float]:
        self._prev_out = {}
        return {n: 0.0 for n in self.alloc.th_ids + self.alloc.fin_ids}
