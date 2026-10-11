"""Live judge for a docking run in the viewer: the same failure list as terminal_docking_control/docking_sim.py (Judge), fed one sample at a time from the
telemetry (ground truth from the fake vehicle's status when available, else the odometry the controller sees). Pure Python, no Qt, no ROS."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE.parent / "terminal_docking_control"))
sys.path.insert(0, str(_HERE.parent / "common"))

from docking_sim import DOCK_POS, NOSE_X, funnel_clearance, hull_points  # noqa: E402
from terminal_docking_core import rot_zyx  # noqa: E402


class LiveJudge:
    def __init__(self, timeout_s: float = 180.0, rpm_cap: float = 1800.0) -> None:
        self.timeout_s = timeout_s
        self.rpm_cap = rpm_cap
        self.reset()

    def reset(self) -> None:
        self.t0: Optional[float] = None
        self.fail: Dict[str, str] = {}
        self.min_clear = math.inf
        self.max_depth_in = 0.0
        self.cross: Dict[str, float] = {}
        self.crossed = False
        self._prev: Optional[tuple] = None
        self.retries = 0
        self.phase = ""
        self.docked_since: Optional[float] = None
        self.pinned: Dict[str, float] = {}
        self.verdict = "waiting"           # waiting | running | PASS | FAIL
        self.final: Dict[str, float] = {}

    @property
    def failures(self) -> List[str]:
        return [f"{k}: {v}" for k, v in self.fail.items()]

    def feed(self, t: float, pos, eul, u: float, phase: str = "", retries: int = 0, cmd: Optional[Dict[str, float]] = None) -> str:
        if self.verdict in ("PASS", "FAIL"):
            return self.verdict
        pos, eul = np.asarray(pos, float), np.asarray(eul, float)
        if self.t0 is None:
            self.t0 = t
        self.verdict = "running"
        self.phase, self.retries = phase, int(retries or 0)
        nose = pos + rot_zyx(*eul) @ np.array([NOSE_X, 0.0, 0.0])
        s_nose = float(DOCK_POS[0] - nose[0])
        clear = min(funnel_clearance(p) for p in hull_points(pos, eul))
        self.min_clear = min(self.min_clear, clear)
        if clear < 0.0:
            self.fail.setdefault("wall contact", f"hull touches the funnel at t={t - self.t0:.1f} s")
        self.max_depth_in = max(self.max_depth_in, -s_nose)
        cur = (s_nose, float(nose[1] - DOCK_POS[1]), float(nose[2] - DOCK_POS[2]), math.degrees(eul[2]), float(u))
        if not self.crossed and self._prev is not None and self._prev[0] > 0.0 >= s_nose:           # nose crossed the mouth plane: interpolate to the crossing
            a = self._prev[0] / (self._prev[0] - s_nose) if self._prev[0] != s_nose else 1.0
            lat, vert, head, spd = (self._prev[i] + a * (cur[i] - self._prev[i]) for i in (1, 2, 3, 4))
            self.crossed, self.cross = True, {"lat": lat, "vert": vert, "heading_deg": head, "speed": spd}
            if abs(lat) > 0.15 or abs(vert) > 0.15:
                self.fail.setdefault("bad entry", f"lateral {lat:+.2f} m / vertical {vert:+.2f} m at the mouth (limit 0.15)")
            if abs(head) > 5.0:
                self.fail.setdefault("bad entry", f"heading {head:+.1f} deg at the mouth (limit 5)")
            if spd > 0.5:
                self.fail.setdefault("bad entry", f"entry speed {spd:.2f} m/s (limit 0.5)")
        self._prev = cur
        tilt = math.degrees(max(abs(eul[0]), abs(eul[1])))
        if tilt > 30.0:
            self.fail.setdefault("attitude limit", f"tilt {tilt:.0f} deg")
        if pos[2] < 0.5 or pos[2] > 8.0:
            self.fail.setdefault("depth limit", f"depth {pos[2]:.2f} m")
        for k, v in (cmd or {}).items():
            if k.startswith("th_"):
                if abs(v) >= 0.98 * self.rpm_cap:
                    self.pinned.setdefault(k, t)
                    if t - self.pinned[k] > 2.0:
                        self.fail.setdefault("actuator pinned", f"{k} at its cap > 2 s")
                else:
                    self.pinned.pop(k, None)
        if self.retries > 0:
            self.fail.setdefault("retry", f"{self.retries} back-off(s): a retry is not a first-try pass")
        if self.crossed and self.max_depth_in > 1.2:
            self.fail.setdefault("overshoot", f"nose {self.max_depth_in:.2f} m into the funnel (limit 1.2)")
        if t - self.t0 > self.timeout_s:
            self.fail.setdefault("timeout", f"not docked after {self.timeout_s:.0f} s")
        if phase == "DOCKED":
            self.docked_since = self.docked_since if self.docked_since is not None else t
            if t - self.docked_since > 4.0:
                depth = -s_nose
                if depth < 0.15:
                    self.fail.setdefault("stopped short", f"final nose depth {depth:.2f} m")
                self.final = {"nose_depth_m": depth, "heading_deg": cur[3]}
                self.verdict = "PASS" if not self.fail else "FAIL"
        else:
            self.docked_since = None
        if self.fail and any(k in self.fail for k in ("wall contact", "attitude limit", "depth limit", "timeout", "bad entry", "overshoot")):
            self.verdict = "FAIL"
        return self.verdict

    def summary(self) -> str:
        if self.verdict == "waiting":
            return "waiting for data"
        head = {"running": f"running ({self.phase})", "PASS": "PASS: docked first try", "FAIL": "FAIL"}[self.verdict]
        bits = [head]
        if self.crossed:
            bits.append(f"entry lat {self.cross['lat']:+.2f} m, vert {self.cross['vert']:+.2f} m, heading {self.cross['heading_deg']:+.1f} deg, {self.cross['speed']:.2f} m/s")
        if math.isfinite(self.min_clear):
            bits.append(f"min wall clearance {self.min_clear:.2f} m")
        bits += self.failures
        return "\n".join(bits)
