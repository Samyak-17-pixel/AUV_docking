"""Run-time control of the offline simulation (pause, step, reset, push, time scale). Pure Python: used by fake_vehicle.py (ROS) and by the
viewer's in-process demo, and tested without ROS.

Commands are JSON objects (one per message):
  {"cmd": "pause", "value": true}                       stop (or resume) the simulated time; odometry keeps being published
  {"cmd": "step", "seconds": 0.5}                       while paused, advance this much simulated time then stop again
  {"cmd": "reset", "pos": [x, y, z], "eul_deg": [r, p, y]}   put the vehicle back (velocities zero); omitted fields use the start pose
  {"cmd": "push", "wrench": [X, Y, Z, K, M, N], "duration": 2.0}   an external force/moment on the vehicle [N, N*m] for `duration` seconds
  {"cmd": "clear_push"}
  {"cmd": "time_scale", "value": 2.0}                   simulated seconds per real second, limited to [0.1, 10]
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

import numpy as np

MIN_SCALE, MAX_SCALE = 0.1, 10.0


class SimControl:
    def __init__(self, start_pos=(0.0, 0.0, 3.0), start_eul_deg=(0.0, 0.0, 0.0)) -> None:
        self.start_pos = list(map(float, start_pos))
        self.start_eul_deg = list(map(float, start_eul_deg))
        self.paused = False
        self.time_scale = 1.0
        self._step_left = 0.0
        self._push = np.zeros(6)
        self._push_left = 0.0
        self._reset: Optional[Dict[str, list]] = None
        self.last_error = ""

    # ------------------------------------------------------------------ commands
    def handle(self, msg: Union[str, Dict[str, Any]]) -> bool:
        """Apply one command. Returns False (and sets last_error) for a malformed or unknown command."""
        try:
            d = json.loads(msg) if isinstance(msg, str) else dict(msg)
            cmd = d["cmd"]
            if cmd == "pause":
                self.paused = bool(d.get("value", True))
                self._step_left = 0.0
            elif cmd == "step":
                self._step_left += max(0.0, float(d.get("seconds", 0.5)))
            elif cmd == "reset":
                pos = d.get("pos", self.start_pos)
                eul = d.get("eul_deg", self.start_eul_deg)
                if len(pos) != 3 or len(eul) != 3:
                    raise ValueError("reset needs pos [x,y,z] and eul_deg [roll,pitch,yaw]")
                self._reset = {"pos": [float(v) for v in pos], "eul_deg": [float(v) for v in eul]}
                self._push[:] = 0.0
                self._push_left = 0.0
            elif cmd == "push":
                w = [float(v) for v in d["wrench"]]
                if len(w) != 6:
                    raise ValueError("push needs a wrench with 6 numbers [X,Y,Z,K,M,N]")
                self._push = np.array(w)
                self._push_left = max(0.0, float(d.get("duration", 1.0)))
            elif cmd == "clear_push":
                self._push[:] = 0.0
                self._push_left = 0.0
            elif cmd == "time_scale":
                self.time_scale = float(np.clip(float(d["value"]), MIN_SCALE, MAX_SCALE))
            else:
                raise ValueError(f"unknown command {cmd!r}")
            self.last_error = ""
            return True
        except Exception as exc:                                             # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    # ------------------------------------------------------------------ used by the simulation loop
    def take_reset(self) -> Optional[Dict[str, list]]:
        r, self._reset = self._reset, None
        return r

    def sim_dt(self, real_dt: float) -> float:
        """Simulated seconds to advance during `real_dt` real seconds (0 when paused and no step is pending)."""
        if self.paused:
            if self._step_left <= 0.0:
                return 0.0
            dt = min(self._step_left, real_dt * self.time_scale)
            self._step_left -= dt
            return dt
        return real_dt * self.time_scale

    def external_wrench(self, sim_dt: float) -> np.ndarray:
        """The pushing wrench for this step; it expires after its duration of SIMULATED time."""
        if self._push_left <= 0.0:
            return np.zeros(6)
        w = self._push.copy()
        self._push_left -= sim_dt
        if self._push_left <= 0.0:
            self._push[:] = 0.0
        return w

    def status(self) -> Dict[str, Any]:
        return {
            "paused": self.paused, "time_scale": self.time_scale, "step_left": round(self._step_left, 3),
            "push_left": round(max(self._push_left, 0.0), 3), "push": [round(float(v), 3) for v in self._push], "error": self.last_error,
        }
