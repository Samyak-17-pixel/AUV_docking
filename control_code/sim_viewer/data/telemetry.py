"""Thread-safe store of what the viewer shows. No ROS, no Qt: fed by ros_link.py (live) or demo_source.py (in-process).

Time base: seconds since the first sample, taken from the monotonic wall clock on ARRIVAL. The real sim's odometry header stamps are 0, so
they cannot be used.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Sample:
    t: float
    pos: np.ndarray        # NED [m]
    eul: np.ndarray        # roll, pitch, yaw [rad]
    nu: np.ndarray         # body velocities [u v w p q r]


class Telemetry:
    def __init__(self, keep_s: float = 600.0) -> None:
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._keep = float(keep_s)
        self._samples: Deque[Sample] = deque()
        self._cmds: Deque[Tuple[float, Dict[str, float]]] = deque()
        self._aligns: Deque[Tuple[float, Dict[str, float]]] = deque()
        self._model: Deque[Sample] = deque()                                   # model-predicted pose (real-vs-model overlay)
        self._ctrl: Deque[Tuple[float, Dict[str, object]]] = deque()          # controller debug (setpoints, mode) from /ctrl_debug
        self._sonar: Dict[str, Deque[Tuple[float, np.ndarray, Dict[str, float]]]] = {"port": deque(maxlen=4000), "starboard": deque(maxlen=4000)}   # (t, intensity uint16, meta) per side
        self.sonar_status: Dict[str, object] = {}
        self._image: Optional[bytes] = None
        self._image_t = 0.0
        self.debug: Dict[str, object] = {}
        self.source_name = ""

    # ------------------------------------------------------------------ time
    def now(self) -> float:
        with self._lock:
            return self._now_locked()

    def _now_locked(self) -> float:
        m = time.monotonic()
        if self._t0 is None:
            self._t0 = m
        return m - self._t0

    def _trim(self, dq: deque, t: float) -> None:
        while dq and (dq[0].t if isinstance(dq[0], Sample) else dq[0][0]) < t - self._keep:
            dq.popleft()

    # ------------------------------------------------------------------ writers
    def push_odom(self, pos, eul, nu, t: Optional[float] = None) -> None:
        with self._lock:
            tt = self._now_locked() if t is None else float(t)
            self._samples.append(Sample(tt, np.asarray(pos, float).copy(), np.asarray(eul, float).copy(), np.asarray(nu, float).copy()))
            self._trim(self._samples, tt)

    def push_model(self, pos, eul, nu, t: Optional[float] = None) -> None:
        """Pose predicted by the vehicle model for the same moment as the latest recorded sample (drawn as a ghost + dashed lines)."""
        with self._lock:
            tt = self._now_locked() if t is None else float(t)
            self._model.append(Sample(tt, np.asarray(pos, float).copy(), np.asarray(eul, float).copy(), np.asarray(nu, float).copy()))
            self._trim(self._model, tt)

    def latest_model(self) -> Optional[Sample]:
        with self._lock:
            return self._model[-1] if self._model else None

    def push_cmd(self, names, values, t: Optional[float] = None) -> None:
        with self._lock:
            tt = self._now_locked() if t is None else float(t)
            self._cmds.append((tt, {str(n): float(v) for n, v in zip(names, values) if n}))
            self._trim(self._cmds, tt)

    def push_align(self, fields: Dict[str, float], t: Optional[float] = None) -> None:
        with self._lock:
            tt = self._now_locked() if t is None else float(t)
            self._aligns.append((tt, dict(fields)))
            self._trim(self._aligns, tt)

    def push_ctrl(self, fields: Dict[str, object], t: Optional[float] = None) -> None:
        """Controller debug message: numeric fields (setpoints) are plotted, text fields (controller name, mode) are shown."""
        with self._lock:
            tt = self._now_locked() if t is None else float(t)
            self._ctrl.append((tt, dict(fields)))
            self._trim(self._ctrl, tt)

    def latest_ctrl(self) -> Tuple[float, Dict[str, object]]:
        with self._lock:
            return self._ctrl[-1] if self._ctrl else (0.0, {})

    def push_sonar(self, side: str, intensity, meta: Dict[str, float], t: Optional[float] = None) -> None:
        """One side-scan ping. meta: start_range_m, range_m, gain_db, altitude_m, pos (3), eul (3), ping_number ..."""
        with self._lock:
            tt = self._now_locked() if t is None else float(t)
            self._sonar[side].append((tt, np.asarray(intensity, np.uint16).copy(), dict(meta)))

    def sonar_rows(self, side: str, n: int = 300):
        """The newest n pings of a side: (list of (t, intensity, meta))."""
        with self._lock:
            d = self._sonar[side]
            return list(d)[-n:]

    def sonar_since(self, t0: float):
        with self._lock:
            return {k: [x for x in v if x[0] >= t0] for k, v in self._sonar.items()}

    def set_image(self, jpeg: bytes) -> None:
        with self._lock:
            self._image = bytes(jpeg)
            self._image_t = self._now_locked()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear(); self._cmds.clear(); self._aligns.clear(); self._model.clear(); self._ctrl.clear(); [v.clear() for v in self._sonar.values()]; self._image = None; self._t0 = None

    # ------------------------------------------------------------------ readers
    def latest(self) -> Optional[Sample]:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def latest_cmd(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._cmds[-1][1]) if self._cmds else {}

    def latest_align(self) -> Tuple[float, Dict[str, float]]:
        with self._lock:
            return self._aligns[-1] if self._aligns else (0.0, {})

    def image(self) -> Optional[bytes]:
        with self._lock:
            return self._image

    def pose_now(self, extrapolate_s: float = 0.4, model: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Latest pose moved forward with its own body velocity, so the 3D view moves smoothly between slow (~4 Hz) odometry samples.
        model=True gives the same for the model-predicted pose (overlay ghost), or None when there is none."""
        from sim_viewer_math import body_to_ned   # local import keeps this module numpy-only at import time

        with self._lock:
            store = self._model if model else self._samples
            if not store:
                return None
            s = store[-1]
            dt = max(0.0, min(self._now_locked() - s.t, float(extrapolate_s)))
        if dt <= 0.0:
            return s.pos, s.eul, s.nu
        return s.pos + body_to_ned(s.eul) @ (s.nu[:3] * dt), s.eul, s.nu

    def window(self, seconds: float) -> Dict[str, np.ndarray]:
        """Arrays for the plots over the last `seconds`. Actuator and dock series are forward-filled onto their own time bases."""
        with self._lock:
            if not self._samples:
                return {}
            t_end = self._samples[-1].t
            samp: List[Sample] = [s for s in self._samples if s.t >= t_end - seconds]
            cmds = [(t, d) for t, d in self._cmds if t >= t_end - seconds]
            aligns = [(t, d) for t, d in self._aligns if t >= t_end - seconds]
            ctrl = [(t, d) for t, d in self._ctrl if t >= t_end - seconds]
            mod: List[Sample] = [s for s in self._model if s.t >= t_end - seconds]
        out: Dict[str, np.ndarray] = {
            "t": np.array([s.t for s in samp]) - t_end,
            "depth": np.array([s.pos[2] for s in samp]),
            "x": np.array([s.pos[0] for s in samp]),
            "y": np.array([s.pos[1] for s in samp]),
            "roll": np.degrees([s.eul[0] for s in samp]),
            "pitch": np.degrees([s.eul[1] for s in samp]),
            "yaw": np.degrees([s.eul[2] for s in samp]),
            "u": np.array([s.nu[0] for s in samp]),
            "v": np.array([s.nu[1] for s in samp]),
            "w": np.array([s.nu[2] for s in samp]),
            "p": np.degrees([s.nu[3] for s in samp]),
            "q": np.degrees([s.nu[4] for s in samp]),
            "r": np.degrees([s.nu[5] for s in samp]),
        }
        if mod:
            out["m_t"] = np.array([s.t for s in mod]) - t_end
            out["m_depth"] = np.array([s.pos[2] for s in mod])
            out["m_x"] = np.array([s.pos[0] for s in mod])
            out["m_y"] = np.array([s.pos[1] for s in mod])
            out["m_roll"] = np.degrees([s.eul[0] for s in mod])
            out["m_pitch"] = np.degrees([s.eul[1] for s in mod])
            out["m_yaw"] = np.degrees([s.eul[2] for s in mod])
            out["m_u"] = np.array([s.nu[0] for s in mod])
            out["m_q"] = np.degrees([s.nu[4] for s in mod])
            out["m_r"] = np.degrees([s.nu[5] for s in mod])
        names = sorted({n for _, d in cmds for n in d})
        out["cmd_t"] = np.array([t for t, _ in cmds]) - t_end
        for n in names:
            last, col = 0.0, []
            for _, d in cmds:
                last = d.get(n, last)
                col.append(last)
            out["cmd_" + n] = np.array(col)
        keys = sorted({k for _, d in aligns for k in d})
        out["align_t"] = np.array([t for t, _ in aligns]) - t_end
        for k in keys:
            last, col = 0.0, []
            for _, d in aligns:
                last = d.get(k, last)
                col.append(last)
            out["align_" + k] = np.array(col)
        nums = sorted({k for _, d in ctrl for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        out["ctrl_t"] = np.array([t for t, _ in ctrl]) - t_end
        for k in nums:
            last, col = float("nan"), []
            for _, d in ctrl:
                v = d.get(k, last)
                last = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else last
                col.append(last)
            out["ctrl_" + k] = np.array(col)
        return out
