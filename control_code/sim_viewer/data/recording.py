"""CSV format of a recorded run, and a writer that needs no ROS.

One row per odometry sample: the state, the last actuator commands (held until the next command arrives) and the last DockAlign values. The file starts
with '# key: value' comment lines. replay.py reads this format (and the CSVs of dof_testing and station_keeping); compare.py and calibrate.py use it to
replay the recorded COMMANDS through the vehicle model.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

STATE_COLUMNS = ["t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg", "u", "v", "w", "p", "q", "r"]     # z = depth (NED, +down); u..r are BODY rates [m/s, rad/s]
ACTUATORS = ["th_01", "th_02", "th_03", "cs_04", "cs_06", "cs_07", "cs_08"]                                  # RPM for th_*, degrees for cs_*
ALIGN_COLUMNS = ["align_valid", "align_lights", "align_err_x_px", "align_err_y_px", "align_radius_px", "align_elev_deg"]
COLUMNS = STATE_COLUMNS + ACTUATORS + ["cmd_age"] + ALIGN_COLUMNS


class CsvRecorder:
    def __init__(self, path: Path, meta: Optional[Dict[str, object]] = None, flush_every: int = 20) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        for k, v in {"recorder": "sim_viewer/data/record_run.py", "started": time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {})}.items():
            self._f.write(f"# {k}: {v}\n")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        self._cmd: Dict[str, float] = {a: 0.0 for a in ACTUATORS}
        self._cmd_t: Optional[float] = None
        self._align: Dict[str, float] = {c: 0.0 for c in ALIGN_COLUMNS}
        self.rows = 0
        self._flush_every = flush_every
        self.t_first: Optional[float] = None
        self.t_last = 0.0

    # ---------------------------------------------------------------- inputs (all times are seconds on one clock)
    def set_cmd(self, names: Sequence[str], values: Sequence[float], t: float) -> None:
        for n, v in zip(names, values):
            if n in self._cmd:
                self._cmd[n] = float(v)
        self._cmd_t = float(t)

    def set_align(self, valid: bool, lights: int, err_x: float, err_y: float, radius: float, elev_deg: float) -> None:
        self._align = dict(zip(ALIGN_COLUMNS, [1.0 if valid else 0.0, float(lights), float(err_x), float(err_y), float(radius), float(elev_deg)]))

    def add_state(self, t: float, pos: Sequence[float], eul_rad: Sequence[float], nu: Sequence[float]) -> None:
        if self.t_first is None:
            self.t_first = float(t)
        age = (float(t) - self._cmd_t) if self._cmd_t is not None else 999.0
        row = [float(t) - self.t_first, pos[0], pos[1], pos[2], math.degrees(eul_rad[0]), math.degrees(eul_rad[1]), math.degrees(eul_rad[2]), *map(float, nu)]
        row += [self._cmd[a] for a in ACTUATORS] + [round(age, 3)] + [self._align[c] for c in ALIGN_COLUMNS]
        self._w.writerow([f"{v:.9g}" if isinstance(v, float) else v for v in row])
        self.rows += 1
        self.t_last = row[0]
        if self.rows % self._flush_every == 0:
            self._f.flush()

    def close(self) -> Path:
        self._f.flush()
        self._f.close()
        return self.path

    @property
    def duration(self) -> float:
        return self.t_last


# ======================================================================================================================
# side-scan sonar pings next to a run (2026-10-11): <run>.csv + <run>_sidescan.npz (or sidescan.npz in a showcase folder)
# ======================================================================================================================
class SonarRecorder:
    """Collects side-scan pings (both sides) with the time since the start of the run and the vehicle pose at the ping; `save` writes one compressed .npz."""

    def __init__(self) -> None:
        self.t: List[float] = []
        self.side: List[int] = []
        self.inten: List[np.ndarray] = []
        self.meta: List[Dict[str, float]] = []

    def add(self, t: float, side: str, intensity, meta: Dict[str, float]) -> None:
        self.t.append(float(t))
        self.side.append(0 if side == "port" else 1)
        self.inten.append(np.asarray(intensity, np.uint16))
        self.meta.append(dict(meta))

    def __len__(self) -> int:
        return len(self.t)

    def save(self, path: Path) -> Path:
        path = Path(path)
        n = len(self.t)
        nb = max((len(x) for x in self.inten), default=0)
        arr = np.zeros((n, nb), np.uint16)
        nbins = np.zeros(n, np.int32)
        for i, x in enumerate(self.inten):
            arr[i, :len(x)] = x
            nbins[i] = len(x)
        g = lambda k, d=0.0: np.array([m.get(k, d) for m in self.meta], np.float64)
        pos = np.array([m.get("pos", [0, 0, 0]) for m in self.meta], np.float64).reshape(n, 3)
        eul = np.array([m.get("eul", [0, 0, 0]) for m in self.meta], np.float64).reshape(n, 3)
        np.savez_compressed(path, t=np.array(self.t), side=np.array(self.side, np.int8), intensity=arr, nbins=nbins, start_range_m=g("start_range_m"), range_m=g("range_m"),
                            gain_db=g("gain_db"), gain_index=g("gain_index", -1.0), altitude_m=g("altitude_m", float("nan")), ping_hz=g("ping_hz"), sound_speed_mps=g("sound_speed_mps", 1500.0),
                            pos=pos, eul=eul)
        return path


class SonarLog:
    """A saved side-scan recording (see SonarRecorder). `until(t)` yields the pings with time <= t that were not yet given (for replay)."""

    def __init__(self, path: Path) -> None:
        z = np.load(path)
        self.path = Path(path)
        self.t = z["t"]
        order = np.argsort(self.t, kind="stable")
        self.order = order
        self.t = self.t[order]
        self.side = z["side"][order]
        self.intensity = z["intensity"][order]
        self.nbins = z["nbins"][order]
        self.meta = {k: z[k][order] for k in ("start_range_m", "range_m", "gain_db", "gain_index", "altitude_m", "ping_hz", "sound_speed_mps", "pos", "eul")}
        self._next = 0

    def __len__(self) -> int:
        return len(self.t)

    def seek(self, t: float) -> None:
        self._next = int(np.searchsorted(self.t, t, side="right"))

    def ping(self, i: int):
        m = {k: (v[i].tolist() if v.ndim > 1 else float(v[i])) for k, v in self.meta.items()}
        m["ping_number"] = i
        return ("port" if self.side[i] == 0 else "starboard"), self.intensity[i, :int(self.nbins[i])], m

    def until(self, t: float):
        out = []
        while self._next < len(self.t) and self.t[self._next] <= t:
            out.append((float(self.t[self._next]),) + self.ping(self._next))
            self._next += 1
        return out


def sonar_sidecar_path(csv_path: Path) -> Optional[Path]:
    """The side-scan file belonging to a run CSV: <stem>_sidescan.npz next to it, or sidescan.npz in the same folder (the showcase layout)."""
    csv_path = Path(csv_path)
    for cand in (csv_path.with_name(csv_path.stem + "_sidescan.npz"), csv_path.with_name("sidescan.npz")):
        if cand.exists():
            return cand
    return None
