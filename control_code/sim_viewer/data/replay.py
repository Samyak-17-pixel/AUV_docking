"""Load recorded runs (record_run.py, dof_testing and station_keeping CSVs) and play them back as a data source for the viewer.

Pure Python (numpy, csv): no ROS and no Qt. A `Run` holds the state and the actuator commands on the recorded time base; `Run.at(t)` interpolates.
"""

from __future__ import annotations

import csv
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from recording import ACTUATORS, ALIGN_COLUMNS


class UnsupportedLog(ValueError):
    """The file is not a run this tool can use (and why)."""


@dataclass
class Run:
    t: np.ndarray                     # [s], increasing
    pos: np.ndarray                   # (N,3) NED [m]
    eul: np.ndarray                   # (N,3) roll, pitch, yaw [rad] (yaw unwrapped)
    nu: np.ndarray                    # (N,6) body velocities [u v w p q r]
    cmd: Dict[str, np.ndarray] = field(default_factory=dict)      # actuator name -> (N,)
    cmd_age: Optional[np.ndarray] = None                           # seconds since the last command (recorder only)
    align: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)
    fmt: str = ""
    path: str = ""

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n > 1 else 0.0

    def index_at(self, t: float) -> int:
        return int(min(max(np.searchsorted(self.t, t, side="right") - 1, 0), self.n - 1))

    def at(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Linearly interpolated (pos, eul, nu); commands are held (zero-order) from the last sample at or before t."""
        t = float(min(max(t, self.t[0]), self.t[-1]))
        i = self.index_at(t)
        j = min(i + 1, self.n - 1)
        a = 0.0 if j == i else (t - self.t[i]) / (self.t[j] - self.t[i])
        lerp = lambda arr: arr[i] + a * (arr[j] - arr[i])            # noqa: E731
        return lerp(self.pos), lerp(self.eul), lerp(self.nu), {k: float(v[i]) for k, v in self.cmd.items()}

    def slice(self, t0: float, t1: float) -> "Run":
        m = (self.t >= t0) & (self.t <= t1)
        return Run(self.t[m] - self.t[m][0] if m.any() else self.t[:0], self.pos[m], self.eul[m], self.nu[m], {k: v[m] for k, v in self.cmd.items()},
                   None if self.cmd_age is None else self.cmd_age[m], {k: v[m] for k, v in self.align.items()}, dict(self.meta), self.fmt, self.path)


def _read_table(path: Path) -> Tuple[Dict[str, str], List[str], List[List[str]]]:
    meta: Dict[str, str] = {}
    lines: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for ln in f:
            if ln.startswith("#"):
                k, _, v = ln[1:].partition(":")
                meta[k.strip()] = v.strip()
            elif ln.strip():
                lines.append(ln)
    if len(lines) < 3:
        raise UnsupportedLog(f"{path.name}: needs a header and at least two data rows")
    rows = list(csv.reader(lines))
    return meta, rows[0], rows[1:]


def load_run(path: Path) -> Run:
    """Read a recorder, dof_testing or station_keeping CSV. Raises UnsupportedLog with the reason if it cannot be used."""
    path = Path(path)
    meta, header, rows = _read_table(path)
    idx = {h: i for i, h in enumerate(header)}

    def col(name: str, default: Optional[float] = None) -> np.ndarray:
        if name not in idx:
            if default is None:
                raise UnsupportedLog(f"{path.name}: missing column '{name}'")
            return np.full(len(rows), default)
        out = np.empty(len(rows))
        for r, row in enumerate(rows):
            try:
                out[r] = float(row[idx[name]])
            except (ValueError, IndexError):
                out[r] = np.nan
        return out

    if "cmd_age" in idx:
        fmt = "recorder"
    elif "phase" in idx and "roll_deg" in idx:
        fmt = "dof_testing"
    elif "depth_err_m" in idx:
        fmt = "station_keeping"
        if "roll_deg" not in idx:
            raise UnsupportedLog(f"{path.name}: this station_keeping log has no pose columns (written by an older version); record it again")
    else:
        raise UnsupportedLog(f"{path.name}: not a recorder, dof_testing or station_keeping log (columns: {', '.join(header[:8])}...)")

    t = col("t")
    z = col("z") if "z" in idx else col("depth")
    pos = np.column_stack([col("x"), col("y"), z])
    eul = np.radians(np.column_stack([col("roll_deg"), col("pitch_deg"), col("yaw_deg")]))
    nu = np.column_stack([col("u"), col("v", 0.0), col("w", 0.0), col("p", 0.0), col("q", 0.0), col("r", 0.0)])
    bad = ~np.isfinite(np.column_stack([t, pos, eul, nu])).all(axis=1)
    if bad.all():
        raise UnsupportedLog(f"{path.name}: no valid rows")
    keep = ~bad
    keep &= np.concatenate([[True], np.diff(t) > 0]) if len(t) > 1 else keep       # strictly increasing time
    eul[keep, 2] = np.unwrap(eul[keep, 2])
    cmd = {a: col(a)[keep] for a in ACTUATORS if a in idx}
    align = {c: col(c)[keep] for c in ALIGN_COLUMNS if c in idx}
    run = Run(t[keep] - t[keep][0], pos[keep], eul[keep], nu[keep], cmd, col("cmd_age")[keep] if fmt == "recorder" else None, align, meta, fmt, str(path))
    if run.n < 3:
        raise UnsupportedLog(f"{path.name}: fewer than 3 usable rows")
    return run


class ReplaySource(threading.Thread):
    """Plays a Run into a Telemetry store at wall-clock speed x `speed`. Same interface as the other sources (ready, error, counts, stop, ...)."""

    def __init__(self, telemetry, run: Run, speed: float = 1.0, loop: bool = False, model: Optional[Run] = None, camera: bool = True) -> None:
        super().__init__(daemon=True, name="replay_source")
        self.tel, self.data, self.model = telemetry, run, model
        self.speed = float(speed)
        self.loop = bool(loop)
        self.playing = True
        self._pos_t = 0.0                       # position in the run [s]
        self._seek_to: Optional[float] = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self.ready = threading.Event()
        self.error: Optional[str] = None
        self.counts = {"samples": 0, "image": 0}
        self.sim_status: dict = {}
        self.sim_controls_available = False
        self._next = 0
        self.camera = bool(camera)
        self.sonar = None                       # side-scan pings saved with the run (sidescan.npz / <run>_sidescan.npz): played back in step with the trajectory
        try:
            from recording import SonarLog, sonar_sidecar_path
            sc = sonar_sidecar_path(Path(run.path))
            if sc is not None:
                self.sonar = SonarLog(sc)
        except Exception:                                                       # noqa: BLE001
            self.sonar = None
        self._last_cam = 0.0
        self._cam = None
        self._det = None
        if self.camera:
            self._camera_init()

    # ------------------------------------------------------------------ interface shared with the other sources
    def send_sim_command(self, d: dict) -> bool:
        return False

    def stop(self) -> None:
        self._stop_evt.set()

    # ------------------------------------------------------------------ replay controls (called from the GUI thread)
    @property
    def position(self) -> float:
        with self._lock:
            return self._pos_t

    @property
    def duration(self) -> float:
        return self.data.duration

    def play(self) -> None:
        with self._lock:
            if self._seek_to is None and self._pos_t >= self.data.duration - 1e-6:        # at the end: play again from the start (unless a seek is pending)
                self._seek_to = 0.0
            self.playing = True

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def seek(self, t: float) -> None:
        with self._lock:
            self._seek_to = float(min(max(t, 0.0), self.data.duration))

    def set_speed(self, s: float) -> None:
        self.speed = float(min(max(s, 0.05), 50.0))

    # ------------------------------------------------------------------ thread
    # ------------------------------------------------------------------ camera: recordings hold no pictures, so they are RE-RENDERED from the recorded pose
    def _camera_init(self) -> None:
        """The recorded run has no camera frames, so the picture pane would stay empty. Draw what the nose camera would have seen from the recorded pose (the synthetic
        camera, the same one the offline stack uses) and run the REAL detector on it for the lights overlay. Falls back to the recorded numbers if the detector (or the
        interfaces package) is not available."""
        self._cam, self._det = None, None
        try:
            from synthetic_camera import SyntheticCamera
            self._cam = SyntheticCamera()
        except Exception:                                                       # noqa: BLE001
            return
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dock_detection_algo"))
            from dock_detector import DockDetector
            self._det = DockDetector()
        except Exception:                                                       # noqa: BLE001
            self._det = None

    def _camera_frame(self, i: int, force: bool = False) -> bool:
        """Render (and detect on) the camera picture for sample i, at most ~15 per wall-clock second. True when the detector's output was pushed."""
        if self._cam is None:
            return False
        now = time.monotonic()
        if not force and now - self._last_cam < 0.066:
            return self._det is not None
        self._last_cam = now
        r = self.data
        import cv2
        bgr = self._cam.render(r.pos[i], np.degrees(r.eul[i]))
        jpeg = self._cam.encode_jpeg(bgr)
        self.tel.set_image(jpeg)
        self.counts["image"] = self.counts.get("image", 0) + 1
        if self._det is None:
            return False
        try:
            m = self._det.process(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), float(r.eul[i][0]), float(r.eul[i][1]), float(r.t[i])).msg
        except Exception:                                                       # noqa: BLE001
            self._det = None
            return False
        self.tel.push_align({"error_x_px": m.error_x_px, "error_y_px": m.error_y_px, "lateral_px": m.lateral_px, "elevation_deg": math.degrees(m.elevation_rad) if m.elevation_valid else 0.0,
                             "radius_px": m.radius_px, "num_lights": float(m.num_lights), "valid": 1.0 if m.valid else 0.0, "top_x": m.top.x, "top_y": m.top.y, "bottom_x": m.bottom.x,
                             "bottom_y": m.bottom.y, "right_x": m.right.x, "right_y": m.right.y, "left_x": m.left.x, "left_y": m.left.y, "center_x": m.center.x, "center_y": m.center.y,
                             "confidence": m.confidence, "spread_px": m.spread_px, "obliqueness": m.obliqueness, "aligned": 1.0 if m.aligned else 0.0, "center_exact": 1.0 if m.center_exact else 0.0,
                             "error_x_norm": m.error_x_norm, "error_y_norm": m.error_y_norm, "search_yaw_norm": m.search_yaw_norm, "search_pitch_norm": m.search_pitch_norm,
                             "search_surge_norm": m.search_surge_norm})
        return True

    def _push(self, i: int, force_cam: bool = False) -> None:
        r = self.data
        self.tel.push_odom(r.pos[i], r.eul[i], r.nu[i])
        if r.cmd:
            self.tel.push_cmd(list(r.cmd), [r.cmd[k][i] for k in r.cmd])
        if self.sonar is not None:
            for (_t, side, inten, meta) in self.sonar.until(float(r.t[i])):
                self.tel.push_sonar(side, inten, meta)
        detected = self._camera_frame(i, force_cam) if self.camera else False
        if not detected and len(r.align) == len(ALIGN_COLUMNS) and np.isfinite(r.align['align_valid'][i]):
            a = r.align
            self.tel.push_align({"valid": a["align_valid"][i], "num_lights": a["align_lights"][i], "error_x_px": a["align_err_x_px"][i],
                                 "error_y_px": a["align_err_y_px"][i], "radius_px": a["align_radius_px"][i], "elevation_deg": a["align_elev_deg"][i]})
        if self.model is not None:
            m = self.model
            self.tel.push_model(m.pos[i], m.eul[i], m.nu[i])
        self.counts["samples"] += 1

    def _loop(self) -> None:
        self.tel.source_name = f"REPLAY {Path(self.data.path).name} ({self.data.fmt}, {self.data.duration:.1f} s)" + (" + model overlay" if self.model is not None else "")
        self.ready.set()
        last = time.monotonic()
        self._push(0, force_cam=True)
        self._next = 1
        while not self._stop_evt.is_set():
            now = time.monotonic()
            dt, last = now - last, now
            with self._lock:
                seek, self._seek_to = self._seek_to, None
                playing = self.playing
                if seek is not None:
                    self._pos_t = seek
                elif playing:
                    self._pos_t += dt * self.speed
                t_play = self._pos_t
            if seek is not None:
                self.tel.clear()
                self._next = self.data.index_at(seek)
                if self.sonar is not None:                                      # show the last few hundred pings before the new position
                    self.sonar._next = max(int(np.searchsorted(self.sonar.t, seek, side="right")) - 500, 0)
                self._push(self._next, force_cam=True)
                self._next += 1
            if playing or seek is not None:
                while self._next < self.data.n and self.data.t[self._next] <= t_play:
                    self._push(self._next)
                    self._next += 1
                if self._next >= self.data.n and playing:
                    if self.loop:
                        with self._lock:
                            self._seek_to = 0.0
                    else:
                        with self._lock:
                            self.playing = False
                            self._pos_t = self.data.duration
            time.sleep(0.01)

    def run(self) -> None:                                                     # noqa: F811  (Thread.run entry point)
        try:
            self._loop()
        except Exception as exc:                                               # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
