#!/usr/bin/env python3
"""Closed-loop docking simulation, in-process (no ROS): offline vehicle + realistic odometry + synthetic camera + the REAL detector + the real controller core.

One call = one docking run from a start pose, judged against the failure list agreed with the user (see `judge`). Used by the grid evaluation
(terminal_docking_eval.py), by the viewer's scenario runner and by the tests. Plant numbers (thruster lag, delays, inertia) and the camera 'look' are ASSUMPTIONS.

World: the dock is at NED (10, 0, 3), mouth plane x = 10, funnel towards +x (x 10 .. 12.5), approach heading 0 (north). The vehicle starts south of it.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_CTRL = _HERE.parent
sys.path[:0] = [str(_d) for _d in [_CTRL / "sim_viewer", *sorted((_CTRL / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # sim_viewer and its sub-folders
for p in (_HERE, _CTRL / "common", _CTRL / "sim_offline", _CTRL / "sim_viewer", _CTRL.parent / "dock_detection_algo"):
    sys.path.insert(0, str(p))

from allocation import Allocator, load_geometry  # noqa: E402
from sensors import OdometryConfig, OdometrySensor  # noqa: E402
from state import State  # noqa: E402
from synthetic_camera import SyntheticCamera  # noqa: E402
from terminal_docking_core import DockObs, TerminalDockingCore, rot_zyx  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

CONFIG = _HERE / "terminal_docking.yaml"
DOCK_POS = np.array([10.0, 0.0, 3.0])
BODY_R = 0.09                                       # hull radius (mesh bounds +-0.09 m)
NOSE_X, TAIL_X = 0.66, -0.69
# inner radius of the funnel versus depth into it (measured from the dock STL: x_d = 0 mouth ... -2.5 end); the tube is 0.15 m radius
FUNNEL_DEPTH = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 2.5])
DOCK_OUTER_R = 1.15                                 # ASSUMED outer radius of the dock body [m] (the light ring is 1.0 m); beyond it the hull is beside the dock
FUNNEL_R_IN = np.array([0.751, 0.512, 0.327, 0.230, 0.170, 0.150, 0.150])

PLANTS: Dict[str, dict] = {                          # the four plants of dock_test/loop_sim.py plus the sensor delay that goes with each
    "nominal": dict(tau=0.2, odom_hz=4.5, odom_latency=0.25, img_delay=0.10, inertia=1.0, drag=1.0),
    "heavy+late": dict(tau=0.3, odom_hz=4.5, odom_latency=0.35, img_delay=0.20, inertia=1.3, drag=1.2),
    "light+fast": dict(tau=0.1, odom_hz=9.0, odom_latency=0.15, img_delay=0.05, inertia=0.7, drag=0.8),
    "stress": dict(tau=0.45, odom_hz=3.0, odom_latency=0.50, img_delay=0.35, inertia=1.8, drag=1.5),
}


@dataclass
class Scenario:
    range_m: float = 7.0              # distance of the vehicle centre in front of the dock plane
    lateral_m: float = 0.0            # + = vehicle east (right of the approach axis)
    heading_deg: float = 0.0          # vehicle heading relative to the approach axis (+ = pointing east of it)
    depth_m: float = 3.0
    plant: str = "nominal"
    vision: str = "detector"          # 'detector' (rendered frames + the real detector) or 'perfect' (projected light centres + 0.5 px noise)
    seed: int = 1
    freeze_interval_s: float = 0.0    # mean time between odometry freezes (0 = none)
    T: float = 180.0
    realistic_odometry: bool = True
    odom_faults: Optional[dict] = None   # OdometryConfig position faults: pos_offset_m, pos_drift_mps, pos_jump_m, pos_jump_t_s, drop_twist


@dataclass
class Result:
    scenario: Scenario
    passed: bool = False
    failures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    log: Optional[np.ndarray] = None
    log_cols: Tuple[str, ...] = ()


def load_cfg(path: Optional[Path] = None, overrides: Optional[dict] = None) -> dict:
    cfg = yaml.safe_load(open(path or CONFIG))
    for sect, d in (overrides or {}).items():
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(cfg[sect].get(k), dict):
                cfg[sect][k].update(v)
            else:
                cfg[sect][k] = v
    return cfg


def funnel_clearance(p_world: np.ndarray, r_body: float = BODY_R) -> float:
    """Gap [m] between a hull point at p_world and the funnel wall (negative = touching). Points not inside the funnel's x range have infinite clearance."""
    depth = p_world[0] - DOCK_POS[0]
    if depth < 0.0 or depth > 2.5:
        return math.inf
    rho = math.hypot(p_world[1] - DOCK_POS[1], p_world[2] - DOCK_POS[2])
    if rho > DOCK_OUTER_R:                                                  # beside the dock, not in it (a vehicle passing 8 m to the side is not touching anything)
        return math.inf
    return float(np.interp(depth, FUNNEL_DEPTH, FUNNEL_R_IN) - rho - r_body)


def hull_points(pos: np.ndarray, eul: np.ndarray, n: int = 15) -> np.ndarray:
    xs = np.linspace(TAIL_X, NOSE_X, n)
    R = rot_zyx(*eul)
    return pos + (R @ np.column_stack([xs, np.zeros(n), np.zeros(n)]).T).T


_CAM: Optional[SyntheticCamera] = None
_DET = None


def _camera() -> SyntheticCamera:
    global _CAM
    if _CAM is None:
        _CAM = SyntheticCamera()
    return _CAM


def _detector(overrides: Optional[dict] = None):
    from dock_detector import DockDetector
    return DockDetector(overrides=overrides)


class DockingSim:
    def __init__(self, scn: Scenario, cfg: Optional[dict] = None, detector_overrides: Optional[dict] = None, plant: Optional[dict] = None) -> None:
        self.scn = scn
        self.cfg = cfg or load_cfg()
        P = plant or PLANTS[scn.plant]
        self.P = P
        geom = copy.deepcopy(load_geometry())
        geom["vehicle"]["gyration_m"] = [geom["vehicle"]["gyration_m"][0], *[g * math.sqrt(P["inertia"]) for g in geom["vehicle"]["gyration_m"][1:]]]
        for k in geom["vehicle"]["drag_quad"]:
            geom["vehicle"]["drag_quad"][k] *= P["drag"]
        s = scn
        self.model = VehicleModel(geom=geom, pos=(DOCK_POS[0] - s.range_m, s.lateral_m, s.depth_m), eul_deg=(0.0, 0.0, s.heading_deg), thruster_tau_s=P["tau"])
        lim = self.cfg["limits"]
        self.alloc = Allocator(rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"], u_fin_min=lim["u_fin_min_mps"], u_fin_off=lim["u_fin_off_mps"], u_fin_full=lim["u_fin_full_mps"], small_force_n=lim.get("small_force_n", 0.0))
        if str(self.cfg["estimator"].get("position_source", "odometry")) == "dead_reckoning" and (self.cfg.get("recover") or {}).get("dock_hint_ned") is not None:
            # dead reckoning has no absolute frame: its origin is the START pose, so the rough dock position the mission plan gives is expressed relative to the start
            self.cfg = copy.deepcopy(self.cfg)
            h = list(self.cfg["recover"]["dock_hint_ned"])
            self.cfg["recover"]["dock_hint_ned"] = [float(h[0]) - float(self.model.pos[0]), float(h[1]) - float(self.model.pos[1])] + h[2:]
        self.core = TerminalDockingCore(self.cfg)
        self.core.pose.set_latency(P["odom_latency"]) if False else None      # the controller is configured with the NOMINAL latency; plants differ on purpose (robustness)
        self.core.image_delay = float(P["img_delay"])
        if s.realistic_odometry:
            self.sensor = OdometrySensor(OdometryConfig(rate_hz=P["odom_hz"], latency_s=P["odom_latency"], freeze_mean_interval_s=s.freeze_interval_s, seed=s.seed, **(s.odom_faults or {})))
        else:
            self.sensor = None
        self.cam = _camera()
        self.det = _detector(detector_overrides) if s.vision == "detector" else None
        self.rng = np.random.default_rng(s.seed + 100)
        self.hist: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=400)       # (t, pos, eul) for image latency
        self.t = 0.0
        self.last_odom: Optional[State] = None
        self.last_det_valid_t = 0.0
        self.frame_cb = None            # optional callback(t_capture, bgr, detector_result, roll_rad) for every rendered frame (the showcase keeps a few)

    # ------------------------------------------------------------------ vision
    def _pose_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        for tt, p, e in reversed(self.hist):
            if tt <= t:
                return p, e
        return self.hist[0][1], self.hist[0][2]

    def _observe(self, t_capture: float) -> DockObs:
        pos, eul = self._pose_at(t_capture)
        cam = self.cam
        if self.scn.vision == "perfect":
            pts = {}
            eul_l = eul.copy()
            eul_l[0] = 0.0                                                # the detector removes roll
            for name in ("top", "bottom", "right", "left"):
                L = next(l for l in cam.lights if l["name"] == name)
                uvz = cam.project(pos, np.degrees(eul_l), L["p"])
                if uvz is None or not (5 < uvz[0] < cam.W - 5 and 5 < uvz[1] < cam.H - 5):
                    return DockObs(fresh=True, valid=False, num_lights=0)
                pts[name] = [uvz[0] + self.rng.normal(0, 0.5), uvz[1] + self.rng.normal(0, 0.5)]
            side = sorted([pts["right"], pts["left"]], key=lambda p: p[0])      # the detector labels the two side lights by IMAGE position
            pix = np.array([pts["top"], pts["bottom"], side[1], side[0]])
            rad = 0.5 * float(np.linalg.norm(pix[0] - pix[1]))
            return DockObs(fresh=True, valid=True, num_lights=4, radius_px=rad, pix=pix)
        import cv2
        bgr = cam.render(pos, np.degrees(eul))
        bgr = cv2.imdecode(np.frombuffer(cam.encode_jpeg(bgr), np.uint8), cv2.IMREAD_COLOR)        # the real path goes through JPEG
        roll, pitch = (self.last_odom.roll, self.last_odom.pitch) if self.last_odom is not None else (0.0, 0.0)   # the IMU comes from odometry in the offline stack
        res = self.det.process(bgr, roll, pitch, t_capture)
        if self.frame_cb is not None:
            self.frame_cb(t_capture, bgr, res, roll)
        return DockObs.from_msg(res.msg)

    # ------------------------------------------------------------------ the run
    def run(self, keep_log: bool = True) -> Result:
        s = self.scn
        m, core = self.model, self.core
        rate = float(self.cfg["node"]["rate_hz"])
        dt_ctl = 1.0 / rate
        n_sub = int(round(dt_ctl / 0.01))
        cam_period = 1.0 / 15.0
        next_cam = 0.0
        pending: Deque[Tuple[float, DockObs]] = deque()
        cmd: Dict[str, float] = {}
        res = Result(s)
        J = Judge(self.cfg, s)
        log: List[list] = []
        t_ctl = 0.0
        first = True
        while self.t < s.T:
            # ---- physics for one control period
            for _ in range(n_sub):
                m.step(0.01)
                self.t += 0.01
                self.hist.append((self.t, m.pos.copy(), m.eul.copy()))
                outs = self.sensor.update(self.t, m.pos, m.eul, m.nu) if self.sensor else [type("S", (), {"pos": m.pos, "eul": m.eul, "nu": m.nu, "frozen": False})()]
                for o in outs:
                    st = State(pos=np.array(o.pos), eul=np.array(o.eul), nu=np.array(o.nu))
                    self.last_odom = st
                    core.pose.push(st, self.t)
                J.step(self.t, m, cmd, dock_visible=False)
            # ---- camera: a frame is captured every 1/15 s and delivered img_delay later
            obs = DockObs(fresh=False)
            if self.t >= next_cam:
                next_cam += cam_period
                if self.last_odom is not None:
                    pending.append((self.t + self.P["img_delay"], self._observe(self.t)))       # captured now, delivered img_delay later
            while pending and self.t >= pending[0][0]:
                obs = pending.popleft()[1]                                                       # (several can be due in one control tick: the newest is used)
                J.note_detection(self.t, m, obs)
            # ---- controller
            st_now = core.pose.at(self.t)
            w, stat = core.update(obs, st_now, dt_ctl, self.t)
            u = st_now.speed_u if st_now is not None else 0.0
            cmd = self.alloc.allocate(w, u) if st_now is not None else {}
            m.set_command(cmd)
            J.note_control(self.t, cmd, stat, core.pose.frozen(self.t))
            if keep_log:
                log.append([self.t, *m.pos, *np.degrees(m.eul), m.nu[0], stat.get("s", np.nan), stat.get("e", np.nan), stat.get("chi_deg", np.nan), stat.get("dz", np.nan),
                            cmd.get("th_01", 0.0), cmd.get("th_02", 0.0), cmd.get("th_03", 0.0), cmd.get("cs_04", 0.0),
                            ["WAIT", "SEARCH", "APPROACH", "TERMINAL", "DOCKED", "RETRY", "SAFE_STOP", "ABORT"].index(str(stat.get("phase", "WAIT"))), float(obs.valid),
                            cmd.get("cs_06", 0.0), cmd.get("cs_07", 0.0), cmd.get("cs_08", 0.0)])
            if J.done(self.t, core.phase):
                break
        J.finish(self.t, core)
        res.passed, res.failures, res.notes, res.metrics = J.verdict()
        if keep_log:
            res.log = np.array(log)
            res.log_cols = ("t", "x", "y", "z", "roll", "pitch", "yaw", "u", "s_est", "e_est", "chi_est", "dz_est", "th_01", "th_02", "th_03", "cs_04", "phase", "valid", "cs_06", "cs_07", "cs_08")
        return res


# ======================================================================================================================
# judging: ground truth only (the controller's own estimate is never used to decide pass/fail)
# ======================================================================================================================
class Judge:
    def __init__(self, cfg: dict, scn: Scenario) -> None:
        self.cfg, self.scn = cfg, scn
        self.fail: Dict[str, str] = {}
        self.notes: List[str] = []
        self.min_clear = math.inf
        self.crossed = False
        self.cross: Dict[str, float] = {}
        self.nose_s_min = math.inf
        self.t_dock_still = None
        self.pinned_since: Dict[str, Optional[float]] = {}
        self.rpm_hist: Deque[Tuple[float, float, float]] = deque(maxlen=400)
        self.invalid_since: Optional[float] = None
        self.e_hist: List[Tuple[float, float, float, float]] = []
        self.retries = 0
        self.stop_t = math.inf
        self.last_phase = "WAIT"
        self.frozen_since: Optional[float] = None
        self.max_tilt = 0.0
        self.min_depth, self.max_depth = math.inf, -math.inf
        self.final: Dict[str, float] = {}
        self.chatter_max = 0.0
        self.max_speed_entry = 0.0
        self.t_end = 0.0
        self.max_e_after_gate = 0.0
        self.pushes_while_frozen = 0.0
        self.osc: List[Tuple[float, float, float, float]] = []      # (s_nose, lateral, vertical, heading_deg) while 0 < s_nose < 3 (before the mouth)
        self.final_nose: Optional[np.ndarray] = None
        self.final_head = 0.0
        self.max_depth_in = 0.0

    def _nose(self, m) -> np.ndarray:
        return m.pos + rot_zyx(*m.eul) @ np.array([NOSE_X, 0.0, 0.0])

    def step(self, t: float, m, cmd: dict, dock_visible: bool) -> None:
        nose = self._nose(m)
        s_nose = DOCK_POS[0] - nose[0]
        self.nose_s_min = min(self.nose_s_min, s_nose)
        self.max_depth_in = max(self.max_depth_in, -s_nose)
        if 0.0 < s_nose < 3.0:
            self.osc.append((s_nose, nose[1] - DOCK_POS[1], nose[2] - DOCK_POS[2], math.degrees(m.eul[2])))
        self.final_nose, self.final_head, self.final_u = nose, math.degrees(m.eul[2]), float(m.nu[0])
        # wall contact: any hull point inside the funnel (or on its rim face) closer than the hull radius
        clear = min(funnel_clearance(p) for p in hull_points(m.pos, m.eul))
        self.min_clear = min(self.min_clear, clear)
        if clear < 0.0 and "wall contact" not in self.fail:
            self.fail["wall contact"] = f"hull touches the funnel at t={t:.1f} s (nose depth {-s_nose:.2f} m)"
        # rim face: crossing the mouth plane with the hull outside the opening
        lat_n, vert_n = nose[1] - DOCK_POS[1], nose[2] - DOCK_POS[2]
        if not self.crossed and s_nose <= 0.0 and math.hypot(lat_n, vert_n) <= DOCK_OUTER_R:        # an entry only counts at the dock, not 8 m to its side
            self.crossed = True
            lat = nose[1] - DOCK_POS[1]
            vert = nose[2] - DOCK_POS[2]
            head = math.degrees(m.eul[2])
            self.cross = {"t": t, "lat": lat, "vert": vert, "heading_deg": head, "speed": float(m.nu[0])}
            if abs(lat) > 0.15 or abs(vert) > 0.15:
                self.fail.setdefault("bad entry", f"lateral {lat:+.2f} m / vertical {vert:+.2f} m at the mouth (limit 0.15)")
            if abs(head) > 5.0:
                self.fail.setdefault("bad entry", f"heading {head:+.1f} deg at the mouth (limit 5)")
            if m.nu[0] > 0.5:
                self.fail.setdefault("bad entry", f"entry speed {m.nu[0]:.2f} m/s (limit 0.5)")
        # safety limits
        tilt = math.degrees(max(abs(m.eul[0]), abs(m.eul[1])))
        self.max_tilt = max(self.max_tilt, tilt)
        self.min_depth, self.max_depth = min(self.min_depth, m.pos[2]), max(self.max_depth, m.pos[2])
        if tilt > 30.0:
            self.fail.setdefault("attitude limit", f"tilt {tilt:.0f} deg at t={t:.1f} s")
        if m.pos[2] < 0.5 or m.pos[2] > 8.0:
            self.fail.setdefault("depth limit", f"depth {m.pos[2]:.2f} m at t={t:.1f} s")
        self.t_end = t

    def note_detection(self, t: float, m, obs: DockObs) -> None:
        s_c = DOCK_POS[0] - m.pos[0]
        if obs.valid:
            self.seen_once = True
        if getattr(self, "seen_once", False) and 2.2 <= s_c <= 4.0 and not obs.valid:     # 'lost' only applies to a dock that has been seen (a search start is not 'lost')
            self.invalid_since = self.invalid_since if self.invalid_since is not None else t
            if t - self.invalid_since > 10.0:
                self.fail.setdefault("lost dock", f"no valid detection for >10 s at 2.2..4 m (t={t:.1f})")
        else:
            self.invalid_since = None

    def note_control(self, t: float, cmd: dict, stat: dict, frozen: bool) -> None:
        phase = str(stat.get("phase", "WAIT"))
        self.retries = int(stat.get("retries", 0))
        self.last_phase = phase
        cap = float(self.cfg["limits"]["rpm_cap"])
        for k, v in cmd.items():
            if k.startswith("th_"):
                pinned = abs(v) >= 0.98 * cap
                st = self.pinned_since.get(k)
                if pinned and st is None:
                    self.pinned_since[k] = t
                elif not pinned:
                    self.pinned_since[k] = None
                elif st is not None and t - st > 2.0:
                    self.fail.setdefault("actuator pinned", f"{k} at its cap for >2 s (t={t:.1f})")
        # chatter near the dock: peak-to-peak of the heave commands inside any 1 s window while within 3 m
        s_est = stat.get("s")
        self.rpm_hist.append((t, cmd.get("th_02", 0.0), cmd.get("th_03", 0.0)))
        if isinstance(s_est, float) and s_est < 3.0 and phase != "DOCKED":
            win = [r for r in self.rpm_hist if r[0] > t - 1.0]
            if len(win) > 10:
                ptp = max(np.ptp([r[1] for r in win]), np.ptp([r[2] for r in win]))
                self.chatter_max = max(self.chatter_max, float(ptp))
                if ptp > 300.0 and phase not in ("RETRY", "SAFE_STOP"):
                    self.fail.setdefault("actuator chatter", f"heave commands swing {ptp:.0f} RPM within 1 s at t={t:.1f}")
        # stale odometry: pushing forward after >1 s of frozen data is acting on stale information
        if frozen:
            self.frozen_since = self.frozen_since if self.frozen_since is not None else t
            if t - self.frozen_since > 1.0 and cmd.get("th_01", 0.0) > 50.0:
                self.fail.setdefault("stale data", f"forward thrust {cmd['th_01']:.0f} RPM on stale odometry (t={t:.1f})")
        else:
            self.frozen_since = None

    def track_true(self, t, m) -> None:        # called from finish for the oscillation check on the log is not needed here
        pass

    def done(self, t: float, phase: str) -> bool:
        if phase == "DOCKED":
            if self.t_dock_still is None:
                self.t_dock_still = t
            return t - self.t_dock_still > 4.0
        self.t_dock_still = None
        return len(self.fail) >= 1 and t > 5.0 and any(k in self.fail for k in ("wall contact", "attitude limit", "depth limit"))

    @staticmethod
    def _crossings(x: np.ndarray, hyst: float) -> int:
        """Zero crossings with hysteresis (a swing from beyond +hyst to beyond -hyst counts once)."""
        side, n = 0, 0
        for v in x:
            if v > hyst and side <= 0:
                n += side < 0
                side = 1
            elif v < -hyst and side >= 0:
                n += side > 0
                side = -1
        return n

    def finish(self, t: float, core: TerminalDockingCore) -> None:
        self.t_end = t
        self.final = {"phase": core.phase}
        self.final_core_phase = core.phase
        self.retries = core.retries
        self.repositions = core.repositions
        if self.retries > int(self.cfg["guidance"]["max_retries"]):
            self.fail.setdefault("too many retries", f"{self.retries} back-offs")
        if core.phase != "DOCKED":
            if t >= self.scn.T - 0.5:
                self.fail.setdefault("timeout", f"not docked after {self.scn.T:.0f} s (phase {core.phase})")
            if not self.crossed:
                self.fail.setdefault("missed", "the nose never reached the dock mouth")
        if self.crossed:
            depth = self.max_depth_in
            fin_depth = -(DOCK_POS[0] - self.final_nose[0]) if self.final_nose is not None else 0.0
            if depth > 1.2:
                self.fail.setdefault("overshoot", f"nose went {depth:.2f} m into the funnel (limit 1.2)")
            if core.phase == "DOCKED":
                if fin_depth < 0.15:
                    self.fail.setdefault("stopped short", f"final nose depth {fin_depth:.2f} m (need >= 0.15)")
                if abs(self.final_head) > 10.0:                      # (the 5 deg limit is checked where the user defined it: at the mouth; here only gross crookedness)
                    self.fail.setdefault("bad final heading", f"{self.final_head:+.1f} deg")
        if len(self.osc) > 20:
            o = np.array(self.osc)
            for col, name, hyst in ((1, "lateral", 0.03), (2, "vertical", 0.03), (3, "heading", 2.0)):
                n = self._crossings(o[:, col], hyst)
                if n >= 6:
                    self.fail.setdefault("oscillation", f"{name} error crossed zero {n} times in the last 3 m")

    def verdict(self) -> Tuple[bool, List[str], List[str], Dict[str, float]]:
        f = [f"{k}: {v}" for k, v in self.fail.items()]
        met: Dict[str, float] = {"t_end": self.t_end, "retries": float(self.retries), "min_clearance_m": self.min_clear if math.isfinite(self.min_clear) else float("nan"),
                                 "max_tilt_deg": self.max_tilt, "chatter_rpm": self.chatter_max, "min_depth": self.min_depth, "max_depth": self.max_depth,
                                 "nose_s_min": self.nose_s_min}
        met.update({f"cross_{k}": v for k, v in self.cross.items()})
        met["repositions"] = float(getattr(self, "repositions", 0))
        met["final_depth_in_m"] = self.max_depth_in
        met["final_heading_deg"] = self.final_head
        return (len(f) == 0 and self.final_core_phase == "DOCKED" and self.retries == 0 and getattr(self, "repositions", 0) == 0), f, self.notes, met
