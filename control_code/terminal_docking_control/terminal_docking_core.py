"""Dock alignment and terminal docking: pure logic, no ROS. Inputs are what the real AUV has: the DockAlign light pixels and its own odometry.

Idea in four lines
  1. DockEstimator: an extended Kalman filter for the dock's pose IN THE WORLD (ring-centre position and the heading of its axis), fed by the four light
     pixels of every valid DockAlign message and by the vehicle's own pose. The dock plane is known to be vertical, so only 4 numbers are unknown
     (the 8 pixel coordinates over-determine them). The dock is static, so the estimate is simply REMEMBERED when the lights leave the image (they do,
     inside ~1.7 m: the 1 m ring no longer fits the 60 deg vertical field of view) and the vehicle keeps docking on its own odometry.
  2. Guidance (line of sight to the dock axis): the vehicle has no sway thruster, so it must steer onto the axis line by yawing while moving
     (the fins need flow). desired heading = axis heading - atan(cross_track / lookahead), limited to +-max_deviation.
  3. Speed profile: cruise, slow to the gate speed, a gate check (lateral, heading, vertical inside tolerance) and then a constant slow terminal
     speed until the stopping envelope brings the nose to rest inside the funnel. A failed gate check backs off and tries again (counted as a retry).
  4. Safety: stale or frozen odometry -> brake open loop and wait; lights lost -> keep going on the remembered dock only while the estimate is trusted.

Frames: NED world; body x fwd, y stbd, z down; heading psi from north towards east. The "approach frame" of the dock: x_a points INTO the funnel
(this is the heading the vehicle must have when it docks), y_a to the right when looking along x_a, z_a down. Cross-track e > 0 means the vehicle is
to the right of the axis line; distance s > 0 means the vehicle is in front of the dock plane.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from loops import HoldLoops  # noqa: E402
from pose_filter import PoseFilter, PoseFilterConfig, PoseTracker  # noqa: E402   (PoseTracker lives in common/pose_filter.py; re-exported here)
from state import State, wrap_pi  # noqa: E402

LIGHT_NAMES = ("top", "bottom", "right", "left")        # IMAGE-based labels used by the detector


def rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body -> world rotation for ZYX Euler angles in radians."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def clamp(v: float, lim: float) -> float:
    return max(-abs(lim), min(abs(lim), v))


def room_needed_m(e_m: float, chi_deg: float, turn_radius_m: float, max_s_curve_deg: float = 40.0) -> float:
    """Distance along the dock axis that a vehicle with minimum turning radius R needs, from lateral offset e [m, + = right of the axis] and heading error chi
    [deg, + = pointing right of the axis], to get onto the axis line pointing along it (a Dubins CSC bound: an S-curve of two arcs and a straight piece; turning
    AWAY from the axis first costs one more arc). The same screening rule the evaluation used (terminal_docking_eval.feasible), now part of the controller."""
    R = float(turn_radius_m)
    e = abs(float(e_m))
    need = 0.0
    if e > 1e-6:
        theta = math.radians(max_s_curve_deg)
        arc_e = 2.0 * R * (1.0 - math.cos(theta))
        straight = max(e - arc_e, 0.0) / math.sin(theta)
        need = 2.0 * R * math.sin(theta) + straight * math.cos(theta)
        if e < arc_e:
            theta = math.acos(max(-1.0, min(1.0, 1.0 - e / (2.0 * R))))
            need = 2.0 * R * math.sin(theta)
    if chi_deg * e_m > 0.0:                                       # pointing away from the axis: one more arc to swing round
        need += R * abs(math.radians(chi_deg))
    return need


# ======================================================================================================================
# observation
# ======================================================================================================================
@dataclass
class DockObs:
    """The DockAlign fields this controller uses. `fresh` is False when no new message arrived since the last tick."""
    fresh: bool = False
    valid: bool = False
    num_lights: int = 0
    radius_px: float = 0.0
    pix: Optional[np.ndarray] = None            # (4, 2) pixel (u, v) of top, bottom, right, left (roll-levelled image)
    search_yaw: float = 0.0                     # the detector's search hints while fewer than four lights are trusted (DockAlign.search_*_norm)
    search_surge: float = 0.0

    @staticmethod
    def from_msg(m, fresh: bool = True) -> "DockObs":
        pix = np.array([[m.top.x, m.top.y], [m.bottom.x, m.bottom.y], [m.right.x, m.right.y], [m.left.x, m.left.y]], float)
        return DockObs(fresh=fresh, valid=bool(m.valid), num_lights=int(m.num_lights), radius_px=float(m.radius_px), pix=pix,
                       search_yaw=float(getattr(m, "search_yaw_norm", 0.0)), search_surge=float(getattr(m, "search_surge_norm", 0.0)))


# ======================================================================================================================
# dock estimator
# ======================================================================================================================
class DockEstimator:
    """EKF over x = [px, py, pz, psi_a]: ring-centre position (NED) and the heading of the approach axis."""

    def __init__(self, cfg: dict) -> None:
        cam, dock, est = cfg["camera"], cfg["dock"], cfg["estimator"]
        self.f = float(cam["f_px"])
        self.cx, self.cy = 0.5 * float(cam["width"]), 0.5 * float(cam["height"])
        self.t_cam = np.array(cam["mount_m"], float)
        self.ring_r = float(dock["ring_radius_m"])
        L = dock["lights_ya_za"]                                   # name -> [y_a, z_a] in the approach frame
        self.obj = np.array([[0.0, *map(float, L[n])] for n in LIGHT_NAMES])      # (4, 3): x_a = 0 (the ring plane)
        self.sigma_px = float(est["pixel_sigma_px"])
        self.gate = float(est["gate_chi2"])
        self.q_pos = float(est["process_pos_m_per_sqrt_s"])
        self.q_yaw = math.radians(float(est["process_yaw_deg_per_sqrt_s"]))
        self.min_radius = float(est["min_radius_px"])
        self.reinit_after = int(est.get("reinit_after_rejects", 12))
        self.prior_yaw_sigma = math.radians(float(est.get("init_yaw_sigma_deg", 30.0)))
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.updates = 0
        self.rejects = 0
        self.last_t = 0.0
        self.last_nis = 0.0

    # ---- model
    def project(self, x: np.ndarray, pos: np.ndarray, eul: np.ndarray) -> np.ndarray:
        """Pixel (u, v) of the 4 lights for dock state x seen from a vehicle at pos with attitude eul (roll removed like the detector does)."""
        c, s = math.cos(x[3]), math.sin(x[3])
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        P = x[:3] + self.obj @ Rz.T
        Rwb = rot_zyx(0.0, eul[1], eul[2])
        pb = (P - pos) @ Rwb - self.t_cam                          # == Rwb.T @ (P - pos) - t_cam, row-wise
        z = np.maximum(pb[:, 0], 1e-3)
        return np.column_stack([self.cx + self.f * pb[:, 1] / z, self.cy + self.f * pb[:, 2] / z]).ravel()

    def _jac(self, x, pos, eul) -> np.ndarray:
        h0 = self.project(x, pos, eul)
        J = np.zeros((8, 4))
        for i, eps in enumerate((1e-4, 1e-4, 1e-4, 1e-5)):
            xp = x.copy(); xp[i] += eps
            J[:, i] = (self.project(xp, pos, eul) - h0) / eps
        return J

    # ---- start-up: batch Gauss-Newton on the first valid frame
    def _init(self, z: np.ndarray, pos: np.ndarray, eul: np.ndarray, radius_px: float) -> None:
        pix = z.reshape(4, 2)
        centre = pix.mean(axis=0)
        d0 = max(self.f * self.ring_r / max(radius_px, self.min_radius), 0.5)
        ray = np.array([1.0, (centre[0] - self.cx) / self.f, (centre[1] - self.cy) / self.f])
        Rwb = rot_zyx(0.0, eul[1], eul[2])
        p0 = pos + Rwb @ (self.t_cam + d0 * ray)
        x = np.array([p0[0], p0[1], p0[2], eul[2]])                # guess: the dock axis equals the vehicle's heading
        W = 1.0 / self.sigma_px ** 2
        prior = np.zeros((4, 4))
        prior[3, 3] = 1.0 / self.prior_yaw_sigma ** 2
        x_prior = x.copy()
        for _ in range(12):
            r = z - self.project(x, pos, eul)
            J = self._jac(x, pos, eul)
            A = W * J.T @ J + prior + 1e-6 * np.eye(4)
            b = W * J.T @ r - prior @ (x - x_prior)
            dx = np.linalg.solve(A, b)
            x = x + np.clip(dx, [-1, -1, -1, -0.3], [1, 1, 1, 0.3])
            if np.linalg.norm(dx[:3]) < 1e-4 and abs(dx[3]) < 1e-5:
                break
        J = self._jac(x, pos, eul)
        self.x = x
        self.P = np.linalg.inv(W * J.T @ J + prior + 1e-6 * np.eye(4))
        self.updates = 1
        self.rejects = 0

    def update(self, obs: DockObs, pos: np.ndarray, eul: np.ndarray, t: float) -> str:
        """One EKF step with the pose AT IMAGE TIME. Returns 'init', 'ok', 'gated', 'skipped'."""
        if self.x is not None:                                      # static dock: only a little process noise so the filter never gets overconfident
            dt = max(t - self.last_t, 0.0)
            self.P = self.P + np.diag([self.q_pos ** 2 * dt] * 3 + [self.q_yaw ** 2 * dt])
        self.last_t = t
        if not (obs.fresh and obs.valid and obs.pix is not None and obs.radius_px >= self.min_radius):
            return "skipped"
        z = obs.pix.ravel()
        if self.x is None:
            self._init(z, pos, eul, obs.radius_px)
            return "init"
        h = self.project(self.x, pos, eul)
        J = self._jac(self.x, pos, eul)
        R = np.eye(8) * self.sigma_px ** 2
        S = J @ self.P @ J.T + R
        innov = z - h
        nis = float(innov @ np.linalg.solve(S, innov))
        self.last_nis = nis
        if nis > self.gate:
            self.rejects += 1
            if self.rejects >= self.reinit_after and self.updates < 5:         # a young estimate that keeps disagreeing was probably wrong: start again
                self._init(z, pos, eul, obs.radius_px)
                return "init"
            return "gated"
        K = self.P @ J.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        I = np.eye(4)
        self.P = (I - K @ J) @ self.P @ (I - K @ J).T + K @ R @ K.T            # Joseph form
        self.updates += 1
        self.rejects = 0
        return "ok"

    # ---- read-outs
    @property
    def ready(self) -> bool:
        return self.x is not None

    def sigmas(self) -> Tuple[float, float]:
        """(position 1-sigma [m], axis-heading 1-sigma [rad])."""
        if self.P is None:
            return math.inf, math.inf
        return float(math.sqrt(np.trace(self.P[:3, :3]) / 3.0)), float(math.sqrt(self.P[3, 3]))


# ======================================================================================================================
# pose bookkeeping: odometry arrives slowly and late; the image pose and the control pose are extrapolated from it
# ======================================================================================================================
# ======================================================================================================================
# the controller
# ======================================================================================================================
PHASES = ("WAIT", "SEARCH", "APPROACH", "TERMINAL", "DOCKED", "RETRY", "SAFE_STOP", "ABORT")


class TerminalDockingCore:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.est = DockEstimator(cfg)
        g, sp, lim, sf = cfg["guidance"], cfg["speed"], cfg["limits"], cfg["safety"]
        self.g, self.sp, self.lim, self.sf = g, sp, lim, sf
        self.loops = HoldLoops(cfg["gains"], heave_ff_n=float(cfg["feedforward"]["heave_n"]))
        self.pose = PoseTracker(float(cfg["estimator"]["odom_latency_s"]), float(sf["stale_odom_s"]), PoseFilterConfig(**cfg.get("pose_filter", {})))
        self.image_delay = float(cfg["estimator"]["image_delay_s"])
        self.phase = "WAIT"
        self.retries = 0
        self.t = 0.0
        self.phase_t = 0.0
        self.last_valid_t = -1e9
        self.gate_ok_since: Optional[float] = None
        self.u_last_good = 0.0
        self._cmd_active = False             # the previous tick commanded real thrust or moments (frozen odometry only matters then)
        self.u_dr = 0.0                      # dead-reckoned forward speed while the odometry is frozen (open-loop braking)
        self.debug_info: Dict[str, float] = {}
        self.nose_x = float(cfg["vehicle"]["nose_m"])
        self.rc = dict(cfg.get("recover", {}))                       # search + reposition behaviour (recover.enabled false = the old behaviour)
        self.rc_on = bool(self.rc.get("enabled", False))
        self.repositions = 0
        # position source: 'odometry' (the pose filter's x, y) or 'dead_reckoning' (x, y integrated from the filtered body speed and attitude, origin = the start pose;
        # no absolute position is used at all, depth still comes from the depth sensor). The dock estimate lives in whichever frame is chosen.
        self.pos_src = str(cfg["estimator"].get("position_source", "odometry"))
        self.dr_on = self.pos_src == "dead_reckoning"
        self._dr_xy = np.zeros(2)
        self._dr_t: Optional[float] = None
        self._dr_hist: deque = deque(maxlen=600)
        self._back_to = float(cfg["guidance"]["retry_back_to_s_m"])
        self._reposition_active = False
        self._t0: Optional[float] = None
        self._search_dir = 1.0

    # ---- dead reckoning (estimator.position_source: dead_reckoning)
    def _dr_view(self, st: State, t: float) -> State:
        """The vehicle state with x, y replaced by the dead-reckoned position (integral of the filtered NED velocity since the start). Depth is kept."""
        if self._dr_t is not None:
            dt = min(max(t - self._dr_t, 0.0), 0.5)
            self._dr_xy = self._dr_xy + st.vel_ned()[:2] * dt
        self._dr_t = t
        self._dr_hist.append((t, self._dr_xy.copy()))
        return State(pos=np.array([self._dr_xy[0], self._dr_xy[1], st.pos[2]]), eul=st.eul.copy(), nu=st.nu.copy())

    def _dr_xy_at(self, t_q: float) -> np.ndarray:
        h = self._dr_hist
        if not h:
            return self._dr_xy.copy()
        if t_q <= h[0][0]:
            return h[0][1].copy()
        for (t0, p0), (t1, p1) in zip(list(h)[-2::-1], list(h)[::-1]):
            if t0 <= t_q <= t1:
                a = 0.0 if t1 <= t0 else (t_q - t0) / (t1 - t0)
                return p0 + a * (p1 - p0)
        return h[-1][1].copy()

    # ---- helpers
    def _set_phase(self, p: str) -> None:
        if p != self.phase:
            self.phase, self.phase_t = p, 0.0

    def geometry(self, st: State) -> Dict[str, float]:
        """Vehicle relative to the estimated dock: s (distance in front of the plane), e (cross-track, + right), chi (heading error), dz (+ below)."""
        x = self.est.x
        psi_a = x[3]
        fa = np.array([math.cos(psi_a), math.sin(psi_a)])
        ra = np.array([-math.sin(psi_a), math.cos(psi_a)])
        d = st.pos[:2] - x[:2]
        return {"s": float(-(d @ fa)), "e": float(d @ ra), "chi": wrap_pi(st.yaw - psi_a), "dz": float(st.pos[2] - x[2]), "psi_a": float(psi_a)}

    def speed_force(self, u: float, u_target: float, brake_ff: bool = False) -> float:
        sp = self.sp
        f = float(sp["kp_n_per_mps"]) * (u_target - u)
        if brake_ff and u > 0.05:
            # force that produces the planned deceleration: the constant decel on the sqrt envelope, or k*u on the linear stopping law (it fades with the speed:
            # a constant feed-forward right down to zero speed drives a light vehicle through zero into reverse)
            f -= float(sp["mass_kg"]) * min(float(sp["decel_mps2"]), float(sp.get("stop_gain_per_s", 0.8)) * u)
        return max(-float(sp["max_brake_n"]), min(float(sp["max_forward_n"]), f))

    def stop_speed(self, s_nose: float) -> float:
        """Largest speed from which the planned deceleration still stops with the nose at stop_nose_s_m."""
        room = s_nose - float(self.g["stop_nose_s_m"])
        if room < 0.0:                                   # past the stop point (inside the funnel): creep back to it
            return max(-float(self.sp.get("back_mps", 0.25)), float(self.sp.get("stop_gain_per_s", 0.8)) * room)
        # the square-root envelope is the fastest speed that can still be stopped; the linear law (speed = kpos * distance) ends it without the brake
        # command reversing the vehicle (a laggy thruster keeps braking after the speed reaches zero and drives the vehicle backwards)
        return min(math.sqrt(2.0 * float(self.sp["decel_mps2"]) * room), float(self.sp.get("stop_gain_per_s", 0.8)) * room)

    # ---- the tick
    def update(self, obs: DockObs, st_now: Optional[State], dt: float, t: float) -> Tuple[np.ndarray, Dict[str, object]]:
        """-> (wrench [X,Y,Z,K,M,N], status). st_now = vehicle state at THIS moment (already extrapolated); None if no odometry yet."""
        w, stat = self._update(obs, st_now, dt, t)
        self._cmd_active = bool(abs(w[0]) > 1.0 or abs(w[2]) > 1.0 or abs(w[4]) > 0.2 or abs(w[5]) > 0.05)
        return w, stat

    def _update(self, obs: DockObs, st_now: Optional[State], dt: float, t: float) -> Tuple[np.ndarray, Dict[str, object]]:
        self.t = t
        self.phase_t += dt
        w = np.zeros(6)
        stat: Dict[str, object] = {"phase": self.phase, "retries": self.retries, "pos_src": self.pos_src}
        if st_now is None:
            return w, stat
        if self.dr_on:
            st_now = self._dr_view(st_now, t)
        u = st_now.speed_u
        frozen = self.pose.frozen(t, expect_motion=self._cmd_active)
        # ---- vision -> estimate (pose at image time = now minus the image delay; extrapolation uses the odometry velocity)
        if obs.fresh:
            st_img = self.pose.at(t - self.image_delay) or st_now
            if self.dr_on:
                xy = self._dr_xy_at(t - self.image_delay)
                st_img = State(pos=np.array([xy[0], xy[1], st_img.pos[2]]), eul=st_img.eul.copy(), nu=st_img.nu.copy())
            r = self.est.update(obs, st_img.pos, st_img.eul, t)
            if r in ("init", "ok") and obs.valid and obs.radius_px > 1.0:
                cam = st_img.pos + rot_zyx(0.0, st_img.eul[1], st_img.eul[2]) @ self.est.t_cam
                stat["range_check_m"] = float(self.est.f * self.est.ring_r / obs.radius_px - np.linalg.norm(self.est.x[:3] - cam))     # picture range minus the EKF range (position-free cross-check)
            if r in ("init", "ok"):
                self.last_valid_t = t
            stat["est"] = r
        else:
            self.est.update(DockObs(fresh=False), st_now.pos, st_now.eul, t)
        if self._t0 is None:
            self._t0 = t
        if not self.est.ready:
            if self.rc_on and t - self._t0 >= float(self.rc.get("search_delay_s", 3.0)):
                return self._search(obs, st_now, dt, stat)
            stat["phase"] = "WAIT"
            return w, stat                                           # never seen the dock: do nothing (without recover.enabled the detector's search hints are not used)
        if self.phase in ("WAIT", "SEARCH"):
            self._set_phase("APPROACH")

        geo = self.geometry(st_now)
        s_nose = geo["s"] - self.nose_x * math.cos(geo["chi"])
        sig_p, sig_a = self.est.sigmas()
        stat.update(s=geo["s"], e=geo["e"], chi_deg=math.degrees(geo["chi"]), dz=geo["dz"], s_nose=s_nose, sig_pos=sig_p, sig_axis_deg=math.degrees(sig_a))

        # ---- safety: no usable odometry -> brake open loop and wait. The vehicle has almost no drag, so it would coast for tens of metres through the dock:
        # the speed is dead-reckoned with a simple model (thrust, mass, quadratic drag) and the axial thruster brakes until it is estimated to be at rest.
        if frozen:
            if self.phase != "DOCKED":
                self._set_phase("SAFE_STOP")
                if self.phase_t < dt * 1.5:
                    self.u_dr = self.u_last_good
                x = self.speed_force(self.u_dr, 0.0) if self.u_dr > float(self.sp.get("dr_rest_mps", 0.04)) else 0.0
                m_eff, c_drag = float(self.sp["mass_kg"]), float(self.sp.get("drag_n_per_mps2", 4.5))
                self.u_dr = max(0.0, self.u_dr + dt * (x - c_drag * self.u_dr * abs(self.u_dr)) / m_eff)
                w[0] = x
                stat.update(phase=self.phase, u_target=0.0, u_dr=self.u_dr)
                return w, stat
            return self._docked(st_now, geo, stat)
        self.u_last_good = u
        if self.phase == "SAFE_STOP":
            self._set_phase("TERMINAL" if self.gate_ok_since is not None else "APPROACH")

        if self.phase == "DOCKED":
            return self._docked(st_now, geo, stat)

        # ---- phase logic
        g = self.g
        if self.phase == "APPROACH" and self.rc_on and self.repositions < int(self.rc.get("max_repositions", 3)):
            # can the vehicle still get onto the axis, pointing along it, before the gate? If not (too close, too far off the axis, pointing the wrong way) it has
            # no sway thruster to slide sideways: back out along the axis, then approach again (a planned manoeuvre, not a failed gate)
            if (sig_a <= math.radians(float(self.rc.get("min_axis_sigma_deg", 8.0))) and sig_p <= float(self.rc.get("min_pos_sigma_m", 0.4))
                    and geo["s"] > float(g["gate_s_m"]) + float(self.rc.get("commit_below_m", 0.8))):
                de, dc = float(self.rc.get("dead_e_m", 0.2)), float(self.rc.get("dead_chi_deg", 4.0))
                e_eff = math.copysign(max(abs(geo["e"]) - de, 0.0), geo["e"])                         # the guidance closes small errors on its own
                chi_eff = math.copysign(max(abs(math.degrees(geo["chi"])) - dc, 0.0), geo["chi"])
                need = room_needed_m(e_eff, chi_eff, float(self.rc.get("turn_radius_m", 2.4)))
                avail = geo["s"] - float(g["gate_s_m"]) - float(self.rc.get("settle_m", 1.0))
                stat["room_need_m"], stat["room_avail_m"] = need, avail
                if need > 0.0 and avail < need * float(self.rc.get("margin", 1.0)):
                    self.repositions += 1
                    self._back_to = min(float(g["gate_s_m"]) + float(self.rc.get("settle_m", 1.0)) + need * float(self.rc.get("target_margin", 1.3)) + float(self.rc.get("extra_back_m", 1.0)),
                                        float(self.rc.get("max_back_to_m", 9.0)))
                    self._back_to = max(self._back_to, float(g["retry_back_to_s_m"]))
                    self._reposition_active = True
                    self._set_phase("RETRY")
        if self.phase == "APPROACH":
            if geo["s"] <= float(g["gate_s_m"]):
                ok = (abs(geo["e"]) <= float(g["gate_lateral_m"]) and abs(math.degrees(geo["chi"])) <= float(g["gate_heading_deg"])
                      and abs(geo["dz"]) <= float(g["gate_vertical_m"]) and sig_a <= math.radians(float(g["gate_axis_sigma_deg"])))
                stat["gate_ok"] = ok
                if ok:
                    self.gate_ok_since = t
                    self._set_phase("TERMINAL")
                elif geo["s"] <= float(g["gate_s_m"]) - float(g["gate_grace_m"]) and self.retries < int(g["max_retries"]):
                    self.retries += 1
                    self._back_to = float(g["retry_back_to_s_m"])
                    self._set_phase("RETRY")
                elif geo["s"] <= float(g["gate_s_m"]) - float(g["gate_grace_m"]):
                    self._set_phase("TERMINAL")                       # out of retries: go in anyway, the harness will judge it
                    self.gate_ok_since = t
        if self.phase == "RETRY":
            if geo["s"] >= self._back_to:
                self._reposition_active = False
                self._set_phase("APPROACH")
        elif self.phase == "TERMINAL":
            if abs(u) < float(g["docked_speed_mps"]) and float(g["stop_nose_s_m"]) - float(g.get("docked_below_m", 0.25)) <= s_nose <= float(g["stop_nose_s_m"]) + float(g["docked_band_m"]):
                self._set_phase("DOCKED")
                return self._docked(st_now, geo, stat)
            if (abs(geo["e"]) > float(g["abort_lateral_m"]) and s_nose > float(g["no_abort_inside_m"]) and self.retries < int(g["max_retries"])):
                self.retries += 1
                self._back_to = float(g["retry_back_to_s_m"])
                self._set_phase("RETRY")

        stat["phase"] = self.phase
        stat["repositions"] = self.repositions
        # ---- guidance
        sgn = -1.0 if self.phase == "RETRY" else 1.0
        if self.phase == "RETRY":
            psi_des = geo["psi_a"]                                    # back straight out along the axis, heading held (steering law uses the signed flow)
        else:
            la = (float(g["lookahead_base_m"]) + float(g["lookahead_per_m"]) * max(geo["s"], 0.0)) if self.phase == "APPROACH" else float(g["terminal_lookahead_m"])
            dev = clamp(math.atan2(geo["e"], max(la, 0.1)), math.radians(float(g["max_deviation_deg"])))
            psi_des = wrap_pi(geo["psi_a"] - dev)
        # ---- speed target
        spd = self.sp
        if self.phase == "APPROACH":
            # faster while the vehicle still has to be steered onto the axis (fin authority grows with speed squared), cruise when aligned, always slowing to the
            # terminal speed over `slow_zone_m` before the gate
            misaligned = abs(geo["e"]) > float(spd.get("align_e_m", 0.3)) or abs(math.degrees(geo["chi"])) > float(spd.get("align_chi_deg", 8.0))
            u_cruise = float(spd.get("align_cruise_mps", spd["cruise_mps"])) if misaligned else float(spd["cruise_mps"])
            frac = max(0.0, min(1.0, (geo["s"] - float(g["gate_s_m"])) / max(float(spd["slow_zone_m"]), 1e-6)))
            u_t = float(spd["terminal_mps"]) + (u_cruise - float(spd["terminal_mps"])) * frac
            u_t = min(u_t, max(self.stop_speed(s_nose), float(spd["terminal_mps"])))
        elif self.phase == "TERMINAL":
            u_t = min(float(spd["terminal_mps"]), self.stop_speed(s_nose))
        else:                                                         # RETRY
            u_t = -float(spd["retry_back_mps"])
        enveloped = self.phase in ("APPROACH", "TERMINAL") and u_t < float(spd["terminal_mps"]) - 1e-6
        w[0] = self.speed_force(u, u_t, brake_ff=enveloped)
        stat["u_target"] = u_t
        # ---- attitude loops (heave to the estimated dock depth, pitch and roll level, heading to the guidance)
        z_sp = float(self.est.x[2]) if sig_p < float(g["depth_trust_sigma_m"]) else st_now.depth
        w[2] = self.loops.heave(st_now, z_sp, dt)
        w[4] = self.loops.pitch(st_now, 0.0, dt)
        w[3] = self.loops.roll(st_now, 0.0, dt)
        n = self.loops.yaw(st_now, psi_des, dt)
        w[5] = n * sgn if False else n                                  # the allocator handles the sign of the flow (see allocation.fins_from_moments)
        stat.update(psi_des_deg=math.degrees(psi_des), z_sp=z_sp)
        return w, stat

    def _search(self, obs: DockObs, st: State, dt: float, stat: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, object]]:
        """No estimate of the dock yet and nothing (or not enough) in view: turn on the spot-ish, slowly, until the whole ring is in the picture.
        The vehicle can only yaw while it moves (fins), so it drives a slow circle (radius about the turning circle) instead of spinning. With one to three
        lights in view the detector's hints are used: yaw toward the side the lights are on, and reverse when the ring is too big for the frame (too close)."""
        rc, w = self.rc, np.zeros(6)
        self._set_phase("SEARCH")
        u = st.speed_u
        u_t = float(rc.get("search_speed_mps", 0.35))
        turn = math.radians(float(rc.get("search_turn_deg", 40.0)))
        # optional rough dock position (from the mission plan): turn toward it, and first get out of its turning-circle zone (a vehicle that cannot sway needs
        # about two turning radii of room to swing round; closer than that it moves straight away from the dock before turning)
        pri = rc.get("dock_hint_ned")
        if pri is not None and len(pri) >= 2:
            if obs.fresh and 1 <= obs.num_lights <= 3 and obs.search_surge < -0.5:       # the detector says the ring is bigger than the picture: too close
                self._hint_back_t = self.t
            forced = self.t - getattr(self, "_hint_back_t", -1e9) < float(rc.get("hint_hold_s", 1.5))
            hxy = np.array([float(pri[0]), float(pri[1])])
            psi_a = math.radians(float(rc.get("dock_axis_deg", 0.0)))                  # the heading a vehicle must have to drive INTO the dock
            fwd = np.array([math.cos(psi_a), math.sin(psi_a)])
            v = st.pos[:2] - hxy
            dist = float(np.hypot(*v))
            d = hxy - st.pos[:2]
            bearing = math.atan2(d[1], d[0])
            rel = wrap_pi(bearing - st.yaw)
            R = float(rc.get("turn_radius_m", 2.4))
            keep = 2.0 * R + float(rc.get("hint_clear_m", 0.8))
            # the lights shine out of the mouth in a cone (the picture only shows them from inside it): from the side or from behind the dock the ring can never be
            # recognised however close the vehicle gets. So: 1) go to a staging point on the dock axis in front of the mouth (round the dock's swing zone),
            # 2) turn to the axis heading there, 3) drive at the dock until the ring appears (and back off if it gets too close to be seen whole).
            cos_view = float(np.dot(v, -fwd)) / max(dist, 1e-6)
            view_ang = math.degrees(math.acos(max(-1.0, min(1.0, cos_view))))
            S = hxy - float(rc.get("hint_stage_m", 7.0)) * fwd
            dS = float(np.hypot(*(S - st.pos[:2])))
            face = math.radians(float(rc.get("hint_face_deg", 35.0)))
            see = float(rc.get("hint_see_m", 4.5))
            keep_nav = float(rc.get("hint_nav_keep_m", 3.5))                   # how close the route to the staging point may pass the dock [m]
            hs = getattr(self, "_hs", "stage")
            ready_to_drive = view_ang <= float(rc.get("hint_cone_deg", 15.0)) and abs(rel) < face and dist >= see - 0.5
            if hs == "stage" and (dS < float(rc.get("hint_reach_m", 1.5)) or ready_to_drive):
                hs = "face"
            if hs == "face" and (ready_to_drive or abs(wrap_pi(st.yaw - psi_a)) < math.radians(12.0)):
                hs = "go"
            if hs != "stage" and (view_ang > 3.0 * float(rc.get("hint_cone_deg", 15.0)) or dist < keep_nav):          # knocked far off the axis, or inside the swing zone: start again
                hs = "stage"
            self._hs = hs
            if hs == "stage":
                aim = S
                if dist < keep_nav:                                           # inside the swing zone of the dock: straight out, no turning
                    aim = hxy + (v / max(dist, 1e-6)) * (keep_nav + 2.0)
                else:
                    seg = S - st.pos[:2]
                    L = float(np.hypot(*seg))
                    if L > 1e-6:
                        t = max(0.0, min(1.0, float(np.dot(hxy - st.pos[:2], seg)) / (L * L)))
                        if float(np.hypot(*(st.pos[:2] + t * seg - hxy))) < keep_nav:        # the straight line to the staging point cuts through the dock's zone: go round it
                            side = 1.0 if (v[0] * fwd[1] - v[1] * fwd[0]) >= 0.0 else -1.0
                            aim = hxy + side * np.array([-fwd[1], fwd[0]]) * (keep_nav + 1.5)
                da = aim - st.pos[:2]
                psi_des = math.atan2(da[1], da[0])
                u_t = float(rc.get("search_speed_mps", 0.35)) if abs(wrap_pi(psi_des - st.yaw)) > math.radians(25.0) else 0.5
                dS_vec = S - st.pos[:2]
                rel_S = wrap_pi(math.atan2(dS_vec[1], dS_vec[0]) - st.yaw)
                if abs(rel_S) > math.radians(90.0) and abs(wrap_pi(st.yaw - psi_a)) < math.radians(100.0) and view_ang < 3.0 * float(rc.get("hint_cone_deg", 15.0)):
                    u_t = -float(rc.get("search_back_mps", 0.3))              # the staging point is behind a vehicle that already points roughly along the axis: reverse straight to it
                    psi_des = psi_a                                           # (turning round would swing it into the dock)
                elif dist < keep_nav:                                     # too close to turn: move straight away from the dock along the heading (reverse if it points at the dock)
                    h_dir = np.array([math.cos(st.yaw), math.sin(st.yaw)])
                    u_t = float(rc.get("search_speed_mps", 0.35)) if float(np.dot(h_dir, v)) >= 0.0 else -float(rc.get("search_back_mps", 0.3))
                    psi_des = st.yaw
            elif hs == "face":
                psi_des = psi_a
                u_t = float(rc.get("search_speed_mps", 0.35))
            else:
                if forced or dist < see:                                  # the whole ring does not fit the picture this close: back straight away
                    u_t = -float(rc.get("search_back_mps", 0.3)) if abs(rel) < 0.5 * math.pi else float(rc.get("search_speed_mps", 0.35))
                    psi_des = st.yaw
                else:
                    psi_des = bearing
                    u_t = min(0.5, 0.25 + 0.1 * dist)
            stat.update(hint_view_deg=view_ang, hint_state={"stage": 1.0, "face": 2.0, "go": 3.0}[hs])
            w[0] = self.speed_force(u, u_t)
            w[2] = self.loops.heave(st, st.depth, dt)
            w[4] = self.loops.pitch(st, 0.0, dt)
            w[3] = self.loops.roll(st, 0.0, dt)
            w[5] = self.loops.yaw(st, psi_des, dt)
            stat.update(phase="SEARCH", u_target=u_t, psi_des_deg=math.degrees(psi_des), hint_dist=dist)
            return w, stat
        hint = obs.fresh and 1 <= obs.num_lights <= 3 and abs(obs.search_yaw) + abs(obs.search_surge) > 1e-6
        if hint:
            self._hint_t = self.t
            self._hint = (obs.search_yaw, obs.search_surge)
        if getattr(self, "_hint_t", -1e9) > self.t - float(rc.get("hint_hold_s", 1.5)):
            yaw_n, surge_n = self._hint
            half_hfov = math.radians(float(rc.get("half_hfov_deg", 37.0)))
            if abs(yaw_n) > 1e-6:
                self._search_dir = 1.0 if yaw_n > 0 else -1.0
            psi_des = wrap_pi(st.yaw + self._search_dir * max(abs(yaw_n), 0.35) * half_hfov)
            if surge_n < -0.5:
                u_t = -float(rc.get("search_back_mps", 0.3))               # the ring is bigger than the frame: back away
        else:
            psi_des = wrap_pi(st.yaw + self._search_dir * turn)
        w[0] = self.speed_force(u, u_t)
        w[2] = self.loops.heave(st, st.depth, dt)
        w[4] = self.loops.pitch(st, 0.0, dt)
        w[3] = self.loops.roll(st, 0.0, dt)
        w[5] = self.loops.yaw(st, psi_des, dt)
        stat.update(phase="SEARCH", u_target=u_t, psi_des_deg=math.degrees(psi_des))
        return w, stat

    def _docked(self, st: State, geo: Dict[str, float], stat: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, object]]:
        """Stop and hold: brake any remaining speed, keep depth and pitch, no steering."""
        w = np.zeros(6)
        s_nose = geo["s"] - self.nose_x * math.cos(geo["chi"])
        hold = max(-0.1, min(0.1, float(self.sp.get("stop_gain_per_s", 0.8)) * (s_nose - float(self.g["stop_nose_s_m"]))))     # creep back to the stop point if it drifted
        w[0] = self.speed_force(st.speed_u, hold, brake_ff=False)
        w[2] = self.loops.heave(st, float(self.est.x[2]), 0.05)
        w[4] = self.loops.pitch(st, 0.0, 0.05)
        stat.update(phase="DOCKED", u_target=0.0)
        return w, stat
