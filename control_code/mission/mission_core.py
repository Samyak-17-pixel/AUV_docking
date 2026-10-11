"""Mission supervisor (pure Python, no ROS): turns a `legs:` list into paths and flies them with the verified WaypointTracker.

Leg types (mission.yaml `legs:`):
  goto        {type: goto, x, y, z}                                        fly to a point
  lawnmower   {type: lawnmower, origin: [x, y], heading_deg, length_m, width_m, spacing_m, z (or depths: [..])}
  orbit       {type: orbit, centre: [x, y], radius_m, revs, clockwise, z}
  spiral      {type: spiral, centre: [x, y], r_start_m, r_end_m, pitch_m, z}
  yoyo        {type: yoyo, from: [x, y], to: [x, y], z_top, z_bottom, cycles}
  hold        {type: hold, seconds}                                        stop the thrust and hold depth (no heading hold: the fins need flow)
  return_home {type: return_home}                                          go to `mission.home` (default: where the mission started) and hold
Every leg may override `acceptance_m`, `depth_acceptance_m`, `speed_mps`, and straight legs are flown as LINES with cross-track control
(`cross_track.enabled`): the vehicle follows the lane, not just the bearing to the next point.

Safety: a geofence box and a dock keep-out circle (`safety.geofence`, `safety.dock_keepout_m`): leaving the box or entering the keep-out ABORTS the mission and the
vehicle returns home (once); `report_odom_loss(seconds)` does the same when the odometry was gone longer than `safety.failsafe.odom_loss_s`. A depth or attitude limit
trip is a hard stop (`safety_reason`).
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "common"))
sys.path.insert(0, str(_HERE.parent / "waypoint_tracking"))
import paths as P  # noqa: E402
from state import State  # noqa: E402
from waypoint_tracking_core import WaypointTracker  # noqa: E402

Waypoint = Tuple[float, float, float]


class Segment:
    """One flown piece: waypoints plus how to fly them."""

    def __init__(self, kind: str, wps: List[Waypoint], leg: int, line: bool = True, acc: Optional[float] = None, depth_acc: Optional[float] = None,
                 cruise: Optional[float] = None, seconds: float = 0.0, label: str = "", pass_by: bool = False) -> None:
        self.kind, self.wps, self.leg, self.line, self.pass_by = kind, wps, leg, line, pass_by
        self.acc, self.depth_acc, self.cruise, self.seconds, self.label = acc, depth_acc, cruise, seconds, label or kind


def _xy(v: Any) -> Tuple[float, float]:
    return float(v[0]), float(v[1])


def expand_legs(cfg: Dict[str, Any], start_xy: Optional[Tuple[float, float]] = None, home: Optional[Tuple[float, float, float]] = None
                ) -> Tuple[List[Segment], List[str]]:
    """Legs -> segments. Returns (segments, warnings). A leg that cannot be built raises ValueError naming the leg."""
    mis, sp = cfg["mission"], cfg["speed"]
    R = float(sp["turn_radius_m"])
    hold_z = float(mis.get("hold_depth_m", 3.0))
    segs: List[Segment] = []
    warns: List[str] = []
    cur = start_xy
    for i, leg in enumerate(cfg.get("legs", [])):
        t = str(leg.get("type", "")).lower()
        common = dict(acc=leg.get("acceptance_m"), depth_acc=leg.get("depth_acceptance_m"), cruise=leg.get("speed_mps"))
        try:
            if t == "goto":
                wp = (float(leg["x"]), float(leg["y"]), float(leg.get("z", hold_z)))
                segs.append(Segment("goto", [wp], i, line=bool(leg.get("line", False)), **common))
            elif t == "lawnmower":
                depths = leg.get("depths") or [leg.get("z", hold_z)]
                for k, z in enumerate(depths):
                    pts, info = P.lawnmower(_xy(leg["origin"]), float(leg.get("heading_deg", 0.0)), float(leg["length_m"]), float(leg["width_m"]),
                                            float(leg["spacing_m"]), float(z), R, int(leg.get("arc_points", 5)))
                    if k % 2 == 1:
                        pts = list(reversed(pts))                       # the next depth is flown back along the same lanes
                    if k == 0 and float(leg.get("leadin_m", 2.0 * R)) > 0:     # a straight run-in on the lane line so the vehicle is lined up when the first lane starts
                        h = math.radians(float(leg.get("heading_deg", 0.0)))
                        lead = float(leg.get("leadin_m", 2.0 * R))
                        pts = [(pts[0][0] - lead * math.cos(h), pts[0][1] - lead * math.sin(h), pts[0][2])] + pts
                    warns += [f"leg {i} (lawnmower): {m}" for m in info["warnings"]]
                    segs.append(Segment("lawnmower", pts, i, line=True, label=f"lawnmower z={z}", **common))
            elif t == "orbit":
                pts, info = P.orbit(_xy(leg["centre"]), float(leg["radius_m"]), float(leg.get("z", hold_z)), float(leg.get("revs", 1.0)),
                                    bool(leg.get("clockwise", True)), float(leg.get("start_deg", 180.0)), int(leg.get("points_per_rev", 16)), R)
                common["acc"] = 1.0 if common["acc"] is None else common["acc"]           # on a polygon of short chords a looser goal mouth lets the vehicle cut the corners smoothly
                segs.append(Segment("orbit", pts, i, line=False, pass_by=True, **common))
            elif t == "spiral":
                pts, info = P.spiral(_xy(leg["centre"]), float(leg["r_start_m"]), float(leg["r_end_m"]), float(leg.get("z", hold_z)), float(leg.get("pitch_m", 3.0)),
                                     bool(leg.get("clockwise", True)), float(leg.get("start_deg", 180.0)), int(leg.get("points_per_rev", 16)), R)
                common["acc"] = 1.0 if common["acc"] is None else common["acc"]
                segs.append(Segment("spiral", pts, i, line=False, pass_by=True, **common))
            elif t == "yoyo":
                if common["depth_acc"] is None:
                    common["depth_acc"] = 0.6                              # the depth extremes are a target to swing towards, not a point to settle on
                pts, info = P.yoyo(_xy(leg["from"]), _xy(leg["to"]), float(leg["z_top"]), float(leg["z_bottom"]), int(leg.get("cycles", 3)))
                segs.append(Segment("yoyo", pts, i, line=True, **common))
            elif t == "hold":
                segs.append(Segment("hold", [], i, seconds=float(leg.get("seconds", 10.0))))
            elif t in ("return_home", "home"):
                segs.append(Segment("return_home", [], i, line=False, **common))          # target filled in at run time
            else:
                raise ValueError(f"unknown leg type {t!r}")
        except KeyError as e:
            raise ValueError(f"leg {i} ({t}): missing parameter {e}") from None
        except ValueError as e:
            raise ValueError(f"leg {i} ({t}): {e}") from None
    return segs, warns


def plan_polyline(cfg: Dict[str, Any], start_xy: Tuple[float, float] = (0.0, 0.0)) -> List[Tuple[float, float, float, str]]:
    """The whole mission as a list of (x, y, z, leg label) for previews (return_home targets the start)."""
    segs, _ = expand_legs(cfg, start_xy)
    out: List[Tuple[float, float, float, str]] = []
    for s in segs:
        if s.kind == "return_home":
            out.append((start_xy[0], start_xy[1], float(cfg["mission"].get("hold_depth_m", 3.0)), s.label))
        for w in s.wps:
            out.append((w[0], w[1], w[2], s.label))
    return out


def validate_mission(cfg: Dict[str, Any], start_xy: Tuple[float, float] = (0.0, 0.0)) -> List[str]:
    """All warnings for a mission config: the generators' own plus check_path on the WHOLE chained path (so the turn from one leg into the next is checked too:
    turning circle, geofence, dock keep-out). Messages name the leg."""
    import re
    try:
        segs, warns = expand_legs(cfg, start_xy)
    except ValueError as e:
        return [str(e)]
    sf = cfg.get("safety", {})
    home = cfg["mission"].get("home")
    chain: List[Waypoint] = []
    owner: List[str] = []
    for s in segs:
        pts = s.wps if s.wps else ([(float(home[0]), float(home[1]), float(cfg["mission"].get("hold_depth_m", 3.0)))] if (s.kind == "return_home" and home) else
                                    [(start_xy[0], start_xy[1], float(cfg["mission"].get("hold_depth_m", 3.0)))] if s.kind == "return_home" else [])
        chain += list(pts)
        owner += [f"leg {s.leg} ({s.label})"] * len(pts)
    msgs = list(warns)
    raw = P.check_path(chain, float(cfg["speed"]["turn_radius_m"]), start_xy, False, sf.get("geofence"), float(sf.get("dock_keepout_m", 0.0)),
                       tol_m=float(cfg["mission"]["acceptance_radius_m"]))
    for m in raw:
        mm = re.match(r"waypoint (\d+)", m)
        who = owner[int(mm.group(1))] if mm and int(mm.group(1)) < len(owner) else "path"
        msgs.append(f"{who}: {m}")
    return msgs


class MissionRunner:
    def __init__(self, cfg: Dict[str, Any], alloc=None) -> None:
        self.cfg = cfg
        self.safety = cfg["safety"]
        self.wt = WaypointTracker(cfg, alloc, allow_empty=True)
        self.ct_on = bool(cfg.get("cross_track", {}).get("enabled", True)) and "cross_track" in self.wt.loops.pids
        self.segments: List[Segment] = []
        self.seg_idx = -1
        self.home: Optional[Tuple[float, float, float]] = None
        self.started = False
        self.done = False
        self.aborted: Optional[str] = None
        self.mode = "wait"
        self.warnings: List[str] = []
        self._hold_left = 0.0
        self._odom_loss_s = 0.0
        self._lengths: List[float] = []
        self._travelled = 0.0
        self._last_xy: Optional[Tuple[float, float]] = None
        self._t = 0.0
        self.status: Dict[str, Any] = {}

    # ------------------------------------------------------------------ setup
    def start(self, st: State) -> None:
        x, y = float(st.pos[0]), float(st.pos[1])
        h = self.cfg["mission"].get("home")
        self.home = (float(h[0]), float(h[1]), float(h[2]) if len(h) > 2 else st.depth) if h else (x, y, float(self.cfg["mission"].get("hold_depth_m", st.depth)))
        self.segments, w = expand_legs(self.cfg, (x, y), self.home)
        self.warnings = validate_mission(self.cfg, (x, y))
        self._lengths = []
        px, py = x, y
        for s in self.segments:
            pts = s.wps if s.wps else ([(self.home[0], self.home[1], self.home[2])] if s.kind == "return_home" else [])
            L = 0.0
            for q in pts:
                L += math.hypot(q[0] - px, q[1] - py)
                px, py = q[0], q[1]
            self._lengths.append(L)
        self.started = True
        self._last_xy = (x, y)
        self._next_segment(st)

    @property
    def total_length(self) -> float:
        return float(sum(self._lengths))

    def progress(self) -> float:
        tot = self.total_length
        if self.done:
            return 1.0
        return 0.0 if tot <= 0 else min(1.0, self._travelled / tot)

    def eta_s(self) -> float:
        cruise = max(float(self.cfg["speed"]["cruise_mps"]), 0.1)
        return max(self.total_length - self._travelled, 0.0) / cruise

    # ------------------------------------------------------------------ safety
    def safety_reason(self, st: State) -> Optional[str]:
        """Hard-stop reasons (depth or attitude out of range): the mission cannot continue."""
        return self.wt.safety_reason(st)

    def _breach(self, st: State) -> Optional[str]:
        gf = self.safety.get("geofence") or {}
        x, y, z = float(st.pos[0]), float(st.pos[1]), float(st.depth)
        for ax, v in (("x", x), ("y", y), ("z", z)):
            lo, hi = gf.get(f"{ax}_min"), gf.get(f"{ax}_max")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                return f"outside the geofence ({ax} = {v:.1f})"
        k = float(self.safety.get("dock_keepout_m", 0.0))
        if k > 0 and math.hypot(x - P.DOCK_XY[0], y - P.DOCK_XY[1]) < k:
            return f"inside the {k:.1f} m dock keep-out"
        return None

    def report_odom_loss(self, seconds: float) -> None:
        """Tell the runner the odometry was gone for `seconds`; if that exceeds safety.failsafe.odom_loss_s the mission is aborted to return-home."""
        self._odom_loss_s = max(self._odom_loss_s, float(seconds))

    def abort(self, why: str) -> None:
        if self.aborted:
            return
        self.aborted = why
        self.segments = [Segment("return_home", [], -1, line=False, label="RETURN HOME (abort)"), Segment("hold", [], -1, seconds=1e9, label="hold at home")]
        self.seg_idx = -1
        self._lengths = [0.0, 0.0]
        self.done = False

    # ------------------------------------------------------------------ flow
    def _next_segment(self, st: State) -> None:
        self.seg_idx += 1
        if self.seg_idx >= len(self.segments):
            self.done = True
            self.mode = "done"
            return
        s = self.segments[self.seg_idx]
        last = self.seg_idx == len(self.segments) - 1
        here = (float(st.pos[0]), float(st.pos[1]))
        prev = self.wt.prev if self.wt.prev is not None else here
        if s.kind == "hold":
            self._hold_left = s.seconds
            self.mode = "hold"
            return
        if s.kind == "return_home":
            self.wt.set_path([self.home], prev=here, line=False, acc=s.acc, depth_acc=s.depth_acc, cruise=s.cruise, stop_at_end=True)
            self.mode = "return_home"
            return
        nxt_is_hold = (not last) and self.segments[self.seg_idx + 1].kind == "hold"
        # a line leg starts where the vehicle IS (the previous leg ended near there), so the line is "here -> first point" for the first piece
        line = bool(s.line and self.ct_on)
        self.wt.set_path(s.wps, prev=here, line=line, acc=s.acc, depth_acc=s.depth_acc, cruise=s.cruise, stop_at_end=bool(last or nxt_is_hold), pass_by=s.pass_by)
        self.mode = s.kind

    def _terrain_follow(self, st: State, dt: float, altitude: Optional[float]) -> None:
        """terrain_follow: hold a fixed ALTITUDE above the sea floor by moving the depth target with the altimeter reading (rate limited, kept between the depth limits)."""
        tf = self.cfg.get("terrain_follow") or {}
        if not tf.get("enabled", False) or altitude is None or not math.isfinite(altitude):
            self.wt.depth_override = None if not tf.get("enabled", False) else self.wt.depth_override
            return
        want = st.depth + (altitude - float(tf.get("altitude_m", 6.0)))                 # deeper if the floor is further away than wanted, shallower if closer
        lo, hi = float(self.safety.get("min_depth_m", 0.4)) + 0.3, float(tf.get("max_depth_m", self.safety.get("max_depth_m", 20.0))) - 0.3
        want = min(max(want, lo), hi)
        cur = self.wt.depth_override if self.wt.depth_override is not None else st.depth
        rate = float(tf.get("rate_mps", 0.4))
        self.wt.depth_override = cur + min(max(want - cur, -rate * dt), rate * dt)
        self.status["altitude_m"] = float(altitude)

    def update(self, st: State, dt: float, altitude: Optional[float] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
        self._terrain_follow(st, dt, altitude)
        if not self.started:
            self.start(st)
        self._t += dt
        xy = (float(st.pos[0]), float(st.pos[1]))
        if self._last_xy is not None:
            self._travelled += math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
        self._last_xy = xy
        if not self.aborted:
            why = self._breach(st)
            if why is None and self._odom_loss_s > float(self.safety.get("failsafe", {}).get("odom_loss_s", 1e9)):
                why = f"odometry lost for {self._odom_loss_s:.0f} s"
            if why:
                self.abort(why)
                self.start_after_abort(st)
        self._odom_loss_s = 0.0
        if self.done:
            return self.wt.neutral(), self._status(st, "done")
        s = self.segments[self.seg_idx]
        if s.kind == "hold":
            self._hold_left -= dt
            if self._hold_left <= 0.0:
                self._next_segment(st)
                return self.update(st, 0.0) if not self.done else (self.wt.neutral(), self._status(st, "done"))
            w = np.zeros(6)
            w[0] = self.wt.loops.speed(st, 0.0, dt)
            w[2] = self.wt.loops.heave(st, float(self.wt.waypoints[-1][2]) if self.wt.waypoints else st.depth, dt)
            return self.wt.alloc.allocate(w, st.speed_u), self._status(st, "hold")
        out, ws = self.wt.update(st, dt)
        if self.wt.done:
            self._next_segment(st)
            if self.done:
                return self.wt.neutral(), self._status(st, "done")
            return self.update(st, 0.0)
        return out, self._status(st, self.mode, ws)

    def start_after_abort(self, st: State) -> None:
        self._next_segment(st)

    def _status(self, st: State, mode: str, ws: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        s = self.segments[self.seg_idx] if 0 <= self.seg_idx < len(self.segments) else None
        self.status = {"mode": mode, "leg": (s.label if s else "-"), "leg_index": (s.leg if s else -1), "segment": self.seg_idx, "segments": len(self.segments),
                       "progress": self.progress(), "eta_s": self.eta_s(), "aborted": self.aborted or "", "wp": self.wt.idx, "wps": len(self.wt.waypoints)}
        if ws:
            for k in ("r_xy", "heading_err_deg", "u", "u_sp", "depth_err_m", "cross_track_m", "fin_authority"):
                if k in ws:
                    self.status[k] = ws[k]
        return self.status

    def neutral(self) -> Dict[str, float]:
        return self.wt.neutral()

    def debug(self) -> Dict[str, Any]:
        """ctrl_debug message: numeric fields are plotted by the viewer (cross-track error, speed vs setpoint, progress ...), text fields are shown."""
        wp = self.wt.waypoints[min(self.wt.idx, len(self.wt.waypoints) - 1)] if self.wt.waypoints else (0.0, 0.0, 0.0)
        st = self.status
        d = {"ctrl": "mission", "mode": st.get("mode", "wait"), "depth_sp": float(wp[2]), "yaw_sp_deg": float(st.get("heading_err_deg", 0.0))}
        d.update({"leg": st.get("leg", "-"), "leg_index": st.get("leg_index", -1), "aborted": st.get("aborted", "")})
        for k in ("progress", "eta_s", "cross_track_m", "heading_err_deg", "u", "u_sp", "depth_err_m", "r_xy"):
            d[k] = float(st.get(k, 0.0) or 0.0)
        d["target_x"], d["target_y"] = float(wp[0]), float(wp[1])
        return d
