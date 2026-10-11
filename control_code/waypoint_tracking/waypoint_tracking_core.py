"""Waypoint tracking engine (pure Python, no ROS), rebuilt on common/ on 2026-10-10.

Why: the first version put a PID straight on fin DEGREES with hard-coded per-fin signs (`fin_yaw_signs`). Offline, on the current 20 kg neutral vessel, that
spun the vehicle after the first waypoint: (1) the signs [1, 1, -1, -1] are the OPPOSITE of the yaw mix the allocator derives from the vessel file
(cs_04, cs_06, cs_07, cs_08 -> -, -, +, +), so the heading loop was positive feedback; (2) the loop gain was not scheduled with speed (fin force ~ speed^2);
(3) surge only had an RPM command and no speed loop or braking, so on a vehicle with almost no drag it coasted away at 1.7 m/s and, with no flow, could not turn.

Now: the heading loop is common/loops.py `yaw` (a moment in N*m) and the allocator turns the moment into fin degrees with the 1/speed^2 gain schedule and the
sign mix from the geometry (the same single source of truth the dof_testing yaw step verifies); depth is the `heave` loop (force) through the allocator;
surge is the `speed` loop with a target that keeps water flowing over the fins and slows towards the waypoint.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from allocation import Allocator  # noqa: E402
from loops import HoldLoops  # noqa: E402
from state import State, wrap_pi  # noqa: E402

Waypoint = Tuple[float, float, float]


def parse_waypoints(items: List[Any], hold_z: float) -> List[Waypoint]:
    out: List[Waypoint] = []
    for item in items:
        if isinstance(item, dict):
            x, y = float(item["x"]), float(item["y"])
            z = float(item["z"]) if item.get("z") is not None else hold_z
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = float(item[0]), float(item[1])
            z = float(item[2]) if len(item) >= 3 else hold_z
        else:
            raise ValueError(f"Bad waypoint entry: {item!r}")
        out.append((x, y, z))
    return out


def unreachable_corners(wps: List[Waypoint], turn_radius_m: float, loop: bool = False, start_xy: Optional[Tuple[float, float]] = None, tol_m: float = 0.0) -> List[str]:
    """Corners the vehicle cannot make: after passing waypoint i on the line from i-1, a target that lies INSIDE the turning circle can never be reached (it orbits
    it for ever). The vehicle cannot strafe or pivot (turn radius ~2.4 m offline: the fins' moment and the hull's yaw drag both scale with speed^2, so slowing
    down does not tighten the circle). Returns one message per bad corner (empty = fine). With start_xy the leg start -> first waypoint is assumed straight and the first corner
    is checked too (the start HEADING is unknown here, so the first leg is taken as the straight line). `tol_m` (default 0 = exact) forgives that much: the vehicle does not have to pass
    through a waypoint exactly, it may cut inside by the acceptance radius, so dense paths made of arcs pass with tol_m = acceptance_radius_m."""
    msgs: List[str] = []
    pts = [tuple(w[:2]) for w in wps]
    if start_xy is not None:
        pts = [tuple(start_xy)] + pts
    off = 1 if start_xy is not None else 0                       # index of a corner in `pts` minus its waypoint number
    n = len(pts)
    R = float(turn_radius_m)
    for i in range(1, n if not loop else n + 1):
        a, b, c = np.array(pts[(i - 1) % n]), np.array(pts[i % n]), np.array(pts[(i + 1) % n])
        if not loop and i + 1 >= n:
            break
        d_in = b - a
        if np.linalg.norm(d_in) < 1e-6:
            continue
        d_in = d_in / np.linalg.norm(d_in)
        cross = d_in[0] * (c - b)[1] - d_in[1] * (c - b)[0]
        side = 1.0 if cross >= 0 else -1.0                      # +1 = the target is on the right-hand side looking along d_in (x north, y east: right of (dx, dy) is (-dy, dx))
        centre = b + R * side * np.array([-d_in[1], d_in[0]])    # centre of the turning circle on that side
        dist = float(np.linalg.norm(c - centre))
        if dist < R - tol_m - 1e-6:
            msgs.append(f"waypoint {(i - off) % len(wps)} -> {(i + 1 - off) % len(wps)}: the target is {dist:.1f} m from the turn centre, inside the {R:.1f} m turning circle: the vehicle would orbit it")
    return msgs


class WaypointTracker:
    """Steers through a waypoint list. With `set_path` the list can be swapped while the loops keep running (the mission controller does that between legs), and a path can be
    followed as a LINE (previous point -> next waypoint) with cross-track control instead of just aiming at the next waypoint (needs `gains.cross_track`)."""

    def __init__(self, cfg: Dict[str, Any], alloc: Optional[Allocator] = None, allow_empty: bool = False) -> None:
        self.cfg = cfg
        lim = cfg["limits"]
        self.alloc = alloc or Allocator(
            rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"], u_fin_min=lim["u_fin_min_mps"], u_fin_off=lim["u_fin_off_mps"],
            u_fin_full=lim["u_fin_full_mps"], small_force_n=lim.get("small_force_n", 0.0),
        )
        self.loops = HoldLoops({k: dict(v) for k, v in cfg["gains"].items()}, heave_ff_n=cfg["feedforward"]["heave_n"])
        self.mission = cfg["mission"]
        self.sp = cfg["speed"]
        self.waypoints = parse_waypoints(cfg.get("waypoints", []), float(self.mission.get("hold_depth_m", 5.0)))
        if not self.waypoints and not allow_empty:
            raise ValueError("waypoint_tracking.yaml: waypoints list is empty")
        self.acc = float(self.mission["acceptance_radius_m"])
        self.depth_acc = float(self.mission["depth_acceptance_m"])
        self.depth_gate = bool(self.mission.get("depth_gate", False))   # True = a waypoint also needs the depth within depth_acceptance_m (old rule: orbits when the depth lags)
        self.pass_radius_m = float(self.mission.get("pass_radius_m", 3.0))   # a waypoint counts as passed when the vehicle is this close and the waypoint is behind it (no strafing: it cannot turn back inside ~2.5 m)
        self._min_r = float("inf")                                       # closest horizontal distance to the current waypoint so far
        self.warnings: List[str] = []
        self.depth_override: Optional[float] = None         # terrain following: a depth target that replaces the waypoint's z (and its depth acceptance) while set
        self.cruise = float(self.sp["cruise_mps"])
        self.loop = bool(self.mission.get("loop", False))
        self.stop_at_end = True                      # False = the last waypoint is not a stop (the next leg continues): keep the flow speed
        self.prev: Optional[Tuple[float, float]] = None
        self.line = False
        self.pass_by = False
        self.ct_err = 0.0                            # signed distance from the current line [m], + = vehicle is to the RIGHT of the direction of travel
        self.idx = 0
        self.done = False
        self.status: Dict[str, Any] = {}

    def set_path(self, wps: List[Waypoint], *, prev: Optional[Tuple[float, float]] = None, line: bool = False, acc: Optional[float] = None,
                 depth_acc: Optional[float] = None, cruise: Optional[float] = None, stop_at_end: bool = True, loop: bool = False, pass_by: bool = False) -> None:
        """Replace the path without touching the loops (no jump in the filters). `prev` is where the first leg starts (for the line mode)."""
        self.waypoints = list(wps)
        self.idx = 0
        self.done = not self.waypoints
        self.prev = prev
        self.line = bool(line) and prev is not None and "cross_track" in self.loops.pids
        self.acc = float(acc) if acc is not None else float(self.mission["acceptance_radius_m"])
        self.depth_acc = float(depth_acc) if depth_acc is not None else float(self.mission["depth_acceptance_m"])
        self.cruise = float(cruise) if cruise is not None else float(self.sp["cruise_mps"])
        self.stop_at_end = bool(stop_at_end)
        self.loop = bool(loop)
        self.pass_by = bool(pass_by)                  # a waypoint counts as reached when the vehicle passes its perpendicular close by (curved legs: do not circle a point)
        self.ct_err = 0.0
        self._min_r = float("inf")
        self.pass_radius_m = self.acc                  # the mission legs keep their own exact acceptance (dense arcs must not skip points)
        if "cross_track" in self.loops.pids:
            self.loops.reset("cross_track")

    # ---------------------------------------------------------------- helpers
    def target_speed(self, r_xy: float, heading_err: float, last: bool) -> float:
        """Speed target [m/s]. Cruise, slowed along a stopping law towards the waypoint. The fins need flow to steer, so the target never drops below the flow speed
        except when stopping on the LAST waypoint; when the heading is far off it is exactly the flow speed (turn while moving, do not stop and get stuck)."""
        s, acc = self.sp, self.acc
        cruise, flow = self.cruise, float(s["flow_mps"])
        v = min(cruise, math.sqrt(2.0 * float(s["decel_mps2"]) * max(r_xy - acc, 0.0)))
        keep_flow = (not last) or r_xy > 3.0 * acc
        if abs(math.degrees(heading_err)) > float(s["align_deg"]):
            return flow
        v *= max(0.0, math.cos(heading_err))              # blend towards zero with misalignment
        return max(v, flow) if keep_flow else v

    def reached(self, st: State) -> bool:
        wx, wy, wz = self.waypoints[self.idx]
        r = math.hypot(wx - st.pos[0], wy - st.pos[1])
        depth_ok = self.depth_override is not None or abs(wz - st.depth) <= self.depth_acc
        if self.depth_gate and not depth_ok:
            return False
        if r <= self.acc:
            return True
        if r < self.pass_radius_m and (wx - st.pos[0]) * math.cos(st.yaw) + (wy - st.pos[1]) * math.sin(st.yaw) < -0.3:     # close, and the waypoint is now BEHIND the vehicle: flew past it, do not circle the point
            self.warnings.append(f"waypoint {self.idx} passed at {r:.1f} m (acceptance {self.acc:.1f} m)")      # (a distance that grows again is NOT used: the latency-compensated pose jitters by ~0.4 m at 1.5 m/s)
            return True
        if (self.line or self.pass_by) and self.prev is not None:   # flew past the end of the leg (a little off it): done, do not turn back or circle the point for it
            ax, ay = self.prev
            L = math.hypot(wx - ax, wy - ay)
            if L > 1e-6:
                ux, uy = (wx - ax) / L, (wy - ay) / L
                rx, ry = st.pos[0] - ax, st.pos[1] - ay
                return (rx * ux + ry * uy) >= L and abs(rx * -uy + ry * ux) <= 4.0 * self.acc
        return False

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
        return None

    # ----------------------------------------------------------------- update
    def update(self, st: State, dt: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """-> (actuator dict {th_XX: RPM, cs_XX: deg}, status). Sets self.done when the last waypoint is reached (then the command is neutral)."""
        if self.done or not self.waypoints:
            self.done = True
            return self.neutral(), {"mode": "done"}
        wxr, wyr, _ = self.waypoints[self.idx]
        r_now = math.hypot(wxr - float(st.pos[0]), wyr - float(st.pos[1]))
        reached = self.reached(st)
        self._min_r = min(self._min_r, r_now)
        if reached:
            self._min_r = float("inf")
            wx0, wy0, _ = self.waypoints[self.idx]
            self.prev = (wx0, wy0)
            self.idx += 1
            self.loops.reset("yaw")
            if self.idx >= len(self.waypoints):
                if self.loop:
                    self.idx = 0
                else:
                    self.done = True
                    return self.neutral(), {"mode": "done"}
        wx, wy, wz = self.waypoints[self.idx]
        if self.depth_override is not None:
            wz = self.depth_override
        dx, dy = wx - float(st.pos[0]), wy - float(st.pos[1])
        r_xy = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        self.ct_err = 0.0
        if self.line and self.prev is not None:           # follow the line prev -> waypoint: aim at a carrot `lookahead_m` ahead of the vehicle's projection on the line
            ax, ay = self.prev
            L = math.hypot(wx - ax, wy - ay)
            if L > 1e-6:
                ux, uy = (wx - ax) / L, (wy - ay) / L
                rx, ry = float(st.pos[0]) - ax, float(st.pos[1]) - ay
                along = rx * ux + ry * uy
                self.ct_err = rx * -uy + ry * ux
                la = float(self.sp.get("lookahead_m", 4.0))
                s_c = min(max(along, 0.0) + la, L)                      # carrot position along the line, never beyond the waypoint
                cx, cy = ax + ux * s_c, ay + uy * s_c
                desired_yaw = math.atan2(cy - float(st.pos[1]), cx - float(st.pos[0]))
                if "cross_track" in self.loops.pids:                      # small integral/proportional trim for a steady push (current)
                    desired_yaw += self.loops.cross_track(-self.ct_err, dt)
        heading_err = wrap_pi(desired_yaw - st.yaw)
        last = self.idx == len(self.waypoints) - 1 and not self.loop and self.stop_at_end
        u_sp = self.target_speed(r_xy, heading_err, last)

        w = np.zeros(6)
        w[0] = self.loops.speed(st, u_sp, dt)
        w[2] = self.loops.heave(st, wz, dt)
        w[5] = self.loops.yaw(st, st.yaw + heading_err, dt)
        out = self.alloc.allocate(w, st.speed_u)
        self.status = {
            "mode": "track", "wp": self.idx, "r_xy": r_xy, "heading_err_deg": math.degrees(heading_err), "u": st.speed_u, "u_sp": u_sp,
            "depth_err_m": wz - st.depth, "cross_track_m": self.ct_err, "fin_authority": self.alloc.fin_authority(st.speed_u), "wrench": w.copy(),
        }
        return out, self.status

    def neutral(self) -> Dict[str, float]:
        return {n: 0.0 for n in self.alloc.th_ids + self.alloc.fin_ids}

    def debug(self) -> Dict[str, Any]:
        wp = self.waypoints[min(self.idx, len(self.waypoints) - 1)]
        return {"ctrl": "waypoint_tracking", "mode": self.status.get("mode", "wait"), "depth_sp": float(wp[2]),
                "yaw_sp_deg": float(self.status.get("heading_err_deg", 0.0))}
