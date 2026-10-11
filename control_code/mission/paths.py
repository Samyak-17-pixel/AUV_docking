"""Path generators for the mission controller (pure Python, no ROS). Every function returns a list of (x north, y east, z depth+down) waypoints that
WaypointTracker can fly. The vehicle cannot strafe or pivot: its smallest circle has a radius `turn_radius_m` (2.5 m offline, an ASSUMPTION until measured on the real sim),
so every generator builds turns that circle can make and `check_path` (unreachable_corners + dock keep-out + geofence) tells you when a path still cannot be flown.

  lawnmower   parallel lanes with semicircular U-turns; lanes closer than 2 turn radii are visited with a SKIP pattern (0, k, 2k, ... then 1, k+1, ...)
  orbit       a circle (polygon) around a centre, any number of revolutions
  spiral      an orbit whose radius grows (or shrinks) each revolution
  yoyo        a straight line flown with the depth going up and down (sawtooth)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "waypoint_tracking"))
from waypoint_tracking_core import Waypoint, unreachable_corners  # noqa: E402

DOCK_XY = (10.0, 0.0)


def _rot(dx: float, dy: float, heading_deg: float) -> Tuple[float, float]:
    """Vector (dx along the heading, dy to the RIGHT of it) -> (north, east)."""
    h = math.radians(heading_deg)
    return dx * math.cos(h) - dy * math.sin(h), dx * math.sin(h) + dy * math.cos(h)


def lane_order(n_lanes: int, skip: int) -> List[int]:
    """Visiting order of the lanes: 0..n-1 for skip 1, else 0, k, 2k, ... then 1, k+1, ... (so every sideways step is at least k lanes)."""
    if skip <= 1:
        return list(range(n_lanes))
    order: List[int] = []
    for off in range(skip):
        order += list(range(off, n_lanes, skip))
    return order


def lawnmower(origin: Tuple[float, float], heading_deg: float, length_m: float, width_m: float, spacing_m: float, z: float,
              turn_radius_m: float = 2.5, arc_points: int = 5, margin: float = 1.05) -> Tuple[List[Waypoint], Dict[str, object]]:
    """Boustrophedon survey. The first lane starts at `origin`, runs `length_m` along `heading_deg`, lanes are `spacing_m` apart to the RIGHT of the heading over
    `width_m`. U-turns are semicircles whose diameter is the sideways step, so they need a step >= 2 * turn radius: when `spacing_m` is smaller the lanes are visited
    with a skip pattern (skip = ceil(2 R * margin / spacing)). Returns (waypoints, info) with info = {lanes, skip, order, length_m, warnings}."""
    if length_m <= 0 or width_m < 0 or spacing_m <= 0:
        raise ValueError("lawnmower: length_m and spacing_m must be positive, width_m >= 0")
    n = int(math.floor(width_m / spacing_m + 1e-9)) + 1
    skip = max(1, int(math.ceil(2.0 * turn_radius_m * margin / spacing_m - 1e-9)))
    order = lane_order(n, skip)
    warns: List[str] = []
    if skip > 1:
        warns.append(f"lane spacing {spacing_m:.1f} m < 2 x turn radius ({2 * turn_radius_m:.1f} m): lanes are visited with a skip of {skip}")
    ox, oy = origin
    pts: List[Waypoint] = []
    forward = True
    for k, lane in enumerate(order):
        y_off = lane * spacing_m
        a, b = (0.0, length_m) if forward else (length_m, 0.0)
        for xa in (a, b):
            e, n_ = _rot(xa, y_off, heading_deg)
            pts.append((ox + e, oy + n_, z))
        if k + 1 < len(order):
            step = (order[k + 1] - lane) * spacing_m               # signed sideways step to the right
            d = abs(step)
            if d + 1e-9 < 2.0 * turn_radius_m:
                warns.append(f"lane {lane} -> {order[k + 1]}: sideways step {d:.1f} m < {2 * turn_radius_m:.1f} m: that U-turn cannot be flown")
            else:                                                      # semicircle of diameter d from the lane end, bulging past the end (forward direction)
                sgn = 1.0 if step > 0 else -1.0
                xe = b                                                 # the end of this lane along the heading
                bulge = 1.0 if forward else -1.0
                for j in range(1, arc_points + 1):
                    ang = math.pi * j / (arc_points + 1)
                    xx = xe + bulge * (d / 2.0) * math.sin(ang)
                    yy = y_off + sgn * (d / 2.0) * (1.0 - math.cos(ang))
                    e, n_ = _rot(xx, yy, heading_deg)
                    pts.append((ox + e, oy + n_, z))
        forward = not forward
    path_len = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
    return pts, {"lanes": n, "skip": skip, "order": order, "length_m": path_len, "warnings": warns}


def orbit(centre: Tuple[float, float], radius_m: float, z: float, revs: float = 1.0, clockwise: bool = True, start_deg: float = 180.0,
          points_per_rev: int = 16, turn_radius_m: float = 2.5, margin: float = 1.1) -> Tuple[List[Waypoint], Dict[str, object]]:
    """A circle around `centre`. `start_deg` is the bearing from the centre to the first point (0 = north of the centre, 90 = east). Clockwise seen from above
    (towards east when starting north of the centre). The radius must be at least `turn_radius_m * margin`, otherwise the vehicle could not follow it."""
    warns: List[str] = []
    if radius_m < turn_radius_m * margin:
        raise ValueError(f"orbit: radius {radius_m:.1f} m is smaller than the turning circle ({turn_radius_m * margin:.1f} m incl. margin)")
    n = max(int(round(points_per_rev * revs)), 3)
    cx, cy = centre
    sgn = 1.0 if clockwise else -1.0
    pts: List[Waypoint] = []
    for i in range(n + 1):
        ang = math.radians(start_deg) + sgn * 2.0 * math.pi * i / points_per_rev
        pts.append((cx + radius_m * math.cos(ang), cy + radius_m * math.sin(ang), z))
    return pts, {"length_m": 2.0 * math.pi * radius_m * revs, "warnings": warns}


def spiral(centre: Tuple[float, float], r_start_m: float, r_end_m: float, z: float, pitch_m: float = 3.0, clockwise: bool = True,
           start_deg: float = 180.0, points_per_rev: int = 16, turn_radius_m: float = 2.5, margin: float = 1.1) -> Tuple[List[Waypoint], Dict[str, object]]:
    """Spiral from r_start to r_end; the radius changes by `pitch_m` per revolution (the gap between neighbouring turns = the swath you cover)."""
    rmin = turn_radius_m * margin
    if min(r_start_m, r_end_m) < rmin:
        raise ValueError(f"spiral: radius {min(r_start_m, r_end_m):.1f} m is smaller than the turning circle ({rmin:.1f} m incl. margin)")
    if pitch_m <= 0:
        raise ValueError("spiral: pitch_m must be positive")
    revs = abs(r_end_m - r_start_m) / pitch_m
    n = max(int(math.ceil(points_per_rev * revs)), 3)
    cx, cy = centre
    sgn = 1.0 if clockwise else -1.0
    pts: List[Waypoint] = []
    for i in range(n + 1):
        f = i / n
        r = r_start_m + (r_end_m - r_start_m) * f
        ang = math.radians(start_deg) + sgn * 2.0 * math.pi * revs * f
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang), z))
    path_len = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
    return pts, {"length_m": path_len, "revs": revs, "warnings": []}


def yoyo(start: Tuple[float, float], end: Tuple[float, float], z_top: float, z_bottom: float, cycles: int = 3,
         z_start: Optional[float] = None) -> Tuple[List[Waypoint], Dict[str, object]]:
    """A straight line from `start` to `end` with the depth swinging between z_top (shallow) and z_bottom (deep): `cycles` full periods, a waypoint at every extreme.
    The depth change per metre is limited by the heave thrust and the fins; check the depth error in the log and lengthen the line if it never reaches the extremes."""
    if cycles < 1 or z_bottom <= z_top:
        raise ValueError("yoyo: need cycles >= 1 and z_bottom > z_top (depth is positive down)")
    n = 2 * cycles
    pts: List[Waypoint] = []
    for i in range(1, n + 1):
        f = i / n
        z = z_bottom if i % 2 == 1 else z_top
        pts.append((start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f, z))
    return pts, {"length_m": math.hypot(end[0] - start[0], end[1] - start[1]), "warnings": []}


def swing_warnings(wps: Sequence[Waypoint], turn_radius_m: float, start_xy: Optional[Tuple[float, float]] = None, geofence: Optional[Dict[str, float]] = None,
                   dock_keepout_m: float = 0.0, dock_xy: Tuple[float, float] = DOCK_XY, min_turn_deg: float = 45.0) -> List[str]:
    """The vehicle cannot turn on the spot: at a corner it sweeps the turning circle on the inside of the turn. For every corner turning more than `min_turn_deg`,
    check that this circle (assuming it arrives along the incoming leg) stays inside the geofence and outside the dock keep-out. A reversal needs a whole circle of room."""
    gf = geofence or {}
    pts = [tuple(w[:2]) for w in wps]
    if start_xy is not None:
        pts = [tuple(start_xy)] + pts
    off = 1 if start_xy is not None else 0
    R = float(turn_radius_m)
    msgs: List[str] = []
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        din = (b[0] - a[0], b[1] - a[1])
        dout = (c[0] - b[0], c[1] - b[1])
        ni, no = math.hypot(*din), math.hypot(*dout)
        if ni < 1e-6 or no < 1e-6:
            continue
        cosang = (din[0] * dout[0] + din[1] * dout[1]) / (ni * no)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        if turn < min_turn_deg:
            continue
        cross = din[0] * dout[1] - din[1] * dout[0]
        side = 1.0 if cross >= 0 else -1.0
        cx, cy = b[0] + R * side * (-din[1] / ni), b[1] + R * side * (din[0] / ni)
        why = []
        for ax, v in (("x", cx), ("y", cy)):
            lo, hi = gf.get(f"{ax}_min"), gf.get(f"{ax}_max")
            if lo is not None and v - R < lo:
                why.append(f"{ax} reaches {v - R:.1f} < geofence {lo}")
            if hi is not None and v + R > hi:
                why.append(f"{ax} reaches {v + R:.1f} > geofence {hi}")
        if dock_keepout_m > 0 and math.hypot(cx - dock_xy[0], cy - dock_xy[1]) - R < dock_keepout_m:
            why.append(f"comes within {math.hypot(cx - dock_xy[0], cy - dock_xy[1]) - R:.1f} m of the dock")
        if why:
            msgs.append(f"waypoint {i - 1 + (1 - off)}: a {turn:.0f} deg turn sweeps a {R:.1f} m circle that " + "; ".join(why))
    return msgs


def check_path(wps: Sequence[Waypoint], turn_radius_m: float, start_xy: Optional[Tuple[float, float]] = None, loop: bool = False,
               geofence: Optional[Dict[str, float]] = None, dock_keepout_m: float = 0.0, dock_xy: Tuple[float, float] = DOCK_XY, tol_m: float = 0.5) -> List[str]:
    """Everything that makes a path unflyable or unsafe: corners inside the turning circle (forgiving `tol_m` = the acceptance radius), points outside the geofence, points too close to the dock."""
    msgs = unreachable_corners(list(wps), turn_radius_m, loop, start_xy, tol_m)
    gf = geofence or {}
    for i, (x, y, z) in enumerate(wps):
        for ax, v, lo, hi in (("x", x, gf.get("x_min"), gf.get("x_max")), ("y", y, gf.get("y_min"), gf.get("y_max")), ("z", z, gf.get("z_min"), gf.get("z_max"))):
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                msgs.append(f"waypoint {i} ({x:.1f}, {y:.1f}, {z:.1f}): {ax} = {v:.1f} is outside the geofence [{lo}, {hi}]")
        if dock_keepout_m > 0 and math.hypot(x - dock_xy[0], y - dock_xy[1]) < dock_keepout_m:
            msgs.append(f"waypoint {i} ({x:.1f}, {y:.1f}): {math.hypot(x - dock_xy[0], y - dock_xy[1]):.1f} m from the dock, inside the {dock_keepout_m:.1f} m keep-out")
    msgs += swing_warnings(wps, turn_radius_m, start_xy, geofence, dock_keepout_m, dock_xy)
    return msgs
