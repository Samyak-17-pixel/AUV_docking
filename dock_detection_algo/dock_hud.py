"""Clean pop-up drawings for the dock detector: pure OpenCV + numpy, no ROS, no windows (so they can be tested and saved as PNG).

  draw_camera_view(bgr, info, hist)    the 'Dock camera' window: banner, the picture with labelled lights, gauges, a top-down mini map and a history strip
  draw_mask_view(mask, info)           the 'Bloom mask' window
  draw_align_view(info, hist)          the separate 'Dock align' window: steering arrows, target, numbers, rolling plots (works from the DockAlign message alone)
  info_from_msg(msg) / info_from_result(res)     the plain dict all of them take
  History(seconds)                     rolling buffer behind the strips and plots

`info` keys: valid, num, status, confidence, err_x_px, err_y_px, err_x_norm, err_y_norm, radius_px, spread_px, lateral_px, obliqueness, elevation_deg, elevation_valid,
aligned, center_exact, w, h, pts (dict top/bottom/left/right -> (x, y) or None), center (x, y) or None, search (yaw, pitch, surge), t (seconds), ring_resid.
Sign conventions are the DockAlign ones: err_x > 0 = the dock is RIGHT of the picture centre (yaw right); err_y > 0 = BELOW the centre (pitch/heave down).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------------------------------------------------------------------- theme (BGR)
BG = (36, 32, 28)
PANEL = (52, 47, 42)
PANEL2 = (64, 58, 52)
EDGE = (88, 80, 72)
TEXT = (238, 234, 230)
DIM = (160, 152, 146)
GOOD = (110, 205, 92)
WARN = (60, 180, 245)
BAD = (80, 80, 235)
INFO = (235, 170, 70)
ACCENT = (210, 200, 70)
LIGHT_COL = {"top": (80, 170, 255), "bottom": (225, 110, 230), "left": (235, 200, 70), "right": (110, 225, 110)}
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_B = cv2.FONT_HERSHEY_SIMPLEX


def _t(img, s, org, scale=0.5, col=TEXT, th=1, font=FONT, anchor="l"):
    (w, h), _ = cv2.getTextSize(s, font, scale, th)
    x, y = int(org[0]), int(org[1])
    if anchor == "c":
        x -= w // 2
    elif anchor == "r":
        x -= w
    cv2.putText(img, s, (x, y), font, scale, col, th, cv2.LINE_AA)
    return w


def _rrect(img, p0, p1, col, r=8, th=-1):
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    r = int(max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if th < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), col, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), col, -1, cv2.LINE_AA)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), col, th, cv2.LINE_AA)


def _alpha(img, p0, p1, col, a):
    x0, y0, x1, y1 = int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
    roi = img[y0:y1, x0:x1]
    if roi.size:
        cv2.addWeighted(np.full_like(roi, col), a, roi, 1.0 - a, 0, roi)


def _arrow(img, p0, p1, col, th=3, tip=0.35):
    cv2.arrowedLine(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, th, cv2.LINE_AA, tipLength=tip)


def _bar(img, x, y, w, h, frac, col, bg=PANEL2, signed=False):
    """Horizontal bar. frac in 0..1 (or -1..1 when signed: grows from the middle)."""
    _rrect(img, (x, y), (x + w, y + h), bg, r=h // 2)
    if signed:
        f = max(-1.0, min(1.0, frac))
        mid = x + w // 2
        x1 = mid + int(f * (w // 2 - 2))
        a, b = (mid, x1) if x1 >= mid else (x1, mid)
        if b - a > 1:
            _rrect(img, (a, y + 2), (b, y + h - 2), col, r=max(1, h // 2 - 2))
        cv2.line(img, (mid, y - 2), (mid, y + h + 2), TEXT, 1, cv2.LINE_AA)
    else:
        f = max(0.0, min(1.0, frac))
        if f > 0.01:
            _rrect(img, (x + 2, y + 2), (x + 2 + int(f * (w - 4)), y + h - 2), col, r=max(1, h // 2 - 2))


# ------------------------------------------------------------------------------------------------------------------------------- data
def _f(m, name, default=0.0):
    return float(getattr(m, name, default))


def info_from_msg(m, vfov_deg: float = 60.0, t: float = 0.0, ring_resid: float = float("nan")) -> Dict:
    """DockAlign message (or any object with the same fields) -> the plain dict the drawings take."""
    w, h = int(getattr(m, "image_width", 640) or 640), int(getattr(m, "image_height", 480) or 480)
    pts = {}
    for n in ("top", "bottom", "left", "right"):
        p = getattr(m, n, None)
        pts[n] = (float(p.x), float(p.y)) if (p is not None and bool(getattr(m, "valid", False))) else None
    c = getattr(m, "center", None)
    center = (float(c.x), float(c.y)) if (c is not None and bool(getattr(m, "valid", False))) else None
    return {"valid": bool(m.valid), "num": int(m.num_lights), "status": str(getattr(m, "status", "")), "confidence": _f(m, "confidence"),
            "err_x_px": _f(m, "error_x_px"), "err_y_px": _f(m, "error_y_px"), "err_x_norm": _f(m, "error_x_norm"), "err_y_norm": _f(m, "error_y_norm"),
            "radius_px": _f(m, "radius_px"), "spread_px": _f(m, "spread_px"), "lateral_px": _f(m, "lateral_px"), "obliqueness": _f(m, "obliqueness"),
            "elevation_deg": math.degrees(_f(m, "elevation_rad")), "elevation_valid": bool(getattr(m, "elevation_valid", False)),
            "aligned": bool(getattr(m, "aligned", False)), "center_exact": bool(getattr(m, "center_exact", False)), "w": w, "h": h, "pts": pts, "center": center,
            "search": (_f(m, "search_yaw_norm"), _f(m, "search_pitch_norm"), _f(m, "search_surge_norm")), "t": float(t),
            "vfov": float(vfov_deg), "ring_resid": float(ring_resid)}


def info_from_result(res, vfov_deg: float = 60.0, t: float = 0.0) -> Dict:
    return info_from_msg(res.msg, vfov_deg, t, getattr(res, "ring_residual", float("nan")))


def focal_px(info: Dict) -> float:
    return 0.5 * info["h"] / math.tan(0.5 * math.radians(info["vfov"]))


def range_m(info: Dict) -> Optional[float]:
    """Quick range from the ring size: f * 1 m / radius_px (assumes a square-on view; an oblique view reads a little long)."""
    if not info["valid"] or info["radius_px"] < 3.0:
        return None
    return focal_px(info) * 1.0 / info["radius_px"]


def bearing_deg(info: Dict) -> Optional[float]:
    return math.degrees(math.atan2(info["err_x_px"], focal_px(info))) if info["valid"] else None


def view_angle_deg(info: Dict) -> Optional[float]:
    """How obliquely the ring is seen: the side lights' spread against the ring height gives cos(angle). Sign is not observable from one frame."""
    p = info["pts"]
    if not info["valid"] or info["radius_px"] < 3.0 or p["left"] is None or p["right"] is None:
        return None
    k = (p["right"][0] - p["left"][0]) / (1.4142 * info["radius_px"])
    return math.degrees(math.acos(max(0.0, min(1.0, k))))


class History:
    """Rolling samples of the numbers the strips and plots show."""

    KEYS = ("err_x_px", "err_y_px", "radius_px", "confidence", "valid", "num", "range", "bearing")

    def __init__(self, seconds: float = 12.0) -> None:
        self.seconds = float(seconds)
        self.d: Deque[Tuple[float, Dict[str, float]]] = deque()

    def add(self, info: Dict) -> None:
        r = range_m(info)
        b = bearing_deg(info)
        row = {"err_x_px": info["err_x_px"] if info["valid"] else float("nan"), "err_y_px": info["err_y_px"] if info["valid"] else float("nan"),
               "radius_px": info["radius_px"] if info["valid"] else float("nan"), "confidence": info["confidence"], "valid": float(info["valid"]),
               "num": float(info["num"]), "range": r if r is not None else float("nan"), "bearing": b if b is not None else float("nan")}
        self.d.append((info["t"], row))
        while self.d and info["t"] - self.d[0][0] > self.seconds:
            self.d.popleft()

    def series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        if not self.d:
            return np.zeros(0), np.zeros(0)
        t = np.array([a for a, _ in self.d])
        return t - t[-1], np.array([r[key] for _, r in self.d])


# ------------------------------------------------------------------------------------------------------------------------------- pieces
def _banner(img, info, x0, y0, x1, y1, title="DOCK DETECTOR"):
    if info["valid"] and info["aligned"]:
        col, word = GOOD, "LOCKED  -  ALIGNED"
    elif info["valid"]:
        col, word = WARN, "LOCKED  -  off centre"
    elif info["num"] > 0:
        col, word = INFO, f"SEARCHING  -  {info['num']} of 4 lights"
    else:
        col, word = BAD, "NO LIGHTS IN VIEW"
    _rrect(img, (x0, y0), (x1, y1), PANEL, r=10)
    cv2.circle(img, (x0 + 26, (y0 + y1) // 2), 10, col, -1, cv2.LINE_AA)
    _t(img, word, (x0 + 48, (y0 + y1) // 2 + 7), 0.7, col, 1)
    _t(img, title, (x0 + 330, (y0 + y1) // 2 + 6), 0.5, DIM)
    if info["valid"]:
        conf = info["confidence"]
        _t(img, f"{info['num']} lights   conf {conf:.2f}", (x1 - 124, (y0 + y1) // 2 + 6), 0.55, TEXT, anchor="r")
        _bar(img, x1 - 112, y0 + 15, 100, 14, conf, GOOD if conf > 0.7 else (WARN if conf > 0.4 else BAD))
    else:
        _t(img, info["status"][:34] or "-", (x1 - 12, (y0 + y1) // 2 + 6), 0.5, DIM, anchor="r")


def _gauge(img, x, y, w, label, value, unit, frac, col, signed=True, fmt="{:+.1f}"):
    _t(img, label, (x, y + 12), 0.42, DIM)
    s = "--" if value is None else fmt.format(value) + unit
    _t(img, s, (x + w, y + 14), 0.62, TEXT if value is not None else DIM, anchor="r")
    _bar(img, x, y + 22, w, 12, 0.0 if value is None else frac, col, signed=signed)


def _top_down(img, x0, y0, x1, y1, info):
    """Top-down mini map: the vehicle at the bottom looking up, the dock placed at the measured range and bearing, the camera's field of view as a wedge."""
    _rrect(img, (x0, y0), (x1, y1), PANEL, r=10)
    _t(img, "TOP-DOWN (vehicle -> dock)", (x0 + 12, y0 + 20), 0.42, DIM)
    cx, by = (x0 + x1) // 2, y1 - 18
    h = by - (y0 + 34)
    scale = h / 12.0                                           # pixels per metre: 12 m to the top
    hf = math.atan(math.tan(math.radians(info["vfov"] / 2.0)) * info["w"] / info["h"])
    for r in (2, 5, 10):
        cv2.ellipse(img, (cx, by), (int(r * scale), int(r * scale)), 0, -90 - math.degrees(hf) - 25, -90 + math.degrees(hf) + 25, EDGE, 1, cv2.LINE_AA)
        _t(img, f"{r} m", (cx + 4, by - int(r * scale) - 2), 0.34, DIM)
    ov = img.copy()
    pts = np.array([[cx, by], [cx + int(12 * scale * math.sin(hf)), by - int(12 * scale * math.cos(hf))], [cx - int(12 * scale * math.sin(hf)), by - int(12 * scale * math.cos(hf))]], np.int32)
    cv2.fillPoly(ov, [pts], (90, 80, 50))
    cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
    tri = np.array([[cx, by - 12], [cx - 8, by + 6], [cx + 8, by + 6]], np.int32)
    cv2.fillPoly(img, [tri], ACCENT)
    r, b = range_m(info), bearing_deg(info)
    if r is not None and b is not None:
        px = cx + int(r * math.sin(math.radians(b)) * scale)
        py = by - int(r * math.cos(math.radians(b)) * scale)
        py = max(y0 + 36, py)
        cv2.line(img, (cx, by - 12), (px, py), DIM, 1, cv2.LINE_AA)
        va = view_angle_deg(info)
        sq = math.cos(math.radians(va)) if va is not None else 1.0
        ew, eh = 4 + int(14 * sq), 14
        cv2.ellipse(img, (px, py), (ew, eh), 0, 0, 360, GOOD if info["aligned"] else WARN, 2, cv2.LINE_AA)       # the ring seen edge-on gets thinner
        cv2.circle(img, (px, py), 3, TEXT, -1, cv2.LINE_AA)
        _t(img, f"{r:.1f} m", (px + 18, py + 4), 0.45, TEXT)
    else:
        _t(img, "dock not located", (cx, (y0 + y1) // 2), 0.5, DIM, anchor="c")


def _history_strip(img, x0, y0, x1, y1, hist: History, which=(("err_x_px", INFO, "err x"), ("err_y_px", WARN, "err y")), second=("confidence", GOOD, "conf")):
    _rrect(img, (x0, y0), (x1, y1), PANEL, r=10)
    _t(img, f"LAST {hist.seconds:.0f} s", (x0 + 12, y0 + 18), 0.42, DIM)
    px0, px1, py0, py1 = x0 + 12, x1 - 12, y0 + 26, y1 - 10
    cv2.rectangle(img, (px0, py0), (px1, py1), PANEL2, 1)
    if not hist.d:
        return
    _plot(img, (px0, py0, px1, py1), hist, [(k, c) for k, c, _ in which], hist.seconds, symmetric=True)
    lx = px0 + 90
    for k, c, name in which:
        cv2.line(img, (lx, y0 + 14), (lx + 16, y0 + 14), c, 3, cv2.LINE_AA)
        _t(img, name, (lx + 22, y0 + 18), 0.4, TEXT)
        lx += 90
    k, c, name = second
    t, v = hist.series(k)
    if len(t) > 1:
        pts = [(int(px1 + (ti / hist.seconds) * (px1 - px0)), int(py1 - float(np.clip(vi, 0, 1)) * (py1 - py0))) for ti, vi in zip(t, v) if not math.isnan(vi)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, c, 1, cv2.LINE_AA)
        cv2.line(img, (lx, y0 + 14), (lx + 16, y0 + 14), c, 1, cv2.LINE_AA)
        _t(img, name + " (0..1)", (lx + 22, y0 + 18), 0.4, TEXT)


def _plot(img, rect, hist: History, series, seconds, symmetric=True, ylim=None):
    px0, py0, px1, py1 = rect
    vals = []
    for k, _ in series:
        t, v = hist.series(k)
        vals.extend([x for x in v if not math.isnan(x)])
    if ylim is not None:
        lo, hi = ylim
    else:
        m = max([abs(x) for x in vals] + [20.0]) if symmetric else max(vals + [1.0])
        lo, hi = (-m, m) if symmetric else (0.0, m)
    if symmetric:
        zy = int(py1 - (0 - lo) / (hi - lo) * (py1 - py0))
        cv2.line(img, (px0, zy), (px1, zy), EDGE, 1, cv2.LINE_AA)
    for k, c in series:
        t, v = hist.series(k)
        pts = [(int(px1 + (ti / seconds) * (px1 - px0)), int(py1 - (vi - lo) / max(hi - lo, 1e-6) * (py1 - py0))) for ti, vi in zip(t, v) if not math.isnan(vi)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, c, 2, cv2.LINE_AA)
    _t(img, f"{hi:+.0f}" if symmetric else f"{hi:.0f}", (px0 + 3, py0 + 12), 0.34, DIM)
    _t(img, f"{lo:+.0f}" if symmetric else f"{lo:.0f}", (px0 + 3, py1 - 3), 0.34, DIM)


def info_from_align_dict(al: Dict[str, float], w: int = 640, h: int = 480, vfov_deg: float = 60.0, t: float = 0.0) -> Dict:
    """The viewer's telemetry `dock_align` dict (ros_link / replay) -> the plain dict the drawings take. Missing keys fall back to neutral values."""
    g = lambda k, d=0.0: float(al.get(k, d))
    valid = bool(g("valid"))
    pts = {}
    for n in ("top", "bottom", "left", "right"):
        x, y = g(f"{n}_x"), g(f"{n}_y")
        pts[n] = (x, y) if valid and (x > 0 or y > 0) else None
    cx, cy = g("center_x"), g("center_y")
    return {"valid": valid, "num": int(g("num_lights")), "status": ("ok" if g("aligned") else "ok_not_aligned") if valid else "search", "confidence": g("confidence"),
            "err_x_px": g("error_x_px"), "err_y_px": g("error_y_px"), "err_x_norm": g("error_x_norm", g("error_x_px") / (0.5 * w)), "err_y_norm": g("error_y_norm", g("error_y_px") / (0.5 * h)),
            "radius_px": g("radius_px"), "spread_px": g("spread_px"), "lateral_px": g("lateral_px"), "obliqueness": g("obliqueness"), "elevation_deg": g("elevation_deg"),
            "elevation_valid": valid, "aligned": bool(g("aligned")), "center_exact": bool(g("center_exact")), "w": int(w), "h": int(h), "pts": pts,
            "center": (cx, cy) if valid and (cx > 0 or cy > 0) else None, "search": (g("search_yaw_norm"), g("search_pitch_norm"), g("search_surge_norm")), "t": float(t),
            "vfov": float(vfov_deg), "ring_resid": g("ring_residual", float("nan"))}


def draw_compact_view(bgr: Optional[np.ndarray], info: Dict, size: Tuple[int, int] = (1000, 400)) -> np.ndarray:
    """One compact dashboard for the viewer's Detection tab (fits under the 3D view): banner, the picture with labelled lights, gauges and the steering lamps, top-down map."""
    W, H = size
    out = np.full((H, W, 3), BG, np.uint8)
    _banner(out, info, 6, 4, W - 6, 44)
    ph = H - 56
    pw = int(ph * info["w"] / info["h"])
    x0, y0 = 6, 50
    if bgr is None:
        _rrect(out, (x0, y0), (x0 + pw, y0 + ph), PANEL, r=8)
        _t(out, "waiting for camera frames ...", (x0 + pw // 2, y0 + ph // 2), 0.6, DIM, anchor="c")
    else:
        out[y0:y0 + ph, x0:x0 + pw] = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        sx, sy = pw / info["w"], ph / info["h"]
        mp = lambda p: (int(x0 + p[0] * sx), int(y0 + p[1] * sy))
        ccx, ccy = x0 + pw // 2, y0 + ph // 2
        cv2.line(out, (ccx - 10, ccy), (ccx + 10, ccy), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(out, (ccx, ccy - 10), (ccx, ccy + 10), (150, 150, 150), 1, cv2.LINE_AA)
        p = info["pts"]
        if info["valid"] and all(p[k] is not None for k in p):
            cv2.polylines(out, [np.array([mp(p["top"]), mp(p["right"]), mp(p["bottom"]), mp(p["left"])], np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)
            for name in ("top", "bottom", "left", "right"):
                q = mp(p[name])
                cv2.circle(out, q, 11, LIGHT_COL[name], 2, cv2.LINE_AA)
                off = {"top": (-5, -15), "bottom": (-5, 24), "left": (-22, 4), "right": (14, 4)}[name]
                _t(out, name[0].upper(), (q[0] + off[0], q[1] + off[1]), 0.45, LIGHT_COL[name])
            if info["center"] is not None:
                c = mp(info["center"])
                cv2.drawMarker(out, c, ACCENT, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    # gauges
    gx, gw = x0 + pw + 12, 250
    r, b, v = range_m(info), bearing_deg(info), view_angle_deg(info)
    _rrect(out, (gx - 4, y0), (gx + gw + 4, y0 + ph), PANEL, r=8)
    _t(out, "RANGE", (gx + 6, y0 + 20), 0.4, DIM)
    wn = _t(out, "--" if r is None else f"{r:.1f}", (gx + 6, y0 + 66), 1.5, TEXT if r is not None else DIM, 2)
    if r is not None:
        _t(out, "m", (gx + 14 + wn, y0 + 66), 0.6, DIM)
    el = info["elevation_deg"] if info["elevation_valid"] and info["valid"] else None
    _gauge(out, gx + 6, y0 + 80, gw - 12, "BEARING (+ right)", b, " deg", (b or 0.0) / 37.0, INFO if b is not None and abs(b) > 3 else GOOD)
    _gauge(out, gx + 6, y0 + 126, gw - 12, "ELEV (+ below)", el, " deg", (el or 0.0) / 30.0, WARN if el is not None and abs(el) > 3 else GOOD)
    _gauge(out, gx + 6, y0 + 172, gw - 12, "VIEW ANGLE", v, " deg", (v or 0.0) / 80.0, WARN if v is not None and v > 40 else GOOD, signed=False, fmt="{:.0f}")
    # steering lamps + top-down map
    rx = gx + gw + 14
    rw = W - 6 - rx
    dead = 10.0
    right = info["valid"] and info["err_x_px"] > dead
    left = info["valid"] and info["err_x_px"] < -dead
    down = info["valid"] and info["err_y_px"] > dead
    up = info["valid"] and info["err_y_px"] < -dead
    _rrect(out, (rx, y0), (rx + rw, y0 + 118), PANEL, r=8)
    for i, (lab, on) in enumerate((("YAW L", left), ("YAW R", right), ("UP", up), ("DOWN", down))):
        bx, by = rx + 8 + (i % 2) * (rw // 2 - 2), y0 + 10 + (i // 2) * 34
        _rrect(out, (bx, by), (bx + rw // 2 - 14, by + 26), WARN if on else PANEL2, r=6)
        _t(out, lab, (bx + (rw // 2 - 14) // 2, by + 19), 0.5, (20, 20, 20) if on else DIM, anchor="c")
    ok = info["valid"] and not (left or right or up or down)
    _t(out, "CENTRED" if ok else ("no lock" if not info["valid"] else "correct"), (rx + rw // 2, y0 + 100), 0.55, GOOD if ok else (INFO if not info["valid"] else WARN), anchor="c")
    _top_down(out, rx, y0 + 126, rx + rw, y0 + ph, info)
    return out


# ------------------------------------------------------------------------------------------------------------------------------- windows
def draw_camera_view(bgr: Optional[np.ndarray], info: Dict, hist: Optional[History] = None, size: Tuple[int, int] = (1120, 736)) -> np.ndarray:
    """The 'Dock camera' window. bgr = the (roll-levelled) camera frame the cores refer to; None draws a waiting panel."""
    W, H = size
    out = np.full((H, W, 3), BG, np.uint8)
    _banner(out, info, 10, 8, W - 10, 52)
    # picture
    ph = 540
    pw = int(ph * info["w"] / info["h"])
    x0, y0 = 10, 62
    if bgr is None:
        _rrect(out, (x0, y0), (x0 + pw, y0 + ph), PANEL, r=10)
        _t(out, "waiting for camera frames ...", (x0 + pw // 2, y0 + ph // 2), 0.8, DIM, anchor="c")
    else:
        pic = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        out[y0:y0 + ph, x0:x0 + pw] = pic
        sx, sy = pw / info["w"], ph / info["h"]
        mp = lambda p: (int(x0 + p[0] * sx), int(y0 + p[1] * sy))
        ccx, ccy = x0 + pw // 2, y0 + ph // 2
        cv2.line(out, (ccx - 14, ccy), (ccx + 14, ccy), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(out, (ccx, ccy - 14), (ccx, ccy + 14), (150, 150, 150), 1, cv2.LINE_AA)
        p = info["pts"]
        if info["valid"] and all(p[k] is not None for k in p):
            ring = np.array([mp(p["top"]), mp(p["right"]), mp(p["bottom"]), mp(p["left"])], np.int32)
            cv2.polylines(out, [ring], True, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(out, mp(p["top"]), mp(p["bottom"]), (120, 120, 120), 1, cv2.LINE_AA)
            for name in ("top", "bottom", "left", "right"):
                q = mp(p[name])
                col = LIGHT_COL[name]
                cv2.circle(out, q, 15, col, 2, cv2.LINE_AA)
                cv2.circle(out, q, 3, col, -1, cv2.LINE_AA)
                tag = name[0].upper()
                off = {"top": (-6, -22), "bottom": (-6, 34), "left": (-34, 5), "right": (20, 5)}[name]
                _rrect(out, (q[0] + off[0] - 4, q[1] + off[1] - 15), (q[0] + off[0] + 16, q[1] + off[1] + 5), (0, 0, 0), r=4)
                _t(out, tag, (q[0] + off[0], q[1] + off[1]), 0.5, col)
            if info["center"] is not None:
                c = mp(info["center"])
                cv2.drawMarker(out, c, ACCENT, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
                if abs(c[0] - ccx) + abs(c[1] - ccy) > 6:
                    _arrow(out, (ccx, ccy), c, ACCENT, 2, 0.12)
        elif info["num"] > 0:
            _t(out, "ring not complete - search commands active", (x0 + 14, y0 + ph - 14), 0.55, INFO)
        _t(out, f"{info['w']}x{info['h']}", (x0 + pw - 8, y0 + 18), 0.4, TEXT, anchor="r")
    # sidebar
    sx0, sx1 = x0 + pw + 12, W - 10
    sw = sx1 - sx0
    _rrect(out, (sx0, y0), (sx1, y0 + 258), PANEL, r=10)
    r, b, v = range_m(info), bearing_deg(info), view_angle_deg(info)
    _t(out, "RANGE", (sx0 + 14, y0 + 24), 0.42, DIM)
    wnum = _t(out, "--" if r is None else f"{r:.1f}", (sx0 + 14, y0 + 78), 1.6, TEXT if r is not None else DIM, 2)
    if r is not None:
        _t(out, "m", (sx0 + 24 + wnum, y0 + 78), 0.7, DIM)
    lim = 37.0
    _gauge(out, sx0 + 14, y0 + 96, sw - 28, "BEARING  (+ = to the right)", b, " deg", (b or 0.0) / lim, INFO if b is not None and abs(b) > 3 else GOOD)
    el = info["elevation_deg"] if info["elevation_valid"] and info["valid"] else None
    _gauge(out, sx0 + 14, y0 + 146, sw - 28, "ELEVATION  (+ = below horizon)", el, " deg", (el or 0.0) / 30.0, WARN if el is not None and abs(el) > 3 else GOOD)
    _gauge(out, sx0 + 14, y0 + 196, sw - 28, "VIEW ANGLE  (ring seen obliquely)", v, " deg", (v or 0.0) / 80.0, WARN if v is not None and v > 40 else GOOD, signed=False, fmt="{:.0f}")
    _top_down(out, sx0, y0 + 268, sx1, y0 + ph, info)
    if hist is not None:
        _history_strip(out, 10, y0 + ph + 10, W - 10, H - 8, hist)
    return out


def draw_mask_view(mask: Optional[np.ndarray], info: Dict, size: Tuple[int, int] = (960, 600)) -> np.ndarray:
    """The 'Bloom mask' window: the binary mask in colour, the chosen lights marked, a legend."""
    W, H = size
    out = np.full((H, W, 3), BG, np.uint8)
    _rrect(out, (10, 8), (W - 10, 48), PANEL, r=10)
    _t(out, "BLOOM MASK  -  what the detector counts as light", (26, 34), 0.62, TEXT)
    if mask is None:
        _t(out, "waiting for camera frames ...", (W // 2, H // 2), 0.8, DIM, anchor="c")
        return out
    ph = H - 66
    pw = int(ph * mask.shape[1] / mask.shape[0])
    m = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
    col = np.zeros((ph, pw, 3), np.uint8)
    col[m > 0] = (150, 120, 60)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    col[edge > 0] = (235, 205, 120)
    x0, y0 = 10, 58
    out[y0:y0 + ph, x0:x0 + pw] = col
    sx, sy = pw / info["w"], ph / info["h"]
    for name, p in info["pts"].items():
        if p is not None:
            q = (int(x0 + p[0] * sx), int(y0 + p[1] * sy))
            cv2.circle(out, q, 12, LIGHT_COL[name], 2, cv2.LINE_AA)
            _t(out, name[0].upper(), (q[0] + 14, q[1] + 5), 0.5, LIGHT_COL[name])
    lx = x0 + pw + 16
    _rrect(out, (lx, y0), (W - 10, y0 + 190), PANEL, r=10)
    _t(out, "LEGEND", (lx + 12, y0 + 22), 0.42, DIM)
    for i, (name, c) in enumerate(LIGHT_COL.items()):
        cv2.circle(out, (lx + 22, y0 + 46 + 26 * i), 8, c, 2, cv2.LINE_AA)
        _t(out, name, (lx + 40, y0 + 52 + 26 * i), 0.5, TEXT)
    _t(out, f"lights {info['num']}  conf {info['confidence']:.2f}", (lx + 12, y0 + 170), 0.5, TEXT)
    if not math.isnan(info.get("ring_resid", float("nan"))):
        _t(out, f"ring fit {info['ring_resid']:.3f}", (lx + 12, y0 + 188 - 2), 0.4, DIM)
    return out


def draw_align_view(info: Dict, hist: Optional[History] = None, age_s: Optional[float] = None, size: Tuple[int, int] = (1000, 680)) -> np.ndarray:
    """The separate 'Dock align' window: what the DockAlign message says, as steering hints. Works from the message alone (no camera frame)."""
    W, H = size
    out = np.full((H, W, 3), BG, np.uint8)
    _banner(out, info, 10, 8, W - 10, 52, "DOCK ALIGN  -  /dock_align")
    # target
    tx0, ty0, ts = 10, 62, 400
    _rrect(out, (tx0, ty0), (tx0 + ts, ty0 + ts), PANEL, r=12)
    cx, cy = tx0 + ts // 2, ty0 + ts // 2
    half = ts // 2 - 34
    for f in (0.25, 0.5, 1.0):
        cv2.rectangle(out, (cx - int(half * f), cy - int(half * f)), (cx + int(half * f), cy + int(half * f)), EDGE, 1, cv2.LINE_AA)
    cv2.line(out, (cx - half, cy), (cx + half, cy), EDGE, 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy - half), (cx, cy + half), EDGE, 1, cv2.LINE_AA)
    ex, ey = (info["err_x_norm"], info["err_y_norm"]) if info["valid"] else (0.0, 0.0)
    dead = 10.0
    ok_x = info["valid"] and abs(info["err_x_px"]) <= dead
    ok_y = info["valid"] and abs(info["err_y_px"]) <= dead
    if info["valid"]:
        px = int(cx + max(-1.1, min(1.1, ex)) * half)
        py = int(cy + max(-1.1, min(1.1, ey)) * half)
        col = GOOD if (ok_x and ok_y) else WARN
        _arrow(out, (cx, cy), (px, py), col, 3, 0.15) if (abs(px - cx) + abs(py - cy)) > 8 else None
        cv2.circle(out, (px, py), 13, col, 3, cv2.LINE_AA)
        cv2.circle(out, (px, py), 3, col, -1, cv2.LINE_AA)
        _t(out, "dock", (px + 18, py + 5), 0.45, col)
    else:
        _t(out, "no valid lock", (cx, cy + 6), 0.7, DIM, anchor="c")
    _t(out, "picture centre = +", (tx0 + 14, ty0 + ts - 12), 0.4, DIM)
    # direction lamps
    def lamp(x, y, label, on, col, big=False):
        _rrect(out, (x - 110, y - 16), (x + 110, y + 16), col if on else PANEL2, r=8)
        _t(out, label, (x, y + 6), 0.55, (20, 20, 20) if on else DIM, anchor="c")
    right = info["valid"] and info["err_x_px"] > dead
    left = info["valid"] and info["err_x_px"] < -dead
    down = info["valid"] and info["err_y_px"] > dead
    up = info["valid"] and info["err_y_px"] < -dead
    lamp(cx, ty0 + 22, "UP  -  pitch / heave up", up, WARN)
    lamp(cx, ty0 + ts - 40, "DOWN  -  pitch / heave down", down, WARN)
    _rrect(out, (tx0 + 6, cy - 16), (tx0 + 86, cy + 16), WARN if left else PANEL2, r=8)
    _t(out, "< YAW L", (tx0 + 46, cy + 6), 0.5, (20, 20, 20) if left else DIM, anchor="c")
    _rrect(out, (tx0 + ts - 86, cy - 16), (tx0 + ts - 6, cy + 16), WARN if right else PANEL2, r=8)
    _t(out, "YAW R >", (tx0 + ts - 46, cy + 6), 0.5, (20, 20, 20) if right else DIM, anchor="c")
    # command card
    if info["valid"]:
        words = []
        words.append("yaw RIGHT" if right else ("yaw LEFT" if left else "heading OK"))
        words.append("pitch/heave DOWN" if down else ("pitch/heave UP" if up else "depth OK"))
        cmd = "  +  ".join(words)
        ccol = GOOD if (ok_x and ok_y) else WARN
    else:
        ys, ps, ss = info["search"]
        cmd = "search: " + (f"yaw {'right' if ys > 0.05 else 'left' if ys < -0.05 else '-'}  pitch {'down' if ps > 0.05 else 'up' if ps < -0.05 else '-'}  " +
                            ("reverse" if ss < -0.05 else "forward" if ss > 0.05 else "hold"))
        ccol = INFO
    _rrect(out, (tx0, ty0 + ts + 10), (tx0 + ts, ty0 + ts + 56), PANEL, r=10)
    _t(out, cmd[:44], (tx0 + ts // 2, ty0 + ts + 40), 0.62, ccol, anchor="c")
    # numbers
    nx0 = tx0 + ts + 14
    nw = W - 10 - nx0
    _rrect(out, (nx0, 62), (W - 10, 62 + 400), PANEL, r=12)
    r, b, v = range_m(info), bearing_deg(info), view_angle_deg(info)
    rows = [("range (f / radius)", None if r is None else f"{r:.2f} m"), ("bearing", None if b is None else f"{b:+.1f} deg"),
            ("elevation", f"{info['elevation_deg']:+.1f} deg" if info["elevation_valid"] and info["valid"] else None),
            ("err x / err y", f"{info['err_x_px']:+.0f} / {info['err_y_px']:+.0f} px" if info["valid"] else None),
            ("ring radius", f"{info['radius_px']:.0f} px" if info["valid"] else None), ("spread", f"{info['spread_px']:.1f} px" if info["valid"] else None),
            ("lateral cue", f"{info['lateral_px']:+.1f} px" if info["valid"] else None), ("obliqueness", f"{info['obliqueness']:+.3f}" if info["valid"] else None),
            ("view angle", None if v is None else f"{v:.0f} deg"), ("lights / conf", f"{info['num']} / {info['confidence']:.2f}"),
            ("message age", "--" if age_s is None else f"{age_s * 1000:.0f} ms")]
    for i, (k, s) in enumerate(rows):
        yy = 92 + i * 33
        _t(out, k, (nx0 + 16, yy), 0.5, DIM)
        _t(out, "--" if s is None else s, (W - 26, yy), 0.62, TEXT if s is not None else DIM, anchor="r")
        cv2.line(out, (nx0 + 12, yy + 10), (W - 22, yy + 10), PANEL2, 1)
    # lamps row
    ly = 62 + 400 - 22
    for i, (name, on, col) in enumerate((("VALID", info["valid"], GOOD), ("ALIGNED", info["aligned"] and info["valid"], GOOD), ("EXACT CENTRE", info["center_exact"] and info["valid"], INFO))):
        xx = nx0 + 20 + i * 120
        cv2.circle(out, (xx, ly), 7, col if on else PANEL2, -1, cv2.LINE_AA)
        _t(out, name, (xx + 12, ly + 5), 0.4, TEXT if on else DIM)
    # search hint bars
    sy0 = 62 + 400 + 10
    _rrect(out, (nx0, sy0), (W - 10, sy0 + 46), PANEL, r=10)
    ys, ps, ss = info["search"]
    labs = (("yaw", ys), ("pitch", ps), ("surge", ss))
    bw = (nw - 40) // 3
    for i, (n, vv) in enumerate(labs):
        bx = nx0 + 14 + i * (bw + 6)
        _t(out, f"search {n}", (bx, sy0 + 15), 0.38, DIM)
        _bar(out, bx, sy0 + 22, bw - 8, 12, vv if not info["valid"] else 0.0, INFO, signed=True)
    # plots
    if hist is not None:
        py0 = 62 + 400 + 66
        _rrect(out, (10, py0), (W - 10, H - 8), PANEL, r=10)
        _t(out, f"LAST {hist.seconds:.0f} s", (24, py0 + 18), 0.42, DIM)
        half_w = (W - 40) // 2
        a = (22, py0 + 28, 22 + half_w - 10, H - 18)
        cv2.rectangle(out, (a[0], a[1]), (a[2], a[3]), PANEL2, 1)
        _plot(out, a, hist, [("err_x_px", INFO), ("err_y_px", WARN)], hist.seconds, True)
        _t(out, "err x (blue)   err y (amber)   px", (a[0] + 110, py0 + 18), 0.4, TEXT)
        b2 = (W // 2 + 4, py0 + 28, W - 24, H - 18)
        cv2.rectangle(out, (b2[0], b2[1]), (b2[2], b2[3]), PANEL2, 1)
        _plot(out, b2, hist, [("range", GOOD)], hist.seconds, False)
        _t(out, "range [m]", (b2[0] + 20, py0 + 18), 0.4, TEXT)
    return out
