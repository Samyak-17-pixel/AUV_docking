"""Heads-up display drawn over the 3D view, a top-view minimap, and the detection overlay for the camera pane. Qt only (QPainter), no VTK, no ROS."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui

GREEN = QtGui.QColor(110, 255, 160)
AMBER = QtGui.QColor(255, 200, 80)
RED = QtGui.QColor(255, 100, 90)
WHITE = QtGui.QColor(235, 245, 255)
BLUE = QtGui.QColor(120, 190, 255)
PANEL = QtGui.QColor(5, 20, 35, 150)


def hud_info(sample, dock_pos: np.ndarray, ctrl: Dict[str, object], align: Dict[str, float], cmd: Dict[str, float]) -> Optional[dict]:
    """Collect what the HUD prints from the latest telemetry sample (None if there is no data yet)."""
    if sample is None:
        return None
    pos, eul, nu = sample.pos, sample.eul, sample.nu
    d = dock_pos - pos
    horiz = math.hypot(d[0], d[1])
    bearing = math.degrees(math.atan2(d[1], d[0]) - eul[2])
    bearing = (bearing + 180.0) % 360.0 - 180.0
    return {
        "depth": float(pos[2]), "heading": math.degrees(eul[2]) % 360.0, "pitch": math.degrees(eul[1]), "roll": math.degrees(eul[0]),
        "speed": float(nu[0]), "dock_range": float(horiz), "dock_bearing": float(bearing), "dock_dz": float(d[2]),
        "mode": str(ctrl.get("mode", "")), "ctrl": str(ctrl.get("ctrl", "")), "retries": ctrl.get("retries"),
        "lights": int(align.get("num_lights", 0)) if align else 0, "valid": bool(align.get("valid", 0)) if align else False,
    }


def draw_hud(p: QtGui.QPainter, rect: QtCore.QRect, info: Optional[dict]) -> None:
    if not info:
        return
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    f = QtGui.QFont("DejaVu Sans Mono", 9)
    f.setBold(True)
    p.setFont(f)
    # --- left-top instrument block
    lines = [f"DEPTH   {info['depth']:5.2f} m", f"HEADING {info['heading']:5.1f} deg", f"SPEED   {info['speed']:5.2f} m/s", f"PITCH   {info['pitch']:5.1f}  ROLL {info['roll']:4.1f}"]
    box = QtCore.QRect(rect.left() + 10, rect.top() + 10, 210, 18 * len(lines) + 10)
    p.fillRect(box, PANEL)
    p.setPen(WHITE)
    for i, t in enumerate(lines):
        p.drawText(box.left() + 8, box.top() + 20 + 18 * i, t)
    # --- dock block
    dock = [f"DOCK    {info['dock_range']:5.2f} m ahead", f"BEARING {info['dock_bearing']:+5.1f} deg", f"LIGHTS  {info['lights']}  " + ("VALID" if info["valid"] else "no lock")]
    box2 = QtCore.QRect(box.left(), box.bottom() + 8, 210, 18 * len(dock) + 10)
    p.fillRect(box2, PANEL)
    for i, t in enumerate(dock):
        p.setPen(GREEN if (i == 2 and info["valid"]) else (AMBER if i == 2 else WHITE))
        p.drawText(box2.left() + 8, box2.top() + 20 + 18 * i, t)
    # --- controller / phase banner, centred at the top
    if info.get("mode") or info.get("ctrl"):
        phase = info.get("mode") or ""
        col = {"DOCKED": GREEN, "TERMINAL": GREEN, "APPROACH": WHITE, "SEARCH": BLUE, "RETRY": AMBER, "SAFE_STOP": RED, "ABORT": RED}.get(phase, WHITE)
        txt = f"{info.get('ctrl', '')}  {phase}" + (f"   retries {info['retries']}" if info.get("retries") else "") + (f"   back-outs {info['repositions']}" if info.get("repositions") else "")
        fm = QtGui.QFontMetrics(p.font())
        w = fm.horizontalAdvance(txt) + 24
        r = QtCore.QRect(rect.center().x() - w // 2, rect.top() + 10, w, 28)
        p.fillRect(r, PANEL)
        p.setPen(col)
        p.drawText(r, QtCore.Qt.AlignCenter, txt)
    p.restore()


def draw_minimap(p: QtGui.QPainter, rect: QtCore.QRect, trail: Sequence[np.ndarray], pos: np.ndarray, yaw: float, dock_pos: np.ndarray,
                 axis_yaw: float = 0.0, extent_m: float = 14.0, ghost_pos: Optional[np.ndarray] = None) -> None:
    """Top view (north up). `rect` is where the map goes. The dock, its approach axis, the trail, the vehicle arrow (and the model ghost)."""
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    p.fillRect(rect, PANEL)
    p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 190), 1))
    p.drawRect(rect)
    cx, cy = rect.center().x(), rect.center().y()
    centre = 0.5 * (pos[:2] + dock_pos[:2])
    scale = min(rect.width(), rect.height()) / extent_m

    def to_px(xy) -> QtCore.QPointF:
        return QtCore.QPointF(cx + (xy[1] - centre[1]) * scale, cy - (xy[0] - centre[0]) * scale)       # east -> right, north -> up

    p.setClipRect(rect)
    # approach axis: from 7 m in front of the dock to the dock
    a = np.array([math.cos(axis_yaw), math.sin(axis_yaw)])
    p.setPen(QtGui.QPen(QtGui.QColor(255, 220, 120, 160), 1, QtCore.Qt.DashLine))
    p.drawLine(to_px(dock_pos[:2] - 9.0 * a), to_px(dock_pos[:2] + 1.0 * a))
    # dock mouth
    left = np.array([-a[1], a[0]])
    p.setPen(QtGui.QPen(QtGui.QColor(220, 230, 240), 3))
    p.drawLine(to_px(dock_pos[:2] - left), to_px(dock_pos[:2] + left))
    # trail
    if len(trail) > 1:
        poly = QtGui.QPolygonF([to_px(t[:2]) for t in trail[-400:]])
        p.setPen(QtGui.QPen(QtGui.QColor(60, 255, 150), 1.5))
        p.drawPolyline(poly)
    if ghost_pos is not None:
        p.setPen(QtGui.QPen(QtGui.QColor(190, 230, 255), 1.2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(to_px(ghost_pos[:2]), 4, 4)
    # vehicle arrow
    c = to_px(pos[:2])
    d = np.array([math.cos(yaw), math.sin(yaw)])
    tip = to_px(pos[:2] + 0.9 * d)
    tail = to_px(pos[:2] - 0.5 * d)
    p.setPen(QtGui.QPen(QtGui.QColor(255, 150, 30), 3))
    p.drawLine(tail, tip)
    p.setBrush(QtGui.QColor(255, 150, 30))
    p.drawEllipse(tip, 3, 3)
    p.setClipping(False)
    p.setPen(WHITE)
    p.setFont(QtGui.QFont("DejaVu Sans", 7))
    p.drawText(rect.left() + 4, rect.top() + 11, "N")
    p.restore()


def draw_detection_overlay(p: QtGui.QPainter, img_rect: QtCore.QRect, src_w: int, src_h: int, align: Dict[str, float]) -> None:
    """Mark the four detected lights, the dock centre and the error vector on the (scaled) camera image. `align` carries the pixel positions (see ros_link/replay)."""
    if not align:
        return
    sx, sy = img_rect.width() / src_w, img_rect.height() / src_h

    def pt(x, y) -> QtCore.QPointF:
        return QtCore.QPointF(img_rect.left() + x * sx, img_rect.top() + y * sy)

    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    valid = bool(align.get("valid", 0))
    col = QtGui.QColor(90, 255, 140) if valid else QtGui.QColor(255, 190, 70)
    names = ("top", "bottom", "right", "left")
    have = all(f"{n}_x" in align for n in names)
    if have:
        for n in names:
            x, y = align[f"{n}_x"], align[f"{n}_y"]
            if x <= 0 and y <= 0:
                continue
            p.setPen(QtGui.QPen(col, 1.6))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(pt(x, y), 9, 9)
            p.setFont(QtGui.QFont("DejaVu Sans", 7))
            p.drawText(pt(x + 10, y - 8), n[0].upper())
    if valid and "center_x" in align:
        c = pt(align["center_x"], align["center_y"])
        mid = pt(src_w / 2, src_h / 2)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1, QtCore.Qt.DashLine))
        p.drawLine(mid, c)
        p.setPen(QtGui.QPen(col, 2))
        p.drawLine(QtCore.QPointF(c.x() - 6, c.y()), QtCore.QPointF(c.x() + 6, c.y()))
        p.drawLine(QtCore.QPointF(c.x(), c.y() - 6), QtCore.QPointF(c.x(), c.y() + 6))
    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 1))
    mid = pt(src_w / 2, src_h / 2)
    p.drawLine(QtCore.QPointF(mid.x() - 8, mid.y()), QtCore.QPointF(mid.x() + 8, mid.y()))
    p.drawLine(QtCore.QPointF(mid.x(), mid.y() - 8), QtCore.QPointF(mid.x(), mid.y() + 8))
    p.setPen(col)
    p.setFont(QtGui.QFont("DejaVu Sans Mono", 8))
    txt = f"lights {int(align.get('num_lights', 0))}  " + ("VALID" if valid else "no lock") + f"  r {align.get('radius_px', 0):.0f}px"
    p.drawText(img_rect.left() + 6, img_rect.bottom() - 6, txt)
    p.restore()
