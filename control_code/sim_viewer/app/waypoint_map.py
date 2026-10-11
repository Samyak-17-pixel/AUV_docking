"""Top-view map widget for the Config tab: shows a planned path (waypoints or a whole mission), the dock, the geofence and keep-out, the vehicle and its trail, the sensor
swath (planned and covered), and the corners the vehicle cannot make. North is up, east is right (the viewer's NED frame).

Editing (edit_mode = True, waypoint lists): left click on empty water adds a point, drag moves one, right click deletes one. Otherwise a left click only emits `clicked`.
Mouse wheel zooms, middle drag (or shift + left drag) pans.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

import coverage as cov

LEG_COLORS = ["#4fc3f7", "#ffb74d", "#81c784", "#ba68c8", "#f06292", "#fff176", "#90a4ae"]


class WaypointMap(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(float, float)
    pointAdded = QtCore.pyqtSignal(float, float)
    pointMoved = QtCore.pyqtSignal(int, float, float)
    pointRemoved = QtCore.pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 260)
        self.setMouseTracking(True)
        self.points: List[Tuple[float, float, float, str]] = []
        self.edit_mode = False
        self.dock_xy: Tuple[float, float] = (10.0, 0.0)
        self.keepout_m = 0.0
        self.geofence: Dict[str, float] = {}
        self.turn_radius_m = 2.5
        self.swath_m = 0.0
        self.bad_points: set = set()                         # indices of points whose corner cannot be made
        self.vehicle: Optional[Tuple[float, float, float]] = None      # x, y, yaw [rad]
        self.trail: List[Tuple[float, float]] = []
        self.start_xy: Tuple[float, float] = (0.0, 0.0)
        self.target_xy: Optional[Tuple[float, float]] = None
        self._center = [3.0, 4.0]                           # world point at the widget centre (x north, y east)
        self._scale = 14.0                                  # pixels per metre
        self._drag: Optional[int] = None
        self._pan: Optional[Tuple[QtCore.QPoint, List[float]]] = None
        self.coverage_pct: Optional[float] = None
        self._cov_cache: Tuple[Optional[tuple], set] = (None, set())

    # ------------------------------------------------------------------ mapping
    def to_px(self, x: float, y: float) -> QtCore.QPointF:
        return QtCore.QPointF(self.width() / 2 + (y - self._center[1]) * self._scale, self.height() / 2 - (x - self._center[0]) * self._scale)

    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        return (self._center[0] - (py - self.height() / 2) / self._scale, self._center[1] + (px - self.width() / 2) / self._scale)

    def fit(self) -> None:
        pts = [(p[0], p[1]) for p in self.points] + [self.start_xy, self.dock_xy]
        if self.trail:
            pts += self.trail[-1:]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs) - 2.0, max(xs) + 2.0, min(ys) - 2.0, max(ys) + 2.0
        self._center = [0.5 * (x0 + x1), 0.5 * (y0 + y1)]
        self._scale = max(2.0, min(self.height() / max(x1 - x0, 1.0), self.width() / max(y1 - y0, 1.0)))
        self.update()

    def set_points(self, points: Sequence[Tuple[float, float, float, str]], bad: Optional[set] = None) -> None:
        self.points = list(points)
        self.bad_points = set(bad or ())
        self.update()

    def set_vehicle(self, pose: Optional[Tuple[float, float, float]], trail: Optional[Sequence[Tuple[float, float]]] = None) -> None:
        self.vehicle = pose
        if trail is not None:
            self.trail = list(trail)
        self.update()

    def coverage_now(self) -> Optional[float]:
        """Share of the planned swath covered by the trail so far (None when no swath width is set)."""
        if self.swath_m <= 0 or not self.points:
            return None
        key = (len(self.trail), len(self.points), self.swath_m)
        if self._cov_cache[0] != key:
            plan = [(p[0], p[1]) for p in self.points]
            self.coverage_pct = 100.0 * cov.coverage_fraction(plan, self.trail, self.swath_m, 0.5) if self.trail else 0.0
            self._cov_cache = (key, set())
        return self.coverage_pct

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#0b1722"))
        self._grid(p)
        if self.geofence:
            self._fence(p)
        if self.keepout_m > 0:
            c = self.to_px(*self.dock_xy)
            p.setPen(QtGui.QPen(QtGui.QColor("#ef5350"), 1, QtCore.Qt.DashLine))
            p.setBrush(QtGui.QColor(239, 83, 80, 30))
            p.drawEllipse(c, self.keepout_m * self._scale, self.keepout_m * self._scale)
        self._dock(p)
        if self.swath_m > 0 and len(self.points) >= 1:
            pen = QtGui.QPen(QtGui.QColor(120, 144, 156, 55), max(1.0, self.swath_m * self._scale))
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QtGui.QPolygonF([self.to_px(q[0], q[1]) for q in self.points]))
            if len(self.trail) >= 2:
                pen = QtGui.QPen(QtGui.QColor(102, 187, 106, 70), max(1.0, self.swath_m * self._scale))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                p.setPen(pen)
                p.drawPolyline(QtGui.QPolygonF([self.to_px(*q) for q in self.trail]))
        self._path(p)
        if len(self.trail) >= 2:
            p.setPen(QtGui.QPen(QtGui.QColor("#e0e0e0"), 1.5))
            p.drawPolyline(QtGui.QPolygonF([self.to_px(*q) for q in self.trail]))
        sx = self.to_px(*self.start_xy)
        p.setPen(QtGui.QPen(QtGui.QColor("#a5d6a7"), 1.5))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(sx, 5, 5)
        if self.vehicle is not None:
            self._vehicle(p)
        if self.target_xy is not None:
            t = self.to_px(*self.target_xy)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffee58"), 2))
            p.drawLine(QtCore.QPointF(t.x() - 6, t.y()), QtCore.QPointF(t.x() + 6, t.y()))
            p.drawLine(QtCore.QPointF(t.x(), t.y() - 6), QtCore.QPointF(t.x(), t.y() + 6))
        p.setPen(QtGui.QColor("#78909c"))
        txt = "N up, E right | scale bar 5 m"
        if self.coverage_now() is not None:
            txt += f" | swath {self.swath_m:g} m: covered {self.coverage_pct:.0f} %"
        p.drawText(8, self.height() - 8, txt)
        y = self.height() - 22
        p.setPen(QtGui.QPen(QtGui.QColor("#78909c"), 2))
        p.drawLine(self.width() - 20 - int(5 * self._scale), y, self.width() - 20, y)

    def _grid(self, p) -> None:
        step = 1.0
        for s in (1.0, 2.0, 5.0, 10.0, 20.0):
            step = s
            if s * self._scale >= 28:
                break
        x_lo, y_lo = self.to_world(0, self.height())
        x_hi, y_hi = self.to_world(self.width(), 0)
        p.setFont(QtGui.QFont("monospace", 7))
        x = math.floor(x_lo / step) * step
        while x <= x_hi:
            a, b = self.to_px(x, y_lo), self.to_px(x, y_hi)
            p.setPen(QtGui.QPen(QtGui.QColor("#16293a" if abs(x) > 1e-9 else "#2e4a63"), 1))
            p.drawLine(a, b)
            p.setPen(QtGui.QColor("#4d6b82"))
            p.drawText(2, int(a.y()) - 2, f"{x:g}")
            x += step
        y = math.floor(y_lo / step) * step
        while y <= y_hi:
            a, b = self.to_px(x_lo, y), self.to_px(x_hi, y)
            p.setPen(QtGui.QPen(QtGui.QColor("#16293a" if abs(y) > 1e-9 else "#2e4a63"), 1))
            p.drawLine(a, b)
            p.setPen(QtGui.QColor("#4d6b82"))
            p.drawText(int(a.x()) + 2, 10, f"{y:g}")
            y += step

    def _fence(self, p) -> None:
        g = self.geofence
        x0, x1 = g.get("x_min", -1e3), g.get("x_max", 1e3)
        y0, y1 = g.get("y_min", -1e3), g.get("y_max", 1e3)
        a, b = self.to_px(x1, y0), self.to_px(x0, y1)
        p.setPen(QtGui.QPen(QtGui.QColor("#ffa726"), 1.5, QtCore.Qt.DashDotLine))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(a, b))

    def _dock(self, p) -> None:
        c = self.to_px(*self.dock_xy)
        p.setPen(QtGui.QPen(QtGui.QColor("#ef5350"), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(c, 1.0 * self._scale, 1.0 * self._scale)
        p.drawLine(QtCore.QPointF(c.x() - 4, c.y()), QtCore.QPointF(c.x() + 4, c.y()))
        p.setPen(QtGui.QColor("#ef9a9a"))
        p.drawText(int(c.x()) + 8, int(c.y()) - 6, "dock")

    def _path(self, p) -> None:
        if not self.points:
            return
        labels: List[str] = []
        for q in self.points:
            if q[3] not in labels:
                labels.append(q[3])
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            col = QtGui.QColor(LEG_COLORS[labels.index(b[3]) % len(LEG_COLORS)])
            p.setPen(QtGui.QPen(col, 2))
            pa, pb = self.to_px(a[0], a[1]), self.to_px(b[0], b[1])
            p.drawLine(pa, pb)
            if math.hypot(pb.x() - pa.x(), pb.y() - pa.y()) > 26:                       # a small arrow head at the middle of long enough segments
                mx, my = 0.5 * (pa.x() + pb.x()), 0.5 * (pa.y() + pb.y())
                ang = math.atan2(pb.y() - pa.y(), pb.x() - pa.x())
                head = QtGui.QPolygonF([QtCore.QPointF(mx + 5 * math.cos(ang), my + 5 * math.sin(ang)),
                                        QtCore.QPointF(mx + 5 * math.cos(ang + 2.5), my + 5 * math.sin(ang + 2.5)),
                                        QtCore.QPointF(mx + 5 * math.cos(ang - 2.5), my + 5 * math.sin(ang - 2.5))])
                p.setBrush(col)
                p.drawPolygon(head)
        many = len(self.points) > 40
        p.setFont(QtGui.QFont("monospace", 7))
        for i, q in enumerate(self.points):
            c = self.to_px(q[0], q[1])
            bad = i in self.bad_points
            if bad:
                p.setPen(QtGui.QPen(QtGui.QColor("#ef5350"), 1, QtCore.Qt.DashLine))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(c, self.turn_radius_m * self._scale, self.turn_radius_m * self._scale)
            if many and not bad and not self.edit_mode:
                continue
            p.setPen(QtGui.QPen(QtGui.QColor("#ef5350" if bad else "#eceff1"), 1))
            p.setBrush(QtGui.QColor("#ef5350" if bad else "#37474f"))
            p.drawEllipse(c, 4, 4)
            if not many or self.edit_mode:
                p.setPen(QtGui.QColor("#cfd8dc"))
                p.drawText(int(c.x()) + 6, int(c.y()) - 5, str(i))

    def _vehicle(self, p) -> None:
        x, y, yaw = self.vehicle
        c = self.to_px(x, y)
        ang = yaw - math.pi / 2          # yaw 0 = north = screen up
        pts = [QtCore.QPointF(c.x() + 10 * math.sin(yaw), c.y() - 10 * math.cos(yaw)),
               QtCore.QPointF(c.x() + 6 * math.sin(yaw + 2.6), c.y() - 6 * math.cos(yaw + 2.6)),
               QtCore.QPointF(c.x() + 6 * math.sin(yaw - 2.6), c.y() - 6 * math.cos(yaw - 2.6))]
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        p.setBrush(QtGui.QColor("#26c6da"))
        p.drawPolygon(QtGui.QPolygonF(pts))

    # ------------------------------------------------------------------ mouse
    def _hit(self, pos: QtCore.QPoint) -> Optional[int]:
        best, bi = 10.0, None
        for i, q in enumerate(self.points):
            c = self.to_px(q[0], q[1])
            d = math.hypot(c.x() - pos.x(), c.y() - pos.y())
            if d < best:
                best, bi = d, i
        return bi

    def mousePressEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MiddleButton or (ev.button() == QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ShiftModifier):
            self._pan = (ev.pos(), list(self._center))
            return
        hit = self._hit(ev.pos()) if self.edit_mode else None
        if ev.button() == QtCore.Qt.RightButton:
            if hit is not None:
                self.pointRemoved.emit(hit)
            return
        if ev.button() == QtCore.Qt.LeftButton:
            if hit is not None:
                self._drag = hit
                return
            x, y = self.to_world(ev.pos().x(), ev.pos().y())
            if self.edit_mode:
                self.pointAdded.emit(round(x, 2), round(y, 2))
            else:
                self.clicked.emit(round(x, 2), round(y, 2))

    def mouseMoveEvent(self, ev) -> None:
        if self._pan is not None:
            start, c0 = self._pan
            self._center = [c0[0] + (ev.pos().y() - start.y()) / self._scale, c0[1] - (ev.pos().x() - start.x()) / self._scale]
            self.update()
        elif self._drag is not None:
            x, y = self.to_world(ev.pos().x(), ev.pos().y())
            q = self.points[self._drag]
            self.points[self._drag] = (round(x, 2), round(y, 2), q[2], q[3])
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if self._drag is not None:
            q = self.points[self._drag]
            self.pointMoved.emit(self._drag, q[0], q[1])
            self._drag = None
        self._pan = None

    def wheelEvent(self, ev) -> None:
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self._scale = max(2.0, min(120.0, self._scale * f))
        self.update()
