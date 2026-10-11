"""The viewer's Detection tab: the dock detector at work, next to the 3D view, so a trajectory and what the detector saw can be checked together.

Draws, from the telemetry the viewer already has (live ROS, the offline stack, or a replay that re-renders the camera and runs the real detector): the nose-camera picture with the
labelled lights, range / bearing / elevation / view-angle gauges, the steering lamps (YAW L/R, UP/DOWN), a top-down map of the dock and a banner (locked / searching / no lights).
`Detach` opens the big version (the same drawings as the detector pop-ups, plus the dock_align window) in its own window. All drawing code is dock_detection_algo/dock_hud.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

_ROOT = Path(__file__).resolve().parents[3]
for p in (_ROOT / "dock_detection_algo", Path(__file__).resolve().parents[2] / "common"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import dock_hud  # noqa: E402
from outdirs import out_dir  # noqa: E402


def _qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QtGui.QImage.Format_RGB888).copy()


class DetectionPanel(QtWidgets.QWidget):
    """Compact dashboard (tab) + a button for the big detached windows."""

    def __init__(self, tel, vfov_deg: float = 60.0, parent=None) -> None:
        super().__init__(parent)
        self.tel = tel
        self.vfov = float(vfov_deg)
        self.hist = dock_hud.History(12.0)
        self._last_t = -1.0
        self._frames = 0
        self._big: Optional["DetachedDetection"] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.view = QtWidgets.QLabel("waiting for the detector ...")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(560, 160)
        self.view.setStyleSheet("background:#1c2024;color:#9ab")
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        lay.addWidget(self.view, 1)
        row = QtWidgets.QHBoxLayout()
        self.note = QtWidgets.QLabel("")
        self.note.setStyleSheet("color:#789")
        self.note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)      # a long status text must never make the window wider
        self.note.setMinimumWidth(10)
        row.addWidget(self.note, 1)
        b1 = QtWidgets.QPushButton("Detach (big windows)")
        b1.clicked.connect(self.detach)
        b2 = QtWidgets.QPushButton("Save picture")
        b2.clicked.connect(self.save_picture)
        row.addWidget(b1)
        row.addWidget(b2)
        lay.addLayout(row)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(120)                                # ~8 Hz, and only draws while the tab is visible

    # ------------------------------------------------------------------ data
    def snapshot(self):
        """-> (bgr levelled or None, info dict) from the latest telemetry; also feeds the history."""
        t, al = self.tel.latest_align()
        bgr = None
        jpeg = self.tel.image()
        s = self.tel.latest()
        roll = float(s.eul[0]) if s is not None else 0.0
        w, h = 640, 480
        if jpeg is not None:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(roll), 1.0)      # the detector's pixels are roll-levelled
                bgr = cv2.warpAffine(img, M, (w, h)) if abs(roll) > 1e-3 else img
        info = dock_hud.info_from_align_dict(al, w, h, self.vfov, t)
        if al and t != self._last_t:
            self._last_t = t
            self.hist.add(info)
        return bgr, info

    def source_note(self) -> str:
        return f"{self.tel.source_name or '(starting)'}   |   detector frames drawn: {self._frames}"

    # ------------------------------------------------------------------ drawing
    def refresh(self) -> None:
        if not self.isVisible():
            return
        bgr, info = self.snapshot()
        size = (max(self.view.width(), 560), max(self.view.height(), 200))
        W = 1000
        H = int(min(max(W * size[1] / size[0], 300), 520))
        frame = dock_hud.draw_compact_view(bgr, info, (W, H))
        pm = QtGui.QPixmap.fromImage(_qimage(frame)).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.view.setPixmap(pm)
        self._frames += 1
        self.note.setText(self.source_note())
        if self._big is not None and self._big.isVisible():
            self._big.refresh(bgr, info, self.hist)

    def render_to_png(self, path: Path, big: bool = False) -> Path:
        bgr, info = self.snapshot()
        img = dock_hud.draw_camera_view(bgr, info, self.hist) if big else dock_hud.draw_compact_view(bgr, info)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return path

    def save_picture(self) -> None:
        d = out_dir("sim_viewer_shots")
        p = self.render_to_png(d / f"detection_{QtCore.QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.png", big=True)
        self.note.setText(f"saved {p}")

    def detach(self) -> None:
        if self._big is None:
            self._big = DetachedDetection()
        self._big.show()
        self._big.raise_()


class DetachedDetection(QtWidgets.QWidget):
    """The detector's big windows in one resizable window: the 'Dock camera' drawing on the left, the 'Dock align' drawing on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dock detection (camera + dock_align)")
        self.resize(1500, 760)
        lay = QtWidgets.QHBoxLayout(self)
        self.a = QtWidgets.QLabel()
        self.b = QtWidgets.QLabel()
        for lab in (self.a, self.b):
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            lay.addWidget(lab, 1)

    def refresh(self, bgr, info, hist) -> None:
        for lab, img in ((self.a, dock_hud.draw_camera_view(bgr, info, hist)), (self.b, dock_hud.draw_align_view(info, hist, None))):
            lab.setPixmap(QtGui.QPixmap.fromImage(_qimage(img)).scaled(lab.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
