"""The viewer's '3D Trajectory' tab: the live path in 3D coloured by speed / depth / time / altitude, the planned path from the Config tab, a saved run for comparison, the seabed
and the side-scan swath footprint. Mouse: left drag = orbit, right drag / wheel = zoom, middle drag or Shift+left = pan."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from traj3d import COLOR_BY, Traj3DScene


class _View(QtWidgets.QWidget):
    def __init__(self, scene: Traj3DScene, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._img: Optional[QtGui.QImage] = None
        self._last = None

    def render_now(self) -> None:
        w, h = max(self.width(), 64), max(self.height(), 64)
        arr = self.scene.render(w, h)
        self._img = QtGui.QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QtGui.QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, _ev) -> None:                                       # noqa: N802
        p = QtGui.QPainter(self)
        if self._img is not None:
            p.drawImage(self.rect(), self._img)
        else:
            p.fillRect(self.rect(), QtGui.QColor(10, 18, 28))

    def mousePressEvent(self, ev) -> None:                                   # noqa: N802
        self._last = ev.pos()

    def mouseMoveEvent(self, ev) -> None:                                    # noqa: N802
        if self._last is None:
            return
        dx, dy = ev.x() - self._last.x(), ev.y() - self._last.y()
        self._last = ev.pos()
        b = ev.buttons()
        if b & QtCore.Qt.MiddleButton or (b & QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ShiftModifier):
            self.scene.pan(dx, dy)
        elif b & QtCore.Qt.RightButton:
            self.scene.zoom(1.0 + 0.01 * dy)
        elif b & QtCore.Qt.LeftButton:
            self.scene.orbit(dx, dy)
        self.render_now()

    def wheelEvent(self, ev) -> None:                                        # noqa: N802
        self.scene.zoom(1.0 + 0.1 * (ev.angleDelta().y() / 120.0))
        self.render_now()


class Trajectory3DPanel(QtWidgets.QWidget):
    def __init__(self, tel, cfg: dict, plan_provider=None, parent=None) -> None:
        super().__init__(parent)
        self.tel, self.cfg = tel, cfg
        self._plan_provider = plan_provider                        # callable -> [(x, y, z, label)] (the Config tab's path)
        terr = None
        try:
            if cfg.get("terrain") is not None:
                from terrain import get_terrain
                terr = get_terrain(cfg.get("terrain"))
        except Exception:                                            # no terrain (or numba trouble): the path still draws
            terr = None
        self.scene = Traj3DScene(cfg, terrain=terr)
        self.view = _View(self.scene)
        self._fitted = False
        self._swath_geo = None
        self._other: Optional[np.ndarray] = None

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        self.color_box = QtWidgets.QComboBox()
        self.color_box.addItems(list(COLOR_BY))
        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems(["iso", "top", "side"])
        self.window_box = QtWidgets.QComboBox()
        self.window_box.addItems(["last 1 min", "last 5 min", "last 10 min"])
        self.window_box.setCurrentIndex(2)
        self.chk_floor = QtWidgets.QCheckBox("Seabed")
        self.chk_floor.setChecked(True)
        self.chk_plan = QtWidgets.QCheckBox("Plan")
        self.chk_plan.setChecked(True)
        self.chk_swath = QtWidgets.QCheckBox("Sonar swath")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_load = QtWidgets.QPushButton("Load run...")
        self.btn_clear_other = QtWidgets.QPushButton("Clear loaded")
        self.btn_save = QtWidgets.QPushButton("Save picture")
        for lab, w in (("Colour by", self.color_box), ("View", self.view_box), ("Trail", self.window_box)):
            bar.addWidget(QtWidgets.QLabel(lab))
            bar.addWidget(w)
        for w in (self.chk_floor, self.chk_plan, self.chk_swath, self.btn_fit, self.btn_load, self.btn_clear_other, self.btn_save):
            bar.addWidget(w)
        bar.addStretch(1)
        self.note = QtWidgets.QLabel("")
        self.note.setStyleSheet("color:#8a9bb0")
        self.note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.note.setMinimumWidth(10)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(bar)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.note)

        self.color_box.currentTextChanged.connect(self._on_color)
        self.view_box.currentTextChanged.connect(lambda n: (self.scene.set_view(n), self._fitted_reset(), self.refresh()))
        self.chk_floor.toggled.connect(lambda on: (self.scene.set_floor_visible(on), self.view.render_now()))
        self.chk_plan.toggled.connect(lambda _on: self.refresh(force=True))
        self.chk_swath.toggled.connect(lambda _on: self.refresh(force=True))
        self.window_box.currentIndexChanged.connect(lambda _i: self.refresh(force=True))
        self.btn_fit.clicked.connect(self._fit)
        self.btn_load.clicked.connect(self._load_dialog)
        self.btn_clear_other.clicked.connect(self.clear_other)
        self.btn_save.clicked.connect(self.save_picture)
        self._last_n = -1

    # ------------------------------------------------------------- helpers
    def _fitted_reset(self) -> None:
        self._fitted = False

    def _on_color(self, name: str) -> None:
        self.scene.color_by = name
        self.refresh(force=True)

    def _seconds(self) -> float:
        return (60.0, 300.0, 600.0)[self.window_box.currentIndex()]

    def _fit(self) -> None:
        w = self.tel.window(self._seconds())
        if w:
            self.scene.fit(np.column_stack([w["x"], w["y"], w["depth"]]))
            self.view.render_now()

    def _swath_params(self):
        if self._swath_geo is None:
            try:
                from sidescan_node import sonar_config_from_yaml
                from synthetic_sidescan import flat_swath_geometry
                g = flat_swath_geometry(1.0, sonar_config_from_yaml(self.cfg))
                self._swath_geo = (g["ground_near_m"], g["ground_far_m"])
            except Exception:
                self._swath_geo = (0.09, 1.43)
        return self._swath_geo

    # ------------------------------------------------------------- data
    def refresh(self, force: bool = False) -> None:
        w = self.tel.window(self._seconds())
        if not w or len(w.get("x", [])) < 2:
            if force:
                self.scene.clear()
                self.view.render_now()
            return
        n = len(w["x"])
        if n == self._last_n and not force:
            return
        self._last_n = n
        pos = np.column_stack([w["x"], w["y"], w["depth"]])
        yaw = np.radians(w["yaw"])
        t = w["t"]
        self.scene.set_path(t, pos, w["u"], eul_yaw=float(yaw[-1]))
        plan = []
        if self.chk_plan.isChecked() and self._plan_provider is not None:
            try:
                plan = self._plan_provider() or []
            except Exception:
                plan = []
        self.scene.set_plan(plan)
        self.scene.set_other_run(self._other)
        if self.chk_swath.isChecked() and self.scene.terrain is not None:
            near, far = self._swath_params()
            alt = np.array([max(self.scene.terrain.altitude(p[0], p[1], p[2]), 0.5) for p in pos])
            self.scene.set_swath(pos, yaw, alt, (near, far, near, far))
        else:
            self.scene.set_swath(np.zeros((0, 3)), np.zeros(0), np.zeros(0))
        if not self._fitted:
            self.scene.fit(pos)
            self._fitted = True
        self.note.setText(f"{n} samples shown  |  path N {pos[-1,0]:.1f} E {pos[-1,1]:.1f} depth {pos[-1,2]:.1f} m" + (f"  |  loaded run: {len(self._other)} samples" if self._other is not None else ""))
        self.view.render_now()

    # ------------------------------------------------------------- saved runs / output
    def load_run(self, path) -> bool:
        from replay import load_run
        run = load_run(Path(path))
        self._other = np.asarray(run.pos, float)
        self.scene.set_other_run(self._other)
        self.scene.fit(self._other)
        self.note.setText(f"loaded {Path(path).name}: {len(self._other)} samples, {run.duration:.0f} s")
        self.view.render_now()
        return True

    def clear_other(self) -> None:
        self._other = None
        self.scene.set_other_run(None)
        self.view.render_now()

    def _load_dialog(self) -> None:
        from history import RUN_DIR
        start = str(RUN_DIR)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load a recorded run (CSV)", start, "CSV (*.csv)")
        if path:
            try:
                self.load_run(path)
            except Exception as exc:
                self.note.setText(f"could not load: {exc}")

    def save_picture(self) -> Optional[str]:
        from outdirs import out_dir
        d = Path(out_dir("sim_viewer_shots"))
        d.mkdir(parents=True, exist_ok=True)
        import time
        path = d / time.strftime("trajectory3d_%Y%m%d_%H%M%S.png")
        if self.view._img is not None:
            self.view._img.save(str(path))
        self.note.setText(f"saved {path}")
        return str(path)
