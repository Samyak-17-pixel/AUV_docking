"""The viewer's Sonar tab (bottom-left tab bar, next to Detection): the side-scan WATERFALL (port | starboard) and the MOSAIC built from the pings, with controls for range, gain and bins
that go to the simulated sonars (/Mako_01/sonar/cmd). Works live (offline stack) and in a replay of a saved survey (sidescan.npz next to the trajectory CSV): the stored pings are played back
in step with the trajectory, so the data can be looked at again later. Drawing code: sidescan_view.py, mosaic: sidescan_mosaic.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE.parent / "common"))
from outdirs import out_dir  # noqa: E402
from sidescan_mosaic import Mosaic  # noqa: E402
from sidescan_view import mosaic_image, waterfall_image  # noqa: E402


def _qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QtGui.QImage.Format_RGB888).copy()


class SonarPanel(QtWidgets.QWidget):
    enlarge = QtCore.pyqtSignal(bool)

    def __init__(self, tel, source, extent=(-150.0, 150.0, -150.0, 150.0), cell_m: float = 0.5, parent=None) -> None:
        super().__init__(parent)
        self.tel, self.source = tel, source
        self.mosaic = Mosaic(extent, cell_m)
        self._seen = {"port": 0, "starboard": 0}                 # how many pings of each side the mosaic has taken (by count of stored rows' newest time)
        self._last_t = {"port": -1.0, "starboard": -1.0}
        self._path: list = []
        self.frozen = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Range [m]"))
        self.range_box = QtWidgets.QDoubleSpinBox()
        self.range_box.setRange(2.0, 150.0)
        self.range_box.setSingleStep(5.0)
        self.range_box.setValue(30.0)
        row.addWidget(self.range_box)
        row.addWidget(QtWidgets.QLabel("Gain"))
        self.gain_box = QtWidgets.QComboBox()
        self.gain_box.addItems(["auto"] + [str(i) for i in range(8)])
        row.addWidget(self.gain_box)
        row.addWidget(QtWidgets.QLabel("Bins"))
        self.bins_box = QtWidgets.QSpinBox()
        self.bins_box.setRange(200, 1200)
        self.bins_box.setSingleStep(100)
        self.bins_box.setValue(600)
        row.addWidget(self.bins_box)
        self.apply_btn = QtWidgets.QPushButton("Apply to sonar")
        self.apply_btn.clicked.connect(self.apply)
        row.addWidget(self.apply_btn)
        self.pause_chk = QtWidgets.QCheckBox("freeze")
        self.pause_chk.toggled.connect(lambda v: setattr(self, "frozen", bool(v)))
        row.addWidget(self.pause_chk)
        clr = QtWidgets.QPushButton("Clear mosaic")
        clr.clicked.connect(self.clear_mosaic)
        row.addWidget(clr)
        save = QtWidgets.QPushButton("Save pictures")
        save.clicked.connect(self.save_pictures)
        row.addWidget(save)
        self.enlarge_btn = QtWidgets.QPushButton("Enlarge panel")
        self.enlarge_btn.setCheckable(True)
        self.enlarge_btn.setToolTip("Give this tab most of the window height")
        self.enlarge_btn.toggled.connect(lambda on: self.enlarge.emit(bool(on)))
        row.addWidget(self.enlarge_btn)
        self.popout_btn = QtWidgets.QPushButton("Pop out mosaic")
        self.popout_btn.setToolTip("Open the mosaic in its own big window")
        self.popout_btn.clicked.connect(self.pop_out)
        row.addWidget(self.popout_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.tabs = QtWidgets.QTabWidget()
        nopings = "no side-scan pings yet: start the sonar node (Controls -> Start offline stack, or ./run_sidescan_sim.sh), or replay a saved survey"
        self.wf, self.mo, self.wf2, self.mo2 = (QtWidgets.QLabel(nopings if i % 2 == 0 else "") for i in range(4))
        for lab in (self.wf, self.mo, self.wf2, self.mo2):
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setWordWrap(True)
            lab.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            lab.setMinimumSize(120, 100)
            lab.setStyleSheet("background:#14100c;color:#a98")
        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)                         # Overview: a narrow waterfall next to a BIG mosaic
        self.split.addWidget(self.wf2)
        self.split.addWidget(self.mo2)
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 5)
        self.split.setSizes([300, 900])
        self.tabs.addTab(self.split, "Overview")
        self.tabs.addTab(self.mo, "Mosaic")
        self.tabs.addTab(self.wf, "Waterfall")
        lay.addWidget(self.tabs, 1)
        self.big: Optional[QtWidgets.QDialog] = None
        self.big_label: Optional[QtWidgets.QLabel] = None
        self.note = QtWidgets.QLabel("")
        self.note.setStyleSheet("color:#8a9bb0")
        self.note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)      # a long status text must never make the window wider
        self.note.setMinimumWidth(10)
        lay.addWidget(self.note)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(120)

    # ------------------------------------------------------------------ big mosaic
    def pop_out(self) -> None:
        """The mosaic in its own big, resizable window (follows the live data)."""
        if self.big is None:
            self.big = QtWidgets.QDialog(self)
            self.big.setWindowTitle("Side-scan mosaic")
            self.big.resize(1100, 900)
            lay = QtWidgets.QVBoxLayout(self.big)
            lay.setContentsMargins(2, 2, 2, 2)
            self.big_label = QtWidgets.QLabel("no mosaic yet")
            self.big_label.setAlignment(QtCore.Qt.AlignCenter)
            self.big_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            self.big_label.setStyleSheet("background:#14100c;color:#a98")
            lay.addWidget(self.big_label)
        self.big.show()
        self.big.raise_()

    # ------------------------------------------------------------------ controls
    def can_command(self) -> bool:
        return hasattr(self.source, "send_sonar_command")

    def apply(self) -> None:
        g = self.gain_box.currentText()
        d = {"range_m": float(self.range_box.value()), "gain": -1 if g == "auto" else int(g), "bins": int(self.bins_box.value())}
        if self.can_command():
            self.source.send_sonar_command(d)
            self.note.setText(f"sent {d}")
        else:
            self.note.setText("this source cannot command the sonar (a replay shows the data as recorded)")

    def clear_mosaic(self) -> None:
        self.mosaic = Mosaic((self.mosaic.x0, self.mosaic.x1, self.mosaic.y0, self.mosaic.y1), self.mosaic.cell)
        self._path.clear()

    # ------------------------------------------------------------------ data
    def _take_new(self) -> None:
        for side in ("port", "starboard"):
            for (t, inten, meta) in self.tel.sonar_rows(side, 400):
                if t > self._last_t[side]:
                    self._last_t[side] = t
                    self.mosaic.add_ping(side, inten, meta)
                    if side == "port":
                        self._path.append((meta["pos"][0], meta["pos"][1]))

    def refresh(self) -> None:
        if not self.isVisible() or self.frozen:
            return
        self._take_new()
        p = [x[1] for x in self.tel.sonar_rows("port", 400)]
        s = [x[1] for x in self.tel.sonar_rows("starboard", 400)]
        last = (self.tel.sonar_rows("port", 1) or self.tel.sonar_rows("starboard", 1) or [None])[-1]
        rng = float(last[2]["range_m"]) if last else float(self.range_box.value())
        cur = self.tabs.currentWidget()
        show_wf = cur in (self.split, self.wf) and (p or s)
        show_mo = (cur in (self.split, self.mo) or (self.big is not None and self.big.isVisible())) and self.mosaic.pings
        if show_wf:
            lab = self.wf2 if cur is self.split else self.wf
            w, h = max(lab.width(), 120), max(lab.height(), 100)
            img = waterfall_image(p, s, (w, h), rng)
            lab.setPixmap(QtGui.QPixmap.fromImage(_qimage(img)))
        if show_mo:
            pts = np.array(self._path) if self._path else None
            crop = None
            if pts is not None and len(pts):
                pad = 15.0
                crop = (max(pts[:, 0].min() - pad, self.mosaic.x0), min(pts[:, 0].max() + pad, self.mosaic.x1), max(pts[:, 1].min() - pad, self.mosaic.y0), min(pts[:, 1].max() + pad, self.mosaic.y1))
            span = (crop[1] - crop[0]) if crop else 1e9
            img = mosaic_image(self.mosaic, 4 if span < 70 else (2 if span < 150 else 1), pts, tuple(pts[-1]) if pts is not None and len(pts) else None, crop)
            pix = QtGui.QPixmap.fromImage(_qimage(img))
            for lab in ((self.mo2 if cur is self.split else self.mo), self.big_label if (self.big is not None and self.big.isVisible()) else None):
                if lab is not None:
                    lab.setPixmap(pix.scaled(lab.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        st = self.tel.sonar_status
        alt = last[2].get("altitude_m") if last else None
        self.note.setText(f"{self.tel.source_name or '(starting)'}   |   pings: port {len(p)} stbd {len(s)} (shown)   range {rng:.0f} m   gain {'auto' if (last and last[2].get('gain_index', -1) < 0) else (int(last[2]['gain_index']) if last else '-')}"
                          f"   altitude {alt if alt is None else round(alt, 1)} m   mosaic coverage {100 * self.mosaic.coverage():.1f}% of the area" + (f"   (node: {st.get('ping_hz', 0):.1f} Hz, {st.get('compute_ms', 0)} ms/ping)" if st else ""))

    def save_pictures(self) -> None:
        d = out_dir("sim_viewer_shots")
        d.mkdir(parents=True, exist_ok=True)
        stamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        p = [x[1] for x in self.tel.sonar_rows("port", 600)]
        s = [x[1] for x in self.tel.sonar_rows("starboard", 600)]
        if p or s:
            cv2.imwrite(str(d / f"waterfall_{stamp}.png"), waterfall_image(p, s, (1200, 700), float(self.range_box.value())))
        if self.mosaic.pings:
            cv2.imwrite(str(d / f"mosaic_{stamp}.png"), mosaic_image(self.mosaic, 1, np.array(self._path) if self._path else None))
        self.note.setText(f"saved to {d}")
