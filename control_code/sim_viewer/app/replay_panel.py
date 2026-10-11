"""Replay tab: play / pause / scrub / speed / loop for a ReplaySource. Qt only, no ROS."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets

SPEEDS = ["0.25", "0.5", "1", "2", "4", "8"]


class ReplayPanel(QtWidgets.QWidget):
    def __init__(self, source, on_seek: Optional[Callable[[], None]] = None, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self._on_seek = on_seek
        self._dragging = False
        lay = QtWidgets.QVBoxLayout(self)
        run = source.data
        info = f"{run.path.split('/')[-1]}\nformat: {run.fmt}   {run.n} rows   {run.duration:.1f} s"
        if run.meta.get("label"):
            info += f"\nlabel: {run.meta['label']}"
        if source.model is not None:
            info += "\nOVERLAY: translucent ghost / dashed lines = vehicle MODEL driven by the recorded commands"
        lab = QtWidgets.QLabel(info)
        lab.setWordWrap(True)
        lay.addWidget(lab)
        row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Pause")
        self.play_btn.clicked.connect(self._toggle)
        row.addWidget(self.play_btn)
        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.restart_btn.clicked.connect(lambda: self._seek(0.0))
        row.addWidget(self.restart_btn)
        lay.addLayout(row)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, int(round(run.duration * 100)))
        self.slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.slider.sliderReleased.connect(self._released)
        lay.addWidget(self.slider)
        self.time_label = QtWidgets.QLabel("0.0 / %.1f s" % run.duration)
        lay.addWidget(self.time_label)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Speed x"))
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems(SPEEDS)
        sp = f"{source.speed:g}"
        if sp not in SPEEDS:
            self.speed_box.addItem(sp)
        self.speed_box.setCurrentText(sp)
        self.speed_box.currentTextChanged.connect(lambda t: source.set_speed(float(t)))
        row2.addWidget(self.speed_box)
        self.loop_box = QtWidgets.QCheckBox("Loop")
        self.loop_box.setChecked(source.loop)
        self.loop_box.toggled.connect(lambda c: setattr(source, "loop", bool(c)))
        row2.addWidget(self.loop_box)
        lay.addLayout(row2)
        lay.addStretch(1)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def _toggle(self) -> None:
        if self.source.playing:
            self.source.pause()
        else:
            self.source.play()
        self.refresh()

    def _seek(self, t: float) -> None:
        self.source.seek(t)
        if self._on_seek:
            self._on_seek()

    def _released(self) -> None:
        self._dragging = False
        self._seek(self.slider.value() / 100.0)

    def refresh(self) -> None:
        pos = self.source.position
        if not self._dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(pos * 100)))
            self.slider.blockSignals(False)
        self.time_label.setText("%.1f / %.1f s%s" % (self.slider.value() / 100.0 if self._dragging else pos, self.source.duration, "" if self.source.playing else "   (paused)"))
        self.play_btn.setText("Pause" if self.source.playing else "Play")
