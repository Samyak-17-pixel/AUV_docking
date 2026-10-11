"""Small reusable widgets for the dark theme: KPI tiles with a coloured state."""

from __future__ import annotations

from typing import Dict, List

from PyQt5 import QtCore, QtWidgets

import theme


class Tile(QtWidgets.QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet(f"color:{theme.MUTED}; font-size:8pt; background:transparent;")
        self.value = QtWidgets.QLabel("-")
        self.value.setStyleSheet(f"color:{theme.TEXT}; font-size:14pt; font-weight:bold; background:transparent;")
        self.value.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)      # long text must not widen the panel
        self.value.setMinimumWidth(10)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(0)
        lay.addWidget(self.title)
        lay.addWidget(self.value)
        self.set("-", "idle")

    def set(self, text: str, level: str = "ok") -> None:
        col = theme.status_colour(level) if level != "idle" else theme.MUTED
        self.value.setText(text)
        self.setStyleSheet(f"Tile {{ background:{theme.PANEL2}; border:1px solid {theme.BORDER}; border-left:4px solid {col}; border-radius:5px; }}")


class TileGrid(QtWidgets.QWidget):
    def __init__(self, names: List[str], columns: int = 2, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.tiles: Dict[str, Tile] = {}
        for i, n in enumerate(names):
            t = Tile(n)
            self.tiles[n] = t
            lay.addWidget(t, i // columns, i % columns)

    def set(self, name: str, text: str, level: str = "ok") -> None:
        self.tiles[name].set(text, level)
