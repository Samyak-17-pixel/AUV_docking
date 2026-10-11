"""A section that folds away: a header button with an arrow and a body widget (keeps long tabs short and tidy)."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


class Collapsible(QtWidgets.QWidget):
    def __init__(self, title: str, body: QtWidgets.QWidget, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.body = body
        if isinstance(body, QtWidgets.QGroupBox):
            body.setTitle("")
            body.setStyleSheet("QGroupBox { margin-top: 2px; padding-top: 4px; }")
        self.btn = QtWidgets.QToolButton()
        self.btn.setStyleSheet("QToolButton { background:#1b2633; border:1px solid #2b3a4b; border-radius:5px; padding:5px 8px; color:#2fb6e8; font-weight:bold; text-align:left; }"
                               "QToolButton:hover { border-color:#2fb6e8; }")
        self.btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.set_expanded)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.btn)
        lay.addWidget(body)
        self.set_expanded(expanded)

    def set_expanded(self, on: bool) -> None:
        self.btn.setChecked(bool(on))
        self.btn.setText(("▾  " if on else "▸  ") + self.title)
        self.body.setVisible(bool(on))

    def is_expanded(self) -> bool:
        return self.body.isVisibleTo(self)
