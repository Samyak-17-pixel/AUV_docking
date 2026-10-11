"""One dark colour theme for the whole viewer: Qt stylesheet, matplotlib style, and the accent colours the plots share.

Why one module: the window, the plots, the sonar/detection panels and the 3D trajectory view all take their colours from here, so a single edit restyles everything.
"""

from __future__ import annotations

from typing import Dict

BG = "#0e141b"          # window
PANEL = "#151d27"       # panels, tab pages
PANEL2 = "#1b2633"      # raised: group titles, inputs
BORDER = "#2b3a4b"
TEXT = "#d7e0ea"
MUTED = "#8a9bb0"
ACCENT = "#2fb6e8"      # cyan: selection, links
ORANGE = "#ff9f43"      # vehicle, setpoints
GREEN = "#3ddc97"       # ok
RED = "#ff5c5c"         # alarm
YELLOW = "#ffd166"      # warning
PURPLE = "#b794f6"

SERIES: Dict[str, str] = {"blue": ACCENT, "orange": ORANGE, "green": GREEN, "red": RED, "yellow": YELLOW, "purple": PURPLE, "grey": "#9aa5b1", "light": "#c9d3de"}

STYLESHEET = f"""
* {{ font-size: 10pt; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QMainWindow, QSplitter, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QLabel {{ background: transparent; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL}; top: -1px; }}
QTabBar {{ background: {BG}; }}
QTabBar::tab {{ background: {PANEL2}; color: {MUTED}; padding: 6px 8px; border: 1px solid {BORDER}; border-bottom: none; margin-right: 2px;
               border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; border-top: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {ACCENT}; font-weight: bold; background: {BG}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: #556; border-color: #222c38; }}
QPushButton:checked {{ background: {ACCENT}; color: #04202c; }}
QPushButton[role="danger"] {{ background: #5a1f24; border-color: {RED}; color: #ffd9d9; font-weight: bold; }}
QPushButton[role="primary"] {{ background: #12506b; border-color: {ACCENT}; color: white; font-weight: bold; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px;
               selection-background-color: {ACCENT}; selection-color: #04202c; }}
QComboBox:hover, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; selection-background-color: {ACCENT}; selection-color: #04202c; border: 1px solid {BORDER}; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {BORDER}; border-radius: 3px; background: {PANEL2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QTableWidget, QTreeWidget, QTableView, QListWidget {{ background: {PANEL}; alternate-background-color: {PANEL2}; gridline-color: {BORDER}; border: 1px solid {BORDER}; }}
QHeaderView::section {{ background: {PANEL2}; color: {ACCENT}; border: 1px solid {BORDER}; padding: 3px; }}
QScrollBar:vertical {{ background: {BG}; width: 12px; }}
QScrollBar:horizontal {{ background: {BG}; height: 12px; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSlider::groove:horizontal {{ height: 5px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QProgressBar {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {ACCENT}; }}
QDockWidget::title {{ background: {PANEL2}; padding: 4px; }}
"""


def apply(app) -> None:
    """Dark Qt style for the whole application and a matching matplotlib style (call before any figure is created)."""
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    apply_matplotlib()


def apply_matplotlib() -> None:
    import warnings
    import matplotlib as mpl
    warnings.filterwarnings("ignore", message="constrained_layout not applied")      # a plot page that is hidden or still tiny while the window lays out
    mpl.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": "#101822", "axes.edgecolor": BORDER, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "grid.color": "#33445a", "grid.alpha": 0.5,
        "legend.facecolor": "#101822", "legend.edgecolor": BORDER, "legend.labelcolor": TEXT, "savefig.facecolor": PANEL,
        "axes.prop_cycle": mpl.cycler(color=[ACCENT, ORANGE, GREEN, RED, YELLOW, PURPLE]),
    })


def status_colour(level: str) -> str:
    return {"ok": GREEN, "warn": YELLOW, "bad": RED}.get(level, MUTED)
