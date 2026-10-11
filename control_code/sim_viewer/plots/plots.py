"""Live plots (matplotlib inside Qt). Two tabs: the vehicle (depth, attitude, speed, actuators, path) and the dock detector's output."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class _Panel:
    """One axes with a fixed set of named lines that are updated in place (no re-creation per frame)."""

    def __init__(self, ax, title: str, ylabel: str, series: List[tuple], invert: bool = False) -> None:
        self.ax = ax
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        self.lines: Dict[str, object] = {}
        self.tkeys: Dict[str, str] = {}
        for item in series:
            key, label, color = item[:3]
            tkey = item[3] if len(item) > 3 else None                 # a series may live on its own time base (setpoints)
            dashed = bool(item[4]) if len(item) > 4 else False
            (ln,) = ax.plot([], [], color=color, lw=1.1 if dashed else 1.2, ls="--" if dashed else "-", label=label)
            self.lines[key] = ln
            if tkey:
                self.tkeys[key] = tkey
        if len(series) > 1:
            ax.legend(fontsize=6, loc="upper left", ncol=3, framealpha=0.6)
        self.invert = invert
        if invert:
            ax.invert_yaxis()

    def set(self, t: np.ndarray, data: Dict[str, np.ndarray], tkey: str, window_s: float) -> None:
        lo, hi = np.inf, -np.inf
        for key, ln in self.lines.items():
            y = data.get(key)
            tt = data.get(self.tkeys.get(key, tkey))
            if y is None or tt is None or len(y) == 0 or len(tt) != len(y):
                ln.set_data([], [])
                continue
            ln.set_data(tt, y)
            lo, hi = min(lo, float(np.nanmin(y))), max(hi, float(np.nanmax(y)))
        self.ax.set_xlim(-window_s, 0.0)
        if np.isfinite(lo) and np.isfinite(hi):
            pad = max(0.05 * (hi - lo), 0.05)
            self.ax.set_ylim((hi + pad, lo - pad) if self.invert else (lo - pad, hi + pad))


class LivePlots(QtWidgets.QTabWidget):
    def __init__(self, window_s: float = 60.0, parent=None) -> None:
        super().__init__(parent)
        self.window_s = float(window_s)
        # --- vehicle tab
        self.fig1 = Figure(figsize=(9, 4), constrained_layout=True)
        self.c1 = FigureCanvasQTAgg(self.fig1)
        ax = self.fig1.subplots(2, 3).ravel()
        self.p_depth = _Panel(ax[0], "Depth (down = deeper)", "m", [("depth", "depth", "#2fb6e8"), ("ctrl_depth_sp", "setpoint", "#2fb6e8", "ctrl_t", True),
                                                                           ("m_depth", "model", "#b9c4d0", "m_t", True)], invert=True)
        self.p_att = _Panel(ax[1], "Attitude", "deg", [("roll", "roll", "#ff5c5c"), ("pitch", "pitch", "#3ddc97"), ("yaw", "yaw", "#2fb6e8"),
                                                                       ("ctrl_pitch_sp_deg", "pitch sp", "#3ddc97", "ctrl_t", True), ("ctrl_yaw_sp_deg", "yaw sp", "#2fb6e8", "ctrl_t", True),
                                                                       ("m_pitch", "model pitch", "#b9c4d0", "m_t", True), ("m_yaw", "model yaw", "#d5dde6", "m_t", True)])
        self.p_speed = _Panel(ax[2], "Surge speed and rates", "m/s  |  deg/s", [("u", "u", "#2fb6e8"), ("q", "q", "#3ddc97"), ("r", "r", "#ff5c5c"), ("m_u", "model u", "#b9c4d0", "m_t", True)])
        self.p_rpm = _Panel(ax[3], "Thruster commands", "RPM", [("cmd_th_01", "th_01 axial", "#2fb6e8"), ("cmd_th_02", "th_02 heave F", "#3ddc97"), ("cmd_th_03", "th_03 heave A", "#ff5c5c")])
        self.p_fin = _Panel(ax[4], "Fin commands", "deg", [("cmd_cs_04", "cs_04", "#2fb6e8"), ("cmd_cs_06", "cs_06", "#3ddc97"), ("cmd_cs_07", "cs_07", "#ff5c5c"), ("cmd_cs_08", "cs_08", "#b794f6")])
        self.xy_ax = ax[5]
        self.xy_ax.set_title("Path (top view)", fontsize=9)
        self.xy_ax.set_xlabel("East y (m)", fontsize=8)
        self.xy_ax.set_ylabel("North x (m)", fontsize=8)
        self.xy_ax.tick_params(labelsize=7)
        self.xy_ax.grid(True, alpha=0.3)
        self.xy_ax.set_aspect("equal", adjustable="datalim")
        (self.xy_line,) = self.xy_ax.plot([], [], color="#3ddc97", lw=1.2)
        (self.xy_now,) = self.xy_ax.plot([], [], "o", color="#ff9f43", ms=5)
        (self.xy_model,) = self.xy_ax.plot([], [], color="#b9c4d0", lw=1.0, ls="--")
        self.xy_ax.plot([0.0], [10.0], "s", color="#9aa5b1", ms=7)         # the dock (north 10 m, east 0 m)
        self.addTab(self.c1, "Vehicle")
        # --- dock tab
        self.fig2 = Figure(figsize=(9, 4), constrained_layout=True)
        self.c2 = FigureCanvasQTAgg(self.fig2)
        bx = self.fig2.subplots(2, 2).ravel()
        self.d_err = _Panel(bx[0], "Dock offset in the image", "px", [("align_error_x_px", "error_x (+ = dock right)", "#ff5c5c"), ("align_error_y_px", "error_y (+ = dock below)", "#2fb6e8")])
        self.d_elev = _Panel(bx[1], "Elevation and side cue", "deg | px", [("align_elevation_deg", "elevation (+ = dock deeper)", "#3ddc97"), ("align_lateral_px", "lateral_px", "#b794f6")])
        self.d_rad = _Panel(bx[2], "Dock size in the image", "px", [("align_radius_px", "radius_px", "#2fb6e8")])
        self.d_cnt = _Panel(bx[3], "Detection", "lights | valid", [("align_num_lights", "lights found", "#ff9f43"), ("align_valid", "valid (1/0)", "#3ddc97")])
        self.addTab(self.c2, "Dock detector")
        # --- mission tab (cross-track error, heading error, speed vs setpoint, depth error, progress, path vs plan)
        self.fig3 = Figure(figsize=(9, 4), constrained_layout=True)
        self.c3 = FigureCanvasQTAgg(self.fig3)
        cx = self.fig3.subplots(2, 3).ravel()
        self.m_ct = _Panel(cx[0], "Cross-track error (+ = right of the lane)", "m", [("ctrl_cross_track_m", "cross-track", "#ff5c5c", "ctrl_t")])
        self.m_hd = _Panel(cx[1], "Heading error", "deg", [("ctrl_heading_err_deg", "heading error", "#b794f6", "ctrl_t"), ("r", "yaw rate", "#2fb6e8")])
        self.m_sp = _Panel(cx[2], "Speed vs setpoint", "m/s", [("u", "u", "#2fb6e8"), ("ctrl_u_sp", "setpoint", "#2fb6e8", "ctrl_t", True)])
        self.m_dz = _Panel(cx[3], "Depth error (setpoint - depth)", "m", [("ctrl_depth_err_m", "depth error", "#3ddc97", "ctrl_t")])
        self.m_pr = _Panel(cx[4], "Mission progress and ETA", "% | s", [("ctrl_progress_pct", "progress %", "#ff9f43", "ctrl_t"), ("ctrl_eta_s", "ETA s", "#9aa5b1", "ctrl_t", True)])
        self.mx = cx[5]
        self.mx.set_title("Path vs plan", fontsize=9)
        self.mx.set_xlabel("East y (m)", fontsize=8)
        self.mx.set_ylabel("North x (m)", fontsize=8)
        self.mx.tick_params(labelsize=7)
        self.mx.grid(True, alpha=0.3)
        self.mx.set_aspect("equal", adjustable="datalim")
        (self.mx_plan,) = self.mx.plot([], [], color="#9aa5b1", lw=1.2, ls="--", label="plan")
        (self.mx_path,) = self.mx.plot([], [], color="#3ddc97", lw=1.4, label="actual")
        self.mx.plot([0.0], [10.0], "s", color="#9aa5b1", ms=7)
        self.mx.legend(fontsize=6, loc="upper left")
        self.addTab(self.c3, "Mission")
        (self.xy_plan,) = self.xy_ax.plot([], [], color="#9aa5b1", lw=1.0, ls="--")
        from plot_pages import build_pages
        self.pages = build_pages(_Panel)
        for pg in self.pages:
            self.addTab(pg.canvas, pg.name)
        self._plan = None
        self.window_box = QtWidgets.QComboBox()                                      # how much history the plots show
        for sec in (15, 30, 60, 120, 300):
            self.window_box.addItem(f"{sec} s", float(sec))
        self.window_box.setCurrentIndex(max(0, [15.0, 30.0, 60.0, 120.0, 300.0].index(self.window_s)) if self.window_s in (15.0, 30.0, 60.0, 120.0, 300.0) else 2)
        self.window_box.currentIndexChanged.connect(lambda _i: setattr(self, "window_s", float(self.window_box.currentData())))
        self.setCornerWidget(self.window_box, QtCore.Qt.TopRightCorner)

    def set_plan(self, points) -> None:
        """The planned path as [(x, y, ...)]; drawn dashed on the path plots (empty list clears it)."""
        self._plan = ([p[1] for p in points], [p[0] for p in points]) if points else None
        xs, ys = self._plan if self._plan else ([], [])
        self.mx_plan.set_data(xs, ys)
        self.xy_plan.set_data(xs, ys)

    def update_plots(self, w: Dict[str, np.ndarray]) -> None:
        if not w:
            return
        ws = self.window_s
        if "ctrl_progress" in w:
            w["ctrl_progress_pct"] = 100.0 * w["ctrl_progress"]
        for panel in (self.p_depth, self.p_att, self.p_speed):
            panel.set(w["t"], w, "t", ws)
        for panel in (self.p_rpm, self.p_fin):
            panel.set(w.get("cmd_t", np.array([])), w, "cmd_t", ws)
        self.xy_line.set_data(w["y"], w["x"])
        self.xy_model.set_data(w.get("m_y", []), w.get("m_x", []))
        if len(w["x"]):
            self.xy_now.set_data([w["y"][-1]], [w["x"][-1]])
            self.xy_ax.relim()
            self.xy_ax.autoscale_view()
        cur = self.currentWidget()
        if cur is self.c1:
            self.c1.draw_idle()
        elif cur is self.c3:
            for panel, tk in ((self.m_ct, "ctrl_t"), (self.m_hd, "t"), (self.m_sp, "t"), (self.m_dz, "ctrl_t"), (self.m_pr, "ctrl_t")):
                panel.set(w.get(tk, np.array([])), w, tk, ws)
            self.mx_path.set_data(w["y"], w["x"])
            if self._plan or len(w["x"]):
                self.mx.relim()
                self.mx.autoscale_view()
            self.c3.draw_idle()
        elif cur is self.c2:
            for panel in (self.d_err, self.d_elev, self.d_rad, self.d_cnt):
                panel.set(w.get("align_t", np.array([])), w, "align_t", ws)
            self.c2.draw_idle()
        else:
            for page in self.pages:
                if page.canvas is cur:
                    page.update(w, ws)
