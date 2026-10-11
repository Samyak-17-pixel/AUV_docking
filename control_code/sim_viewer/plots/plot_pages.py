"""Extra live-plot pages (Pose, Actuators, Tracking, Overview) for plots.LivePlots. Each page is one matplotlib canvas with a few `_Panel`s and an update function that
derives what it shows from the telemetry window dict (so a page works for the live sim, the offline stack and a replay alike)."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import theme

A, O, G, R, Y, P, GR, LT = (theme.SERIES[k] for k in ("blue", "orange", "green", "red", "yellow", "purple", "grey", "light"))
FIN_CAP_DEG = 25.0                      # the soft fin cap in the controllers' yaml (limits.fin_deg_cap)


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def _along(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(len(x))
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


class Page:
    """canvas + the callable that fills it. `panels` are plots._Panel objects (time series); `custom` draws the others."""

    def __init__(self, name: str, fig: Figure, update: Callable[[Dict[str, np.ndarray], float], None]) -> None:
        self.name = name
        self.fig = fig
        self.canvas = FigureCanvasQTAgg(fig)
        self._update = update

    def update(self, w: Dict[str, np.ndarray], window_s: float) -> None:
        self._update(w, window_s)
        self.canvas.draw_idle()


def build_pages(Panel) -> List[Page]:
    pages: List[Page] = []

    # ------------------------------------------------------------------ Pose
    fig = Figure(figsize=(9, 4), constrained_layout=True)
    ax = fig.subplots(2, 3).ravel()
    pn = [Panel(ax[0], "North", "m", [("x", "x north", A)]), Panel(ax[1], "East", "m", [("y", "y east", O)]),
          Panel(ax[2], "Depth (down = deeper)", "m", [("depth", "depth", G)], invert=True),
          Panel(ax[3], "Roll / pitch", "deg", [("roll", "roll", R), ("pitch", "pitch", G)]),
          Panel(ax[4], "Heading", "deg", [("yaw", "yaw", A)]),
          Panel(ax[5], "Body velocity", "m/s", [("u", "surge u", A), ("v", "sway v", O), ("w", "heave w", G)])]
    pages.append(Page("Pose", fig, lambda w, ws: [p.set(w["t"], w, "t", ws) for p in pn]))

    # ------------------------------------------------------------------ Actuators
    fig = Figure(figsize=(9, 4), constrained_layout=True)
    ax = fig.subplots(2, 2).ravel()
    th = Panel(ax[0], "Thruster commands", "RPM", [("cmd_th_01", "th_01 axial", A), ("cmd_th_02", "th_02 heave F", G), ("cmd_th_03", "th_03 heave A", R)])
    fin = Panel(ax[1], "Fin commands (dashed = cap)", "deg", [("cmd_cs_04", "cs_04", A), ("cmd_cs_06", "cs_06", G), ("cmd_cs_07", "cs_07", R), ("cmd_cs_08", "cs_08", P),
                                                              ("cap_hi", "cap", GR, "cmd_t", True), ("cap_lo", "", GR, "cmd_t", True)])
    sat = Panel(ax[2], "Fins at the cap", "% of fins", [("fin_sat", "saturated", Y)])
    eff = Panel(ax[3], "Thruster effort (relative, sum |rpm|^3)", "% of max", [("effort", "effort", O)])

    def upd_act(w, ws):
        t = w.get("cmd_t", np.array([]))
        names = [f"cmd_cs_0{i}" for i in (4, 6, 7, 8)]
        fins = [w[n] for n in names if n in w and len(w[n]) == len(t)]
        w["cap_hi"], w["cap_lo"] = np.full(len(t), FIN_CAP_DEG), np.full(len(t), -FIN_CAP_DEG)
        w["fin_sat"] = (100.0 * np.mean([np.abs(f) >= 0.98 * FIN_CAP_DEG for f in fins], axis=0)) if fins else np.zeros(len(t))
        ths = [w[n] for n in ("cmd_th_01", "cmd_th_02", "cmd_th_03") if n in w and len(w[n]) == len(t)]
        w["effort"] = (100.0 * np.sum([np.abs(x / 1800.0) ** 3 for x in ths], axis=0) / 3.0) if ths else np.zeros(len(t))
        for p in (th, fin, sat, eff):
            p.set(t, w, "cmd_t", ws)
    pages.append(Page("Actuators", fig, upd_act))

    # ------------------------------------------------------------------ Tracking errors
    fig = Figure(figsize=(9, 4), constrained_layout=True)
    ax = fig.subplots(2, 3).ravel()
    e_d = Panel(ax[0], "Depth error (setpoint - depth)", "m", [("e_depth", "error", G)])
    e_h = Panel(ax[1], "Heading error", "deg", [("e_head", "error", P)])
    e_s = Panel(ax[2], "Speed error (setpoint - u)", "m/s", [("e_speed", "error", A)])
    e_c = Panel(ax[3], "Cross-track error", "m", [("ctrl_cross_track_m", "cross-track", R, "ctrl_t")])
    e_r = Panel(ax[4], "Distance to the next waypoint", "m", [("e_dist", "distance", O, "ctrl_t")])
    e_a = Panel(ax[5], "Altitude above the seabed", "m", [("altitude", "altitude", Y, "ctrl_t")])

    def upd_trk(w, ws):
        t, ct = w["t"], w.get("ctrl_t", np.array([]))
        if len(ct) and "ctrl_depth_err_m" in w:
            w["e_depth"] = w["ctrl_depth_err_m"]
        elif len(ct) and "ctrl_depth_sp" in w:
            w["e_depth"] = w["ctrl_depth_sp"] - np.interp(ct, t, w["depth"])
        else:
            w["e_depth"] = np.full(len(ct), np.nan)
        if len(ct) and "ctrl_heading_err_deg" in w:
            w["e_head"] = w["ctrl_heading_err_deg"]
        elif len(ct) and "ctrl_yaw_sp_deg" in w:
            w["e_head"] = _wrap_deg(w["ctrl_yaw_sp_deg"] - np.interp(ct, t, w["yaw"]))
        else:
            w["e_head"] = np.full(len(ct), np.nan)
        w["e_speed"] = (w["ctrl_u_sp"] - np.interp(ct, t, w["u"])) if len(ct) and "ctrl_u_sp" in w else np.full(len(ct), np.nan)
        w["e_dist"] = w.get("ctrl_distance_m", w.get("ctrl_r_xy", np.full(len(ct), np.nan)))
        w["altitude"] = w.get("ctrl_altitude_m", np.full(len(ct), np.nan))
        for p, k in ((e_d, "ctrl_t"), (e_h, "ctrl_t"), (e_s, "ctrl_t"), (e_c, "ctrl_t"), (e_r, "ctrl_t"), (e_a, "ctrl_t")):
            p.set(ct, w, k, ws)
    pages.append(Page("Tracking", fig, upd_trk))

    # ------------------------------------------------------------------ Overview: top view, depth profile, speed profile, odometry rate
    fig = Figure(figsize=(9, 4), constrained_layout=True)
    ax = fig.subplots(2, 2).ravel()
    ax[0].set_title("Top view (path coloured by speed)", fontsize=9)
    ax[0].set_xlabel("East y (m)", fontsize=8); ax[0].set_ylabel("North x (m)", fontsize=8); ax[0].set_aspect("equal", adjustable="datalim"); ax[0].grid(True, alpha=0.3)
    ax[0].tick_params(labelsize=7)
    sc = ax[0].scatter([], [], c=[], s=6, cmap="viridis")
    (now,) = ax[0].plot([], [], "o", color=O, ms=6)
    ax[0].plot([0.0], [10.0], "s", color=GR, ms=7)
    ax[1].set_title("Depth profile along the path", fontsize=9)
    ax[1].set_xlabel("distance along the path (m)", fontsize=8); ax[1].set_ylabel("depth (m)", fontsize=8); ax[1].invert_yaxis(); ax[1].grid(True, alpha=0.3); ax[1].tick_params(labelsize=7)
    (dp,) = ax[1].plot([], [], color=A, lw=1.3)
    sp = Panel(ax[2], "Speed", "m/s", [("u", "u", A)])
    od = Panel(ax[3], "Odometry sample interval", "s", [("odo_dt", "dt", Y)])

    def upd_over(w, ws):
        x, y = w["x"], w["y"]
        if len(x):
            sc.set_offsets(np.column_stack([y, x]))
            sc.set_array(np.asarray(w["u"], float))
            sc.set_clim(0.0, max(float(np.nanmax(w["u"])), 0.5))
            now.set_data([y[-1]], [x[-1]])
            ax[0].relim(); ax[0].autoscale_view()
            d = _along(x, y)
            dp.set_data(d, w["depth"])
            lo, hi = float(np.nanmin(w["depth"])), float(np.nanmax(w["depth"]))
            pad = max(0.1 * (hi - lo), 0.2)
            ax[1].set_xlim(0.0, max(float(d[-1]), 1.0))
            ax[1].set_ylim(hi + pad, lo - pad)
        t = w["t"]
        w["odo_dt"] = np.concatenate([[0.0], np.diff(t)]) if len(t) else np.array([])
        sp.set(t, w, "t", ws)
        od.set(t, w, "t", ws)
    pages.append(Page("Overview", fig, upd_over))
    return pages
