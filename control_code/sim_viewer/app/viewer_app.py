#!/usr/bin/env python3
"""Desktop sim viewer: 3D scene (real vehicle and dock meshes), live plots, the camera image and a status panel.

  ./run_sim_viewer.sh              # live: shows whatever is on the ROS topics (fake vehicle or the real mavsim)
  ./run_sim_viewer.sh --demo       # no ROS needed: scripted manoeuvres in-process, to see what the viewer does

Mouse in the 3D view: left drag = orbit, middle drag or Shift+left drag = pan, right drag or wheel = zoom.
Config: sim_viewer.yaml (section "viewer").
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

# opencv-python points Qt at ITS OWN plugin folder when imported, and that copy cannot start the desktop ("Could not load the Qt platform plugin xcb").
# PyQt5 must use its own plugins, so drop the override before any Qt object exists.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "cv2" in os.environ.get(_var, ""):
        os.environ.pop(_var)

_HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HERE)] + [str(_d) for _d in sorted(_HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(_HERE.parent / "common"))
from outdirs import out_dir  # noqa: E402

from controls_panel import ControlsPanel  # noqa: E402
import hud  # noqa: E402
from plots import LivePlots  # noqa: E402
import theme  # noqa: E402
from process_manager import ProcessManager  # noqa: E402
from scene3d import VIEWS, Scene3D  # noqa: E402
from telemetry import Telemetry  # noqa: E402

DEFAULT_CONFIG = _HERE / "sim_viewer.yaml"


class SceneView(QtWidgets.QWidget):
    """Shows the offscreen-rendered VTK scene and turns mouse input into camera moves."""

    def __init__(self, scene: Scene3D, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._last: Optional[QtCore.QPoint] = None
        self._image: Optional[QtGui.QImage] = None
        self.last_render_ms = 0.0
        self.overlay = None                       # callable(QPainter, QRect) set by the main window: HUD and minimap

    def render_now(self) -> None:
        w, h = max(self.width(), 64), max(self.height(), 64)
        t = time.perf_counter()
        arr = self.scene.render(w, h)
        self.last_render_ms = 1000.0 * (time.perf_counter() - t)
        self._image = QtGui.QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QtGui.QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, _ev) -> None:                                       # noqa: N802
        p = QtGui.QPainter(self)
        if self._image is not None:
            p.drawImage(self.rect(), self._image)
        else:
            p.fillRect(self.rect(), QtGui.QColor(10, 30, 50))
        if self.overlay is not None:
            self.overlay(p, self.rect())

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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: dict, tel: Telemetry, source, scene: Scene3D, view: str = "follow", screenshot_dir: Optional[str] = None) -> None:
        super().__init__()
        self.cfg, self.tel, self.source, self.scene = cfg, tel, source, scene
        v = cfg["viewer"]
        self.setWindowTitle("AUV sim viewer")
        scr = QtWidgets.QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr is not None else QtCore.QRect(0, 0, 1920, 1080)
        self.resize(min(int(v["window"]["width"]), int(avail.width() * 0.97)), min(int(v["window"]["height"]), int(avail.height() * 0.95)))   # never bigger than the screen
        self._extrap = float(v["extrapolate_s"])
        self._paused = False
        self._last_pos: Optional[np.ndarray] = None
        self._frames = 0
        self._fps_t, self._fps = time.perf_counter(), 0.0

        self.view3d = SceneView(scene)
        self.view3d.overlay = self._draw_overlay
        self._show_hud = bool(v.get("hud", True))
        self.plots = LivePlots(window_s=float(v["plot_window_s"]))
        self.bottom = QtWidgets.QTabWidget()                                      # bottom-left: Plots | Detection | Sonar | 3D Trajectory
        self.bottom.addTab(self.plots, "Plots")
        from detection_panel import DetectionPanel
        self.detection = DetectionPanel(tel, float(cfg.get("camera", {}).get("vfov_deg", 60.0)))
        self.bottom.addTab(self.detection, "Detection")
        from sonar_panel import SonarPanel
        tcfg = cfg.get("terrain") or {}
        cx, cy = (tcfg.get("center_m") or [0.0, 0.0])
        half = float(tcfg.get("size_m", 300.0)) / 2.0
        self.sonar_panel = SonarPanel(tel, source, (float(cx) - half, float(cx) + half, float(cy) - half, float(cy) + half))
        self.bottom.addTab(self.sonar_panel, "Sonar")
        self.sonar_panel.enlarge.connect(lambda on: self._enlarge_bottom(on))
        from traj3d_panel import Trajectory3DPanel
        self.traj3d = Trajectory3DPanel(tel, cfg, plan_provider=self._current_plan)
        self.bottom.addTab(self.traj3d, "3D Trajectory")
        self._bottom_tab_touched = False
        self._sonar_shown = False
        self.bottom.tabBarClicked.connect(lambda _i: setattr(self, "_bottom_tab_touched", True))
        self.bottom.setMinimumSize(320, 160)                                      # explicit minimums: a panel's own size hint (e.g. a row of buttons) must never push the window wider
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.setMinimumWidth(480)
        left.addWidget(self.view3d)
        left.addWidget(self.bottom)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)

        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(2, 2, 2, 2)
        self.cam_label = QtWidgets.QLabel("no camera image")                                  # optional small preview (Status tab); the real camera view lives in the Detection tab
        self.cam_label.setFixedSize(320, 240)                                                 # FIXED size: a picture or a long text must never resize the pane or the window
        self.cam_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.cam_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_label.setWordWrap(True)
        self.cam_label.setStyleSheet("background:#050a10;color:#789;border:1px solid #2b3a4b")
        self.cam_label.setVisible(False)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setUsesScrollButtons(False)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setElideMode(QtCore.Qt.ElideRight)
        status_tab = QtWidgets.QWidget()
        stl = QtWidgets.QVBoxLayout(status_tab)
        from widgets import TileGrid
        self.tiles = TileGrid(["DEPTH", "SPEED", "HEADING", "ODOMETRY", "LIGHTS", "MODE"])
        stl.addWidget(self.tiles)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("View:"))
        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems(VIEWS)
        self.view_box.setCurrentText(view)
        self.view_box.currentTextChanged.connect(self._on_view)
        row.addWidget(self.view_box, 1)
        stl.addLayout(row)
        for text, slot in (("Reset trail", self._reset_trail), ("Overview camera", self._overview), ("Pause display", self._toggle_pause), ("Save screenshot", self._screenshot)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            if text == "Pause display":
                self.pause_btn = b
            stl.addWidget(b)
        self.status = QtWidgets.QLabel("")
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.status.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.status.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.status.setWordWrap(True)
        self.chk_cam = QtWidgets.QCheckBox("Small camera preview")
        self.chk_cam.setToolTip("The full camera view is in the Detection tab")
        self.chk_cam.setStyleSheet("QCheckBox { spacing: 6px }")
        self.chk_cam.toggled.connect(self.cam_label.setVisible)
        stl.addWidget(self.chk_cam)
        stl.addWidget(self.cam_label, 0, QtCore.Qt.AlignHCenter)
        stl.addWidget(self.status, 1)
        status_scroll = QtWidgets.QScrollArea()
        status_scroll.setWidgetResizable(True)
        status_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        status_scroll.setWidget(status_tab)
        self.tabs.addTab(status_scroll, "Status")
        self.procs = ProcessManager(self)
        self.controls = ControlsPanel(source, self.procs)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.controls)
        self.tabs.addTab(scroll, "Controls")
        from config_panel import ConfigPanel
        self.config_panel = ConfigPanel(self.controls, tel)
        cfg_scroll = QtWidgets.QScrollArea()
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        cfg_scroll.setWidget(self.config_panel)
        self.tabs.addTab(cfg_scroll, "Config")
        self.tabs.currentChanged.connect(lambda _i: self.config_panel.refresh_from_controls())
        from scenario_panel import ScenarioPanel
        self.scenarios = ScenarioPanel(source, tel, self.controls)
        sc_scroll = QtWidgets.QScrollArea()
        sc_scroll.setWidgetResizable(True)
        sc_scroll.setWidget(self.scenarios)
        self.tabs.addTab(sc_scroll, "Scenarios")
        from history import HistoryPanel, HistoryRecorder
        self.history = HistoryPanel()
        self.tabs.addTab(self.history, "History")
        self.recorder = HistoryRecorder() if bool(v.get("auto_record", True)) and not hasattr(source, "seek") else None
        self.replay_panel = None
        if hasattr(source, "seek"):                       # a ReplaySource: add the transport controls
            from replay_panel import ReplayPanel
            self.replay_panel = ReplayPanel(source, on_seek=self._reset_trail)
            self.tabs.insertTab(1, self.replay_panel, "Replay")
            self.tabs.setCurrentIndex(1)
        sl.addWidget(self.tabs, 1)
        side.setMinimumWidth(380)
        side.setMaximumWidth(760)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(side)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)
        split.setSizes([1280, 500])
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        self._shot_dir = screenshot_dir
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000.0 / float(v["refresh_hz"])))
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._tick_plots)
        self.plot_timer.start(int(1000.0 / float(v["plot_refresh_hz"])))
        self.scene.set_view(view)

    # ---------------------------------------------------------------- slots
    def _on_view(self, name: str) -> None:
        self.scene.set_view(name)

    def _reset_trail(self) -> None:
        self.scene.reset_trail()

    def _overview(self) -> None:
        self.view_box.setCurrentText("free")
        self.scene.set_view("free")
        self.scene.reset_camera_overview()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_btn.setText("Resume display" if self._paused else "Pause display")

    def _screenshot(self) -> Optional[str]:
        d = Path(self._shot_dir or out_dir("sim_viewer_shots"))
        d.mkdir(parents=True, exist_ok=True)
        path = d / time.strftime("viewer_%Y%m%d_%H%M%S.png")
        self.grab().save(str(path))
        self.status.setText(self.status.text() + f"\nsaved {path}")
        return str(path)

    def _draw_overlay(self, p, rect) -> None:
        if not self._show_hud:
            return
        s = self.tel.latest()
        if s is None:
            return
        _, ctl = self.tel.latest_ctrl()
        _, al = self.tel.latest_align()
        info = hud.hud_info(s, self.scene.dock_pos, ctl, al, self.tel.latest_cmd())
        hud.draw_hud(p, rect, info)
        size = max(120, min(200, rect.width() // 4))
        mm = QtCore.QRect(rect.right() - size - 10, rect.top() + 10, size, size)
        pose = self.tel.pose_now(self._extrap)
        g = self.tel.pose_now(self._extrap, model=True)
        axis_yaw = 0.0
        hud.draw_minimap(p, mm, self.scene._trail, pose[0] if pose else s.pos, float((pose[1] if pose else s.eul)[2]), self.scene.dock_pos, axis_yaw,
                         ghost_pos=g[0] if g else None)

    # ---------------------------------------------------------------- periodic updates
    def _tick(self) -> None:
        if self._paused:
            return
        pose = self.tel.pose_now(self._extrap)
        if pose is not None:
            pos, eul, _ = pose
            if self._last_pos is not None and float(np.linalg.norm(pos - self._last_pos)) > 2.0:
                self.scene.reset_trail()          # the vehicle was reset or teleported: do not draw a line across the scene
            self._last_pos = pos.copy()
            self.scene.set_pose(pos, eul)
            self.scene.add_trail_point(pos)
            self.scene.set_actuators(self.tel.latest_cmd())
        now = time.perf_counter()
        self.scene.update_environment(min(now - getattr(self, "_env_t", now), 0.2))
        self._env_t = now
        ghost = self.tel.pose_now(self._extrap, model=True)
        if ghost is not None:
            self.scene.set_ghost(ghost[0], ghost[1])
        else:
            self.scene.hide_ghost()
        if self.recorder is not None:
            self.recorder.observe(self._running_controller(), self.tel)
            self._summarize_finished()
        self.view3d.render_now()
        self._update_camera_image()
        self._update_status()
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self._fps, self._fps_t = self._frames / (now - self._fps_t), now
            self._frames = 0

    def _running_controller(self) -> Optional[str]:
        mine = self.controls.running_controllers()
        if mine:
            return mine[0]
        ext = self.controls.external_controllers()
        if ext:
            return next(iter(ext))
        return None

    def _summarize_finished(self) -> None:
        """When a mission / waypoint run has just been closed by the recorder, write its summary figure next to the CSV (path error, coverage, speed, depth, actuators)."""
        done = getattr(self.recorder, "finished", [])
        if len(done) <= getattr(self, "_n_summarized", 0):
            return
        self._n_summarized = len(done)
        path = done[-1]
        try:
            from history import read_meta
            ctl = read_meta(path).get("controller", "")
            if ctl not in ("mission", "waypoint_tracking"):
                return
            import run_summary
            from replay import load_run
            doc = self.config_panel.docs.get(ctl)
            run = load_run(path)
            plan = doc.path_points((float(run.pos[0, 0]), float(run.pos[0, 1]))) if doc is not None else None
            swath = float((doc.data.get("survey", {}) if doc is not None else {}).get("swath_width_m", 0.0) or 0.0)
            out = path.with_name(path.stem + "_summary.png")
            s = run_summary.save_figure(run, plan, out, swath, path.name)
            self._summary_note = f"run summary: {out}\n" + run_summary.format_summary(s)
        except Exception as exc:                            # a summary problem must never disturb the viewer
            self._summary_note = f"run summary failed: {exc}"

    def _maybe_show_sonar(self) -> None:
        """The first time side-scan pings arrive, bring the Sonar tab to the front (unless the user already picked a bottom tab)."""
        if self._sonar_shown or self._bottom_tab_touched:
            return
        if self.tel.sonar_rows("port", 1) or self.tel.sonar_rows("starboard", 1):
            self._sonar_shown = True
            self.bottom.setCurrentWidget(self.sonar_panel)

    def _enlarge_bottom(self, on: bool) -> None:
        """Give the bottom-left tab area most of the height (the 3D view shrinks to a strip) and back."""
        sp = self.centralWidget().widget(0)
        h = max(sp.height(), 400)
        sp.setSizes([int(h * 0.2), int(h * 0.8)] if on else [int(h * 0.6), int(h * 0.4)])

    def _current_plan(self):
        doc = self.config_panel.doc
        return doc.path_points() if doc is not None and doc.name in ("mission", "waypoint_tracking") else []

    def _tick_plots(self) -> None:
        self._maybe_show_sonar()
        if not self._paused:
            if self.bottom.currentWidget() is self.plots:
                self.plots.update_plots(self.tel.window(self.plots.window_s))
            elif self.bottom.currentWidget() is self.traj3d:
                self.traj3d.refresh()
        try:
            self.config_panel.update_live()
            plan = self.config_panel.doc.path_points() if self.config_panel.doc is not None and self.config_panel.doc.name in ("mission", "waypoint_tracking") else []
            if plan != getattr(self, "_plan_shown", None):
                self._plan_shown = plan
                self.plots.set_plan(plan)
        except Exception:                                   # a bad edit in the Config tab must never stop the live plots
            pass

    def _camera_help(self) -> str:
        """What the empty camera pane says: why there is no picture and what to do (the usual cause is a ROS domain that does not match the camera node's)."""
        counts = getattr(self.source, "counts", None) or {}
        dom = os.environ.get("ROS_DOMAIN_ID", "?")
        lines = ["no camera image", "", f"source: {self.tel.source_name or '(starting)'}", f"ROS domain {dom}: camera frames {counts.get('image', 0)}, odometry {counts.get('odom', 0)}", ""]
        if dom == "42":
            lines.append("domain 42 = the real mavsim bridge: it publishes camera_03 only\nwhile a session is running and the controller is up.")
        elif counts.get("odom", 0) == 0:
            lines.append("nothing on this domain yet: Controls tab -> Start offline stack\n(fake vehicle + synthetic camera + detector), on THIS domain.")
        else:
            lines.append("odometry arrives but no camera: start the synthetic camera\n(Controls tab -> Start offline stack, or sim_viewer/camera/camera_node.py).")
        return "\n".join(lines)

    def _update_camera_image(self) -> None:
        jpeg = self.tel.image()
        if jpeg is None:
            if self.cam_label.pixmap() is None or self.cam_label.pixmap().isNull():
                self.cam_label.setText(self._camera_help())
            return
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(img).scaled(QtCore.QSize(self.cam_label.width() - 2, self.cam_label.height() - 2), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        _, al = self.tel.latest_align()
        if al and self._show_hud:
            pr = QtGui.QPainter(pix)
            hud.draw_detection_overlay(pr, QtCore.QRect(0, 0, pix.width(), pix.height()), rgb.shape[1], rgb.shape[0], al)
            pr.end()
        self.cam_label.setPixmap(pix)

    def _update_tiles(self, s) -> None:
        T = self.tiles.set
        if s is None:
            for n in ("DEPTH", "SPEED", "HEADING", "ODOMETRY"):
                T(n, "-", "idle")
        else:
            age = self.tel.now() - s.t
            T("DEPTH", f"{s.pos[2]:.2f} m", "ok" if 0.3 < s.pos[2] < 60.0 else "warn")
            T("SPEED", f"{s.nu[0]:.2f} m/s", "ok")
            T("HEADING", f"{math.degrees(s.eul[2]) % 360.0:.0f} deg", "ok")
            T("ODOMETRY", f"{age:.1f} s old", "ok" if age < 1.0 else ("warn" if age < 2.0 else "bad"))
        _, al = self.tel.latest_align()
        if al:
            n = int(al.get("num_lights", 0))
            T("LIGHTS", f"{n} / 4" + ("  lock" if al.get("valid") else ""), "ok" if al.get("valid") else ("warn" if n else "bad"))
        else:
            T("LIGHTS", "-", "idle")
        _, ctl = self.tel.latest_ctrl()
        T("MODE", f"{ctl.get('ctrl', '')} {ctl.get('mode', '')}".strip() or "-", "ok" if ctl else "idle")

    def _update_status(self) -> None:
        s = self.tel.latest()
        self._update_tiles(s)
        lines = [f"source : {self.tel.source_name or '(starting)'}", f"assets : {Path(self.scene.asset_source).name}", ""]
        err = getattr(self.source, "error", None)
        if err:
            lines.append(f"!! source error: {err}")
        if s is None:
            lines.append("waiting for odometry...")
        else:
            age = self.tel.now() - s.t
            e = np.degrees(s.eul)
            lines += [
                f"pos  N {s.pos[0]:7.2f}  E {s.pos[1]:6.2f}  D {s.pos[2]:5.2f} m",
                f"att  r {e[0]:6.1f}  p {e[1]:6.1f}  y {e[2]:7.1f} deg",
                f"vel  u {s.nu[0]:6.2f}  q {math.degrees(s.nu[4]):6.1f}  r {math.degrees(s.nu[5]):6.1f}",
                f"dock distance (plane) {self.scene.dock_pos[0] - s.pos[0] - 0.575:6.2f} m",
                f"odometry age {age:4.1f} s" + ("   <-- STALE" if age > 2.0 else ""),
            ]
        cmd = self.tel.latest_cmd()
        if cmd:
            lines += ["", "commands:"] + [f"  {k}: {v:8.1f}" for k, v in sorted(cmd.items())]
        tc, ctl = self.tel.latest_ctrl()
        if ctl:
            lines += ["", f"controller: {ctl.get('ctrl', '?')}  {ctl.get('mode', '')}" + (f"  d={ctl['distance_m']:.2f} m" if isinstance(ctl.get("distance_m"), (int, float)) else "")]
        ta, al = self.tel.latest_align()
        if al:
            lines += ["", f"dock_align ({self.tel.now() - ta:4.1f} s ago):",
                      f"  valid {int(al.get('valid', 0))}  lights {int(al.get('num_lights', 0))}",
                      f"  err_x {al.get('error_x_px', 0):7.1f}  err_y {al.get('error_y_px', 0):7.1f} px",
                      f"  elev {al.get('elevation_deg', 0):6.1f} deg  radius {al.get('radius_px', 0):6.1f} px"]
        if getattr(self, "_summary_note", ""):
            lines += ["", self._summary_note]
        counts = getattr(self.source, "counts", None)
        if counts:
            lines += ["", "messages: " + "  ".join(f"{k} {v}" for k, v in counts.items())]
        lines += ["", f"view {self.view3d.last_render_ms:5.1f} ms render, {self._fps:4.1f} fps"]
        self.status.setText("\n".join(lines))

    def closeEvent(self, ev) -> None:                                        # noqa: N802
        try:
            if self.recorder is not None:
                self.recorder.close()
            self.procs.stop_all()
            self.source.stop()
        finally:
            super().closeEvent(ev)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="AUV sim viewer")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="in-process scripted demo, no ROS needed")
    src.add_argument("--ros", action="store_true", help="read the ROS topics (default)")
    ap.add_argument("--replay", type=Path, default=None, metavar="CSV", help="play back a recorded run (record_run.py, dof_testing or station_keeping CSV)")
    ap.add_argument("--overlay", action="store_true", help="with --replay: also run the vehicle model on the recorded commands and show it as a ghost + dashed lines")
    ap.add_argument("--overlay-mode", choices=["free", "segments"], default="segments", help="free = one run from the first sample; segments = restart from the recording every --segment-s")
    ap.add_argument("--segment-s", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--view", choices=VIEWS, default="follow")
    ap.add_argument("--bottom-tab", default="", help="show this tab of the bottom-left plot area first: Vehicle, 'Dock detector', Mission or Detection")
    ap.add_argument("--side-tab", default="", help="show this tab of the right-hand panel first: Status, Controls, Config, Scenarios, History")
    ap.add_argument("--size", default="", metavar="WxH", help="window size in pixels (default: the viewer.window block of the yaml, never larger than the screen)")
    ap.add_argument("--run-seconds", type=float, default=0.0, help="close by itself after this long (used by the tests)")
    ap.add_argument("--screenshot", type=Path, default=None, help="with --run-seconds: save the window to this PNG before closing")
    ap.add_argument("--time-scale", type=float, default=1.0, help="demo speed-up")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    theme.apply(app)
    tel = Telemetry()
    if args.replay:
        from replay import ReplaySource, UnsupportedLog, load_run
        try:
            run = load_run(args.replay)
            model = None
            if args.overlay:
                from compare import simulate
                model = simulate(run, mode=args.overlay_mode, segment_s=args.segment_s)
        except UnsupportedLog as exc:
            print(f"viewer: {exc}", file=sys.stderr)
            return 2
        source = ReplaySource(tel, run, speed=args.speed, loop=args.loop, model=model)
    elif args.demo:
        from demo_source import DemoSource
        source = DemoSource(tel, cfg, time_scale=args.time_scale)
    else:
        from ros_link import RosLink
        source = RosLink(tel, cfg)
    source.start()
    source.ready.wait(5.0)
    if getattr(source, "error", None):
        print(f"viewer: data source failed: {source.error}", file=sys.stderr)
    scene = Scene3D(cfg)
    win = MainWindow(cfg, tel, source, scene, view=args.view)
    if args.bottom_tab:
        want = args.bottom_tab.lower()
        for i in range(win.bottom.count()):                                    # Plots | Detection | Sonar | 3D Trajectory
            if win.bottom.tabText(i).lower() == want:
                win.bottom.setCurrentIndex(i)
                win._bottom_tab_touched = True
        for i in range(win.plots.count()):                                     # or one of the plot pages: Vehicle, 'Dock detector', Mission, Pose, Actuators, Tracking, Overview
            if win.plots.tabText(i).lower() == want:
                win.bottom.setCurrentWidget(win.plots)
                win.plots.setCurrentIndex(i)
                win._bottom_tab_touched = True
    if args.size:
        w_, h_ = (int(x) for x in args.size.lower().split("x"))
        win.resize(w_, h_)
    for i in range(win.tabs.count()):
        if args.side_tab and win.tabs.tabText(i).lower() == args.side_tab.lower():
            win.tabs.setCurrentIndex(i)
    win.show()
    if args.run_seconds > 0:
        def _finish() -> None:
            if args.screenshot:
                win.tick_for_shot = True
                win.grab().save(str(args.screenshot))
            win.close()
            app.quit()
        QtCore.QTimer.singleShot(int(args.run_seconds * 1000), _finish)
    rc = app.exec_()
    source.stop()
    source.join(3.0)                       # let the ROS thread leave rclpy before the interpreter exits (otherwise: "terminate called ... core dumped")
    try:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except ImportError:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
