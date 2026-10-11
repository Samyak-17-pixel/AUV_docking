"""The 'Controls' tab of the viewer: drive the offline simulation (pause, step, reset, push, time scale) and start/stop controllers with edited gains.

Simulation controls only work with the offline fake vehicle (or the in-process demo); the real mavsim cannot be paused or reset from here, so they stay
disabled there. Starting a controller while connected to the real mavsim domain (42) asks for confirmation first, because the vehicle will move.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

import gains_editor
import live_gains
import yaml_edit
from collapsible import Collapsible
from process_manager import ProcessManager

ROOT = Path(__file__).resolve().parents[2]            # control_code/
REPO = ROOT.parent                                    # AUV_docking/
HERE = Path(__file__).resolve().parents[1]

DOF_NAMES = ["surge", "heave", "pitch", "yaw", "roll", "sway"]

# name -> (script, default config, extra-argument builder taking the panel)
CONTROLLERS: Dict[str, dict] = {
    "dof_testing": {"script": ROOT / "dof_testing" / "dof_testing.py", "config": ROOT / "dof_testing" / "dof_testing.yaml", "uses_dof": True},
    "station_keeping": {"script": ROOT / "station_keeping" / "station_keeping.py", "config": ROOT / "station_keeping" / "station_keeping.yaml"},
    "terminal_docking": {"script": ROOT / "terminal_docking_control" / "terminal_docking.py", "config": ROOT / "terminal_docking_control" / "terminal_docking.yaml"},
    "dock_test": {"script": ROOT / "dock_test" / "dock_test.py", "config": ROOT / "dock_test" / "dock_test.yaml"},
    "depth_control": {"script": ROOT / "depth_control" / "depth_control.py", "config": ROOT / "depth_control" / "depth_control.yaml"},
    "waypoint_tracking": {"script": ROOT / "waypoint_tracking" / "waypoint_tracking.py", "config": ROOT / "waypoint_tracking" / "waypoint_tracking.yaml"},
    "mission": {"script": ROOT / "mission" / "mission.py", "config": ROOT / "mission" / "mission.yaml"},
}

PUSH_PRESETS = {
    "sideways shove (Y 6 N, 1.5 s)": ([0, 6, 0, 0, 0, 0], 1.5),
    "downward push (Z +8 N, 1 s)": ([0, 0, 8, 0, 0, 0], 1.0),
    "upward push (Z -8 N, 1 s)": ([0, 0, -8, 0, 0, 0], 1.0),
    "forward shove (X 6 N, 1 s)": ([6, 0, 0, 0, 0, 0], 1.0),
    "nose-up kick (M +1 N*m, 0.5 s)": ([0, 0, 0, 0, 1, 0], 0.5),
    "yaw kick (N +0.5 N*m, 1 s)": ([0, 0, 0, 0, 0, 0.5], 1.0),
}
WRENCH_LABELS = ["X [N]", "Y [N]", "Z [N] +dn", "K [N*m]", "M [N*m] +up", "N [N*m]"]
STACK = ("fake_vehicle", "camera_node", "detector", "sidescan")


def find_external_controllers(skip_pids=()) -> Dict[str, List[int]]:
    """Controllers running OUTSIDE the viewer (started in a terminal): name -> pids, found by looking for their script path in /proc/*/cmdline.
    Only our own known controller scripts are matched, and only processes of the same user."""
    found: Dict[str, List[int]] = {}
    me = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return found
    for e in entries:
        if not e.isdigit() or int(e) in (me, *skip_pids):
            continue
        try:
            with open(f"/proc/{e}/cmdline", "rb") as f:
                parts = f.read().split(b"\0")
        except OSError:
            continue
        for name, spec in CONTROLLERS.items():
            if any(p.decode("utf-8", "ignore").endswith(str(Path(spec["script"]).name)) and str(Path(spec["script"]).parent.name) in p.decode("utf-8", "ignore") for p in parts[:4]):
                found.setdefault(name, []).append(int(e))
    return found


def _spin(lo: float, hi: float, val: float, step: float = 0.1, dec: int = 2) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setMaximumWidth(92)
    return s


class ControlsPanel(QtWidgets.QWidget):
    def __init__(self, source, procs: ProcessManager, parent=None, confirm: Optional[Callable[[str, str], bool]] = None) -> None:
        super().__init__(parent)
        self.source = source
        self.procs = procs
        self._confirm = confirm or self._ask
        self._ext_finder: Callable[[], Dict[str, List[int]]] = lambda: find_external_controllers([int(p.processId()) for p in getattr(self.procs, "_procs", {}).values()])
        self._pending_start: Optional[str] = None               # controller to start as soon as the one being stopped has exited
        self._external: Dict[str, List[int]] = {}
        self._edited: Dict[str, dict] = {}                       # controller name -> edited config dict
        self._temp: Dict[str, Path] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # ---------------------------------------------------------------- simulation
        self.sim_box = QtWidgets.QGroupBox("Simulation (offline fake vehicle / demo)")
        sl = QtWidgets.QGridLayout(self.sim_box)
        self.sim_status = QtWidgets.QLabel("")
        self.sim_status.setWordWrap(True)
        sl.addWidget(self.sim_status, 0, 0, 1, 4)
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_step = QtWidgets.QPushButton("Step 0.5 s")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_step.clicked.connect(self._on_step)
        sl.addWidget(self.btn_pause, 1, 0, 1, 2)
        sl.addWidget(self.btn_step, 1, 2, 1, 2)
        sl.addWidget(QtWidgets.QLabel("Time scale"), 2, 0)
        self.scale = QtWidgets.QComboBox()
        for v in ("0.25", "0.5", "1", "2", "4", "8"):
            self.scale.addItem(v + "x", float(v))
        self.scale.setCurrentIndex(2)
        self.scale.currentIndexChanged.connect(self._on_scale)
        sl.addWidget(self.scale, 2, 1)
        sl.addWidget(QtWidgets.QLabel("Reset pose:"), 3, 0)
        self.reset_spins = {k: _spin(lo, hi, v) for k, (lo, hi, v) in {"x": (-50, 50, 0.0), "y": (-50, 50, 0.8), "z": (0, 50, 3.0), "yaw": (-180, 180, 0.0)}.items()}
        for i, (k, w) in enumerate(self.reset_spins.items()):
            sl.addWidget(QtWidgets.QLabel(k + (" [deg]" if k == "yaw" else " [m]")), 4 + i // 2, (i % 2) * 2)
            sl.addWidget(w, 4 + i // 2, (i % 2) * 2 + 1)
        self.btn_reset = QtWidgets.QPushButton("Reset vehicle to this pose")
        self.btn_reset.clicked.connect(self._on_reset)
        sl.addWidget(self.btn_reset, 6, 0, 1, 4)
        sl.addWidget(QtWidgets.QLabel("Push (external force / moment):"), 7, 0, 1, 4)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("(custom)")
        for name in PUSH_PRESETS:
            self.preset.addItem(name)
        self.preset.currentTextChanged.connect(self._on_preset)
        sl.addWidget(self.preset, 8, 0, 1, 4)
        self.wrench = [_spin(-100, 100, 0.0, 0.5) for _ in range(6)]
        for i, w in enumerate(self.wrench):
            sl.addWidget(QtWidgets.QLabel(WRENCH_LABELS[i]), 9 + i // 2, (i % 2) * 2)
            sl.addWidget(w, 9 + i // 2, (i % 2) * 2 + 1)
        sl.addWidget(QtWidgets.QLabel("Duration [s]"), 12, 0)
        self.duration = _spin(0.05, 60, 1.0, 0.25)
        sl.addWidget(self.duration, 12, 1)
        self.btn_push = QtWidgets.QPushButton("Push")
        self.btn_clear = QtWidgets.QPushButton("Clear push")
        self.btn_push.clicked.connect(self._on_push)
        self.btn_clear.clicked.connect(lambda: self._send({"cmd": "clear_push"}))
        sl.addWidget(self.btn_push, 12, 2)
        sl.addWidget(self.btn_clear, 12, 3)
        self.sec_sim = Collapsible("Simulation (offline fake vehicle / demo)", self.sim_box, expanded=False)
        self.sim_box.setTitle("")

        # ---------------------------------------------------------------- controllers
        cb = QtWidgets.QGroupBox("Controllers")
        cl = QtWidgets.QGridLayout(cb)
        cl.addWidget(QtWidgets.QLabel("Controller"), 0, 0)
        self.ctrl = QtWidgets.QComboBox()
        self.ctrl.addItems(list(CONTROLLERS))
        self.ctrl.currentTextChanged.connect(self._on_ctrl_changed)
        cl.addWidget(self.ctrl, 0, 1, 1, 3)
        self.dof_label = QtWidgets.QLabel("dof / mode")
        self.dof = QtWidgets.QComboBox()
        self.dof.addItems(DOF_NAMES)
        self.dof.setCurrentText("heave")
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["step", "hold"])
        cl.addWidget(self.dof_label, 1, 0)
        cl.addWidget(self.dof, 1, 1)
        cl.addWidget(self.mode, 1, 2)
        self.btn_gains = QtWidgets.QPushButton("Edit gains...")
        self.btn_gains.clicked.connect(self._on_edit_gains)
        self.gains_label = QtWidgets.QLabel("file values")
        cl.addWidget(self.btn_gains, 2, 0, 1, 2)
        cl.addWidget(self.gains_label, 2, 2, 1, 2)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop running controller")
        self.btn_estop = QtWidgets.QPushButton("EMERGENCY STOP")
        self.btn_estop.setStyleSheet("QPushButton{background:#b3261e;color:white;font-weight:bold}QPushButton:disabled{background:#999}")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_estop.clicked.connect(self._on_estop)
        cl.addWidget(self.btn_start, 3, 0, 1, 2)
        cl.addWidget(self.btn_stop, 3, 2, 1, 2)
        cl.addWidget(self.btn_estop, 4, 0, 1, 4)
        self.ctrl_state = QtWidgets.QLabel("no controller running")
        self.ctrl_state.setWordWrap(True)
        cl.addWidget(self.ctrl_state, 5, 0, 1, 4)
        self.sec_ctrl = Collapsible("Controllers", cb, expanded=True)
        cb.setTitle("")

        # ---------------------------------------------------------------- live gains
        gb = QtWidgets.QGroupBox("Live gains (change a number while the controller runs)")
        gl = QtWidgets.QGridLayout(gb)
        self.lg_loop = QtWidgets.QComboBox()
        self.lg_key = QtWidgets.QComboBox()
        self.lg_key.addItems(["kp", "kd", "ki", "lpf_tau_s", "sp_tau_s", "i_max", "max"])
        self.lg_value = QtWidgets.QDoubleSpinBox()
        self.lg_value.setDecimals(4)
        self.lg_value.setRange(0.0, 100000.0)
        self.lg_value.setSingleStep(0.1)
        self.lg_auto = QtWidgets.QCheckBox("apply while I change it")
        self.lg_auto.setChecked(True)
        self.lg_apply = QtWidgets.QPushButton("Apply now")
        self.lg_down = QtWidgets.QPushButton("x 0.8")
        self.lg_up = QtWidgets.QPushButton("x 1.25")
        self.lg_save = QtWidgets.QPushButton("Save as new default...")
        self.lg_info = QtWidgets.QLabel("")
        self.lg_info.setWordWrap(True)
        gl.addWidget(self.lg_loop, 0, 0)
        gl.addWidget(self.lg_key, 0, 1)
        gl.addWidget(self.lg_value, 0, 2, 1, 2)
        gl.addWidget(self.lg_down, 1, 0)
        gl.addWidget(self.lg_up, 1, 1)
        gl.addWidget(self.lg_apply, 1, 2)
        gl.addWidget(self.lg_auto, 1, 3)
        gl.addWidget(self.lg_save, 2, 0, 1, 4)
        gl.addWidget(self.lg_info, 3, 0, 1, 4)
        self._lg_sent: Dict[tuple, float] = {}
        self.lg_loop.currentTextChanged.connect(self._lg_load)
        self.lg_key.currentTextChanged.connect(self._lg_load)
        self.lg_value.valueChanged.connect(self._lg_changed)
        self.lg_apply.clicked.connect(self._lg_send)
        self.lg_down.clicked.connect(lambda: self.lg_value.setValue(self.lg_value.value() * 0.8))
        self.lg_up.clicked.connect(lambda: self.lg_value.setValue(self.lg_value.value() * 1.25))
        self.lg_save.clicked.connect(self._lg_save)
        self.sec_gains = Collapsible("Live gains (change a number while the controller runs)", gb, expanded=False)
        gb.setTitle("")

        # ---------------------------------------------------------------- offline stack
        sb = QtWidgets.QGroupBox("Offline closed-loop stack")
        stl = QtWidgets.QVBoxLayout(sb)
        self.stack_note = QtWidgets.QLabel("Starts the fake vehicle (at the reset pose above), the synthetic camera (Detection tab), the real detector (no windows unless ticked) and the simulated side-scan sonars (Sonar tab).")
        self.stack_note.setWordWrap(True)
        stl.addWidget(self.stack_note)
        self.chk_realistic = QtWidgets.QCheckBox("Realistic odometry")
        self.chk_realistic.setToolTip("4.5 Hz, 0.25 s latency, noise and freezes like the real sim")
        self.chk_realistic.setChecked(True)
        stl.addWidget(self.chk_realistic)
        self.chk_sonar = QtWidgets.QCheckBox("Side-scan sonars (Sonar tab)")
        self.chk_sonar.setToolTip("Start the simulated side-scan sonars too: pings for the Sonar tab waterfall + mosaic, and the altimeter")
        self.chk_sonar.setChecked(True)
        stl.addWidget(self.chk_sonar)
        self.chk_det_windows = QtWidgets.QCheckBox("Detector pop-up windows")
        self.chk_det_windows.setToolTip("Dock camera, Bloom mask and Dock align windows (needs a display)")
        self.chk_det_windows.setChecked(False)
        stl.addWidget(self.chk_det_windows)
        row = QtWidgets.QHBoxLayout()
        self.btn_stack = QtWidgets.QPushButton("Start offline stack")
        self.btn_stack_stop = QtWidgets.QPushButton("Stop stack")
        self.btn_stack.clicked.connect(self._on_stack_start)
        self.btn_stack_stop.clicked.connect(self._on_stack_stop)
        row.addWidget(self.btn_stack)
        row.addWidget(self.btn_stack_stop)
        stl.addLayout(row)
        self.sec_stack = Collapsible("Offline closed-loop stack", sb, expanded=True)
        sb.setTitle("")
        for sec in (self.sec_stack, self.sec_ctrl, self.sec_gains, self.sec_sim):          # most used first
            outer.addWidget(sec)
        self.btn_stack.setProperty("role", "primary")
        self.btn_estop.setProperty("role", "danger")
        self.btn_start.setProperty("role", "primary")
        outer.addStretch(0)
        # ---------------------------------------------------------------- log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setMinimumHeight(140)
        outer.addWidget(QtWidgets.QLabel("Process output"))
        outer.addWidget(self.log, 1)

        self.procs.output.connect(self._on_output)
        self.procs.state_changed.connect(lambda *_: self._refresh())
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(500)
        self._on_ctrl_changed(self.ctrl.currentText())
        self._refresh()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ask(title: str, text: str) -> bool:
        return QtWidgets.QMessageBox.question(None, title, text, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes

    def _say(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _on_output(self, name: str, text: str) -> None:
        self._say(f"[{name}] {text}")

    @property
    def domain(self) -> str:
        return os.environ.get("ROS_DOMAIN_ID", "0")

    def _on_real_domain(self) -> bool:
        return self.domain == "42"

    def _send(self, d: dict) -> None:
        if not self.source.sim_controls_available:
            self._say("[viewer] simulation controls need the offline fake vehicle; nothing sent")
            return
        self.source.send_sim_command(d)

    # ------------------------------------------------------------------ simulation slots
    def _on_pause(self) -> None:
        paused = bool(self.source.sim_status.get("paused", False)) if self.source.sim_status else False
        self._send({"cmd": "pause", "value": not paused})

    def _on_step(self) -> None:
        self._send({"cmd": "pause", "value": True})
        self._send({"cmd": "step", "seconds": 0.5})

    def _on_scale(self) -> None:
        self._send({"cmd": "time_scale", "value": float(self.scale.currentData())})

    def _on_reset(self) -> None:
        r = {k: float(w.value()) for k, w in self.reset_spins.items()}
        self._send({"cmd": "reset", "pos": [r["x"], r["y"], r["z"]], "eul_deg": [0.0, 0.0, r["yaw"]]})

    def _on_preset(self, name: str) -> None:
        if name in PUSH_PRESETS:
            w, dur = PUSH_PRESETS[name]
            for sp, v in zip(self.wrench, w):
                sp.setValue(float(v))
            self.duration.setValue(float(dur))

    def _on_push(self) -> None:
        self._send({"cmd": "push", "wrench": [float(w.value()) for w in self.wrench], "duration": float(self.duration.value())})

    # ------------------------------------------------------------------ controllers
    def _spec(self) -> dict:
        return CONTROLLERS[self.ctrl.currentText()]

    def _on_ctrl_changed(self, name: str) -> None:
        uses = bool(CONTROLLERS[name].get("uses_dof"))
        for w in (self.dof_label, self.dof, self.mode):
            w.setVisible(uses)
        self.gains_label.setText("edited (%d values)" % len(gains_editor.changed_leaves(gains_editor.load_yaml(CONTROLLERS[name]["config"]), self._edited[name])) if name in self._edited else "file values")
        if hasattr(self, "lg_loop"):
            self._lg_refill_loops(name)
        if hasattr(self, "btn_start"):
            self._refresh()                     # Start/Stop must reflect the NEWLY selected controller at once, not after the next timer tick

    # ------------------------------------------------------------------ live gains
    def _lg_gains(self, name: str) -> dict:
        base = self._edited.get(name) or gains_editor.load_yaml(CONTROLLERS[name]["config"])
        return base.get("gains", {}) if isinstance(base, dict) else {}

    def _lg_refill_loops(self, name: str) -> None:
        cur = self.lg_loop.currentText()
        loops = [k for k, v in self._lg_gains(name).items() if isinstance(v, dict) and "kp" in v]
        self.lg_loop.blockSignals(True)
        self.lg_loop.clear()
        self.lg_loop.addItems(loops)
        if cur in loops:
            self.lg_loop.setCurrentText(cur)
        self.lg_loop.blockSignals(False)
        self._lg_load()

    def _lg_load(self, *_) -> None:
        name, loop, key = self.ctrl.currentText(), self.lg_loop.currentText(), self.lg_key.currentText()
        v = self._lg_sent.get((name, loop, key))
        if v is None:
            v = self._lg_gains(name).get(loop, {}).get(key, 0.0)
        self.lg_value.blockSignals(True)
        self.lg_value.setValue(float(v))
        self.lg_value.blockSignals(False)

    def _lg_changed(self, _v: float) -> None:
        if self.lg_auto.isChecked():
            self._lg_send()

    def _lg_send(self) -> None:
        name, loop, key, val = self.ctrl.currentText(), self.lg_loop.currentText(), self.lg_key.currentText(), float(self.lg_value.value())
        if not loop:
            return
        sender = getattr(self.source, "send_gain", None)
        if not callable(sender) or not sender(live_gains.make_message(name, loop, key, val)):
            self.lg_info.setText("not sent: this data source cannot reach a running controller (use the ROS source with a controller running)")
            return
        self._lg_sent[(name, loop, key)] = val
        self.lg_info.setText(f"sent {name}: {loop}.{key} = {val:g}  (the controller logs 'live gain ...' when it applies it)")

    def _lg_save(self) -> None:
        name, loop, key, val = self.ctrl.currentText(), self.lg_loop.currentText(), self.lg_key.currentText(), float(self.lg_value.value())
        path = CONTROLLERS[name]["config"]
        if not self._confirm("Save as default", f"Write {loop}.{key} = {val:g} into\n{path}\n\nThe comments in the file are kept. Continue?"):
            self.lg_info.setText("not saved")
            return
        try:
            yaml_edit.save_gain(path, loop, key, val)
        except (KeyError, OSError) as exc:
            self.lg_info.setText(f"not saved: {exc}")
            return
        self._edited.pop(name, None)
        self.lg_info.setText(f"saved {loop}.{key} = {val:g} to {Path(path).name}")

    def _on_edit_gains(self) -> None:
        name = self.ctrl.currentText()
        path = CONTROLLERS[name]["config"]
        base = gains_editor.load_yaml(path)
        dlg = gains_editor.make_dialog(self._edited.get(name, base), f"Edit {name} parameters", self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            edited = dlg.edited_config()
            n = len(gains_editor.changed_leaves(base, edited))
            if n:
                self._edited[name] = edited
            else:
                self._edited.pop(name, None)
            self._on_ctrl_changed(name)

    def config_for_start(self, name: str) -> Path:
        """The YAML the controller is started with: the edited temporary copy if there are edits, else the original file."""
        if name in self._edited:
            self._temp[name] = gains_editor.write_temp_yaml(self._edited[name], name)
            return self._temp[name]
        return CONTROLLERS[name]["config"]

    def controller_argv(self, name: str) -> List[str]:
        spec = CONTROLLERS[name]
        argv = [sys.executable, "-u", str(spec["script"]), "--config", str(self.config_for_start(name))]
        if spec.get("uses_dof"):
            argv += ["--dof", self.dof.currentText(), "--mode", self.mode.currentText()]
        return argv

    def running_controllers(self) -> List[str]:
        """Controllers started from this window that are still running."""
        return [n for n in CONTROLLERS if self.procs.is_running(n)]

    def external_controllers(self) -> Dict[str, List[int]]:
        return {n: pids for n, pids in self._external.items() if n not in self.running_controllers()}

    def _start_now(self, name: str) -> None:
        if self._on_real_domain() and not self.source.sim_controls_available:
            if not self._confirm("Real simulator", f"ROS domain 42 is the REAL mavsim. Starting {name} will move the vehicle.\n\nMake sure teleop is stopped and the vehicle is in open water.\n\nStart it?"):
                self._say("[viewer] cancelled")
                return
        self.procs.start(name, self.controller_argv(name), cwd=str(Path(CONTROLLERS[name]["script"]).parent))

    def _on_start(self) -> None:
        name = self.ctrl.currentText()
        others = [n for n in self.running_controllers() if n != name] + [n for n in self.external_controllers()]
        if name in self.running_controllers():
            self._say(f"[viewer] {name} is already running")
            return
        if others:
            if not self._confirm("Switch controller", f"{', '.join(others)} is running and only one controller may publish actuator_cmd.\n\nStop it and start {name}?"):
                self._say("[viewer] cancelled")
                return
            self._pending_start = name
            self._stop_all_controllers()
            self._say(f"[viewer] stopping {', '.join(others)}, then starting {name}")
            return
        self._start_now(name)

    def _stop_all_controllers(self, grace_s: float = 3.0) -> None:
        for n in self.running_controllers():
            self.procs.stop(n, grace_s)
        for n, pids in list(self.external_controllers().items()):
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGINT)                        # the controllers publish neutral on Ctrl-C
                except (ProcessLookupError, PermissionError):
                    pass
            self._say(f"[viewer] sent Ctrl-C to {n} (started outside the viewer, pid {', '.join(map(str, pids))})")

    def _on_stop(self) -> None:
        self._pending_start = None
        self._stop_all_controllers()

    def _on_estop(self) -> None:
        """Stop everything NOW: Ctrl-C with a 1 s grace, then kill, and publish neutral commands ourselves."""
        self._pending_start = None
        self._stop_all_controllers(grace_s=1.0)
        zero = getattr(self.source, "publish_zero", None)
        if callable(zero):
            zero()
        self._say("[viewer] EMERGENCY STOP: controllers stopped, neutral actuator command sent")

    def _maybe_start_pending(self) -> None:
        if self._pending_start and not self.running_controllers() and not self.external_controllers():
            name, self._pending_start = self._pending_start, None
            self._start_now(name)

    # ------------------------------------------------------------------ stack
    def stack_argvs(self) -> Dict[str, List[str]]:
        r = {k: w.value() for k, w in self.reset_spins.items()}
        return {
            "fake_vehicle": [sys.executable, "-u", str(ROOT / "sim_offline" / "fake_vehicle.py"), "--x", str(r["x"]), "--y", str(r["y"]), "--z", str(r["z"]), "--yaw", str(r["yaw"])] + (["--realistic"] if self.chk_realistic.isChecked() else []),
            "camera_node": [sys.executable, "-u", str(HERE / "camera" / "camera_node.py"), "--config", str(HERE / "sim_viewer.yaml")],
            "sidescan": [sys.executable, "-u", str(HERE / "sonar" / "sidescan_node.py"), "--config", str(HERE / "sim_viewer.yaml")],
            "detector": [sys.executable, "-u", str(REPO / "dock_detection_algo" / "live_dock_lights.py")] + (["--align-window"] if self.chk_det_windows.isChecked() else ["--no-gui"])
            + ["--config", str(REPO / "dock_detection_algo" / "dock_detection.yaml")],
        }

    def _on_stack_start(self) -> None:
        if self._on_real_domain():
            self._say("[viewer] refused: ROS domain 42 is the real mavsim. Restart the viewer with ./run_sim_viewer.sh (it now defaults to the private domain 77)")
            return
        for name, argv in self.stack_argvs().items():
            if name == "sidescan" and not self.chk_sonar.isChecked():
                continue
            if not self.procs.is_running(name):
                self.procs.start(name, argv, cwd=str(ROOT))

    def _on_stack_stop(self) -> None:
        for name in reversed(STACK):
            self.procs.stop(name)

    # ------------------------------------------------------------------ periodic
    def _refresh(self) -> None:
        avail = bool(self.source.sim_controls_available)
        self.sim_box.setEnabled(avail)
        st = self.source.sim_status or {}
        if avail:
            self.btn_pause.setText("Resume" if st.get("paused") else "Pause")
            idx = self.scale.findData(float(st.get("time_scale", 1.0)))
            if idx >= 0 and idx != self.scale.currentIndex():
                self.scale.blockSignals(True)               # show the real state without sending a command back
                self.scale.setCurrentIndex(idx)
                self.scale.blockSignals(False)
            bits = ["PAUSED" if st.get("paused") else "running", f"time scale {st.get('time_scale', 1.0):g}x"]
            if st.get("push_left", 0) > 0:
                bits.append(f"pushing {st['push_left']:.1f} s")
            if st.get("error"):
                bits.append("error: " + str(st["error"]))
            self.sim_status.setText(" | ".join(bits))
        else:
            self.sim_status.setText("Not available: no offline fake vehicle answering. The real mavsim cannot be paused or reset from here. Use 'Start offline stack' (private ROS domain) or run the viewer with --demo.")
        name = self.ctrl.currentText()
        self._external = self._ext_finder()
        mine, ext = self.running_controllers(), self.external_controllers()
        self.btn_start.setEnabled(name not in mine)
        self.btn_stop.setEnabled(bool(mine or ext))                     # stops WHATEVER runs, not only the controller selected in the list
        self.btn_estop.setEnabled(True)
        bits = [f"{n} (this window)" for n in mine] + [f"{n} (started in a terminal, pid {', '.join(map(str, p))})" for n, p in ext.items()]
        self.ctrl_state.setText((("RUNNING: " + "; ".join(bits)) if bits else "no controller running") + f"   [ROS domain {self.domain}]")
        self._maybe_start_pending()
        wanted = [n for n in STACK if n != "sidescan" or self.chk_sonar.isChecked()]
        stack_running = [n for n in wanted if self.procs.is_running(n)]
        self.btn_stack.setEnabled(len(stack_running) < len(wanted) and not self._on_real_domain())
        self.btn_stack_stop.setEnabled(bool(stack_running))
