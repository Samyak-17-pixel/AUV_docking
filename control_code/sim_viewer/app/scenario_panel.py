"""The 'Scenarios' tab: run a docking scenario live on the offline stack with a pass/fail judge, or run the whole docking grid as a batch and show a scoreboard.

Live: puts the fake vehicle at a start pose (range / lateral offset / heading offset relative to the dock axis), starts `terminal_docking` and judges the run
with the failure list agreed for this project (scenario_judge.LiveJudge: wall contact, bad entry, overshoot, retry, attitude/depth limits, pinned thrusters, timeout).
Batch: runs terminal_docking_control/terminal_docking_eval.py in a separate process (in-process closed loop, ground-truth judge) and lists every run.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

_HERE_SP = Path(__file__).resolve().parents[1]
if str(_HERE_SP.parent / "common") not in sys.path:
    sys.path.insert(0, str(_HERE_SP.parent / "common"))
from outdirs import out_dir  # noqa: E402

from PyQt5 import QtCore, QtGui, QtWidgets

from scenario_judge import LiveJudge

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "terminal_docking_control" / "terminal_docking_eval.py"
DOCK_X = 10.0
START_DEPTH = 3.0


def _spin(lo: float, hi: float, val: float, step: float, suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(2)
    s.setSuffix(suffix)
    return s


class ScenarioPanel(QtWidgets.QWidget):
    def __init__(self, source, tel, controls, parent=None) -> None:
        super().__init__(parent)
        self.source, self.tel, self.controls = source, tel, controls
        self.judge = LiveJudge()
        self.clock: Callable[[], float] = time.monotonic          # injectable for tests
        self._running = False
        self._last_truth_t = -1.0
        self._last_sample_t = -1.0
        self.batch_proc: Optional[QtCore.QProcess] = None
        self._batch_csv: Optional[Path] = None
        lay = QtWidgets.QVBoxLayout(self)

        # ------------------------------------------------------------ live
        gb = QtWidgets.QGroupBox("Live docking scenario (offline stack + terminal_docking)")
        g = QtWidgets.QGridLayout(gb)
        self.range = _spin(3.0, 14.0, 7.0, 0.5, " m")
        self.lateral = _spin(-5.0, 5.0, 0.0, 0.25, " m")
        self.heading = _spin(-60.0, 60.0, 0.0, 5.0, " deg")
        for i, (lab, w) in enumerate((("Range to dock", self.range), ("Lateral offset (+ east)", self.lateral), ("Heading offset (+ east of axis)", self.heading))):
            g.addWidget(QtWidgets.QLabel(lab), i, 0, 1, 2)
            g.addWidget(w, i, 2)
        self.btn_run = QtWidgets.QPushButton("Run scenario")
        self.btn_abort = QtWidgets.QPushButton("Abort")
        self.btn_run.clicked.connect(self.run_live)
        self.btn_abort.clicked.connect(self.abort_live)
        g.addWidget(self.btn_run, 3, 0, 1, 2)
        g.addWidget(self.btn_abort, 3, 2)
        self.verdict = QtWidgets.QLabel("no run yet")
        self.verdict.setWordWrap(True)
        self.verdict.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.verdict.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        g.addWidget(self.verdict, 4, 0, 1, 3)
        lay.addWidget(gb)

        # ------------------------------------------------------------ batch
        bb = QtWidgets.QGroupBox("Batch scoreboard (in-process closed loop, ground-truth judge)")
        b = QtWidgets.QGridLayout(bb)
        self.grid_box = QtWidgets.QComboBox()
        self.grid_box.addItems(["smoke", "moderate", "wide"])
        self.grid_box.setCurrentText("moderate")
        self.plants_box = QtWidgets.QComboBox()
        self.plants_box.addItems(["all", "nominal", "heavy+late", "light+fast", "stress"])
        self.vision_box = QtWidgets.QComboBox()
        self.vision_box.addItems(["perfect", "detector"])
        for i, (lab, w) in enumerate((("Grid", self.grid_box), ("Plants", self.plants_box), ("Vision", self.vision_box))):
            b.addWidget(QtWidgets.QLabel(lab), 0, 2 * i)
            b.addWidget(w, 0, 2 * i + 1)
        self.btn_batch = QtWidgets.QPushButton("Run batch")
        self.btn_batch_stop = QtWidgets.QPushButton("Stop")
        self.btn_export = QtWidgets.QPushButton("Export CSV...")
        self.btn_batch.clicked.connect(self.run_batch)
        self.btn_batch_stop.clicked.connect(self.stop_batch)
        self.btn_export.clicked.connect(self.export_csv)
        b.addWidget(self.btn_batch, 1, 0, 1, 2)
        b.addWidget(self.btn_batch_stop, 1, 2, 1, 2)
        b.addWidget(self.btn_export, 1, 4, 1, 2)
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        b.addWidget(self.summary, 2, 0, 1, 6)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["plant", "range", "lateral", "heading", "result", "why"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(180)
        b.addWidget(self.table, 3, 0, 1, 6)
        lay.addWidget(bb, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(200)
        self._refresh_buttons()

    # ------------------------------------------------------------------ live run
    def start_pose(self) -> Dict[str, list]:
        return {"pos": [DOCK_X - float(self.range.value()), float(self.lateral.value()), START_DEPTH], "eul_deg": [0.0, 0.0, float(self.heading.value())]}

    def run_live(self) -> None:
        if not self.source.sim_controls_available:
            self.verdict.setText("Not available: no offline fake vehicle answering. Use the Controls tab's 'Start offline stack' first (private ROS domain).")
            return
        c = self.controls
        c._pending_start = None
        pose = self.start_pose()
        self.source.send_sim_command({"cmd": "pause", "value": False})
        self.source.send_sim_command({"cmd": "reset", **pose})
        self.judge.reset()
        self._running, self._last_truth_t = True, -1.0
        self.verdict.setText("starting...")
        c.ctrl.setCurrentText("terminal_docking")
        if c.running_controllers() or c.external_controllers():
            c._pending_start = "terminal_docking"                        # started automatically once the old controller has exited
            c._stop_all_controllers()
        else:
            QtCore.QTimer.singleShot(600, lambda: c._start_now("terminal_docking"))      # let the reset arrive first
        self._refresh_buttons()

    def abort_live(self) -> None:
        self.controls._on_stop()
        self._running = False
        self.verdict.setText(self.verdict.text() + "\n(aborted)")
        self._refresh_buttons()

    def _sample(self):
        """(t, pos, eul, u) from the fake vehicle's ground truth if present, else from odometry."""
        st = getattr(self.source, "sim_status", None) or {}
        tr = st.get("truth")
        now = self.clock()
        if tr and len(tr) >= 7:
            key = tuple(round(v, 6) for v in tr)
            if key != getattr(self, "_last_truth", None):
                self._last_truth = key
                return now, tr[:3], tr[3:6], tr[6], "truth"
            return None
        s = self.tel.latest()
        if s is not None and s.t != self._last_sample_t:
            self._last_sample_t = s.t
            return now, s.pos, s.eul, float(s.nu[0]), "odometry"
        return None

    def _tick(self) -> None:
        if self._running:
            smp = self._sample()
            if smp is not None:
                t, pos, eul, u, how = smp
                _, ctl = self.tel.latest_ctrl()
                self.judge.feed(t, pos, eul, u, str(ctl.get("mode", "")), ctl.get("retries") or 0, self.tel.latest_cmd())
                self.verdict.setText(self.judge.summary() + f"\n(judged on {how})")
                if self.judge.verdict in ("PASS", "FAIL"):
                    self._running = False
                    self.controls._on_stop()
                    self._refresh_buttons()
        self._refresh_buttons()

    # ------------------------------------------------------------------ batch
    def batch_argv(self, csv_path: Path) -> List[str]:
        return [sys.executable, "-u", str(EVAL), "--grid", self.grid_box.currentText(), "--plants", self.plants_box.currentText(),
                "--vision", self.vision_box.currentText(), "--csv", str(csv_path), "--show-pass"]

    def run_batch(self) -> None:
        if self.batch_proc is not None and self.batch_proc.state() != QtCore.QProcess.NotRunning:
            return
        self._batch_csv = out_dir("sim_viewer_runs") / time.strftime("scoreboard_%Y%m%d_%H%M%S.csv")
        self._batch_csv.parent.mkdir(parents=True, exist_ok=True)
        self.table.setRowCount(0)
        self.summary.setText("running ... (the wide grid with the detector takes several minutes)")
        p = QtCore.QProcess(self)
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        p.finished.connect(self._batch_done)
        p.readyReadStandardOutput.connect(lambda: self._batch_out(p))
        self._batch_text = ""
        self.batch_proc = p
        argv = self.batch_argv(self._batch_csv)
        p.start(argv[0], argv[1:])
        self._refresh_buttons()

    def _batch_out(self, p: QtCore.QProcess) -> None:
        self._batch_text += bytes(p.readAllStandardOutput()).decode("utf-8", "replace")

    def stop_batch(self) -> None:
        if self.batch_proc is not None and self.batch_proc.state() != QtCore.QProcess.NotRunning:
            self.batch_proc.kill()

    def _batch_done(self, code: int, status) -> None:
        head = [l for l in self._batch_text.splitlines() if l.startswith(("PASS", "failure reasons", "by plant"))]
        self.load_csv(self._batch_csv, head)
        self._refresh_buttons()

    def load_csv(self, path: Optional[Path], head: Optional[List[str]] = None) -> None:
        rows: List[dict] = []
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        except (OSError, TypeError):
            self.summary.setText("no results (the batch did not finish)\n" + "\n".join((self._batch_text or "").splitlines()[-6:]))
            return
        self.table.setRowCount(len(rows))
        n_pass = 0
        for i, r in enumerate(rows):
            ok = r.get("pass") == "1"
            n_pass += ok
            why = r.get("failures") or ("retries=%s" % r.get("retries") if r.get("retries") not in (None, "", "0.0", "0") else "")
            vals = [r.get("plant"), r.get("range"), r.get("lateral"), r.get("heading"), "PASS" if ok else "FAIL", why]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                if j == 4:
                    it.setForeground(QtGui.QColor(30, 150, 60) if ok else QtGui.QColor(200, 50, 40))
                self.table.setItem(i, j, it)
        self.summary.setText(f"{n_pass}/{len(rows)} first-try passes" + ("\n" + "\n".join(head) if head else ""))

    def export_csv(self) -> None:
        if not self._batch_csv or not Path(self._batch_csv).exists():
            self.summary.setText("nothing to export yet")
            return
        self.summary.setText(self.summary.text() + f"\nresults are in {self._batch_csv}")

    def _refresh_buttons(self) -> None:
        batch_on = self.batch_proc is not None and self.batch_proc.state() != QtCore.QProcess.NotRunning
        self.btn_batch.setEnabled(not batch_on)
        self.btn_batch_stop.setEnabled(batch_on)
        self.btn_run.setEnabled(not self._running)
        self.btn_abort.setEnabled(self._running)
