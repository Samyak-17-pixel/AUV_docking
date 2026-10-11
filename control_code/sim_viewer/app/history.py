"""Run history: every controller run seen by the viewer is recorded to outputs/sim_viewer_runs/auto_<controller>_<time>.csv, and the 'History' tab lists them for
replay, overlay with the model, comparison of two runs, and deletion. Qt for the panel; the recorder logic (HistoryRecorder) is plain Python."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from recording import CsvRecorder

HERE = Path(__file__).resolve().parents[1]
_HERE_O = Path(__file__).resolve().parents[1]
if str(_HERE_O.parent / "common") not in sys.path:
    sys.path.insert(0, str(_HERE_O.parent / "common"))
from outdirs import out_dir  # noqa: E402
RUN_DIR = out_dir("sim_viewer_runs")


class HistoryRecorder:
    """Call `observe` regularly: it starts a file when a controller starts running, feeds every new odometry sample, and closes the file shortly after it stops."""

    def __init__(self, directory: Optional[Path] = None, idle_close_s: float = 3.0, min_rows: int = 10) -> None:
        self.dir = Path(directory) if directory else RUN_DIR
        self.idle_close_s = idle_close_s
        self.min_rows = min_rows
        self.rec: Optional[CsvRecorder] = None
        self.path: Optional[Path] = None
        self._last_sample_t = -1.0
        self._idle_since: Optional[float] = None
        self.finished: List[Path] = []

    @property
    def recording(self) -> bool:
        return self.rec is not None

    def observe(self, controller: Optional[str], tel, now: Optional[float] = None, extra_meta: Optional[dict] = None) -> None:
        now = time.monotonic() if now is None else now
        if controller:
            self._idle_since = None
            if self.rec is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                self.path = self.dir / f"auto_{controller}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                self.rec = CsvRecorder(self.path, {"recorder": "sim_viewer history", "controller": controller, "source": tel.source_name, **(extra_meta or {})})
                self._last_sample_t = -1.0
                from recording import SonarRecorder
                self.sonar_rec = SonarRecorder()
                self._sonar_seen = {"port": -1.0, "starboard": -1.0}
                self._t_first = None
        elif self.rec is not None:
            self._idle_since = self._idle_since if self._idle_since is not None else now
            if now - self._idle_since >= self.idle_close_s:
                self.close()
                return
        if self.rec is None:
            return
        s = tel.latest()
        if s is None or s.t == self._last_sample_t:
            return
        self._last_sample_t = s.t
        cmd = tel.latest_cmd()
        if cmd:
            self.rec.set_cmd(list(cmd), list(cmd.values()), s.t)
        _, al = tel.latest_align()
        if al:
            self.rec.set_align(bool(al.get("valid", 0)), int(al.get("num_lights", 0)), al.get("error_x_px", 0.0), al.get("error_y_px", 0.0), al.get("radius_px", 0.0), al.get("elevation_deg", 0.0))
        self.rec.add_state(s.t, s.pos, s.eul, s.nu)
        if getattr(self, "_t_first", None) is None:
            self._t_first = s.t
        for side in ("port", "starboard"):                    # side-scan pings seen during the run are saved next to the CSV (<run>_sidescan.npz)
            for (tt, inten, meta) in tel.sonar_rows(side, 60):
                if tt > self._sonar_seen[side]:
                    self._sonar_seen[side] = tt
                    self.sonar_rec.add(max(tt - self._t_first, 0.0), side, inten, meta)

    def close(self) -> Optional[Path]:
        if self.rec is None:
            return None
        rows, path = self.rec.rows, self.path
        self.rec.close()
        sr = getattr(self, "sonar_rec", None)
        if sr is not None and len(sr) > 0 and path is not None and rows >= self.min_rows:
            sr.save(path.with_name(path.stem + "_sidescan.npz"))
        self.sonar_rec = None
        self.rec, self.path = None, None
        if rows < self.min_rows and path is not None:                  # a start that was stopped at once is not worth keeping
            try:
                path.unlink()
            except OSError:
                pass
            return None
        self.finished.append(path)
        return path


def read_meta(path: Path) -> dict:
    meta = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.startswith("#"):
                    break
                k, _, v = ln[1:].partition(":")
                meta[k.strip()] = v.strip()
    except OSError:
        pass
    return meta


def list_runs(directory: Optional[Path] = None) -> List[Path]:
    d = Path(directory) if directory else RUN_DIR
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.csv") if not p.name.startswith("scoreboard_")], key=lambda p: p.stat().st_mtime, reverse=True)


class HistoryPanel(QtWidgets.QWidget):
    def __init__(self, directory: Optional[Path] = None, confirm: Optional[Callable[[str, str], bool]] = None, parent=None) -> None:
        super().__init__(parent)
        self.dir = Path(directory) if directory else RUN_DIR
        self._confirm = confirm or (lambda t, m: QtWidgets.QMessageBox.question(None, t, m) == QtWidgets.QMessageBox.Yes)
        self.children_procs: List[QtCore.QProcess] = []
        lay = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["run", "controller", "started", "rows"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_replay = QtWidgets.QPushButton("Replay")
        self.btn_overlay = QtWidgets.QPushButton("Replay + model")
        self.btn_compare = QtWidgets.QPushButton("Compare 2 selected")
        self.btn_delete = QtWidgets.QPushButton("Delete")
        for b in (self.btn_refresh, self.btn_replay, self.btn_overlay, self.btn_compare, self.btn_delete):
            row.addWidget(b)
        lay.addLayout(row)
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(170)
        self.info.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(self.info)
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.image)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_replay.clicked.connect(lambda: self.replay(False))
        self.btn_overlay.clicked.connect(lambda: self.replay(True))
        self.btn_compare.clicked.connect(self.compare)
        self.btn_delete.clicked.connect(self.delete)
        self.refresh()

    def selected(self) -> List[Path]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.dir / self.table.item(r, 0).text() for r in rows]

    def refresh(self) -> None:
        runs = list_runs(self.dir)
        self.table.setRowCount(len(runs))
        for i, p in enumerate(runs):
            m = read_meta(p)
            vals = [p.name, m.get("controller", m.get("label", "")), m.get("started", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))), str(max(0, sum(1 for _ in open(p, "rb")) - len([1 for k in m]) - 1))]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(v))

    def _launch(self, args: List[str]) -> QtCore.QProcess:
        p = QtCore.QProcess(self)
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.remove("QT_QPA_PLATFORM_PLUGIN_PATH")
        p.setProcessEnvironment(env)
        p.start(sys.executable, args)
        self.children_procs.append(p)
        return p

    def replay(self, overlay: bool) -> None:
        sel = self.selected()
        if not sel:
            self.info.setPlainText("select a run first")
            return
        self._launch(["-u", str(HERE / "app" / "viewer_app.py"), "--replay", str(sel[0])] + (["--overlay"] if overlay else []))
        self.info.setPlainText(f"opened {sel[0].name} in a new viewer window")

    def compare(self) -> None:
        sel = self.selected()
        if len(sel) != 2:
            self.info.setPlainText("select exactly two runs (Ctrl+click)")
            return
        import compare_runs
        from replay import UnsupportedLog, load_run
        try:
            a, b = load_run(sel[0]), load_run(sel[1])
        except UnsupportedLog as exc:
            self.info.setPlainText(str(exc))
            return
        png = self.dir / "compare_last.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        compare_runs.plot_two(a, b, png, sel[0].name, sel[1].name)
        self.info.setPlainText(compare_runs.format_table(compare_runs.difference_table(a, b)))
        pix = QtGui.QPixmap(str(png))
        self.image.setPixmap(pix.scaledToWidth(max(300, self.width() - 30), QtCore.Qt.SmoothTransformation))

    def delete(self) -> None:
        sel = self.selected()
        if not sel or not self._confirm("Delete runs", "Delete %d file(s)?\n%s" % (len(sel), "\n".join(p.name for p in sel))):
            return
        for p in sel:
            try:
                p.unlink()
            except OSError:
                pass
        self.refresh()

    def closeEvent(self, ev) -> None:                                         # noqa: N802
        for p in self.children_procs:
            if p.state() != QtCore.QProcess.NotRunning:
                p.terminate()
        super().closeEvent(ev)
