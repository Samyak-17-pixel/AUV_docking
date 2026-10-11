"""Start and stop helper programs (fake vehicle, camera, detector, controllers) from the viewer and capture their output.

Stopping sends SIGINT first: the controllers catch Ctrl-C and publish zero commands on the way out; a process that ignores it is killed after a grace
period. Qt only (QProcess); no ROS.
"""

from __future__ import annotations

import os
import signal
from typing import Dict, List, Optional

from PyQt5 import QtCore


class ProcessManager(QtCore.QObject):
    output = QtCore.pyqtSignal(str, str)            # name, text
    state_changed = QtCore.pyqtSignal(str, bool)    # name, running

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._procs: Dict[str, QtCore.QProcess] = {}
        self._buf: Dict[str, str] = {}

    def is_running(self, name: str) -> bool:
        p = self._procs.get(name)
        return p is not None and p.state() != QtCore.QProcess.NotRunning

    def running(self) -> List[str]:
        return [n for n in self._procs if self.is_running(n)]

    def start(self, name: str, argv: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> bool:
        """Start `argv` under `name`. Returns False (and prints a line to the log) if that name is already running or the program cannot start."""
        if self.is_running(name):
            self.output.emit(name, "already running")
            return False
        p = QtCore.QProcess(self)
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        if cwd:
            p.setWorkingDirectory(cwd)
        e = QtCore.QProcessEnvironment.systemEnvironment()
        for k, v in (env or {}).items():
            e.insert(k, v)
        p.setProcessEnvironment(e)
        p.readyReadStandardOutput.connect(lambda n=name, pr=p: self._read(n, pr))
        p.finished.connect(lambda code, status, n=name: self._finished(n, code, status))
        p.errorOccurred.connect(lambda err, n=name: self.output.emit(n, f"process error {int(err)}") if err == QtCore.QProcess.FailedToStart else None)
        self._procs[name] = p
        self._buf[name] = ""
        p.start(argv[0], argv[1:])
        if not p.waitForStarted(3000):
            self.output.emit(name, f"could not start: {' '.join(argv)}")
            self._procs.pop(name, None)
            return False
        self.output.emit(name, "started: " + " ".join(argv))
        self.state_changed.emit(name, True)
        return True

    def stop(self, name: str, grace_s: float = 3.0) -> None:
        p = self._procs.get(name)
        if p is None or p.state() == QtCore.QProcess.NotRunning:
            return
        pid = int(p.processId())
        if pid > 0:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                return
        QtCore.QTimer.singleShot(int(grace_s * 1000), lambda pr=p: pr.kill() if pr.state() != QtCore.QProcess.NotRunning else None)

    def stop_all(self, wait_ms: int = 4000) -> None:
        for n in self.running():
            self.stop(n)
        for p in list(self._procs.values()):
            if p.state() != QtCore.QProcess.NotRunning:
                p.waitForFinished(wait_ms)
                if p.state() != QtCore.QProcess.NotRunning:
                    p.kill()
                    p.waitForFinished(1000)

    # ------------------------------------------------------------------ internals
    def _read(self, name: str, p: QtCore.QProcess) -> None:
        text = bytes(p.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buf[name] += text
        *lines, rest = self._buf[name].split("\n")
        self._buf[name] = rest
        for ln in lines:
            if ln.strip():
                self.output.emit(name, ln.rstrip())

    def _finished(self, name: str, code: int, status) -> None:
        if self._buf.get(name, "").strip():
            self.output.emit(name, self._buf[name].rstrip())
            self._buf[name] = ""
        how = "crashed" if status == QtCore.QProcess.CrashExit else f"exit {code}"
        self.output.emit(name, f"stopped ({how})")
        self.state_changed.emit(name, False)
