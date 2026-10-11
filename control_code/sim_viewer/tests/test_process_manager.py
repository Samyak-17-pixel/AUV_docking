import sys
import time

import pytest
from PyQt5 import QtCore, QtWidgets

from process_manager import ProcessManager

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(cond, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_output_and_exit_are_reported():
    pm = ProcessManager()
    lines, states = [], []
    pm.output.connect(lambda n, t: lines.append((n, t)))
    pm.state_changed.connect(lambda n, r: states.append((n, r)))
    assert pm.start("echo", [sys.executable, "-u", "-c", "print('hello'); print('world')"])
    assert wait_until(lambda: ("echo", False) in states)
    texts = [t for n, t in lines if n == "echo"]
    assert "hello" in texts and "world" in texts and any(t.startswith("stopped (exit 0") for t in texts)
    assert not pm.is_running("echo")


def test_stop_sends_sigint_so_a_ros_style_cleanup_can_run():
    pm = ProcessManager()
    out = []
    pm.output.connect(lambda n, t: out.append(t))
    code = "import time\ntry:\n    print('ready', flush=True)\n    time.sleep(60)\nexcept KeyboardInterrupt:\n    print('cleanup on SIGINT', flush=True)\n"
    assert pm.start("sleeper", [sys.executable, "-u", "-c", code])
    assert wait_until(lambda: "ready" in out)
    assert pm.is_running("sleeper") and pm.running() == ["sleeper"]
    pm.stop("sleeper")
    assert wait_until(lambda: not pm.is_running("sleeper"), 6.0)
    assert "cleanup on SIGINT" in out                                         # it got Ctrl-C, not a kill


def test_a_process_that_ignores_sigint_is_killed_after_the_grace_period():
    pm = ProcessManager()
    code = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nprint('up', flush=True)\ntime.sleep(60)\n"
    out = []
    pm.output.connect(lambda n, t: out.append(t))
    assert pm.start("stubborn", [sys.executable, "-u", "-c", code])
    assert wait_until(lambda: "up" in out)
    pm.stop("stubborn", grace_s=0.5)
    assert wait_until(lambda: not pm.is_running("stubborn"), 5.0)


def test_starting_the_same_name_twice_and_bad_programs_are_refused():
    pm = ProcessManager()
    out = []
    pm.output.connect(lambda n, t: out.append(t))
    assert pm.start("a", [sys.executable, "-c", "import time; time.sleep(30)"])
    assert not pm.start("a", [sys.executable, "-c", "pass"]) and "already running" in out
    assert not pm.start("b", ["/definitely/not/a/program"]) and any("could not start" in t for t in out)
    pm.stop_all()
    assert pm.running() == []
