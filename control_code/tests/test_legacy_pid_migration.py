"""depth_control and waypoint_tracking used to carry a private copy of Pid; depth_control now imports common/pid.py (2026-10-10), and waypoint_tracking was rebuilt
on common/loops.py (which uses the same Pid; see waypoint_tracking/test_waypoint_tracking_core.py).

The migration is only safe if the shared class gives the SAME numbers as the old copy for the way depth_control calls it (update(error, dt), no rate). The old
class is reproduced here verbatim as the reference.
"""

from __future__ import annotations

import importlib
import random
import re
import sys
from pathlib import Path
from typing import Optional

import pytest

_CTRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CTRL / "common"))

from pid import Pid as SharedPid  # noqa: E402


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class LegacyPid:                       # verbatim copy of the class that was in depth_control.py / waypoint_tracking.py
    def __init__(self, kp, ki, kd, i_max, d_filter_tau_s=0.05):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_max = abs(i_max)
        self.d_filter_tau_s = max(0.0, d_filter_tau_s)
        self.integral = 0.0
        self._prev_error: Optional[float] = None
        self._d_filtered = 0.0

    def update(self, error, dt):
        if dt <= 0.0:
            return self.kp * error
        self.integral += error * dt
        if self.ki > 1e-12:
            self.integral = _clamp(self.integral, -self.i_max / self.ki, self.i_max / self.ki)
        else:
            self.integral = 0.0
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        if self.d_filter_tau_s > 0.0:
            alpha = dt / (self.d_filter_tau_s + dt)
            self._d_filtered += alpha * (derivative - self._d_filtered)
            derivative = self._d_filtered
        return self.kp * error + self.ki * self.integral + self.kd * derivative


@pytest.mark.parametrize("gains", [
    dict(kp=400.0, ki=20.0, kd=300.0, i_max=200.0, d_filter_tau_s=0.05),     # depth-like
    dict(kp=2.0, ki=0.0, kd=0.5, i_max=80.0, d_filter_tau_s=0.0),            # no integral, no filter
    dict(kp=1.0, ki=5.0, kd=0.0, i_max=10.0, d_filter_tau_s=0.2),            # integral hits its clamp
])
def test_shared_pid_matches_the_old_private_copy(gains):
    rng = random.Random(7)
    a, b = LegacyPid(**gains), SharedPid(**gains)
    for i in range(2000):
        e = 3.0 * rng.uniform(-1, 1) * (1 if i % 400 < 300 else 20)      # includes large errors that saturate the integral
        dt = 0.05 if i % 97 else 0.0                                     # and the dt <= 0 branch
        assert a.update(e, dt) == pytest.approx(b.update(e, dt), rel=1e-12, abs=1e-12)
    assert a.integral == pytest.approx(b.integral, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("rel", ["depth_control/depth_control.py", "waypoint_tracking/waypoint_tracking.py", "waypoint_tracking/waypoint_tracking_core.py"])
def test_no_node_defines_its_own_pid(rel):
    src = (_CTRL / rel).read_text()
    assert not re.search(r"^class Pid\b", src, re.M)


def test_depth_control_uses_the_shared_pid_import():
    src = (_CTRL / "depth_control/depth_control.py").read_text()
    assert re.search(r"^from pid import Pid, clamp", src, re.M)


@pytest.mark.parametrize("module,rel", [("depth_control", "depth_control")])
def test_the_nodes_import_and_use_the_shared_class(module, rel):
    pytest.importorskip("rclpy")
    pytest.importorskip("interfaces.msg")
    sys.path.insert(0, str(_CTRL / rel))
    try:
        mod = importlib.import_module(module)
    finally:
        sys.path.remove(str(_CTRL / rel))
    assert mod.Pid is SharedPid and mod.clamp is not None
