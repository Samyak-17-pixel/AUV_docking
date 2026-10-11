"""assess() of real_sim_preflight.py (no ROS)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_sim_preflight import assess  # noqa: E402


def _s(i, z=3.0):
    return (i * 0.22, (0.01 * i, 0.0, z, 0.0, 0.0, 0.0, 1.0, 0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_healthy_odometry_is_ready():
    ok, msgs = assess([_s(i) for i in range(36)], 8.0)
    assert ok, msgs


def test_no_messages_means_sim_not_there():
    ok, msgs = assess([], 8.0)
    assert not ok and "no odometry" in msgs[0]


def test_frozen_odometry_is_flagged():
    base = _s(1)[1]
    ok, msgs = assess([(i * 0.22, base) for i in range(36)], 8.0)
    assert not ok and any("FROZEN" in m for m in msgs)


def test_too_slow_and_too_shallow_are_flagged():
    ok, msgs = assess([(i * 1.0, _s(i)[1]) for i in range(6)], 8.0)            # one message per second
    assert not ok and any("rate below" in m for m in msgs)
    ok, msgs = assess([_s(i, z=0.4) for i in range(36)], 8.0)
    assert not ok and any("surface" in m for m in msgs)


def test_slow_discovery_at_the_start_does_not_count_as_a_slow_topic():
    late = [(2.5 + i * 0.22, _s(i)[1]) for i in range(8)]                       # first message only after 2.5 s, then 4.5 Hz
    ok, msgs = assess(late, 4.0)
    assert ok, msgs
