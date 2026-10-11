"""FreezeDetector logic of odom_freeze_monitor.py (no ROS)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odom_freeze_monitor import FreezeDetector  # noqa: E402


def _feed(det, seq, dt=0.2):
    t, out = 0.0, []
    for s in seq:
        e = det.push(t, s)
        if e:
            out.append(e)
        t += dt
    return out


def test_no_freeze_when_every_message_differs():
    d = FreezeDetector(1.0)
    _feed(d, [(i,) for i in range(100)])
    assert d.episodes == [] and not d.frozen


def test_one_freeze_is_found_with_its_duration():
    d = FreezeDetector(1.0)
    seq = [(i,) for i in range(10)] + [(99,)] * 50 + [(i + 200,) for i in range(10)]     # 50 identical messages = 49 intervals of 0.2 s = 9.8 s
    eps = _feed(d, seq)
    assert len(eps) == 1 and abs(eps[0]["duration"] - 9.8) < 0.25
    assert abs(eps[0]["start"] - 10 * 0.2 + 0.2 * 0) < 0.45                                  # starts at the first repeated sample (the last live one)


def test_a_short_repeat_is_not_a_freeze_and_an_open_freeze_is_reported_as_frozen():
    d = FreezeDetector(1.0)
    _feed(d, [(1,), (2,), (2,), (2,), (3,)])            # 0.4 s of repeats: below min_s
    assert d.episodes == []
    d2 = FreezeDetector(1.0)
    _feed(d2, [(1,)] + [(2,)] * 20)
    assert d2.frozen and d2.episodes == []              # not finished yet
