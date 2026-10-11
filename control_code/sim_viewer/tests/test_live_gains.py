"""Live gain changes (HoldLoops.set_gain, message format) and in-place YAML saving that keeps comments."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from live_gains import apply_message, make_message
from loops import HoldLoops
from state import State
from yaml_edit import save_gain, save_gains, set_gain_in_text

GAINS = {"heave": {"kp": 10.0, "ki": 0.0, "kd": 20.0, "i_max": 5.0, "max": 25.0}, "pitch": {"kp": 1.0, "ki": 0.0, "kd": 4.0, "i_max": 1.0, "max": 4.0}}


def test_set_gain_changes_the_running_pid():
    h = HoldLoops({k: dict(v) for k, v in GAINS.items()})
    st = State(pos=np.array([0.0, 0.0, 3.0]))
    a = h.heave(st, 4.0, 0.05)
    assert h.set_gain("heave", "kp", 20.0) and h.pids["heave"].kp == 20.0
    h.reset()
    b = h.heave(st, 4.0, 0.05)
    assert b > 1.5 * a                                       # doubled kp (the D term is zero here)
    assert not h.set_gain("heave", "nonsense", 1.0) and not h.set_gain("nothere", "kp", 1.0) and not h.set_gain("heave", "kp", -1.0) and not h.set_gain("heave", "kp", float("nan"))
    assert h.gains["heave"]["kp"] == 20.0


def test_message_round_trip_and_addressing():
    h = HoldLoops({k: dict(v) for k, v in GAINS.items()})
    line = apply_message(make_message("terminal_docking", "pitch", "kd", 6.5), "terminal_docking", h)
    assert "pitch.kd" in line and h.pids["pitch"].kd == 6.5
    assert apply_message(make_message("station_keeping", "pitch", "kd", 9.0), "terminal_docking", h) is None      # addressed to another controller
    assert h.pids["pitch"].kd == 6.5
    assert apply_message(json.dumps({"loop": "heave", "key": "kp", "value": 3.0}), "anyone", h) and h.pids["heave"].kp == 3.0     # no addressee = everyone
    assert apply_message("not json", "x", h) is None and apply_message("[1]", "x", h) is None and apply_message('{"loop": "heave"}', "x", h) is None
    assert apply_message(make_message("", "heave", "kp", -5.0), "x", h).startswith("rejected")


FLOW = """node:
  rate_hz: 20  # keep

gains:    # all loops
  heave: {kp: 7.5, ki: 0.009, kd: 20.0, i_max: 4.0, max: 25.0}   # N per m
  pitch: {kp: 0.444, ki: 0.001, kd: 4.667}
other:
  heave: {kp: 99.0}
"""
BLOCK = """gains:
  heave:
    kp: 60.0       # N per m: stiffness
    ki: 0.5
    kd: 60.0       # damping
  pitch:
    kp: 6.0
"""


def test_flow_style_edit_keeps_comments_and_only_touches_the_gains_section():
    out = set_gain_in_text(FLOW, "heave", "kp", 12.25)
    assert "heave: {kp: 12.25, ki: 0.009, kd: 20.0, i_max: 4.0, max: 25.0}   # N per m" in out
    assert "other:\n  heave: {kp: 99.0}" in out and "# keep" in out and "# all loops" in out
    assert yaml.safe_load(out)["gains"]["heave"]["kp"] == 12.25


def test_block_style_edit_keeps_the_comment_next_to_the_number():
    out = set_gain_in_text(BLOCK, "heave", "kd", 33.0)
    assert "    kd: 33  " in out and "# damping" in out and "kp: 60.0       # N per m: stiffness" in out
    assert yaml.safe_load(out)["gains"]["pitch"]["kp"] == 6.0


def test_edit_errors_leave_the_file_unchanged(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(FLOW)
    for loop, key in (("roll", "kp"), ("heave", "zz")):
        with pytest.raises(KeyError):
            save_gain(f, loop, key, 1.0)
    with pytest.raises(KeyError):
        save_gains(f, [("heave", "kp", 1.0), ("heave", "nope", 2.0)])                     # all or nothing
    assert f.read_text() == FLOW
    assert save_gains(f, [("heave", "kp", 8.0), ("pitch", "kd", 5.0)]) == 2
    d = yaml.safe_load(f.read_text())["gains"]
    assert d["heave"]["kp"] == 8.0 and d["pitch"]["kd"] == 5.0


def test_every_shipped_controller_yaml_can_be_edited_for_its_gain_blocks():
    root = Path(__file__).resolve().parents[2]
    for rel, loop in (("terminal_docking_control/terminal_docking.yaml", "heave"), ("dof_testing/dof_testing.yaml", "heave"), ("station_keeping/station_keeping.yaml", "heave")):
        text = (root / rel).read_text()
        out = set_gain_in_text(text, loop, "kp", 1.2345)
        assert yaml.safe_load(out)["gains"][loop]["kp"] == 1.2345, rel
        assert out.count("\n") == text.count("\n")                                            # no line added or lost
