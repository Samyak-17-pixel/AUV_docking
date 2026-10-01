"""End-to-end (no ROS): DofTest and StationKeeper against the offline vehicle model."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from dof_testing import DOFS, DEFAULT_CONFIG, DofTest, load_config, run_offline
from state import State
from station_keeping_core import StationKeeper, shrink
from vehicle_model import VehicleModel

SK_CFG = Path(__file__).resolve().parents[1] / "station_keeping" / "station_keeping.yaml"


def test_vehicle_floats_up_without_thrust():
    m = VehicleModel()
    for _ in range(500):
        m.step(0.01)
    assert m.pos[2] < 3.0                                            # z decreases = rises (NED)


def test_negative_rpm_on_heave_thrusters_dives():
    m = VehicleModel()
    m.set_command({"th_02": -1800, "th_03": -1800})
    for _ in range(500):
        m.step(0.01)
    assert m.pos[2] > 3.0


@pytest.mark.parametrize("dof", DOFS)
@pytest.mark.parametrize("mode", ["step", "hold"])
def test_every_dof_passes_offline(dof, mode):
    t = run_offline(load_config(DEFAULT_CONFIG), dof, mode, VehicleModel())
    assert t.result["verdict"] == "PASS", t.result


def test_wrong_sign_actuator_is_caught_by_step_test():
    cfg = load_config(DEFAULT_CONFIG)
    t = DofTest(cfg, "heave", "step")
    orig = t.alloc.force_to_rpm
    t.alloc.force_to_rpm = lambda n, f: -orig(n, f)                 # simulate heave thrusters wired backwards
    m = VehicleModel()
    while not t.done and t.t < 60:
        out, _ = t.update(State.from_model(m), 0.05)
        m.set_command(out)
        for _ in range(5):
            m.step(0.01)
    assert t.result["verdict"] == "FAIL"


def _run_sk(cfg, T, ext=None, ext_window=(5.0, 15.0)):
    m = VehicleModel()
    sk = StationKeeper(cfg)
    st = State.from_model(m)
    while not sk.feed_capture(st, 0.05):
        st = State.from_model(m)
    t, peak_depth, peak_ahead = 0.0, 0.0, 0.0
    while t < T:
        st = State.from_model(m)
        m.ext[:] = 0
        if ext is not None and ext_window[0] <= t < ext_window[1]:
            m.ext[:] = ext
        out, s = sk.update(st, 0.05)
        m.set_command(out)
        for _ in range(5):
            m.step(0.01)
        peak_depth = max(peak_depth, abs(s["depth_err_m"]))
        peak_ahead = max(peak_ahead, abs(s["ahead_err_m"]))
        t += 0.05
    return sk, s, peak_depth, peak_ahead


def test_station_keeping_holds_still_water():
    cfg = yaml.safe_load(open(SK_CFG))
    _, s, pd, pa = _run_sk(cfg, 30)
    assert pd < 0.1 and pa < 0.05 and abs(s["pitch_err_deg"]) < 0.5


def test_station_keeping_rejects_downward_push_and_current():
    cfg = yaml.safe_load(open(SK_CFG))
    _, s, pd, _ = _run_sk(cfg, 60, ext=[0, 0, 6, 0, 0, 0])
    assert pd < 0.2 and abs(s["depth_err_m"]) < 0.05
    _, s, _, pa = _run_sk(cfg, 60, ext=[4, 0, 0, 0, 0, 0])
    assert pa < 0.6 and abs(s["ahead_err_m"]) < 0.1


def test_low_rpm_cap_cannot_hold_depth():
    """Documents the finding: 800 RPM cap cannot counter buoyancy + a 6 N push."""
    cfg = yaml.safe_load(open(SK_CFG))
    cfg["limits"]["rpm_cap"] = 800
    _, _, pd, _ = _run_sk(cfg, 60, ext=[0, 0, 6, 0, 0, 0])
    assert pd > 1.0


def test_heading_not_held_without_flow():
    cfg = yaml.safe_load(open(SK_CFG))
    sk, s, _, _ = _run_sk(cfg, 30, ext=[0, 0, 0, 0, 0, 0.3])
    assert s["fin_authority"] == 0.0 and abs(s["heading_err_deg"]) > 20


def test_deadband_shrink():
    assert shrink(0.01, 0.02) == 0.0
    assert shrink(0.05, 0.02) == pytest.approx(0.03)
    assert shrink(-0.05, 0.02) == pytest.approx(-0.03)
    assert shrink(0.5, 0.0) == 0.5


def test_safety_leash_and_depth():
    cfg = yaml.safe_load(open(SK_CFG))
    sk = StationKeeper(cfg)
    st = State.from_model(VehicleModel())
    sk.capture(st)
    assert sk.safety_reason(st) is None
    st2 = State.from_model(VehicleModel(pos=(cfg["safety"]["max_drift_m"] + 1, 0, 3)))
    assert "drifted" in sk.safety_reason(st2)
    assert "shallow" in sk.safety_reason(State.from_model(VehicleModel(pos=(0, 0, 0.1))))


def _captured_depth(trim: bool) -> float:
    cfg = yaml.safe_load(open(SK_CFG))
    cfg["capture"]["average_s"] = 3.0
    m = VehicleModel()
    sk = StationKeeper(cfg)
    st = State.from_model(m)
    while not sk.feed_capture(st, 0.05):
        m.set_command(sk.capture_command() if trim else {})
        for _ in range(5):
            m.step(0.01)
        st = State.from_model(m)
    return sk.sp["depth"]


def test_capture_window_buoyancy_trim_reduces_float_up():
    """Offline: without the trim the 3 s capture window floats the hold point ~28 cm too shallow; with it ~11 cm
    (the rest is thruster spin-up lag from rest)."""
    no_trim, trim = 3.0 - _captured_depth(False), 3.0 - _captured_depth(True)
    assert no_trim > 0.2
    assert trim < 0.5 * no_trim
