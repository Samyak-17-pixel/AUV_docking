"""End-to-end (no ROS): DofTest and StationKeeper against the offline vehicle model."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from dof_testing import DOFS, DEFAULT_CONFIG, DofTest, load_config, run_offline
from allocation import load_geometry
from state import State
from station_keeping_core import StationKeeper, shrink
from vehicle_model import VehicleModel

SK_CFG = Path(__file__).resolve().parents[1] / "station_keeping" / "station_keeping.yaml"


def _buoyant_geom(extra_buoyancy_kg: float) -> dict:
    """The vessel data with extra buoyancy (the old Mako_01.mavsim had +0.74 kg, i.e. +7.25 N)."""
    g = copy.deepcopy(load_geometry())
    g["vehicle"]["buoyancy_mass_kg"] = g["vehicle"]["mass_kg"] + extra_buoyancy_kg
    return g


def test_vehicle_is_neutrally_buoyant():
    """'Mako (1).mavsim': mass 20 kg = buoyancy 20 kg, so with no thrust it stays where it is."""
    m = VehicleModel()
    for _ in range(500):
        m.step(0.01)
    assert abs(m.pos[2] - 3.0) < 0.01


def test_positive_net_buoyancy_floats_up():
    m = VehicleModel(geom=_buoyant_geom(0.7399))
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
    # re-tuned 2026-10-10 for realistic sensing: softer than the old 60 N/m, so a 10 s, 6 N push moves it ~0.7 m (was 0.2 m) but it is pushed back to ~1 cm (the integral)
    assert pd < 0.9 and abs(s["depth_err_m"]) < 0.05
    _, s, _, pa = _run_sk(cfg, 60, ext=[4, 0, 0, 0, 0, 0])
    assert pa < 0.6 and abs(s["ahead_err_m"]) < 0.1


def test_low_rpm_cap_cannot_hold_depth():
    """Documents the limit: an 800 RPM cap gives the two heave thrusters about 4.6 N in total, so an 8 N push wins."""
    cfg = yaml.safe_load(open(SK_CFG))
    cfg["limits"]["rpm_cap"] = 800
    _, _, pd, _ = _run_sk(cfg, 60, ext=[0, 0, 8, 0, 0, 0])
    assert pd > 1.3
    cfg["limits"]["rpm_cap"] = 1800
    _, s, pd, _ = _run_sk(cfg, 60, ext=[0, 0, 8, 0, 0, 0])
    assert pd < 1.1 and abs(s["depth_err_m"]) < 0.1


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
    """Capture window on a vehicle with +7.25 N net buoyancy (the old vessel file), trim = that value."""
    cfg = yaml.safe_load(open(SK_CFG))
    cfg["capture"]["average_s"] = 3.0
    cfg["feedforward"]["heave_n"] = 7.25
    m = VehicleModel(geom=_buoyant_geom(0.7399))
    sk = StationKeeper(cfg)
    st = State.from_model(m)
    while not sk.feed_capture(st, 0.05):
        m.set_command(sk.capture_command() if trim else {})
        for _ in range(5):
            m.step(0.01)
        st = State.from_model(m)
    return sk.sp["depth"]


def test_capture_window_buoyancy_trim_reduces_float_up():
    """On a buoyant vehicle: without the trim the 3 s capture window floats the hold point ~28 cm too shallow; with it ~11 cm
    (the rest is thruster spin-up lag from rest). On the current neutral vessel the trim is 0 and this is not needed."""
    no_trim, trim = 3.0 - _captured_depth(False), 3.0 - _captured_depth(True)
    assert no_trim > 0.2
    assert trim < 0.5 * no_trim


def test_wrong_heave_trim_sinks_a_neutral_vehicle():
    """The 7.25 N trim of the old vessel file is wrong for the neutral 20 kg one: the depth loop must fight it."""
    cfg = yaml.safe_load(open(SK_CFG))
    assert cfg["feedforward"]["heave_n"] == 0.0
    m = VehicleModel()
    m.set_command(StationKeeper(cfg).alloc.thrusters_from_wrench([0, 0, 7.25, 0, 0, 0]))
    for _ in range(500):
        m.step(0.01)
    assert m.pos[2] - 3.0 > 0.5                                      # sinks > 0.5 m in 5 s (offline drag; ~3 m with no drag)


class _FrozenModel(VehicleModel):
    """The live sim sometimes stops advancing while odometry keeps arriving."""

    def step(self, dt):
        pass


def test_frozen_state_gives_invalid_not_a_false_pass():
    t = run_offline(load_config(DEFAULT_CONFIG), "pitch", "hold", _FrozenModel())
    assert t.result["verdict"] == "INVALID"
    t = run_offline(load_config(DEFAULT_CONFIG), "heave", "step", _FrozenModel())
    assert t.result["verdict"] == "INVALID"


class _NoFinModel(VehicleModel):
    """A DOF that is locked or has dead actuators: the fins do nothing."""

    def step(self, dt):
        self.cmd = {k: v for k, v in self.cmd.items() if k.startswith("th_")}
        super().step(dt)


def test_no_response_step_fails_with_a_locked_dof_hint():
    t = run_offline(load_config(DEFAULT_CONFIG), "yaw", "step", _NoFinModel())
    assert t.result["verdict"] == "FAIL" and "hint" in t.result and "LOCKED" in t.result["hint"]


def test_controllers_expose_their_setpoints_for_the_viewer():
    cfg = yaml.safe_load(open(SK_CFG))
    sk = StationKeeper(cfg)
    assert sk.debug() == {}                                                    # nothing before the hold point is captured
    st = State.from_model(VehicleModel(pos=(0, 0, 3.7), eul_deg=(0, 3, 40)))
    sk.capture(st)
    d = sk.debug()
    assert d["ctrl"] == "station_keeping" and d["depth_sp"] == pytest.approx(3.7) and d["pitch_sp_deg"] == pytest.approx(3.0) and d["yaw_sp_deg"] == pytest.approx(40.0)
    t = DofTest(load_config(DEFAULT_CONFIG), "heave", "hold")
    assert "baseline" in t.debug()["mode"] and "depth_sp" not in t.debug()
    m = VehicleModel(pos=(0, 0, 3.0))
    while t.phase != "run":
        out, _ = t.update(State.from_model(m), 0.05)
        m.set_command(out)
        for _ in range(5):
            m.step(0.01)
    assert t.debug()["depth_sp"] == pytest.approx(3.0 + t.cfg["hold"]["heave_delta_m"], abs=0.05) and t.debug()["pitch_sp_deg"] == 0.0
