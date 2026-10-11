"""VehicleModel: a CG-CB offset gives a passive righting moment (cg_minus_cb_m was declared but unused before 2026-10-10)."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

_CTRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CTRL / "common"))
sys.path.insert(0, str(_CTRL / "sim_offline"))

from allocation import load_geometry  # noqa: E402
from pendulum_analysis import free_swing, period_and_range  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402


def _model(cg, pitch_deg=0.0, roll_deg=0.0):
    geom = copy.deepcopy(load_geometry())
    geom["vehicle"]["cg_minus_cb_m"] = list(cg)
    for k in ("K", "M", "N", "X", "Y", "Z"):
        geom["vehicle"]["drag_quad"][k] = 0.0
        geom["vehicle"]["drag_lin"][k] = 0.0
    return VehicleModel(geom=geom, pos=(0, 0, 3), eul_deg=(roll_deg, pitch_deg, 0.0))


def test_default_geometry_has_no_righting_moment():
    m = _model([0.0, 0.0, 0.0], pitch_deg=20.0)
    for _ in range(500):
        m.step(0.01)
    assert abs(m.nu[4]) < 1e-12 and abs(math.degrees(m.eul[1]) - 20.0) < 1e-9


def test_cg_below_cb_is_stable_and_oscillates_at_the_pendulum_frequency():
    d = 0.01
    m = _model([0.0, 0.0, d], pitch_deg=2.0)               # small amplitude: linear pendulum
    ts, th = [], []
    for _ in range(6000):
        m.step(0.01)
        ts.append(m.t)
        th.append(math.degrees(m.eul[1]))
    per, mn, mx = period_and_range(np.array(ts), np.array(th))
    inertia = m.M[4]
    expected = 2 * math.pi * math.sqrt(inertia / (m.weight_n * d))
    assert abs(per - expected) / expected < 0.03, (per, expected)
    assert mx <= 2.1 and mn >= -2.1                          # energy is conserved: it never exceeds the start amplitude


def test_cg_above_cb_is_unstable_and_roll_is_restored_too():
    m = _model([0.0, 0.0, -0.01], pitch_deg=1.0)
    for _ in range(300):
        m.step(0.01)
    assert math.degrees(m.eul[1]) > 1.0                      # grows away from level
    r = _model([0.0, 0.0, 0.01], roll_deg=5.0)
    for _ in range(10):                                      # roll inertia is tiny (0.12 kg*m^2): look at the first 0.1 s, before it swings back
        r.step(0.01)
    assert r.nu[3] < 0.0                                     # starboard-down roll -> restoring moment turns it back (negative p)


def test_fitted_pendulum_reproduces_the_observed_swing():
    ts, th = free_swing(0.0065, -46.0, 7.0, T=60.0)
    per, mn, mx = period_and_range(ts, th)
    assert abs(per - 11.0) < 0.6 and abs(mn - 7.0) < 2.0 and abs(mx - 85.0) < 3.0
