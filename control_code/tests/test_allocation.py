"""Allocation sign conventions. These encode facts from Mako_01.mavsim: heave thrusters point UP."""
import math

import numpy as np
import pytest

from allocation import Allocator, eul_to_rotm


@pytest.fixture
def alloc():
    return Allocator()


def test_heave_thruster_axis_points_up(alloc):
    # orientation pitch=+90 -> thrust axis = body -z (up in NED)
    axis = eul_to_rotm([0, 90, 0]) @ np.array([1.0, 0.0, 0.0])
    assert axis == pytest.approx([0, 0, -1], abs=1e-9)


def test_dive_needs_negative_rpm_on_both_heave_thrusters(alloc):
    out = alloc.thrusters_from_wrench([0, 0, 5.0, 0, 0, 0])      # +Z = down
    assert out["th_02"] < 0 and out["th_03"] < 0
    assert out["th_02"] == pytest.approx(out["th_03"])
    assert out["th_01"] == 0.0


def test_rise_needs_positive_rpm(alloc):
    out = alloc.thrusters_from_wrench([0, 0, -5.0, 0, 0, 0])
    assert out["th_02"] > 0 and out["th_03"] > 0


def test_nose_up_moment_drives_forward_heave_thruster_up(alloc):
    out = alloc.thrusters_from_wrench([0, 0, 0, 0, 1.0, 0])      # +M = nose up
    assert out["th_02"] > 0 > out["th_03"]                       # th_02 is the forward one (x=+0.349)
    assert out["th_02"] == pytest.approx(-out["th_03"])


def test_surge_uses_only_axial_thruster_and_reverses(alloc):
    fwd = alloc.thrusters_from_wrench([5, 0, 0, 0, 0, 0])
    rev = alloc.thrusters_from_wrench([-5, 0, 0, 0, 0, 0])
    assert fwd["th_01"] > 0 > rev["th_01"]
    assert fwd["th_02"] == fwd["th_03"] == 0.0


def test_unreachable_dofs_ignored_by_thrusters(alloc):
    out = alloc.thrusters_from_wrench([0, 7, 0, 3, 0, 4])        # sway, roll, yaw
    assert all(v == 0.0 for v in out.values())


def test_rpm_force_roundtrip_and_cap(alloc):
    rpm = alloc.force_to_rpm("th_02", 3.0)
    assert alloc.thrust_n("th_02", rpm) == pytest.approx(3.0, rel=1e-6)
    capped = Allocator(rpm_cap=500).force_to_rpm("th_02", 1e6)
    assert capped == 500


def test_fin_modes_are_orthogonal_roll_pitch_yaw(alloc):
    B = alloc.B_fin
    # columns are cs_04, cs_06, cs_07, cs_08; rows K, M, N
    assert np.sign(B[0]).tolist() == [1, 1, 1, 1]                 # roll: all same sign
    assert np.sign(B[1]).tolist() == [1, 1, -1, -1]               # pitch: top pair vs bottom pair
    assert np.sign(B[2]).tolist() == [-1, 1, 1, -1]               # yaw: left/right pairs
    G = B @ B.T
    assert np.allclose(G - np.diag(np.diag(G)), 0, atol=1e-9)     # decoupled


def test_fin_moment_roundtrip(alloc):
    a = Allocator(u_fin_off=0.0, u_fin_full=0.0)
    u = 1.0
    deltas = a.fins_from_moments(0.05, 0.0, 0.5, u)
    w = a.wrench_from_fins(deltas, u)
    assert w[3] == pytest.approx(0.05, rel=1e-6)                  # K
    assert w[5] == pytest.approx(0.5, rel=1e-6)                   # N
    assert w[4] == pytest.approx(0.0, abs=1e-9)


def test_fins_scale_with_speed_squared_and_fade_out(alloc):
    big = max(abs(v) for v in alloc.fins_from_moments(0, 0, 0.3, 1.0).values())
    small = max(abs(v) for v in alloc.fins_from_moments(0, 0, 0.3, 2.0).values())
    assert small == pytest.approx(big / 4, rel=1e-6)              # force ~ U^2
    assert all(v == 0 for v in alloc.fins_from_moments(0, 0, 0.3, 0.05).values())   # no flow -> off


def test_fin_cap(alloc):
    assert max(abs(v) for v in alloc.fins_from_moments(0, 0, 100.0, 1.0).values()) <= alloc.fin_cap_deg + 1e-9
