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
    assert np.sign(B[1]).tolist() == [1, -1, -1, 1]               # pitch: top pair vs bottom pair
    assert np.sign(B[2]).tolist() == [-1, -1, 1, 1]               # yaw: left/right pairs
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


def test_fin_ids_match_the_vessel_file():
    """'Mako (1).mavsim': id4 (+y,-z) 45 deg, id8 (+y,+z) 135 deg, id7 (-y,+z) 225 deg, id6 (-y,-z) -45 deg.
    With cs_06 and cs_08 the other way round, a yaw command comes out as a pitch moment."""
    fins = Allocator().g["fins"]
    expect = {"cs_04": (1, -1, 45.0), "cs_08": (1, 1, 135.0), "cs_07": (-1, 1, 225.0), "cs_06": (-1, -1, -45.0)}
    for name, (sy, sz, roll) in expect.items():
        loc, ori = fins[name]["location"], fins[name]["orientation"]
        assert np.sign(loc[1]) == sy and np.sign(loc[2]) == sz, name
        assert ori[0] == pytest.approx(roll), name


def test_yaw_command_produces_yaw_and_no_pitch_or_roll():
    a = Allocator(u_fin_off=0.0, u_fin_full=0.0)
    w = a.wrench_from_fins(a.fins_from_moments(0.0, 0.0, 0.5, 1.0), 1.0)
    assert w[5] == pytest.approx(0.5, rel=1e-6)
    assert abs(w[3]) < 1e-9 and abs(w[4]) < 1e-9


def test_fins_reverse_their_deflection_for_the_same_moment_in_reverse_flow():
    from allocation import Allocator
    a = Allocator()
    fwd = a.fins_from_moments(0.0, 0.0, 0.5, 0.8)
    rev = a.fins_from_moments(0.0, 0.0, 0.5, -0.8)
    assert all(abs(fwd[k] + rev[k]) < 1e-9 for k in fwd) and any(abs(v) > 1e-3 for v in fwd.values())
    # and the physical model closes the loop: the delivered yaw moment has the demanded sign in both flow directions
    for u, d in ((0.8, fwd), (-0.8, rev)):
        assert a.wrench_from_fins(d, u)[5] == pytest.approx(0.5, rel=0.05)


def test_small_force_region_is_linear_continuous_and_changes_nothing_above_the_joint():
    base = Allocator()
    soft = Allocator(small_force_n=1.0)
    for f in (1.0, 2.0, 5.0, -3.0):
        assert soft.force_to_rpm("th_02", f) == pytest.approx(base.force_to_rpm("th_02", f))          # identical at and above the joint
    # continuous at the joint, linear below it, and far gentler than the square root near zero
    r1, r_half, r_small = soft.force_to_rpm("th_02", 1.0), soft.force_to_rpm("th_02", 0.5), soft.force_to_rpm("th_02", 0.05)
    assert r_half == pytest.approx(0.5 * r1) and r_small == pytest.approx(0.05 * r1)
    assert abs(base.force_to_rpm("th_02", 0.05)) > 3.0 * abs(r_small)
    assert soft.force_to_rpm("th_02", -0.5) == pytest.approx(-r_half)
