"""PoseFilter / PoseTracker: accuracy under realistic odometry, latency compensation, freezes, yaw wrap, re-initialisation."""
import math

import numpy as np
import pytest

from pose_filter import PoseFilter, PoseFilterConfig, PoseTracker
from sensors import OdometryConfig, OdometrySensor
from state import State
from vehicle_model import VehicleModel


def _drive(freeze=0.0, T=60.0, yaw0=0.0, seed=3, cmd=None):
    m = VehicleModel(pos=(0, 0, 3), eul_deg=(0, 0, yaw0))
    m.set_command(cmd or {"th_01": 500.0, "cs_04": 3.0, "cs_06": 3.0, "cs_07": -3.0, "cs_08": -3.0, "th_02": -80.0, "th_03": -80.0})
    sens = OdometrySensor(OdometryConfig(freeze_mean_interval_s=freeze, seed=seed))
    tr = PoseTracker(0.25, 1.0)
    t, errs = 0.0, []
    while t < T:
        m.step(0.01)
        t += 0.01
        for o in sens.update(t, m.pos, m.eul, m.nu):
            tr.push(State(pos=np.array(o.pos), eul=np.array(o.eul), nu=np.array(o.nu)), t)
        if t > 8.0 and int(round(t * 100)) % 5 == 0 and not tr.frozen(t):
            e = tr.at(t)
            d_yaw = math.atan2(math.sin(e.eul[2] - m.eul[2]), math.cos(e.eul[2] - m.eul[2]))
            errs.append([*(e.pos - m.pos), d_yaw, e.nu[0] - m.nu[0], e.nu[5] - m.nu[5]])
    return np.array(errs), tr


def test_filtered_state_is_close_to_the_truth_despite_slow_late_noisy_odometry():
    e, _ = _drive()
    rms = np.sqrt((e ** 2).mean(axis=0))
    assert rms[2] < 0.02                         # depth: 2 cm (raw samples are 25 cm old and 1.5 cm noisy; moving at ~0.1 m/s vertical)
    assert rms[0] < 0.08 and rms[1] < 0.08        # horizontal position while moving at ~0.5 m/s
    assert rms[3] < math.radians(1.0) and rms[4] < 0.03 and rms[5] < math.radians(1.5)


def test_it_beats_the_raw_samples_by_a_wide_margin():
    m = VehicleModel(pos=(0, 0, 3))
    m.set_command({"th_01": 700.0})
    sens = OdometrySensor(OdometryConfig(freeze_mean_interval_s=0.0, seed=1))
    tr = PoseTracker(0.25, 1.0)
    t, raw, filt = 0.0, [], []
    last = None
    while t < 30.0:
        m.step(0.01)
        t += 0.01
        for o in sens.update(t, m.pos, m.eul, m.nu):
            last = o
            tr.push(State(pos=np.array(o.pos), eul=np.array(o.eul), nu=np.array(o.nu)), t)
        if t > 8.0 and last is not None and int(round(t * 100)) % 5 == 0:
            raw.append(abs(last.pos[0] - m.pos[0]))            # the newest raw sample, un-corrected
            filt.append(abs(tr.at(t).pos[0] - m.pos[0]))
    assert np.mean(filt) < 0.35 * np.mean(raw)


def test_yaw_wrap_does_not_confuse_the_filter():
    e, _ = _drive(yaw0=175.0, T=40.0, cmd={"th_01": 500.0, "cs_04": -4.0, "cs_06": -4.0, "cs_07": 4.0, "cs_08": 4.0})   # turns across +-180 deg
    assert np.abs(e[:, 2]).max() < math.radians(5.0)


def test_identical_frozen_samples_are_detected_and_do_not_pull_the_rate_to_zero():
    f = PoseTracker(0.25, 1.0)
    t = 0.0
    for k in range(20):                                           # a vehicle moving at 0.5 m/s
        t += 0.22
        f.push(State(pos=np.array([0.5 * (t - 0.25), 0, 3.0]), eul=np.zeros(3), nu=np.array([0.5, 0, 0, 0, 0, 0.0])), t)
        f.frozen(t)                                                # the controller asks every tick
    s = f.state
    v_before = f.filt.x[0, 1]
    for k in range(8):                                            # the real sim's freeze: the very same sample (with the twist zeroed) keeps arriving
        t += 0.22
        f.push(State(pos=s.pos.copy(), eul=s.eul.copy(), nu=s.nu.copy()), t)
        f.frozen(t)
    assert f.frozen(t) and abs(f.filt.x[0, 1] - v_before) < 0.05
    f.push(State(pos=s.pos + np.array([5.0, 0, 0]), eul=s.eul.copy(), nu=np.array([0.5, 0, 0, 0, 0, 0.0])), t + 0.22)      # recovery (a big jump)
    assert not f.frozen(t + 0.3)


def test_a_long_gap_reinitialises_instead_of_extrapolating_for_ever():
    f = PoseFilter(PoseFilterConfig(max_gap_s=1.5))
    f.update(State(pos=np.array([0, 0, 3.0]), eul=np.zeros(3), nu=np.array([0.5, 0, 0, 0, 0, 0.0])), 0.0)
    f.update(State(pos=np.array([100.0, 0, 3.0]), eul=np.zeros(3), nu=np.array([0.0, 0, 0, 0, 0, 0.0])), 30.0)
    assert f.state_at(30.0).pos[0] == pytest.approx(100.0, abs=0.2)


def test_state_can_be_asked_for_a_slightly_earlier_time():
    _, tr = _drive(T=20.0)
    now = tr.at(tr.t_arrival)
    earlier = tr.at(tr.t_arrival - 0.1)
    v = tr.filt.x[:3, 1]                                               # the filter's own NED velocity estimate
    assert np.allclose(now.pos - earlier.pos, 0.1 * v, atol=0.02)        # asking 0.1 s earlier moves back along it


def test_tracker_reports_staleness_and_missing_data():
    tr = PoseTracker(0.25, 1.0)
    assert tr.frozen(0.0) and tr.at(0.0) is None and tr.age(5.0) == math.inf
    for k in range(5):
        tr.push(State(pos=np.array([0.1 * k, 0, 3.0]), eul=np.zeros(3), nu=np.zeros(6)), 0.2 * k)
    assert not tr.frozen(1.0) and tr.frozen(3.0)


def test_a_vehicle_at_rest_in_a_noise_free_sim_is_not_frozen_unless_it_is_being_driven():
    tr = PoseTracker(0.25, 1.0)
    st = State(pos=np.array([1.0, 2.0, 3.0]), eul=np.zeros(3), nu=np.zeros(6))
    t = 0.0
    for _ in range(40):                                              # 10 s of bit-identical samples, nothing commanded: that is just a still vehicle
        t += 0.25
        tr.push(State(pos=st.pos.copy(), eul=st.eul.copy(), nu=st.nu.copy()), t)
        assert not tr.frozen(t, expect_motion=False)
    # the controller starts driving: the old identical samples must not count (else it would be declared frozen at once and never move)
    assert not tr.frozen(t + 0.05, expect_motion=True)
    for k in range(3):
        t += 0.25
        tr.push(State(pos=st.pos.copy(), eul=st.eul.copy(), nu=st.nu.copy()), t)
        assert not tr.frozen(t, expect_motion=True)                  # < 1 s of identical samples since the commands began
    for k in range(3):
        t += 0.25
        tr.push(State(pos=st.pos.copy(), eul=st.eul.copy(), nu=st.nu.copy()), t)
    assert tr.frozen(t, expect_motion=True)                          # > 1 s while commanding motion: frozen


def test_frozen_stays_declared_until_a_different_sample_arrives_even_if_the_command_drops_to_neutral():
    tr = PoseTracker(0.25, 1.0)
    t = 0.0
    for k in range(3):
        t += 0.25
        tr.push(State(pos=np.array([0.1 * k, 0, 3.0]), eul=np.zeros(3), nu=np.zeros(6)), t)
        tr.frozen(t, expect_motion=True)
    s = tr.state
    for _ in range(8):
        t += 0.25
        tr.push(State(pos=s.pos.copy(), eul=s.eul.copy(), nu=s.nu.copy()), t)
    assert tr.frozen(t, expect_motion=True)
    assert tr.frozen(t, expect_motion=False)                          # the controller went neutral because of the freeze: still frozen (no flicker)
    t += 0.25
    tr.push(State(pos=s.pos + 0.01, eul=s.eul.copy(), nu=s.nu.copy()), t)
    assert not tr.frozen(t, expect_motion=False)                      # a different sample: alive again
