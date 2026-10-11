"""Odometry sensor model: rate, jitter, latency, noise, freezes. No ROS."""
import numpy as np
import pytest

from sensors import OdometryConfig, OdometrySensor, quat_from_eul
from vehicle_model import VehicleModel


def _run(cfg, T=300.0, dt=0.01, vel=0.2):
    s = OdometrySensor(cfg)
    out, t = [], 0.0
    while t < T:
        t += dt
        out += s.update(t, [vel * t, 0.0, 3.0], [0.0, 0.0, 0.0], [vel, 0, 0, 0, 0, 0])
    return s, out


def test_rate_and_jitter_match_the_real_sim():
    s, out = _run(OdometryConfig(freeze_mean_interval_s=0.0))
    ts = np.array([o.t for o in out])
    assert len(out) / 300.0 == pytest.approx(4.5, abs=0.25)
    d = np.diff(ts)
    assert 0.16 < d.min() and d.max() < 0.29 and d.std() > 0.005            # irregular, not a metronome


def test_latency_and_noise_levels():
    cfg = OdometryConfig(freeze_mean_interval_s=0.0, latency_s=0.25, pos_noise_m=0.02)
    s, out = _run(cfg, vel=0.5)
    err = np.array([o.pos[0] - 0.5 * (o.t - 0.25) for o in out[5:]])        # true position at t - latency
    assert abs(err.mean()) < 0.01                                          # the state is 0.25 s old
    assert err.std() == pytest.approx(0.02, rel=0.25)                       # plus the configured noise


def test_freezes_repeat_exact_values_with_zero_twist_and_end():
    cfg = OdometryConfig(freeze_mean_interval_s=20.0, freeze_min_s=5.0, freeze_max_s=10.0, seed=2)
    s, out = _run(cfg, T=200.0)
    assert s.freezes >= 3
    fz = [o for o in out if o.frozen]
    assert len(fz) > 20 and all(np.all(o.nu == 0.0) for o in fz)
    # a run of frozen samples carries exactly the same pose
    i = next(k for k, o in enumerate(out) if o.frozen)
    j = i
    while j + 1 < len(out) and out[j + 1].frozen:
        j += 1
    assert j > i and all(np.array_equal(out[k].pos, out[i].pos) for k in range(i, j + 1))
    assert not out[-1].frozen or s.freezes >= 1
    assert any(not o.frozen for o in out[j + 1:])                           # it recovers
    spans = []
    k = 0
    while k < len(out):
        if out[k].frozen:
            m = k
            while m + 1 < len(out) and out[m + 1].frozen:
                m += 1
            spans.append(out[m].t - out[k].t)
            k = m
        k += 1
    assert all(4.0 < sp < 11.0 for sp in spans[:-1])


def test_freeze_can_be_disabled_and_runs_are_reproducible():
    s, out = _run(OdometryConfig(freeze_mean_interval_s=0.0), T=100.0)
    assert s.freezes == 0 and not any(o.frozen for o in out)
    _, a = _run(OdometryConfig(seed=5), T=30.0)
    _, b = _run(OdometryConfig(seed=5), T=30.0)
    assert all(np.array_equal(x.pos, y.pos) for x, y in zip(a, b))


def test_quaternion_matches_the_vehicle_model():
    m = VehicleModel(pos=(0, 0, 3), eul_deg=(10, -20, 130))
    assert quat_from_eul(m.eul) == pytest.approx(m.quaternion(), abs=1e-12)
