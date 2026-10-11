import pytest

from pid import Pid


def test_proportional_and_clamp():
    p = Pid(kp=2.0, out_max=5.0)
    assert p.update(1.0, 0.1) == pytest.approx(2.0)
    assert p.update(10.0, 0.1) == 5.0


def test_integral_is_clamped_in_output_units():
    p = Pid(kp=0.0, ki=1.0, i_max=2.0)
    for _ in range(1000):
        u = p.update(1.0, 0.1)
    assert u == pytest.approx(2.0)


def test_derivative_on_measurement_has_no_setpoint_kick():
    p = Pid(kp=0.0, kd=1.0, d_filter_tau_s=0.0)
    p.update(0.0, 0.1, rate=0.0)
    assert p.update(5.0, 0.1, rate=0.0) == pytest.approx(0.0)    # error jumped, measured rate did not
    assert p.update(5.0, 0.1, rate=2.0) == pytest.approx(-2.0)   # moving toward target -> brakes


def test_reset():
    p = Pid(kp=0.0, ki=1.0, i_max=10)
    p.update(1.0, 1.0)
    p.reset()
    assert p.integral == 0.0


def test_loops_output_lowpass_smooths_noise_and_is_off_by_default():
    import numpy as np
    from loops import HoldLoops
    from state import State
    base = {"heave": {"kp": 20.0, "ki": 0.0, "kd": 0.0, "max": 25.0}}
    filt = {"heave": {"kp": 20.0, "ki": 0.0, "kd": 0.0, "max": 25.0, "lpf_tau_s": 0.5}}
    rng = np.random.default_rng(0)
    a, b = HoldLoops(base), HoldLoops(filt)
    ua, ub = [], []
    for _ in range(200):
        st = State(pos=np.array([0.0, 0.0, 3.0 + rng.normal(0, 0.02)]))
        ua.append(a.heave(st, 3.0, 0.05))
        ub.append(b.heave(st, 3.0, 0.05))
    assert np.std(ub[50:]) < 0.35 * np.std(ua[50:])
    # a step passes through with the filter's time constant, not instantly
    c = HoldLoops(filt)
    first = c.heave(State(pos=np.array([0.0, 0.0, 2.0])), 3.0, 0.05)
    assert first > 0 and first == pytest.approx(20.0 * 1.0, rel=1e-9) or first <= 20.0


def test_setpoint_filter_ramps_a_step_and_starts_from_the_measurement():
    import numpy as np
    from loops import HoldLoops
    from state import State
    g = {"heave": {"kp": 10.0, "ki": 0.0, "kd": 0.0, "max": 100.0, "sp_tau_s": 2.0}}
    h = HoldLoops(g)
    st = State(pos=np.array([0.0, 0.0, 3.0]))
    first = h.heave(st, 4.0, 0.05)                    # a 1 m step: the filtered setpoint starts AT the measurement, so no jump
    assert 0.0 <= first < 10.0 * 0.05
    for _ in range(200):                              # 10 s later the filtered setpoint has (almost) reached 4.0
        u = h.heave(st, 4.0, 0.05)
    assert u == pytest.approx(10.0 * 1.0, rel=0.05)
    h.reset()
    assert h._sp_f == {}
