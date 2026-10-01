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
