"""Shared PID with integral clamp, optional derivative low-pass and derivative-on-measurement."""

from __future__ import annotations

from typing import Optional


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class Pid:
    """u = kp*e + ki*int(e) + kd*de/dt.

    If `rate` is passed to update(), the derivative term uses -rate (the measured rate of
    the controlled quantity, e.g. a gyro/odometry twist) instead of differentiating the error.
    That avoids derivative kick on setpoint changes and is much less noisy.
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        i_max: float = 0.0,
        d_filter_tau_s: float = 0.05,
        out_max: Optional[float] = None,
    ) -> None:
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.i_max = abs(float(i_max))          # clamp on the integral CONTRIBUTION (output units)
        self.d_filter_tau_s = max(0.0, float(d_filter_tau_s))
        self.out_max = None if out_max is None else abs(float(out_max))
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self._prev_error: Optional[float] = None
        self._d_filtered = 0.0

    def update(self, error: float, dt: float, rate: Optional[float] = None) -> float:
        if dt <= 0.0:
            return self.kp * error

        if self.ki > 1e-12:
            self.integral += error * dt
            lim = self.i_max / self.ki
            self.integral = clamp(self.integral, -lim, lim)
        else:
            self.integral = 0.0

        if rate is not None:
            derivative = -float(rate)
        elif self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        if rate is None and self.d_filter_tau_s > 0.0:
            alpha = dt / (self.d_filter_tau_s + dt)
            self._d_filtered += alpha * (derivative - self._d_filtered)
            derivative = self._d_filtered

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.out_max is not None:
            u = clamp(u, -self.out_max, self.out_max)
        return u
