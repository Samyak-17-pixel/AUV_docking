"""The tuning tools themselves: scoring is sane, the shipped gains pass, the OLD roll gains are caught, sensitivity has the expected shape. No ROS."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_tuner as lt  # noqa: E402

CTRL = Path(__file__).resolve().parents[1]


def _gains(name):
    return yaml.safe_load(open(CTRL / name))["gains"]


def test_shipped_yaw_roll_speed_gains_pass_on_every_plant():
    g = _gains("dof_testing/dof_testing.yaml")
    for plant in lt.PLANTS:
        assert lt.score_yaw(g, plant, seed=1).ok, plant
        assert lt.score_roll(g, plant, seed=1).ok, plant


def test_the_old_roll_gains_are_caught_as_unstable_with_realistic_odometry():
    g = _gains("dof_testing/dof_testing.yaml")
    g["roll"] = {"kp": 0.5, "ki": 0.02, "kd": 0.2, "i_max": 0.1, "max": 0.5}               # the values shipped before 2026-10-10
    sc = lt.score_roll(g, "nominal", seed=1)
    assert not sc.ok and sc.detail["fin_p2p_deg"] > 6.0                                      # the fins chatter


def test_hover_gains_reject_a_steady_push_and_keep_chatter_bounded_on_the_main_plants():
    g = _gains("station_keeping/station_keeping.yaml")
    for plant in ("nominal", "heavy+late", "light+fast"):
        p = lt.score_push(g, plant, small_force_n=0.3)
        assert p.detail["push_peak_m"] < 0.7 and p.detail["push_ss_m"] < 0.08, (plant, p.detail)
        assert lt.score_push_pitch(g, plant, small_force_n=0.3).ok
        assert lt.score_hold(g, plant, small_force_n=0.3).detail["chatter_rpm"] < 260


def test_noise_free_simulation_is_quiet_and_noisy_one_is_not():
    g = _gains("station_keeping/station_keeping.yaml")
    quiet = lt.simulate(g, "nominal", T=30.0, odom_noise=False, depth_step=0.0, pitch_step_deg=0.0)["d"]
    assert abs(quiet[:, 1] - 3.0).max() < 0.01
    noisy = lt.simulate(g, "nominal", T=30.0, odom_noise=True, depth_step=0.0, pitch_step_deg=0.0)["d"]
    assert noisy[:, 5].std() > quiet[:, 5].std()


def test_sensitivity_returns_the_requested_factors_and_a_known_trend():
    g = _gains("station_keeping/station_keeping.yaml")
    r = lt.loop_sensitivity(g, only=("heave", "kd"), seeds=(1,))
    row = r["heave.kd"]["factors"]
    assert set(row) == {"0.5", "0.75", "1.0", "1.5", "2.0"} and r["heave.kd"]["value"] == g["heave"]["kd"]
    assert row["0.5"]["overshoot_pct"] > row["2.0"]["overshoot_pct"]                          # less damping rings more
