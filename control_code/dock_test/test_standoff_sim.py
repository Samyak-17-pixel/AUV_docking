"""Closed-loop regression for the heading error at the standoff (standoff_sim.py). No ROS. Statistical: a population of runs, not one."""

from __future__ import annotations

import copy

import standoff_sim as S

STARTS = [(plant, yaw0) for plant in ("nominal", "light+fast") for yaw0 in (10.0, -25.0)]


def _rest(cfg):
    out = []
    for plant, yaw0 in STARTS:
        r = S.run(cfg=cfg, yaw0_deg=yaw0, T=100.0, seed=3, **S.PLANTS[plant])
        out.append((r["rest_frac"], r["min_dist_m"]))
    return out


def test_realign_gets_most_starts_to_a_rest_and_stays_clear_of_the_dock():
    res = _rest(S.load_cfg())
    assert min(d for _, d in res) > 1.9                             # never into the dock (hold distance ~2.3 m)
    assert sum(1 for rest, _ in res if rest >= 0.5) >= 2            # offline: ~75% of the last 20 s at rest on average (see dock_test.yaml)


def test_negative_control_the_old_behaviour_does_not_come_to_rest():
    cfg = copy.deepcopy(S.load_cfg())
    cfg["speed"]["flow_min_mps"] = -1.0                              # no backing away
    cfg["speed"]["settle_yaw_rate_rad_s"] = 9.0
    cfg["deadband"]["hold_extra_px"] = 0.0
    cfg["gains"]["yaw_nm_per_px"], cfg["gains"]["yaw_kd"], cfg["deadband"]["error_x_px"] = 0.03, 2.0, 10.0        # and the old yaw gains
    res = _rest(cfg)
    assert sum(1 for rest, _ in res if rest >= 0.5) <= 1


# ---- close start with the dock partly out of frame: the detector keeps asking for a FORWARD search surge (blind_surge=+1), the old search drove into the dock ----
CLOSE = [(yaw0, plant) for yaw0 in (26.0, 35.0, -30.0) for plant in ("nominal", "heavy+late")]


def _close_min_dist(cfg):
    return [S.run(cfg=cfg, yaw0_deg=yaw0, standoff_m=3.0, blind_surge=1.0, T=70.0, seed=3, **S.PLANTS[plant])["min_dist_m"] for yaw0, plant in CLOSE]


def test_blind_search_budget_keeps_the_vehicle_out_of_the_dock_from_a_close_start():
    d = _close_min_dist(S.load_cfg())
    assert min(d) > 1.5, d                                         # the hold distance is ~2.3 m, the dock plane is at 0; the heavy+late plant overshoots most (~1.7 m)


def test_negative_control_without_the_blind_budget_the_vehicle_drives_through_the_dock():
    cfg = copy.deepcopy(S.load_cfg())
    cfg["speed"]["blind_max_forward_m"] = -1.0                       # the old behaviour
    d = _close_min_dist(cfg)
    assert max(d) < 1.0, d


def test_blind_budget_does_not_apply_once_a_distance_is_known():
    # a dock seen whole first (valid view, then lost): the remembered-distance envelope handles it, the blind budget must not fire (mode stays lost_close/search, never search_back)
    r = S.run(cfg=S.load_cfg(), yaw0_deg=10.0, standoff_m=3.0, T=40.0, seed=3, **S.PLANTS["nominal"])
    assert "search_back" not in r["modes"]
