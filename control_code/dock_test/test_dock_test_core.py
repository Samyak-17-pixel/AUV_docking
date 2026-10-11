"""Sign checks for the standoff wrench. No ROS.

    python3 test_dock_test_core.py
"""

from __future__ import annotations

import copy
import math
import sys
from collections import deque
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "common"))
sys.path.insert(0, str(_HERE.parent / "sim_offline"))

from dock_test_core import DockTestCore, DockView, VehicleSnap

CFG = yaml.safe_load((Path(__file__).resolve().parent / "dock_test.yaml").read_text())
FF = float(CFG["feedforward"]["heave_n"])


def _core() -> DockTestCore:
    return DockTestCore(CFG)


def _still() -> VehicleSnap:
    return VehicleSnap()


def test_positive_elevation_dives():
    view = DockView(fresh=True, valid=True, elevation_valid=True, elevation_rad=0.1)
    wrench, status = _core().update(view, _still())
    assert wrench[2] > FF
    assert status["mode"] == "standoff"
    assert wrench[0] == 0.0


def test_dock_below_pitches_nose_down():
    view = DockView(fresh=True, valid=True, elevation_valid=True, error_y_px=40.0)
    wrench, _ = _core().update(view, _still())
    assert wrench[4] < 0.0


def test_dock_right_yaws_right():
    view = DockView(fresh=True, valid=True, elevation_valid=True, error_x_px=120.0)      # beyond the speed band (deadband + hold_extra_px)
    wrench, status = _core().update(view, _still())
    assert wrench[5] > 0.0
    assert wrench[0] > 0.0
    assert status["mode"] == "creep"


def test_negative_lateral_yaws_left():
    view = DockView(fresh=True, valid=True, elevation_valid=True, lateral_px=-20.0)
    wrench, _ = _core().update(view, _still())
    assert wrench[5] < 0.0


def test_squared_pose_has_zero_surge():
    view = DockView(
        fresh=True, valid=True, elevation_valid=True,
        elevation_rad=0.005, error_x_px=3.0, error_y_px=-4.0, lateral_px=2.0,
        radius_px=80.0,
    )
    wrench, status = _core().update(view, _still())
    assert wrench[0] == 0.0
    assert status["mode"] == "standoff"


def test_too_close_reverses():
    view = DockView(fresh=True, valid=True, elevation_valid=True, radius_px=250.0)
    wrench, status = _core().update(view, _still())
    assert wrench[0] < 0.0
    assert status["mode"] == "backup"


def test_partial_detection_does_not_heave_on_elevation():
    view = DockView(
        fresh=True, valid=False, elevation_valid=True, elevation_rad=0.4,
        search_surge_norm=-1.0, search_yaw_norm=0.0, search_pitch_norm=1.0,
    )
    wrench, status = _core().update(view, _still())
    assert status["mode"] == "search"
    assert abs(wrench[2] - FF) < 1e-6
    assert wrench[0] < 0.0
    assert wrench[4] < 0.0


def test_stale_message_is_neutral():
    wrench, status = _core().update(DockView(fresh=False, valid=True, elevation_rad=0.5), _still())
    assert status["mode"] == "stale"
    assert wrench.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# --- closed-loop approach on the offline vehicle (surge only) ---------------------------------------
DOCK_X = 10.0          # dock origin; the light circle lies in the plane x = 10 (dock_02 in the vessel file)
CAM_X = 0.575          # camera_03 sits at the nose, 0.575 m ahead of the vehicle origin


def _approach(x0: float, u0: float, view_kwargs: dict, seconds: float = 120.0, odom_delay_s: float = 0.25):
    """Fly the speed loop towards the dock with worst-case assumptions: almost no drag (config X_u = 0.01),
    a 0.25 s stale speed reading, the thruster lag of the model. Returns (closest distance, final distance, final speed, modes)."""
    from allocation import Allocator, load_geometry
    from vehicle_model import VehicleModel

    geom = copy.deepcopy(load_geometry())
    geom["vehicle"]["drag_quad"]["X"] = 0.0
    m = VehicleModel(geom=geom, pos=(x0, 0.0, 3.0))
    m.nu[0] = u0
    core = _core()
    lim = CFG["limits"]
    alloc = Allocator(rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"])
    dt, inner = 0.05, 5
    lag = deque([u0] * max(1, int(odom_delay_s / dt)), maxlen=max(1, int(odom_delay_s / dt)))
    dmin, modes = 1e9, set()
    for _ in range(int(seconds / dt)):
        d = DOCK_X - (m.pos[0] + CAM_X)
        dmin = min(dmin, d)
        lag.append(float(m.nu[0]))
        view = DockView(fresh=True, valid=True, elevation_valid=True, radius_px=core.fy * core.dock_radius_m / max(d, 0.05),
                        **view_kwargs)
        wrench, status = core.update(view, VehicleSnap(speed_u=lag[0]))
        modes.add(status["mode"])
        # Surge loop only: a constant fake error_x would otherwise spin the vehicle, which a real closed yaw loop does not.
        m.set_command({"th_01": alloc.allocate(wrench, float(m.nu[0]))["th_01"]})
        for _ in range(inner):
            m.step(dt / inner)
    return dmin, DOCK_X - (m.pos[0] + CAM_X), float(m.nu[0]), modes


def test_misaligned_approach_stops_before_the_dock():
    dmin, dfin, ufin, _ = _approach(0.0, 0.0, {"error_x_px": 60.0})
    assert dmin > core_hold() - 0.4
    assert dfin > 1.9
    assert abs(ufin) < 0.6          # since 2026-10-10 a heading error at the hold distance is worked on by backing away and creeping, not by sitting still


def test_fast_arrival_is_braked_in_time():
    dmin, dfin, ufin, modes = _approach(3.5, 1.5, {"error_x_px": 60.0})      # 1.5 m/s at ~5.9 m: coasting in
    assert dmin > 1.9, dmin
    assert abs(ufin) < 0.6          # the misaligned fixed 60 px keeps the realign cycle going (this harness has no yaw dynamics); the point is the distance


def test_aligned_approach_holds_standoff_without_pushing():
    dmin, dfin, ufin, modes = _approach(5.5, 1.0, {})                         # square and in frame, still moving
    assert dmin > 2.0 and abs(ufin) < 0.05 and "standoff" in modes


def test_old_fixed_push_would_have_hit_the_dock():
    """Documents why a speed loop is needed: a fixed 4 N creep and a 4 N reverse at radius 200 px does not stop."""
    from allocation import load_geometry
    from vehicle_model import VehicleModel

    geom = copy.deepcopy(load_geometry())
    geom["vehicle"]["drag_quad"]["X"] = 0.0
    m = VehicleModel(geom=geom, pos=(0.0, 0.0, 3.0))
    core = _core()
    dmin = 1e9
    for _ in range(int(120 / 0.05)):
        d = DOCK_X - (m.pos[0] + CAM_X)
        dmin = min(dmin, d)
        radius = core.fy * core.dock_radius_m / max(d, 0.05)
        f = -4.0 if radius >= core.too_close_radius_px else 4.0
        m.set_command({"th_01": math.copysign(60.0 * math.sqrt(abs(f) / (1000.0 * 0.1 ** 4)), f)})
        for _ in range(5):
            m.step(0.01)
    assert dmin < 0.5, dmin


def test_speed_loop_never_exceeds_its_force_caps():
    core = _core()
    assert core.speed_force(0.0, 5.0) <= core.fwd_max_n + 1e-9
    assert core.speed_force(3.0, 0.0) >= -core.brake_max_n - 1e-9


def _valid(radius: float, **kw) -> DockView:
    return DockView(fresh=True, valid=True, elevation_valid=True, radius_px=radius, error_x_px=60.0, **kw)


def _search(surge_norm: float = 0.35) -> DockView:
    return DockView(fresh=True, valid=False, search_surge_norm=surge_norm, search_yaw_norm=-1.0)


def test_lost_close_backs_away_instead_of_surging_in():
    """Closed-loop sim finding 2026-10-08: the dock left the frame at ~2.5 m, search asked for forward flow, and the vehicle
    drove through the dock. After a close sighting, losing the lights must back away, never go forward."""
    core = _core()
    core.update(_valid(190.0), VehicleSnap())                      # dock seen at ~2.2 m
    wrench, status = core.update(_search(0.35), VehicleSnap(speed_u=0.0))
    assert status["mode"] == "lost_close" and wrench[0] < 0.0 and status["u_target"] < 0.0


def test_search_speed_follows_the_stopping_envelope_when_the_dock_was_seen_recently():
    core = _core()
    core.update(_valid(160.0), VehicleSnap())                      # ~2.6 m: envelope allows only ~0.36 m/s
    _, status = core.update(_search(0.35), VehicleSnap())
    assert 0.0 < status["u_target"] < core.search_mps


def test_spurious_tiny_detection_after_a_good_one_is_rejected():
    core = _core()
    core.update(_valid(175.0), VehicleSnap())
    wrench, status = core.update(_valid(14.0), VehicleSnap())      # 'dock at 29 m' one frame later: impossible
    assert status["mode"] in ("search", "lost_close") and wrench[0] <= 0.0
    assert status["d_m"] < 3.0                                     # the remembered distance was kept


def test_wrong_distance_after_a_dropout_is_rejected_both_ways():
    """Closed-loop sim 2026-10-08: after a 7 s dropout four blobs said 4.4 m while the dock was 1.9 m away."""
    core = _core()
    core.update(_valid(190.0), VehicleSnap())                      # ~2.2 m
    for _ in range(int(7 / 0.05)):
        core.update(_search(0.0), VehicleSnap(speed_u=0.0))        # dropout, not moving
    _, status = core.update(_valid(94.0), VehicleSnap())           # claims 4.4 m
    assert status["mode"] in ("search", "lost_close")
    assert status["d_m"] < 2.6                                     # memory kept, not overwritten by 4.4 m
    _, status = core.update(_valid(185.0), VehicleSnap())          # a consistent view is accepted again
    assert status["mode"] not in ("search", "lost_close")


def test_distance_memory_is_dead_reckoned_and_then_forgotten():
    core = _core()
    core.update(_valid(100.0), VehicleSnap())                      # ~4.2 m
    d0 = core.update(_search(0.0), VehicleSnap(speed_u=0.0))[1]["d_m"]
    for _ in range(40):                                            # 2 s at 0.5 m/s closes ~1 m
        _, status = core.update(_search(0.0), VehicleSnap(speed_u=0.5))
    assert 0.8 < d0 - status["d_m"] < 1.2
    for _ in range(int(25 / 0.05)):                                # 25 s without a good view: memory dropped
        _, status = core.update(_search(0.35), VehicleSnap(speed_u=0.0))
    assert status["d_m"] != status["d_m"]                          # NaN = unknown
    assert status["u_target"] == core.search_mps                   # free to search forward again


def test_heave_damping_opposes_vertical_speed():
    core = _core()
    view = DockView(fresh=True, valid=True, elevation_valid=True, elevation_rad=0.1)
    still = core.update(view, VehicleSnap(heave_rate=0.0))[0][2]
    sinking = core.update(view, VehicleSnap(heave_rate=0.2))[0][2]          # already moving down at 0.2 m/s
    rising = core.update(view, VehicleSnap(heave_rate=-0.2))[0][2]
    assert sinking < still < rising
    assert still - sinking == pytest.approx(core.kd_elev * 0.2, rel=1e-6)


def test_shipped_gains_settle_without_oscillation_on_every_plant():
    """loop_sim.py: the REAL core + allocator + vehicle model with thruster lag, ~4.5 Hz odometry and image delay."""
    import loop_sim
    for name, plant in loop_sim.PLANTS.items():
        r = loop_sim.run({}, **plant)
        assert r["pitch_amp"] < 0.3 and r["depth_amp"] < 0.02, (name, r["pitch_amp"], r["depth_amp"])
        assert r["settle_s"] < 10.0, (name, r["settle_s"])


def test_the_old_gains_really_did_oscillate():
    """Documents the bug the viewer exposed: pitch_nm_per_px 0.04, pitch_kd 4 and no heave damping -> a sustained limit cycle."""
    import loop_sim
    r = loop_sim.run(loop_sim.OLD_GAINS, **loop_sim.PLANTS["nominal"])
    assert r["pitch_amp"] > 5.0 and r["rpm_swing"] > 800


def test_heave_damping_is_needed_for_robustness():
    """The new pitch gains alone are not enough on a slow, heavy plant: without elevation damping the heave spring rings into pitch."""
    import loop_sim
    stress = loop_sim.PLANTS["stress"]
    assert loop_sim.run({"elevation_kd_n_per_mps": 0.0}, **stress)["pitch_amp"] > 3.0
    assert loop_sim.run({"elevation_kd_n_per_mps": 30.0}, **stress)["pitch_amp"] < 0.3


def test_deadband_is_continuous_at_the_edge():
    """A measurement flickering across a deadband edge must not make a step in the command (it became +-350 RPM thruster spikes)."""
    core = _core()
    band = core.elev_db
    inside = core.update(DockView(fresh=True, valid=True, elevation_valid=True, elevation_rad=band - 1e-6), VehicleSnap())[0][2]
    outside = core.update(DockView(fresh=True, valid=True, elevation_valid=True, elevation_rad=band + 1e-6), VehicleSnap())[0][2]
    assert abs(outside - inside) < 0.01                                       # was ~0.9 N with the hard gate
    band_y = core.y_db
    m_in = core.update(DockView(fresh=True, valid=True, elevation_valid=True, error_y_px=band_y - 1e-6), VehicleSnap())[0][4]
    m_out = core.update(DockView(fresh=True, valid=True, elevation_valid=True, error_y_px=band_y + 1e-6), VehicleSnap())[0][4]
    assert abs(m_out - m_in) < 1e-4
    far = core.update(DockView(fresh=True, valid=True, elevation_valid=True, error_y_px=band_y + 20.0), VehicleSnap())[0][4]
    assert far == pytest.approx(-core.kp_y * 20.0, rel=1e-6)                   # only the part beyond the band counts


def core_hold() -> float:
    return _core().hold_distance_m


def main() -> None:
    tests = [
        test_positive_elevation_dives,
        test_dock_below_pitches_nose_down,
        test_dock_right_yaws_right,
        test_negative_lateral_yaws_left,
        test_squared_pose_has_zero_surge,
        test_too_close_reverses,
        test_partial_detection_does_not_heave_on_elevation,
        test_stale_message_is_neutral,
        test_misaligned_approach_stops_before_the_dock,
        test_fast_arrival_is_braked_in_time,
        test_aligned_approach_holds_standoff_without_pushing,
        test_old_fixed_push_would_have_hit_the_dock,
        test_speed_loop_never_exceeds_its_force_caps,
        test_lost_close_backs_away_instead_of_surging_in,
        test_search_speed_follows_the_stopping_envelope_when_the_dock_was_seen_recently,
        test_spurious_tiny_detection_after_a_good_one_is_rejected,
        test_wrong_distance_after_a_dropout_is_rejected_both_ways,
        test_distance_memory_is_dead_reckoned_and_then_forgotten,
        test_heave_damping_opposes_vertical_speed,
        test_shipped_gains_settle_without_oscillation_on_every_plant,
        test_the_old_gains_really_did_oscillate,
        test_heave_damping_is_needed_for_robustness,
        test_deadband_is_continuous_at_the_edge,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()


# ---- heading error at the hold distance (2026-10-10): back away to get fin flow instead of sitting with a wrong heading
BIG = 150.0          # px: beyond the deadband + hold_extra_px (61) band


def _at_hold(core, **kw):
    """A fresh view at the hold distance (apparent radius for hold_distance_m)."""
    radius = core.fy * core.dock_radius_m / core.hold_distance_m
    return DockView(fresh=True, valid=True, elevation_valid=True, radius_px=radius, **kw)


def test_heading_error_at_hold_distance_backs_away():
    core = _core()
    wrench, status = core.update(_at_hold(core, error_x_px=BIG), _still())
    assert status["mode"] == "realign"
    assert wrench[0] < 0.0                      # reverse thrust for fin flow
    assert status["u_target"] < 0.0
    assert wrench[5] > 0.0                      # dock to the right -> yaw right (the allocator flips the fin sign for reverse flow)


def test_small_heading_error_inside_the_speed_band_still_holds_but_still_yaws():
    core = _core()
    wrench, status = core.update(_at_hold(core, error_x_px=core.x_db + 0.5 * core.hold_extra_px), _still())
    assert status["mode"] == "standoff" and wrench[0] == 0.0
    assert wrench[5] > 0.0                      # the yaw moment itself uses the narrow band (it acts as soon as fins have flow)


def test_realign_ends_when_square_and_still_or_out_of_room():
    core = _core()
    d = core.hold_distance_m
    snap = VehicleSnap(speed_u=-0.4, dt=0.05)
    modes = []
    for _ in range(400):                       # back away with a fixed large heading error: it must stop backing at hold + realign_max_back_m
        view = DockView(fresh=True, valid=True, elevation_valid=True, error_x_px=BIG, radius_px=core.fy * core.dock_radius_m / d)
        _, st = core.update(view, snap)
        modes.append(st["mode"])
        d += 0.4 * 0.05
    assert modes[0] == "realign" and modes[-1] == "creep"
    k = modes.index("creep")
    assert abs(k * 0.4 * 0.05 - core.realign_max_back_m) < 0.15
    core2 = _core()                            # square (error inside the band) and not rotating: leaves realign at once
    core2.update(_at_hold(core2, error_x_px=BIG), _still())
    _, st = core2.update(_at_hold(core2, error_x_px=0.0), VehicleSnap(speed_u=-0.3, dt=0.05))
    assert st["mode"] == "standoff"


def test_rotating_vehicle_is_not_square():
    view = DockView(fresh=True, valid=True, elevation_valid=True, radius_px=80.0)
    _, status = _core().update(view, VehicleSnap(yaw_rate=0.3))
    assert status["mode"] != "standoff"


def test_realign_can_be_switched_off():
    cfg = copy.deepcopy(CFG)
    cfg["speed"]["flow_min_mps"] = -1.0
    core = DockTestCore(cfg)
    _, status = core.update(_at_hold(core, error_x_px=BIG), _still())
    assert status["mode"] == "creep"           # the old behaviour: no backing away
