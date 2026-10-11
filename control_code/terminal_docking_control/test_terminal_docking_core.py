"""Terminal docking core: estimator accuracy, memory, outlier gating, guidance signs, phase transitions, stale-odometry safety, and a closed-loop docking."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_d) for _d in [HERE.parent / "sim_viewer", *sorted((HERE.parent / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # sim_viewer and its sub-folders
for p in (HERE, HERE.parent / "common", HERE.parent / "sim_offline", HERE.parent / "sim_viewer", HERE.parent.parent / "dock_detection_algo"):
    sys.path.insert(0, str(p))

from docking_sim import DOCK_POS, DockingSim, Scenario, load_cfg  # noqa: E402
from state import State  # noqa: E402
from terminal_docking_core import DockEstimator, DockObs, PoseTracker, TerminalDockingCore, rot_zyx  # noqa: E402

CFG = load_cfg()
GATE = float(CFG["guidance"]["gate_s_m"])
GRACE = float(CFG["guidance"]["gate_grace_m"])


def synth_obs(est: DockEstimator, pos, eul, dock_xyz, psi_a, noise_px=0.0, rng=None):
    x = np.array([*dock_xyz, psi_a])
    pix = est.project(x, np.asarray(pos, float), np.asarray(eul, float)).reshape(4, 2)
    if noise_px:
        pix = pix + (rng or np.random.default_rng(0)).normal(0, noise_px, pix.shape)
    side = sorted([pix[2], pix[3]], key=lambda p: p[0])          # the detector labels the side lights by image position
    pix = np.array([pix[0], pix[1], side[1], side[0]])
    return DockObs(fresh=True, valid=True, num_lights=4, radius_px=0.5 * float(np.linalg.norm(pix[0] - pix[1])), pix=pix)


# ----------------------------------------------------------------------------------------------------------------- estimator
@pytest.mark.parametrize("pos,yaw_deg", [((3.0, 0.0, 3.0), 0.0), ((3.0, 1.5, 3.0), 10.0), ((4.0, -1.5, 3.1), -12.0), ((2.0, 0.5, 2.9), 5.0)])
def test_estimator_recovers_the_dock_pose_from_the_light_pixels(pos, yaw_deg):
    est = DockEstimator(CFG)
    rng = np.random.default_rng(1)
    eul = np.array([0.0, 0.0, math.radians(yaw_deg)])
    for k in range(40):
        est.update(synth_obs(est, pos, eul, DOCK_POS, 0.0, 0.7, rng), np.asarray(pos), eul, 0.1 * k)
    assert est.ready
    assert np.allclose(est.x[:3], DOCK_POS, atol=0.08)
    assert abs(est.x[3]) < math.radians(4.0)


def test_estimator_keeps_its_answer_without_observations_and_grows_uncertainty_slowly():
    est = DockEstimator(CFG)
    pos, eul = np.array([4.0, 0.0, 3.0]), np.zeros(3)
    for k in range(20):
        est.update(synth_obs(est, pos, eul, DOCK_POS, 0.0, 0.5), pos, eul, 0.1 * k)
    x0, s0 = est.x.copy(), est.sigmas()[0]
    for k in range(20, 220):
        est.update(DockObs(fresh=False), pos, eul, 0.1 * k)
    assert np.allclose(est.x, x0)                                    # static dock: remembered as it was
    assert est.sigmas()[0] > s0 and est.sigmas()[0] < 0.2             # more uncertain, but still usable after 20 s


def test_estimator_gates_a_wild_detection():
    est = DockEstimator(CFG)
    pos, eul = np.array([4.0, 0.0, 3.0]), np.zeros(3)
    for k in range(25):
        est.update(synth_obs(est, pos, eul, DOCK_POS, 0.0, 0.5), pos, eul, 0.1 * k)
    before = est.x.copy()
    bad = synth_obs(est, pos, eul, DOCK_POS + np.array([0.0, 2.0, 0.5]), 0.3)
    assert est.update(bad, pos, eul, 2.6) == "gated"
    assert np.allclose(est.x, before)


def test_estimator_uses_the_vehicle_pose_it_is_given():
    """Seen from a different pose the same dock must give the same world estimate (this is what makes the memory work)."""
    est = DockEstimator(CFG)
    for k, (pos, yaw) in enumerate([((3.0, 1.0, 3.0), 8.0), ((4.0, 0.6, 3.0), 4.0), ((5.0, 0.3, 3.0), 0.0), ((6.0, 0.1, 3.0), -2.0)] * 6):
        eul = np.array([0.0, 0.0, math.radians(yaw)])
        est.update(synth_obs(est, pos, eul, DOCK_POS, 0.0, 0.5), np.array(pos), eul, 0.1 * k)
    assert np.allclose(est.x[:3], DOCK_POS, atol=0.06)


# ----------------------------------------------------------------------------------------------------------------- pose tracker
def _state(x, y, z, yaw=0.0, u=0.0):
    return State(pos=np.array([x, y, z]), eul=np.array([0.0, 0.0, yaw]), nu=np.array([u, 0, 0, 0, 0, 0.0]))


def test_pose_tracker_detects_frozen_and_stale_odometry():
    pt = PoseTracker(0.25, 1.0)
    assert pt.frozen(0.0)                                             # nothing yet
    t = 0.0
    for k in range(10):
        t += 0.22
        pt.push(_state(0.1 * k, 0.0, 3.0 + 0.01 * k, u=0.4), t)
    assert not pt.frozen(t + 0.1)
    assert pt.frozen(t + 1.5)                                         # nothing arrived for 1.5 s
    for _ in range(8):                                                # the real sim's freeze: identical samples keep arriving
        t += 0.22
        pt.push(_state(0.9, 0.0, 3.09, u=0.0), t)
    s = pt.state
    for _ in range(8):
        t += 0.22
        pt.push(State(pos=s.pos.copy(), eul=s.eul.copy(), nu=s.nu.copy()), t)
    assert pt.frozen(t)


def test_pose_tracker_extrapolates_smoothly_between_samples():
    pt = PoseTracker(0.25, 1.0)
    t = 0.0
    for k in range(30):
        t += 0.2
        pt.push(_state(0.4 * (t - 0.25), 0.0, 3.0, u=0.4), t)
    a, b = pt.at(t + 0.05), pt.at(t + 0.10)
    assert b.pos[0] > a.pos[0] and (b.pos[0] - a.pos[0]) == pytest.approx(0.4 * 0.05, rel=0.2)


# ----------------------------------------------------------------------------------------------------------------- guidance and phases
def _core_with_dock(psi_a=0.0, dock=DOCK_POS):
    core = TerminalDockingCore(load_cfg())
    core.est.x = np.array([*dock, psi_a])
    core.est.P = np.diag([0.0004] * 3 + [math.radians(1.0) ** 2])
    core.est.updates = 30
    return core


def _run_state(core, st, steps=60, dt=0.05):
    """Feed one odometry sample every 0.25 s (with a hair of noise: a bit-identical sample means 'frozen' to the tracker), then ask the controller."""
    t = 100.0
    for k in range(steps):
        t += dt
        if k % 5 == 0:
            core.pose.push(State(pos=st.pos + 1e-5 * k, eul=st.eul.copy(), nu=st.nu.copy()), t)
        w, stat = core.update(DockObs(fresh=False), core.pose.at(t), dt, t)
    return w, stat


def test_vehicle_right_of_the_axis_steers_left_and_dock_below_pulls_down():
    core = _core_with_dock()
    st = State(pos=np.array([DOCK_POS[0] - 6.0, 1.0, DOCK_POS[2] + 0.3]), eul=np.zeros(3), nu=np.array([0.6, 0, 0, 0, 0, 0.0]))
    w, stat = _run_state(core, st)
    assert stat["e"] == pytest.approx(1.0, abs=0.05) and 5.2 < stat["s"] < 6.0       # (the filter extrapolates the 0.6 m/s forward over the 3 s of the test)
    assert stat["psi_des_deg"] < -5.0                                  # to the right of the line: aim left of the axis heading
    assert w[5] < 0.0                                                  # N < 0: bow to port
    assert w[2] < 0.0                                                  # vehicle is 0.3 m BELOW the dock centre: Z < 0 pushes up
    assert w[0] > 0.0                                                  # still below cruise speed? (0.6 < 0.7): keep pushing


def test_vehicle_left_of_the_axis_steers_right():
    core = _core_with_dock()
    st = State(pos=np.array([DOCK_POS[0] - 6.0, -1.0, DOCK_POS[2]]), eul=np.zeros(3), nu=np.array([0.6, 0, 0, 0, 0, 0.0]))
    w, stat = _run_state(core, st)
    assert stat["psi_des_deg"] > 5.0 and w[5] > 0.0


def test_axis_heading_other_than_north_is_respected():
    core = _core_with_dock(psi_a=math.radians(20.0))
    # sitting exactly on the 20 deg axis line, 6 m in front, heading along it: nothing to correct
    fa = np.array([math.cos(math.radians(20.0)), math.sin(math.radians(20.0))])
    p = DOCK_POS[:2] - 6.0 * fa
    st = State(pos=np.array([p[0], p[1], DOCK_POS[2]]), eul=np.array([0.0, 0.0, math.radians(20.0)]), nu=np.array([0.6, 0, 0, 0, 0, 0.0]))
    w, stat = _run_state(core, st)
    assert abs(stat["e"]) < 0.02 and abs(stat["chi_deg"]) < 0.5 and abs(w[5]) < 0.05


def test_gate_pass_goes_terminal_and_gate_fail_backs_off():
    core = _core_with_dock()
    good = State(pos=np.array([DOCK_POS[0] - (GATE - 0.05), 0.02, DOCK_POS[2]]), eul=np.zeros(3), nu=np.array([0.45, 0, 0, 0, 0, 0.0]))
    _run_state(core, good)
    assert core.phase == "TERMINAL"
    core2 = _core_with_dock()
    bad = State(pos=np.array([DOCK_POS[0] - (GATE - GRACE - 0.1), 0.6, DOCK_POS[2]]), eul=np.zeros(3), nu=np.array([0.45, 0, 0, 0, 0, 0.0]))
    w, stat = _run_state(core2, bad)
    assert core2.phase == "RETRY" and core2.retries == 1 and w[0] < 0.0     # too far off at the gate: reverse out and try again


def test_retries_are_bounded():
    core = _core_with_dock()
    core.retries = int(core.g["max_retries"])
    bad = State(pos=np.array([DOCK_POS[0] - (GATE - GRACE - 0.1), 0.6, DOCK_POS[2]]), eul=np.zeros(3), nu=np.array([0.45, 0, 0, 0, 0, 0.0]))
    _run_state(core, bad)
    assert core.phase != "RETRY" and core.retries == int(core.g["max_retries"])


def test_stopped_inside_the_funnel_is_docked_and_thrust_off():
    core = _core_with_dock()
    core.phase = "TERMINAL"
    # centre 0.16 m before the mouth plane = nose 0.50 m inside the funnel, at rest
    inside = State(pos=np.array([DOCK_POS[0] - 0.16, 0.0, DOCK_POS[2]]), eul=np.zeros(3), nu=np.zeros(6))
    w, stat = _run_state(core, inside)
    assert core.phase == "DOCKED" and abs(w[0]) < 3.0 and w[5] == 0.0


def test_frozen_odometry_brakes_open_loop_and_never_pushes_forward():
    core = _core_with_dock()
    st = State(pos=np.array([DOCK_POS[0] - 4.0, 0.0, DOCK_POS[2]]), eul=np.zeros(3), nu=np.array([0.5, 0, 0, 0, 0, 0.0]))
    _run_state(core, st, steps=40)
    t = 100.0 + 40 * 0.05
    for k in range(80):                                              # nothing arrives for 4 s
        t += 0.05
        w, stat = core.update(DockObs(fresh=False), core.pose.at(t), 0.05, t)
    assert stat["phase"] == "SAFE_STOP" and w[0] <= 0.0 and w[5] == 0.0


def test_no_dock_seen_means_no_command():
    core = TerminalDockingCore(load_cfg())
    st = State(pos=np.array([0.0, 0.0, 3.0]), eul=np.zeros(3), nu=np.zeros(6))
    t = 100.0
    core.pose.push(st, t)
    w, stat = core.update(DockObs(fresh=False), core.pose.at(t + 0.1), 0.05, t + 0.1)
    assert not np.any(w) and stat["phase"] == "WAIT"


# ----------------------------------------------------------------------------------------------------------------- closed loop
def test_straight_approach_docks_first_try_on_the_nominal_plant():
    r = DockingSim(Scenario(range_m=7.0, vision="perfect")).run(keep_log=False)
    assert r.passed, r.failures
    assert abs(r.metrics["cross_lat"]) < 0.1 and r.metrics["min_clearance_m"] > 0.1 and r.metrics["retries"] == 0


def test_offset_and_crooked_start_docks_first_try():
    r = DockingSim(Scenario(range_m=9.0, lateral_m=1.0, heading_deg=-15.0, vision="perfect")).run(keep_log=False)
    assert r.passed, r.failures


def test_judge_flags_a_wall_collision():
    """The judge must fail a deliberately bad run: aim 0.5 m off the axis with the guidance disabled."""
    cfg = load_cfg(overrides={"guidance": {"max_deviation_deg": 0.0, "gate_lateral_m": 5.0, "gate_heading_deg": 90.0, "abort_lateral_m": 99.0}})
    r = DockingSim(Scenario(range_m=7.0, lateral_m=0.5, vision="perfect"), cfg).run(keep_log=False)
    assert not r.passed and any(("wall contact" in f) or ("bad entry" in f) for f in r.failures)


def test_frozen_odometry_dead_reckons_the_speed_and_brakes_until_at_rest():
    core = _core_with_dock()
    st = State(pos=np.array([DOCK_POS[0] - 5.0, 0.0, DOCK_POS[2]]), eul=np.zeros(3), nu=np.array([0.6, 0, 0, 0, 0, 0.0]))
    _run_state(core, st, steps=40)
    t = 100.0 + 40 * 0.05
    forces, u_dr = [], []
    for k in range(400):                                             # 20 s without a new odometry sample
        t += 0.05
        w, stat = core.update(DockObs(fresh=False), core.pose.at(t), 0.05, t)
        forces.append(w[0]); u_dr.append(stat.get("u_dr", 0.0))
    assert stat["phase"] == "SAFE_STOP"
    assert min(forces) < -2.0 and max(forces) <= 0.0                  # braked, never pushed forward
    first = next(v for v in u_dr if v > 0)                            # (the first second is not yet 'frozen': the sample is only 1 s old)
    assert first > 0.3 and u_dr[-1] < 0.05                          # the estimated speed ran down to rest
    assert forces[-1] == 0.0                                          # and the thruster is off again (it does not reverse the vehicle)


def test_past_the_stop_point_the_controller_creeps_back_instead_of_staying():
    core = _core_with_dock()
    core.phase = "TERMINAL"
    deep = State(pos=np.array([DOCK_POS[0] + 1.0, 0.0, DOCK_POS[2]]), eul=np.zeros(3), nu=np.zeros(6))      # nose 1.7 m inside, stop point is 0.5 m
    w, stat = _run_state(core, deep, steps=40)
    assert stat["u_target"] < 0.0 and w[0] < 0.0
