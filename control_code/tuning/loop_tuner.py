#!/usr/bin/env python3
"""Single-loop gain tuning against the offline vehicle with REALISTIC sensing, on four plants. No ROS, no GUI.

The loops under test are the real ones (common/loops.py HoldLoops + common/allocation.py) driven the way the docking controller drives them:
odometry arrives at ~4.5 Hz, ~0.25 s late, noisy; the controller extrapolates it to 'now' with its own velocity (PoseTracker logic), runs at 20 Hz, and the
thrusters lag. Plants: nominal / heavy+late / light+fast / stress (thruster lag, odometry rate and latency, inertia, drag): all ASSUMED numbers.

  python3 loop_tuner.py show                     # score the gains in a yaml on all plants
  python3 loop_tuner.py tune heave pitch         # coordinate search, prints the best passing gains
Acceptance (user-chosen defaults): depth within 5 cm, pitch within 2 deg, overshoot < 10 %, no oscillation, no thruster chatter above 200 RPM p-p
(measured on the heave commands in 1 s windows), on every plant.
"""

from __future__ import annotations

import sys as _sys0
from pathlib import Path as _P0
_sys0.path.insert(0, str(_P0(__file__).resolve().parents[1] / "common"))
from outdirs import tmp_dir  # noqa: E402
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_CTRL = _HERE.parent
for p in (_CTRL / "common", _CTRL / "sim_offline", _CTRL / "terminal_docking_control"):
    sys.path.insert(0, str(p))

from allocation import Allocator, load_geometry  # noqa: E402
from loops import HoldLoops  # noqa: E402
from sensors import OdometryConfig, OdometrySensor  # noqa: E402
from state import State, wrap_pi  # noqa: E402
from terminal_docking_core import PoseTracker  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

PLANTS: Dict[str, dict] = {
    "nominal": dict(tau=0.2, odom_hz=4.5, latency=0.25, inertia=1.0, drag=1.0),
    "heavy+late": dict(tau=0.3, odom_hz=4.5, latency=0.35, inertia=1.3, drag=1.2),
    "light+fast": dict(tau=0.1, odom_hz=9.0, latency=0.15, inertia=0.7, drag=0.8),
    "stress": dict(tau=0.45, odom_hz=3.0, latency=0.50, inertia=1.8, drag=1.5),
}

ACCEPT = dict(depth_ss_m=0.05, pitch_ss_deg=2.0, overshoot_frac=0.10, chatter_rpm=200.0, yaw_ss_deg=3.0, yaw_overshoot_frac=0.10)


def make_model(P: dict, pos=(0.0, 0.0, 3.0), eul_deg=(0.0, 0.0, 0.0)) -> VehicleModel:
    geom = copy.deepcopy(load_geometry())
    g = geom["vehicle"]["gyration_m"]
    geom["vehicle"]["gyration_m"] = [g[0], g[1] * math.sqrt(P["inertia"]), g[2] * math.sqrt(P["inertia"])]
    for k in geom["vehicle"]["drag_quad"]:
        geom["vehicle"]["drag_quad"][k] *= P["drag"]
    return VehicleModel(geom=geom, pos=pos, eul_deg=eul_deg, thruster_tau_s=P["tau"])


@dataclass
class LoopScore:
    ok: bool
    cost: float
    detail: Dict[str, float]


def simulate(gains: Dict[str, dict], plant: str, T: float = 45.0, seed: int = 1, small_force_n: float = 1.0, extrapolate: bool = True,
             depth_step: float = 0.4, pitch_step_deg: float = 6.0, speed_mps: float = 0.0, yaw_step_deg: float = 0.0, odom_noise: bool = True, push_z_n: float = 0.0, push_m_nm: float = 0.0, filt: Optional[dict] = None) -> dict:
    """Depth step and pitch step at t = 5 s (with both loops active), optionally with forward speed and a yaw step (needs speed). Returns time series + metrics."""
    P = PLANTS[plant]
    m = make_model(P, eul_deg=(0, 0, 0))
    alloc = Allocator(rpm_cap=1800.0, fin_deg_cap=25.0, small_force_n=small_force_n)
    loops = HoldLoops(gains, heave_ff_n=0.0)
    cfg = OdometryConfig(rate_hz=P["odom_hz"], latency_s=P["latency"], freeze_mean_interval_s=0.0, seed=seed)
    if not odom_noise:
        cfg.pos_noise_m = cfg.att_noise_deg = cfg.vel_noise_mps = cfg.rate_noise_dps = 0.0
    sens = OdometrySensor(cfg)
    from pose_filter import PoseFilterConfig
    pose = PoseTracker(0.25, 1e9, PoseFilterConfig(**(filt or {})))      # the controller assumes the NOMINAL 0.25 s latency; the plants differ (robustness)
    dt = 0.05
    t, nxt = 0.0, 0.0
    cmd: Dict[str, float] = {}
    rows = []
    z0 = 3.0
    while t < T:
        for _ in range(5):
            m.step(0.01)
            t += 0.01
            for o in sens.update(t, m.pos, m.eul, m.nu):
                pose.push(State(pos=np.array(o.pos), eul=np.array(o.eul), nu=np.array(o.nu)), t)
        st = pose.at(t) if extrapolate else pose.state
        if st is None:
            continue
        z_sp = z0 + (depth_step if t > 5.0 else 0.0)
        p_sp = math.radians(pitch_step_deg) if t > 5.0 else 0.0
        w = np.zeros(6)
        w[2] = loops.heave(st, z_sp, dt)
        w[4] = loops.pitch(st, p_sp, dt)
        if "speed" in gains and speed_mps > 0:
            w[0] = loops.speed(st, speed_mps, dt)
        if yaw_step_deg != 0.0 and "yaw" in gains:
            w[5] = loops.yaw(st, math.radians(yaw_step_deg) if t > 15.0 else 0.0, dt)
        if "roll" in gains:
            w[3] = loops.roll(st, 0.0, dt)
        cmd = alloc.allocate(w, st.speed_u)
        m.set_command(cmd)
        m.ext[:] = 0.0
        if push_z_n and t > 5.0:
            m.ext[2] = push_z_n                                   # a steady downward push (a current, a trim error), body Z
            m.ext[4] = push_m_nm                                  # a steady pitching moment (CG offset, a hanging cable)
        rows.append([t, m.pos[2], math.degrees(m.eul[1]), math.degrees(m.eul[2]), m.nu[0], cmd.get("th_02", 0.0), cmd.get("th_03", 0.0), z_sp, math.degrees(p_sp)])
    return {"t": np.array([r[0] for r in rows]), "d": np.array(rows)}


def _settle(err: np.ndarray, t: np.ndarray, band: float, t0: float) -> float:
    idx = np.where(np.abs(err) > band)[0]
    idx = idx[t[idx] >= t0]
    return float(t[idx[-1]] - t0) if len(idx) else 0.0


def _metrics(d: np.ndarray, sp_col: int, val_col: int, step: float) -> dict:
    t = d[:, 0]
    err = d[:, val_col] - d[:, sp_col]
    after = t > 5.0
    over = max(0.0, float(np.max(err[after] * np.sign(step)))) / abs(step)
    late = t > t[-1] - 12.0
    return {"over": over, "ss": float(np.max(np.abs(err[late]))), "settle": err, "t": t}


def score_hold(gains: Dict[str, dict], plant: str, seed: int = 1, **kw) -> LoopScore:
    """Two runs: a 0.4 m depth step with pitch held at 0, and a 6 deg pitch step with depth held. Cross-coupling must stay small too."""
    rd = simulate(gains, plant, seed=seed, depth_step=0.4, pitch_step_deg=0.0, **kw)["d"]
    rp = simulate(gains, plant, seed=seed, depth_step=0.0, pitch_step_deg=6.0, **kw)["d"]
    md = _metrics(rd, 7, 1, 0.4)
    mp_ = _metrics(rp, 8, 2, 6.0)
    t = rd[:, 0]
    cross_pitch = float(np.max(np.abs(rd[t > 5.0, 2])))                    # pitch error during the depth step [deg]
    cross_depth = float(np.max(np.abs(rp[t > 5.0, 1] - 3.0)))               # depth error during the pitch step [m]
    chat = 0.0
    for d in (rd, rp):
        w = np.where(d[:, 0] > 15.0)[0]
        for i in range(w[0], len(d) - 20, 5):
            chat = max(chat, float(np.max(np.ptp(d[i:i + 20, 5:7], axis=0))))
    st_d = _settle(md["settle"], md["t"], 0.05, 5.0)
    st_p = _settle(mp_["settle"], mp_["t"], 1.0, 5.0)
    a = ACCEPT
    fail = {"depth_ss": md["ss"] > a["depth_ss_m"], "pitch_ss": mp_["ss"] > a["pitch_ss_deg"], "over_d": md["over"] > a["overshoot_frac"],
            "over_p": mp_["over"] > a["overshoot_frac"], "chatter": chat > a["chatter_rpm"], "cross_pitch": cross_pitch > a["pitch_ss_deg"], "cross_depth": cross_depth > 0.05}
    cost = st_d + st_p + 40 * md["ss"] + 5 * mp_["ss"] + 30 * max(md["over"], mp_["over"]) + 0.05 * chat + 3 * cross_pitch + 20 * cross_depth + 100.0 * sum(fail.values())
    return LoopScore(not any(fail.values()), cost, {"settle_d_s": st_d, "settle_p_s": st_p, "ss_depth_m": md["ss"], "ss_pitch_deg": mp_["ss"], "over_d": md["over"], "over_p": mp_["over"],
                                                     "chatter_rpm": chat, "cross_pitch_deg": cross_pitch, "cross_depth_m": cross_depth, **{f"FAIL_{k}": 1.0 for k, v in fail.items() if v}})


def score_push(gains: Dict[str, dict], plant: str, seed: int = 1, push_n: float = 5.0, **kw) -> LoopScore:
    """Disturbance rejection for HOVER use (station_keeping): a steady 5 N downward push from t = 5 s must be pushed back to within 5 cm, with a bounded peak."""
    d = simulate(gains, plant, seed=seed, depth_step=0.0, pitch_step_deg=0.0, push_z_n=push_n, T=60.0, **kw)["d"]
    t = d[:, 0]
    err = d[:, 1] - 3.0
    peak = float(np.max(np.abs(err[t > 5.0])))
    late = t > t[-1] - 10.0
    ss = float(np.max(np.abs(err[late])))
    chat = 0.0
    idx = np.where(t > 40.0)[0]
    for i in range(idx[0], len(t) - 20, 5):
        chat = max(chat, float(np.max(np.ptp(d[i:i + 20, 5:7], axis=0))))
    fail = {"push_peak": peak > 0.6, "push_ss": ss > 0.05, "chatter": chat > 200.0}
    return LoopScore(not any(fail.values()), 20 * peak + 60 * ss + 0.05 * chat + 100.0 * sum(fail.values()), {"push_peak_m": peak, "push_ss_m": ss, "chatter_rpm": chat, **{f"FAIL_{k}": 1.0 for k, v in fail.items() if v}})


def score_push_pitch(gains: Dict[str, dict], plant: str, seed: int = 1, push_nm: float = 0.3, **kw) -> LoopScore:
    """A steady 0.3 N*m nose-up moment (an offset centre of gravity of 1.5 cm would do) must be held to within 2 deg, peak below 10 deg."""
    d = simulate(gains, plant, seed=seed, depth_step=0.0, pitch_step_deg=0.0, push_m_nm=push_nm, T=60.0, **kw)["d"]
    t = d[:, 0]
    peak = float(np.max(np.abs(d[t > 5.0, 2])))
    ss = float(np.max(np.abs(d[t > t[-1] - 10.0, 2])))
    fail = {"pitch_push_peak": peak > 10.0, "pitch_push_ss": ss > 2.0}
    return LoopScore(not any(fail.values()), 3 * peak + 20 * ss + 100.0 * sum(fail.values()), {"pitch_push_peak_deg": peak, "pitch_push_ss_deg": ss, **{f"FAIL_{k}": 1.0 for k, v in fail.items() if v}})


def score_hover(gains: Dict[str, dict], plant: str, seed: int = 1, small_force_n: float = 0.3) -> LoopScore:
    """Hover (station_keeping / dof_testing) criteria = the step responses + the two steady-disturbance tests."""
    a = score_hold(gains, plant, seed=seed, small_force_n=small_force_n)
    b = score_push(gains, plant, seed=seed, small_force_n=small_force_n)
    c = score_push_pitch(gains, plant, seed=seed, small_force_n=small_force_n)
    return LoopScore(a.ok and b.ok and c.ok, a.cost + b.cost + c.cost, {**a.detail, **b.detail, **c.detail})


def _hover_cost(args):
    g, sf = args
    tot = 0.0
    for plant in PLANTS:
        w = 0.4 if plant == "stress" else 1.0                  # the stress plant only has to stay sane
        if plant == "stress":
            continue                                             # (excluded from the search for speed; checked afterwards)
        for sd in (1,):
            sc = score_hover(g, plant, seed=sd, small_force_n=sf)
            tot += w * (sc.cost + (0.0 if sc.ok else 300.0))
    return tot


def search_hover(gains: Dict[str, dict], rounds: int = 6, workers: int = 20, sf: float = 0.3) -> Dict[str, dict]:
    import multiprocessing as mp
    g = copy.deepcopy(gains)
    keys = [(lp, par) for lp in ("heave", "pitch") for par in ("kp", "kd", "ki", "lpf_tau_s")]
    best = _hover_cost((g, sf))
    step = 0.6
    with mp.Pool(workers) as pool:
        for _ in range(rounds):
            improved = False
            for loop, par in keys:
                cands = []
                for f in (1 + step, 1 / (1 + step), 1 + 2 * step, 1 / (1 + 2 * step)):
                    c = copy.deepcopy(g)
                    c[loop][par] = (g[loop].get(par, 0.0) * f) if g[loop].get(par, 0.0) > 0 else 0.05 * (f - 1.0 > 0)
                    cands.append(c)
                costs = pool.map(_hover_cost, [(c, sf) for c in cands])
                i = int(np.argmin(costs))
                if costs[i] < best - 1e-9:
                    best, g, improved = costs[i], cands[i], True
            if not improved:
                step *= 0.5
    return g


def score_all(gains: Dict[str, dict], seeds=(1, 2), **kw) -> Dict[str, LoopScore]:
    out = {}
    for plant in PLANTS:
        ss = [score_hold(gains, plant, seed=s, **kw) for s in seeds]
        worst = max(ss, key=lambda x: x.cost)
        out[plant] = LoopScore(all(x.ok for x in ss), worst.cost, worst.detail)
    return out


def total_cost(gains, **kw) -> float:
    sc = score_all(gains, **kw)
    return sum(s.cost for s in sc.values()) + 1000.0 * sum(not s.ok for s in sc.values())


def _cost_job(args):
    g, kw = args
    return total_cost(g, **kw)


def coordinate_search(gains: Dict[str, dict], keys: List[tuple], rounds: int = 4, workers: int = 24, **kw) -> Dict[str, dict]:
    """keys: [(loop, param)], multiplicative steps, shrinking; all candidates of a sweep are scored in parallel."""
    import multiprocessing as mp
    g = copy.deepcopy(gains)
    best = total_cost(g, **kw)
    step = 0.5
    with mp.Pool(workers) as pool:
        for _ in range(rounds):
            improved = False
            for loop, par in keys:
                cands = []
                for f in (1 + step, 1 / (1 + step), 1 + 2 * step, 1 / (1 + 2 * step)):
                    c = copy.deepcopy(g)
                    c[loop][par] = (g[loop].get(par, 0.0) * f) if g[loop].get(par, 0.0) > 0 else 0.1 * (f - 1.0 > 0)
                    cands.append(c)
                costs = pool.map(_cost_job, [(c, kw) for c in cands])
                i = int(np.argmin(costs))
                if costs[i] < best - 1e-9:
                    best, g, improved = costs[i], cands[i], True
            if not improved:
                step *= 0.5
    return g


def simulate_yaw(gains: Dict[str, dict], plant: str, seed: int = 1, T: float = 70.0, speed: float = 0.8, step_deg: float = 30.0, small_force_n: float = 0.3, odom_noise: bool = True, roll_step_deg: float = 0.0):
    """Heading step at forward speed (speed loop + heave/pitch/roll holds active). Returns rows [t, yaw_deg, sp_deg, u, cs_04, th_01]."""
    P = PLANTS[plant]
    m = make_model(P)
    alloc = Allocator(rpm_cap=1800.0, fin_deg_cap=25.0, small_force_n=small_force_n)
    loops = HoldLoops(gains, heave_ff_n=0.0)
    cfg = OdometryConfig(rate_hz=P["odom_hz"], latency_s=P["latency"], freeze_mean_interval_s=0.0, seed=seed)
    if not odom_noise:
        cfg.pos_noise_m = cfg.att_noise_deg = cfg.vel_noise_mps = cfg.rate_noise_dps = 0.0
    sens, pose = OdometrySensor(cfg), PoseTracker(0.25, 1e9)
    dt, t, rows = 0.05, 0.0, []
    while t < T:
        for _ in range(5):
            m.step(0.01)
            t += 0.01
            for o in sens.update(t, m.pos, m.eul, m.nu):
                pose.push(State(pos=np.array(o.pos), eul=np.array(o.eul), nu=np.array(o.nu)), t)
        st = pose.at(t)
        if st is None:
            continue
        sp = math.radians(step_deg) if t > 25.0 else 0.0
        w = np.zeros(6)
        w[0] = loops.speed(st, speed if t > 3.0 else 0.0, dt)
        w[2] = loops.heave(st, 3.0, dt)
        w[3] = loops.roll(st, math.radians(roll_step_deg) if t > 25.0 else 0.0, dt)
        w[4] = loops.pitch(st, 0.0, dt)
        w[5] = loops.yaw(st, sp, dt)
        cmd = alloc.allocate(w, st.speed_u)
        m.set_command(cmd)
        rows.append([t, math.degrees(m.eul[2]), math.degrees(sp), m.nu[0], cmd.get("cs_04", 0.0), cmd.get("th_01", 0.0), math.degrees(m.eul[0])])
    return np.array(rows)


def score_yaw(gains: Dict[str, dict], plant: str, seed: int = 1, **kw) -> LoopScore:
    d = simulate_yaw(gains, plant, seed=seed, **kw)
    t, y, sp = d[:, 0], d[:, 1], d[:, 2]
    step = float(sp.max()) or 1.0
    after = t > 25.0
    over = max(0.0, float(np.max(y[after] - step)) / step)
    err = y - sp
    late = t > t[-1] - 12.0
    ss = float(np.max(np.abs(err[late])))
    settle = _settle(err, t, 3.0, 25.0)
    chat = 0.0
    idx = np.where(t > 45.0)[0]
    for i in range(idx[0], len(t) - 20, 5):
        chat = max(chat, float(np.ptp(d[i:i + 20, 4])))
    pre = (t > 3.0) & (t < 24.0)
    u_over = max(0.0, float(np.max(d[pre, 3]) - 0.8)) / 0.8                      # speed overshoot on the 0 -> 0.8 m/s start
    fail = {"yaw_ss": ss > ACCEPT["yaw_ss_deg"], "yaw_over": over > ACCEPT["yaw_overshoot_frac"], "fin_chatter": chat > 6.0, "speed": abs(d[late, 3].mean() - 0.8) > 0.1, "speed_over": u_over > 0.10}
    cost = settle + 4 * ss + 40 * over + chat + 60 * u_over + 100.0 * sum(fail.values())
    return LoopScore(not any(fail.values()), cost, {"settle_yaw_s": settle, "ss_yaw_deg": ss, "over_yaw": over, "fin_p2p_deg": chat, "u_late": float(d[late, 3].mean()), "u_over": u_over, **{f"FAIL_{k}": 1.0 for k, v in fail.items() if v}})


def _yaw_cost_job(args):
    g, kw = args
    sc = [score_yaw(g, p, seed=s, **kw) for p in PLANTS for s in (1, 2)]
    return sum(x.cost for x in sc) + 1000.0 * sum(not x.ok for x in sc)


def search_yaw(gains: Dict[str, dict], rounds: int = 5, workers: int = 24, **kw) -> Dict[str, dict]:
    import multiprocessing as mp
    g = copy.deepcopy(gains)
    keys = [("yaw", "kp"), ("yaw", "kd"), ("yaw", "ki"), ("yaw", "lpf_tau_s"), ("speed", "kp"), ("speed", "ki"), ("speed", "sp_tau_s")]
    best = _yaw_cost_job((g, kw))
    step = 0.5
    with mp.Pool(workers) as pool:
        for _ in range(rounds):
            improved = False
            for loop, par in keys:
                cands = []
                for f in (1 + step, 1 / (1 + step), 1 + 2 * step, 1 / (1 + 2 * step)):
                    c = copy.deepcopy(g)
                    c[loop][par] = (g[loop].get(par, 0.0) * f) if g[loop].get(par, 0.0) > 0 else 0.1 * (f - 1.0 > 0)
                    cands.append(c)
                costs = pool.map(_yaw_cost_job, [(c, kw) for c in cands])
                i = int(np.argmin(costs))
                if costs[i] < best - 1e-9:
                    best, g, improved = costs[i], cands[i], True
            if not improved:
                step *= 0.5
    return g


def score_roll(gains: Dict[str, dict], plant: str, seed: int = 1, speed: float = 1.0, step_deg: float = 15.0) -> LoopScore:
    """Roll step at speed (the yaw loop holds heading 0). Roll has almost no inertia (0.06 kg*m^2): the loop is stiff and noise-sensitive."""
    d = simulate_yaw(gains, plant, seed=seed, speed=speed, step_deg=0.0, roll_step_deg=step_deg, T=60.0)
    t, roll = d[:, 0], d[:, 6]
    err = roll - np.where(t > 25.0, step_deg, 0.0)
    late = t > t[-1] - 12.0
    ss = float(np.max(np.abs(err[late])))
    over = max(0.0, float(np.max(roll[t > 25.0] - step_deg)) / step_deg)
    chat = 0.0
    idx = np.where(t > 40.0)[0]
    for i in range(idx[0], len(t) - 20, 5):
        chat = max(chat, float(np.ptp(d[i:i + 20, 4])))
    settle = _settle(err, t, 3.0, 25.0)
    yaw_dev = float(np.max(np.abs(d[t > 25.0, 1])))
    fail = {"roll_ss": ss > ACCEPT["yaw_ss_deg"], "roll_over": over > 0.15, "fin_chatter": chat > 6.0, "yaw_dev": yaw_dev > 6.0}
    return LoopScore(not any(fail.values()), settle + 4 * ss + 40 * over + chat + 5 * yaw_dev + 100.0 * sum(fail.values()),
                     {"settle_roll_s": settle, "ss_roll_deg": ss, "over_roll": over, "fin_p2p_deg": chat, "yaw_dev_deg": yaw_dev, **{f"FAIL_{k}": 1.0 for k, v in fail.items() if v}})


def _roll_cost_job(args):
    g = args
    sc = [score_roll(g, p, seed=s) for p in PLANTS for s in (1, 2)]
    return sum(x.cost for x in sc) + 1000.0 * sum(not x.ok for x in sc)


def search_roll(gains: Dict[str, dict], rounds: int = 5, workers: int = 14) -> Dict[str, dict]:
    import multiprocessing as mp
    g = copy.deepcopy(gains)
    keys = [("roll", "kp"), ("roll", "kd"), ("roll", "ki"), ("roll", "lpf_tau_s")]
    best = _roll_cost_job(g)
    step = 0.6
    with mp.Pool(workers) as pool:
        for _ in range(rounds):
            improved = False
            for loop, par in keys:
                cands = []
                for f in (1 + step, 1 / (1 + step), 1 + 2 * step, 1 / (1 + 2 * step)):
                    c = copy.deepcopy(g)
                    c[loop][par] = (g[loop].get(par, 0.0) * f) if g[loop].get(par, 0.0) > 0 else 0.05 * (f - 1.0 > 0)
                    cands.append(c)
                costs = pool.map(_roll_cost_job, cands)
                i = int(np.argmin(costs))
                if costs[i] < best - 1e-9:
                    best, g, improved = costs[i], cands[i], True
            if not improved:
                step *= 0.5
    return g


def _sens_job(args):
    g, (loop, par) = args
    full = loop_sensitivity(g, only=(loop, par))
    return full


def loop_sensitivity(gains: Dict[str, dict], plant: str = "nominal", factors=(0.5, 0.75, 1.0, 1.5, 2.0), seeds=(1, 2), only=None) -> Dict[str, dict]:
    """Measured effect of each gain on the step response (the numbers quoted in the yaml comments). Returns {loop.param: {factor: metrics}}."""
    out: Dict[str, dict] = {}
    plan = {"heave": ("kp", "kd", "ki", "lpf_tau_s"), "pitch": ("kp", "kd", "ki", "lpf_tau_s"), "yaw": ("kp", "kd", "ki", "lpf_tau_s"), "roll": ("kp", "kd", "ki", "lpf_tau_s"), "speed": ("kp", "ki")}
    for loop, params in plan.items():
        for par in params:
            if only is not None and (loop, par) != tuple(only):
                continue
            base = gains[loop].get(par, 0.0)
            if base <= 0:
                continue
            row = {}
            for f in factors:
                g = copy.deepcopy(gains)
                g[loop][par] = base * f
                ms = []
                for sd in seeds:
                    if loop == "heave" or loop == "pitch":
                        d = score_hold(g, plant, seed=sd).detail
                        ms.append({"overshoot_pct": 100 * d["over_d" if loop == "heave" else "over_p"], "settle_s": d["settle_d_s" if loop == "heave" else "settle_p_s"], "chatter_rpm": d["chatter_rpm"]})
                    elif loop == "yaw":
                        d = score_yaw(g, plant, seed=sd).detail
                        ms.append({"overshoot_pct": 100 * d["over_yaw"], "settle_s": d["settle_yaw_s"], "fin_p2p_deg": d["fin_p2p_deg"]})
                    elif loop == "roll":
                        d = score_roll(g, plant, seed=sd).detail
                        ms.append({"overshoot_pct": 100 * d["over_roll"], "settle_s": d["settle_roll_s"], "fin_p2p_deg": d["fin_p2p_deg"]})
                    else:
                        d = simulate_yaw(g, plant, seed=sd, step_deg=0.0)
                        t, u = d[:, 0], d[:, 3]
                        post = t > 3.0
                        over = max(0.0, float(np.max(u[post]) - 0.8)) / 0.8
                        idx = np.where(np.abs(u - 0.8) > 0.04)[0]
                        idx = idx[t[idx] > 3.0]
                        ms.append({"overshoot_pct": 100 * over, "settle_s": float(t[idx[-1]] - 3.0) if len(idx) else 0.0, "ss_error_mps": float(abs(np.mean(u[t > t[-1] - 10.0]) - 0.8))})
                row[str(f)] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            out[f"{loop}.{par}"] = {"value": base, "factors": row}
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = yaml.safe_load(open(_CTRL / "terminal_docking_control" / "terminal_docking.yaml"))
    gains = cfg["gains"]
    if not argv or argv[0] == "show":
        for plant, sc in score_all(gains).items():
            print(f"{plant:11s} {'OK ' if sc.ok else 'FAIL'} cost {sc.cost:7.1f}  " + "  ".join(f"{k}={v:.3g}" for k, v in sc.detail.items()))
        return 0
    if argv[0] == "sens":
        import json
        S = str(tmp_dir()) + "/"
        g = yaml.safe_load(open(S + "final_gains.yaml"))
        import multiprocessing as mp
        plan = [("heave", "kp"), ("heave", "kd"), ("heave", "ki"), ("heave", "lpf_tau_s"), ("pitch", "kp"), ("pitch", "kd"), ("pitch", "ki"), ("pitch", "lpf_tau_s"),
                ("yaw", "kp"), ("yaw", "kd"), ("yaw", "ki"), ("yaw", "lpf_tau_s"), ("roll", "kp"), ("roll", "kd"), ("roll", "ki"), ("speed", "kp"), ("speed", "ki")]
        with mp.Pool(14) as pool:
            parts = pool.map(_sens_job, [(g, lp) for lp in plan])
        res = {}
        for r in parts:
            res.update(r)
        json.dump(res, open(S + "loop_sens.json", "w"), indent=1)
        print(json.dumps(res, indent=1)[:3000])
        return 0
    if argv[0] == "hover":
        S = str(tmp_dir()) + "/"
        g = yaml.safe_load(open(S + "final_gains.yaml"))
        g["heave"].update(kp=15.0, ki=0.5, kd=25.0, lpf_tau_s=0.5, i_max=10.0)          # start from a stiffer point than the docking set (i_max must exceed the push to cancel)
        g["pitch"].update(kp=1.4, ki=0.05, kd=6.0, lpf_tau_s=0.5, i_max=1.5)
        g2 = search_hover(g)
        yaml.safe_dump(g2, open(S + "hover_gains.yaml", "w"))
        print(yaml.safe_dump({k: g2[k] for k in ("heave", "pitch")}, sort_keys=False))
        for plant in PLANTS:
            for sd in (1, 2, 3):
                sc = score_hover(g2, plant, seed=sd)
                print(f"{plant:11s} seed {sd} {'OK ' if sc.ok else 'FAIL'} " + "  ".join(f"{k}={v:.3g}" for k, v in sc.detail.items()))
        return 0
    if argv[0] == "yaw":
        g2 = search_yaw(gains)
        print(yaml.safe_dump({k: g2[k] for k in ("yaw", "speed") if k in g2}, sort_keys=False))
        for plant in PLANTS:
            for sd in (1, 2, 3):
                sc = score_yaw(g2, plant, seed=sd)
                print(f"{plant:11s} seed {sd} {'OK ' if sc.ok else 'FAIL'} " + "  ".join(f"{k}={v:.3g}" for k, v in sc.detail.items()))
        return 0
    if argv[0] == "tune":
        sel = argv[1:] or ["heave", "pitch"]
        keys = [(loop, p) for loop in sel for p in ("kp", "kd", "ki") if p in gains[loop]]
        g2 = coordinate_search(gains, keys)
        print(yaml.safe_dump({k: g2[k] for k in sel}, sort_keys=False))
        for plant, sc in score_all(g2).items():
            print(f"{plant:11s} {'OK ' if sc.ok else 'FAIL'} cost {sc.cost:7.1f}  " + "  ".join(f"{k}={v:.3g}" for k, v in sc.detail.items()))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
