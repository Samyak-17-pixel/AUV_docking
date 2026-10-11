#!/usr/bin/env python3
"""Generate the fully documented terminal_docking.yaml from the current values + the measured sensitivity (sensitivity.py --out sens.json). Dev tool.

  python3 write_docking_yaml.py sens.json            # rewrites terminal_docking_control/terminal_docking.yaml (values are kept, comments regenerated)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

YAML = Path(__file__).resolve().parents[1] / "terminal_docking_control" / "terminal_docking.yaml"

D = {   # path -> (explanation, analogy or '')
    "topics.odometry": ("Odometry topic (nav_msgs/Odometry, NED pose, BODY twist).", ""), "topics.dock_align": ("Detector output (interfaces/DockAlign).", ""), "topics.actuator_cmd": ("Actuator command topic (interfaces/Actuator: th_XX in RPM, cs_XX in degrees).", ""), "topics.ctrl_debug": ("JSON status for the viewer: controller, phase, distance, setpoints.", ""),
    "node.rate_hz": ("Control loop rate. The loops are slow (seconds), so 20 Hz is plenty; the odometry only arrives at ~4.5 Hz and is smoothed by the pose filter in between.", "How many times per second the driver looks at the road."),
    "node.ros_domain_id": ("ROS domain used when ROS_DOMAIN_ID is not already set. 42 = the real mavsim bridge. For offline work export ROS_DOMAIN_ID=77 (run_terminal_docking.sh respects it).", ""),
    "node.vessel": ("Vessel name in the topic names.", ""),
    "camera.f_px": ("Focal length of camera_03 in pixels: 0.5 * 480 / tan(30 deg) = 415.7 (the 60 deg field of view of the vessel file is VERTICAL). Turns a pixel into a bearing; a 10% error here is a 10% error in every distance.", ""),
    "camera.width": ("Image width [px].", ""), "camera.height": ("Image height [px].", ""),
    "camera.mount_m": ("Camera position in the body frame [m]: on the nose, 0.575 m ahead of the vehicle centre.", ""),
    "dock.ring_radius_m": ("Radius of the circle of dock lights [m] (vessel file: 1 m). With the pixel radius it gives the range: d = f * 1 m / radius_px.", ""),
    "vehicle.nose_m": ("Nose tip ahead of the vehicle centre [m] (mesh bound 0.66). The stop point is defined for the NOSE.", ""),
    "estimator.pixel_sigma_px": ("Assumed noise of one light centre [px]. Lower = the filter trusts each detection more (faster, jumpier); higher = smoother and slower to converge.", "How much you believe each tape-measure reading."),
    "estimator.gate_chi2": ("A detection whose normalised innovation (a chi-square with 8 degrees of freedom) is above this is REJECTED as an outlier (a reflection, a wrong label). 40 = about 5.6 sigma.", "A bouncer who throws out any reading that disagrees wildly with what you already know."),
    "estimator.process_pos_m_per_sqrt_s": ("How much the dock position estimate is allowed to wander [m per sqrt(s)]. The dock is static, so this is tiny; it only stops the filter from becoming overconfident.", ""),
    "estimator.process_yaw_deg_per_sqrt_s": ("The same for the dock's axis heading [deg per sqrt(s)].", ""),
    "estimator.position_source": ("Where the VEHICLE position (x, y) comes from: odometry = the pose filter's x, y (default); dead_reckoning = integrated from the filtered body speed and attitude since the start, NO absolute position is used (depth still from the depth sensor). The dock position is triangulated from the lights either way, so a constant odometry offset never mattered; what dead reckoning removes is the dependence on odometry DRIFT and JUMPS while the dock is only remembered (the last ~2 m). With dead_reckoning, recover.dock_hint_ned is relative to the START pose. The speed and heading it needs still come from the odometry twist/IMU (unverified on the real sim).", ""),
    "estimator.min_radius_px": ("Detections whose light circle is smaller than this [px] (range > f*1/20 = 20 m) are ignored.", ""),
    "estimator.odom_latency_s": ("Age of an odometry sample when it arrives [s] (ASSUMED 0.25; the real value is unknown).", ""),
    "estimator.image_delay_s": ("Age of a camera frame when its detection arrives [s] (ASSUMED 0.10). The detection is matched with the vehicle pose of that moment.", ""),
    "estimator.init_yaw_sigma_deg": ("Prior uncertainty of the dock axis heading at the first detection [deg]: from far away the axis direction is almost unobservable, so the first guess is 'same as the vehicle heading +-30 deg'.", ""),
    "estimator.reinit_after_rejects": ("A young estimate (fewer than 5 updates) that rejects this many detections in a row is thrown away and rebuilt (it was probably initialised on a wrong detection).", ""),
    "pose_filter.accel_noise_lin": ("Pose filter process noise, positions [m/s^2]: how fast the vehicle may change speed (10 N / 20 kg = 0.5). Higher = follows new odometry faster, noisier.", ""),
    "pose_filter.accel_noise_ang": ("The same for angles [rad/s^2].", ""),
    "pose_filter.pos_sigma_m": ("Assumed odometry position noise [m].", ""),
    "pose_filter.ang_sigma_rad": ("Assumed odometry angle noise [rad] (0.25 deg).", ""),
    "pose_filter.vel_sigma_mps": ("Assumed velocity noise [m/s].", ""), "pose_filter.rate_sigma_rps": ("Assumed angular-rate noise [rad/s].", ""),
    "guidance.gate_s_m": ("Distance of the vehicle centre in front of the dock plane at which the ALIGNMENT GATE is checked [m]. Before it: APPROACH (steer onto the axis); after a passed gate: TERMINAL. A larger gate gives more room to retry but less room to align.", "A checkpoint on the runway: you must be lined up by here or you go around."),
    "guidance.gate_grace_m": ("If the gate check fails, wait this much further before giving up and backing off [m] (alignment keeps improving while the vehicle moves).", ""),
    "guidance.gate_lateral_m": ("Maximum cross-track error at the gate [m].", ""),
    "guidance.gate_heading_deg": ("Maximum heading error at the gate [deg]. The requirement at the mouth is 5 deg; this looser value is allowed because the heading keeps converging for the last metre.", ""),
    "guidance.gate_vertical_m": ("Maximum depth error at the gate [m].", ""),
    "guidance.gate_axis_sigma_deg": ("The dock-axis estimate must be at least this good (1-sigma) at the gate [deg], otherwise the vehicle is steering at a guess.", ""),
    "guidance.max_retries": ("How many times it may back off and try again. A run that needed a retry is reported as 'success with retry', NOT as a first-try pass.", ""),
    "guidance.retry_back_to_s_m": ("A retry reverses until the vehicle is this far in front of the dock [m], then approaches again.", ""),
    "guidance.lookahead_base_m": ("Line-of-sight guidance: the vehicle aims at the axis point this far ahead of its own projection, plus lookahead_per_m times its distance. SMALL = aggressive (steers hard onto the axis, can overshoot); LARGE = gentle (takes many metres to converge).", "Looking at a point just ahead of the car's bumper versus far down the road when you change lanes."),
    "guidance.lookahead_per_m": ("Extra lookahead per metre of distance to the dock (farther = gentler).", ""),
    "guidance.max_deviation_deg": ("Largest angle between the heading and the axis that guidance may ask for [deg]. It limits how sideways the vehicle may point to correct a large lateral error.", ""),
    "guidance.terminal_lookahead_m": ("Lookahead after the gate [m]. Large = the vehicle only trims lightly inside the last metres (changing heading with little speed is weak anyway).", ""),
    "guidance.stop_nose_s_m": ("Where the NOSE should stop relative to the dock mouth plane [m]; negative = inside the funnel (-0.5 = 0.5 m in).", ""),
    "guidance.docked_speed_mps": ("Speed below which the vehicle counts as stopped [m/s].", ""),
    "guidance.docked_band_m": ("DOCKED is declared when the nose is within this band beyond the stop point [m].", ""),
    "guidance.docked_below_m": ("... and not more than this far past it [m] (deeper than that it creeps back instead).", ""),
    "guidance.abort_lateral_m": ("In TERMINAL, a cross-track error above this [m] triggers a back-off (while the nose is still more than no_abort_inside_m from the mouth).", ""),
    "guidance.no_abort_inside_m": ("Inside this distance of the mouth [m] the vehicle commits (backing out of a funnel it is already entering would only scrape the wall).", ""),
    "guidance.depth_trust_sigma_m": ("The dock depth estimate is used as the depth target once its 1-sigma is below this [m]; before that the vehicle holds its current depth.", ""),
    "speed.cruise_mps": ("Approach speed when aligned [m/s]. Fin authority grows with speed squared, so slower = weaker steering; faster = more to brake at the end.", ""),
    "speed.align_cruise_mps": ("Approach speed while still misaligned [m/s] (a bit faster: more steering authority).", ""),
    "speed.align_e_m": ("Cross-track error above which the vehicle counts as misaligned [m].", ""),
    "speed.align_chi_deg": ("Heading error above which the vehicle counts as misaligned [deg].", ""),
    "speed.terminal_mps": ("Speed in the terminal phase and at the gate [m/s]. Below ~0.3 the fins lose authority (the heading can no longer be corrected); the entry limit is 0.5.", ""),
    "speed.slow_zone_m": ("The speed falls linearly from the cruise speed to the terminal speed over this distance before the gate [m].", ""),
    "speed.retry_back_mps": ("Reverse speed during a retry [m/s].", ""),
    "speed.kp_n_per_mps": ("Axial force per m/s of speed error [N]. The vehicle has almost no drag, so speed changes only when you push or brake.", ""),
    "speed.max_forward_n": ("Largest forward force [N].", ""), "speed.max_brake_n": ("Largest braking/reverse force [N] (the thruster could give ~90 N).", ""),
    "speed.decel_mps2": ("Planned deceleration for the stopping-distance envelope [m/s^2]: the fastest speed from which the vehicle can still stop at the stop point is sqrt(2*decel*distance).", "How hard the driver is willing to brake."),
    "speed.mass_kg": ("Vehicle mass for the braking feed-forward and the open-loop speed model [kg] (20 kg + a bit of added mass).", ""),
    "speed.drag_n_per_mps2": ("Quadratic drag used only when the odometry is frozen and the speed is dead-reckoned [N per (m/s)^2].", ""),
    "speed.dr_rest_mps": ("While frozen, braking stops when the dead-reckoned speed falls below this [m/s].", ""),
    "speed.back_mps": ("Largest creep-back speed when the nose went past the stop point [m/s].", ""),
    "speed.stop_gain_per_s": ("Final approach law: speed = this * distance to the stop point [1/s]. 0.8 = the vehicle closes the last metre exponentially with a 1.25 s time constant, no hard brake.", "Easing up to a kerb instead of braking at the last moment."),
    "speed.ff_fade_mps": ("(kept for reference) speed below which the braking feed-forward fades.", ""),
    "feedforward.heave_n": ("Net buoyancy compensation [N, + = down]. 0: the vehicle (20 kg) is neutrally buoyant.", ""),
    "limits.rpm_cap": ("Soft cap on every thruster [RPM] (hardware 2668).", ""), "limits.fin_deg_cap": ("Soft cap on each fin [deg] (hardware 35).", ""),
    "limits.small_force_n": ("Below this thrust the force-to-RPM map is linear instead of a square root [N]. Thrust ~ RPM^2, so the exact inverse has an infinite slope at zero and turns a +-0.5 N flicker into a +-370 RPM flip. 0.3 N cuts the thruster chatter from 380-560 to 100-250 RPM p-p.", ""),
    "limits.u_fin_min_mps": ("Speed floor in the 1/speed^2 fin scheduling [m/s].", ""), "limits.u_fin_off_mps": ("Below this speed the fins are switched off [m/s].", ""),
    "limits.u_fin_full_mps": ("Above this speed the fins have full authority [m/s]; in between they fade in.", ""),
    "recover.enabled": ('false = the old behaviour: do nothing until the dock is seen (phase WAIT) and only retry at the gate. true = SEARCH when the dock has never been seen, and REPOSITION (back out along the axis, then approach again) when the remembered dock pose says the vehicle cannot get onto the axis in time.', ""),
    "recover.search_delay_s": ('Wait this long after start before searching (lets the first detections arrive) [s].', ""),
    "recover.search_speed_mps": ('Surge while searching [m/s]. The fins need flow to turn the vehicle, so it drives a slow circle. Faster = wider circle (it needs more room).', ""),
    "recover.search_turn_deg": ('Heading offset asked of the yaw loop while searching [deg]; the vehicle turns at its fin-limited rate (about u/2.4 m rad/s) and so sweeps the camera round.', ""),
    "recover.search_back_mps": ('Reverse speed when the detector says the ring is bigger than the frame (too close) [m/s].', ""),
    "recover.hint_hold_s": ('A detector search hint is kept for this long [s] (lights flicker in and out of the frame).', ""),
    "recover.half_hfov_deg": ("Horizontal half field of view [deg], converts the detector's normalised yaw hint into a heading change.", ""),
    "recover.dock_hint_ned": ('Optional rough dock position [x, y, z] in NED metres from the mission plan (null = none: the search is then blind). With it the search turns toward the dock and, closer than 2 turning radii + hint_clear_m, first backs straight away (a vehicle that cannot sway would otherwise swing into it). It is only a prompt: the real position comes from the lights.', ""),
    "recover.hint_face_deg": ('Pointing within this angle of the hinted dock counts as facing it: the vehicle drives at it instead of retreating [deg].', ""),
    "recover.hint_see_m": ('Closer to the hinted dock than this the 2 m ring does not fit the picture (it needs about 2.3 m, more when seen at an angle), so the vehicle backs away first [m].', ""),
    "recover.dock_axis_deg": ('With dock_hint_ned: the heading [deg, from north toward east] a vehicle needs to drive INTO the dock (0 = the funnel points north, as in the mavsim world).', ""),
    "recover.hint_cone_deg": ('The dock lights shine out of the mouth in a cone; the search drives straight at the dock only from within this angle off the axis [deg], otherwise it first goes to a staging point on the axis.', ""),
    "recover.hint_nav_keep_m": ('The route to the staging point stays this far from the hinted dock [m] (about the turning circle plus a margin).', ""),
    "recover.hint_reach_m": ('The staging point counts as reached within this distance [m].', ""),
    "recover.hint_stage_m": ('Distance in front of the dock mouth, on its axis, of that staging point [m].', ""),
    "recover.hint_clear_m": ('Extra distance beyond two turning radii that the hint logic keeps from the dock before it starts to turn [m].', ""),
    "recover.turn_radius_m": ('Tightest turning circle of the vehicle [m] (ASSUMED from the fin authority; measure it on the real sim: mission/ and the yaw-step checklist). Larger = backs out earlier and further.', ""),
    "recover.settle_m": ('Straight run wanted between finishing the turn and the gate [m].', ""),
    "recover.margin": ('Back out when the room available is below margin x the room needed. 1.2 = more cautious (more back-outs).', ""),
    "recover.dead_e_m": ('Lateral errors below this [m] are ignored by the room check (the line-of-sight guidance removes them on its own).', ""),
    "recover.dead_chi_deg": ('Heading errors below this [deg] are ignored by the room check.', ""),
    "recover.target_margin": ('The back-out goes this far x the room needed (more than the trigger margin, so it does not trigger again straight away).', ""),
    "recover.extra_back_m": ('Back out this much further than strictly needed [m].', ""),
    "recover.max_back_to_m": ("Never back out beyond this distance in front of the dock [m] (the dock leaves the detector's range, ~12 m).", ""),
    "recover.max_repositions": ('How many planned back-outs per run (separate from guidance.max_retries, which counts failed gates).', ""),
    "recover.min_axis_sigma_deg": ('Only judge feasibility once the dock axis estimate is this good (1-sigma) [deg]; the first guess is +-30 deg and would trigger needless back-outs.', ""),
    "recover.min_pos_sigma_m": ('... and the dock position this good [m].', ""),
    "recover.commit_below_m": ('Closer than gate_s_m + this the vehicle is committed: only the gate check and the retry apply [m].', ""),
    "safety.stale_odom_s": ("Odometry older than this, or the same sample repeated for this long [s], counts as FROZEN: the controller brakes open loop and waits (SAFE_STOP).", ""),
    "safety.min_depth_m": ("Abort if shallower than this [m].", ""), "safety.max_depth_m": ("Abort if deeper than this [m].", ""), "safety.max_tilt_deg": ("Abort if roll or pitch exceeds this [deg].", ""),
    "safety.dock_lost_timeout_s": ("(reserved) seconds without a valid detection before the controller gives up.", ""),
    "logging.dir": ("Where the CSV log of a run goes (one file per run, written on exit).", ""), "logging.status_period_s": ("Seconds between status lines.", ""),
}
GAIN_TXT = {"heave": "depth hold -> Z force [N, + = down]", "pitch": "pitch hold (level) -> M moment [N*m]", "yaw": "heading hold -> N moment [N*m] via the fins (needs forward speed)", "roll": "roll hold (level) -> K moment [N*m] via the fins (needs forward speed)"}


def fmt(v):
    if v is None:
        return "null"
    return f"{v:g}" if isinstance(v, float) else str(v)


def effect(sens, path):
    row = sens.get(path)
    if not row:
        return ""
    ps = row["passes"]
    n = ps["1.0"][1]
    parts = [f"x{f}: {ps[f][0]}/{n}" for f in ("0.5", "0.75", "1.0", "1.25", "1.5") if f in ps]
    return "MEASURED first-try passes on the 240-run grid when changed: " + " | ".join(parts)


def main(argv):
    sens = json.load(open(argv[0])) if argv else {}
    cfg = yaml.safe_load(open(YAML))
    out = ["# Terminal docking control for Mako_01: align with the dock from any start inside the camera's field of view, then enter the funnel and stop.",
           "#",
           "# WHAT IT DOES (vision + odometry only: the dock's world position is NOT hard-coded)",
           "#   1. A Kalman filter turns the four light pixels of every valid DockAlign message + the vehicle pose into the dock's pose in the WORLD (position and axis heading).",
           "#      The dock is static, so the estimate is REMEMBERED when the lights leave the image (inside ~1.7 m the 1 m ring no longer fits the 60 deg vertical field of view).",
           "#   2. APPROACH: steer onto the dock axis line by yawing while moving (no sway thruster; the fins need flow), at cruise speed, slowing to the terminal speed before the gate.",
           "#   3. GATE: lateral, heading and depth errors and the axis-estimate quality are checked at gate_s_m. Pass -> TERMINAL. Fail -> back off and try again (counted as a retry).",
           "#   4. TERMINAL: constant slow speed, then the final approach law (speed proportional to the distance left) stops the NOSE at stop_nose_s_m inside the funnel -> DOCKED.",
           "#   Safety: stale or frozen odometry -> brake open loop (dead-reckoned speed) and wait; lost lights -> keep going on the remembered dock.",
           "#   RECOVER (2026-10-11): dock never seen -> SEARCH (a slow circle, or with recover.dock_hint_ned: go to a staging point on the dock axis, turn to the axis heading, drive at it);",
           "#   too close / too far off the axis to line up in time -> back out along the axis (REPOSITION, planned, separate from the gate retries), then approach again. recover.enabled: false = the old behaviour.",
           "#",
           "# HOW IT WAS TUNED (2026-10-10) and WHAT IT MEANS",
           "#   In-process closed loop (terminal_docking_control/docking_sim.py): the offline vehicle + realistic odometry (4.5 Hz, 0.25 s late, noise) + the synthetic camera + the REAL detector,",
           "#   over a grid of start poses (range 5/7/9 m x lateral -2..+2 m x heading -30..+30 deg + random combinations, dock inside the field of view), on four plants (nominal, heavy+late, light+fast, stress).",
           "#   A run FAILS on: wall contact, bad entry (speed > 0.5 m/s, heading > 5 deg, lateral/vertical > 15 cm at the mouth), overshoot/stopping short, lost dock, timeout, more than 2 back-offs,",
           "#   depth/attitude limits, a thruster pinned at its cap > 2 s, heave chatter > 300 RPM p-p near the dock, oscillation, driving on stale odometry. A retry is NOT a pass.",
           "#   RESULT: see the RESULTS block at the end of this header (generated by terminal_docking_eval.py). EVERYTHING IS ON THE ASSUMED OFFLINE MODEL (common/mako_geometry.yaml) AND AN ASSUMED CAMERA LOOK:",
           "#   it is not validated on the real simulator. Re-tune there: run terminal_docking_eval.py after changing the model, ros_dock_confirm.py over ROS.",
           "#   The numbers 'MEASURED first-try passes' below were produced by control_code/tuning/sensitivity.py: each value changed alone by x0.5 ... x1.5, passes counted over the same 240 runs.",
           "#"]
    res_path = Path(__file__).resolve().parents[2] / "outputs" / "docking_sweeps" / "docking_results.txt"
    if res_path.exists():
        out += ["# RESULTS (wide grid, perfect vision, 4 plants x 60 starts):"] + [f"#   {l}" for l in res_path.read_text().strip().splitlines()] + ["#"]
    sec_doc = {"node": "ROS node settings", "topics": "ROS topic names", "camera": "Nose camera camera_03 (pinhole, looking along the body x axis)", "dock": "The dock's light ring in its own frame",
               "vehicle": "Vehicle geometry used by the controller", "estimator": "Dock estimator (Kalman filter on the light pixels)", "pose_filter": "Pose filter (common/pose_filter.py): smooth state from the slow, late, noisy odometry",
               "guidance": "Alignment, gate and final approach", "speed": "Speed profile and the axial-thruster loop", "feedforward": "Feed-forward", "gains": "Hold loops (common/loops.py): PID on the measurement, output low-pass lpf_tau_s, setpoint filter sp_tau_s",
               "limits": "Actuator limits and allocation", "recover": "Search for the dock and back out when there is no room to line up   [NEW 2026-10-11; ASSUMED turning circle 2.4 m]", "safety": "Safety trips and the frozen-odometry rule", "logging": "Logging"}
    for sec, body in cfg.items():
        out += ["", f"# {'-' * 75}", f"# {sec.upper()}  -  {sec_doc.get(sec, '')}", f"# {'-' * 75}", f"{sec}:"]
        if sec == "gains":
            for loop, g in body.items():
                out.append(f"  {loop}: {{{', '.join(f'{k}: {fmt(v)}' for k, v in g.items())}}}   # {GAIN_TXT.get(loop, '')}")
            out += ["  # kp [N or N*m per unit error], ki (integral), kd (on the measured RATE), i_max (integral cap), max (output cap), lpf_tau_s (output low-pass [s]).",
                    "  # heave/pitch: tuned on four plants with realistic odometry (control_code/tuning/loop_tuner.py): depth and pitch hold within 1 cm / 1 deg, no oscillation, heave chatter < 200 RPM.",
                    "  # roll: the OLD shipped values (kp 0.5, kd 0.2) oscillated with realistic odometry (fins +-20 deg); roll inertia is only 0.06 kg*m^2, so the loop must be gentle (kp 0.10, kd 0.125).",
                    "  # yaw: tuned together with the guidance (see guidance.*); heading step response, 30 deg at 0.8 m/s: settles in ~4 s, overshoot < 2% (loop_tuner.py yaw)."]
            continue
        for k, v in body.items():
            path = f"{sec}.{k}"
            what, ana = D.get(path, ("", ""))
            if isinstance(v, dict):
                items = ", ".join(f"{a}: {fmt(b) if not isinstance(b, list) else b}" for a, b in v.items())
                out.append(f"  {k}: {{{items}}}   # {what}")
                continue
            line = f"  {k}: {yaml.safe_dump(v, default_flow_style=True).strip().splitlines()[0] if isinstance(v, list) else fmt(v)}"
            txt = what + (f" Analogy: {ana}" if ana else "")
            eff = effect(sens, path)
            out.append(((line + " ").ljust(34) + f"# {txt}") if txt.strip() else line)
            if eff:
                out.append(" " * 34 + f"#   {eff}")
    YAML.write_text("\n".join(out) + "\n")
    chk = yaml.safe_load(open(YAML))
    assert chk == cfg, "values changed while writing the comments"
    print("wrote", YAML, "values unchanged;", sum(1 for _ in open(YAML)), "lines")


if __name__ == "__main__":
    main(sys.argv[1:])
