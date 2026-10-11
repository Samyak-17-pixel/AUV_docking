#!/usr/bin/env python3
"""Write the tuned hold-loop gains and their MEASURED effects into station_keeping.yaml and dof_testing.yaml (comments included). Dev tool, run once per tuning.

  python3 write_gain_docs.py final_gains.yaml loop_sens.json
It edits the 'gains:' section of both files in place; everything else in the files is untouched.
"""
import json
import re
import sys
from pathlib import Path

import yaml

CTRL = Path(__file__).resolve().parents[1]
UNITS = {"heave": "N", "pitch": "N*m", "yaw": "N*m", "roll": "N*m", "speed": "N"}
PER = {"heave": "per metre of depth error", "pitch": "per radian of pitch error", "yaw": "per radian of heading error", "roll": "per radian of roll error", "speed": "per m/s of speed error"}
TEST = {"heave": "0.4 m dive", "pitch": "6 deg pitch step", "yaw": "30 deg heading step at 0.8 m/s", "roll": "15 deg roll step at 1 m/s", "speed": "0 -> 0.8 m/s start"}
WHAT = {
    ("heave", "kp"): "Stiffness. Analogy: a spring pulling the vehicle back to the depth; stiffer = faster but it rings, and the noise of the depth reading is multiplied by it.",
    ("heave", "kd"): "Damping on the depth RATE (the vehicle has almost no natural damping, so this is what stops the ringing). Analogy: the shock absorber.",
    ("heave", "ki"): "Integral. Only removes a steady push (a current, a trim error). Analogy: leaning on the rope a little longer each second you are still off target.",
    ("heave", "lpf_tau_s"): "Time constant of a low-pass on the loop OUTPUT: keeps measurement noise from flicking the thrusters. Analogy: a heavy flywheel between the sensor and the thrusters.",
    ("pitch", "kp"): "Stiffness of the pitch hold (no righting moment on this vehicle: CG == CB, so only this loop keeps it level). Analogy: balancing a plank on a pivot.",
    ("pitch", "kd"): "Damping on the pitch rate. Never 0: with no natural damping the pitch keeps ringing.",
    ("pitch", "ki"): "Integral; only for a steady pitching moment.",
    ("pitch", "lpf_tau_s"): "Output low-pass (noise filter), see heave.",
    ("yaw", "kp"): "Stiffness of the heading hold (needs fin flow: forward speed above ~0.4 m/s). Analogy: how hard the driver turns the wheel per degree of heading error.",
    ("yaw", "kd"): "Damping on the yaw rate. Too much slows the turn; too little overshoots.",
    ("yaw", "ki"): "Integral; the vehicle has no steady yaw moment, keep it tiny.",
    ("yaw", "lpf_tau_s"): "Output low-pass: smooths the fin command (fins at 1 m/s turn 1 deg of noise into several degrees of deflection).",
    ("roll", "kp"): "Stiffness of the roll hold. Roll inertia is tiny (0.06 kg*m^2) so the loop is very fast: the SHIPPED values (kp 0.5, kd 0.2) made the fins chatter 15-40 deg and the roll overshoot 50-190% once the odometry was realistic.",
    ("roll", "kd"): "Damping on the roll rate. Below ~0.1 the roll rings (x0.5 = 19% overshoot).",
    ("roll", "ki"): "Integral; keep tiny.",
    ("speed", "kp"): "Axial force per m/s of speed error (used to give the fins flow in dof_testing). The vehicle has almost no drag, so speed changes only when you push.",
    ("speed", "ki"): "Integral of the speed error; too much overshoots the speed target (the vehicle keeps coasting).",
}
MAXNOTE = {"heave": "Total heave force cap [N]; the two thrusters give about 23 N at 1800 RPM.", "pitch": "Pitch moment cap [N*m]: about half what the two heave thrusters can make.",
           "yaw": "Yaw moment cap [N*m] (the fins give 0.7 N*m at 0.5 m/s and 2.7 at 1 m/s).", "roll": "Roll moment cap [N*m].", "speed": "Axial force cap [N]."}


def measured(sens, loop, par):
    key = f"{loop}.{par}"
    if key not in sens:
        return None
    parts = []
    for f in ("0.5", "0.75", "1.0", "1.5", "2.0"):
        m = sens[key]["factors"].get(f)
        if m is None:
            continue
        lab = "x1 (this)" if f == "1.0" else f"x{f}"
        bits = f"overshoot {m['overshoot_pct']:.0f}%, settle {m['settle_s']:.0f} s"
        if "chatter_rpm" in m:
            bits += f", thruster chatter {m['chatter_rpm']:.0f} RPM"
        if "fin_p2p_deg" in m:
            bits += f", fin swing {m['fin_p2p_deg']:.1f} deg"
        parts.append(f"{lab} -> {bits}")
    return parts


def block_style(loop, g, sens, comment_head):
    u, per = UNITS[loop], PER[loop]
    out = [f"  {loop}:{' ' * max(1, 24 - len(loop))}# {comment_head}"]
    for par in ("kp", "ki", "kd", "i_max", "max", "lpf_tau_s", "sp_tau_s"):
        if par not in g:
            continue
        v = g[par]
        vs = f"{v:g}"
        if par in ("kp", "ki", "kd"):
            unit = f"{u} {per}" if par == "kp" else (f"{u} per (unit*s)" if par == "ki" else f"{u} per unit/s of the rate")
            lines = [f"    {par}: {vs}".ljust(29) + f"# {unit}. {WHAT.get((loop, par), '')}"]
            m = measured(sens, loop, par)
            if m:
                lines.append(" " * 29 + f"#   MEASURED ({TEST[loop]}, nominal plant, 4.5 Hz odometry 0.25 s late with noise; 0.5x .. 2x of this value):")
                for p in m:
                    lines.append(" " * 29 + f"#     {p}")
                lines.append(" " * 29 + "#   A change of about 1.5x-2x is a considerable change. Checked on three more plants (heavy+late, light+fast, stress): the first two pass the same criteria; the stress plant stays stable but with more overshoot/chatter (see the header).")
        elif par == "i_max":
            lines = [f"    i_max: {vs}".ljust(29) + f"# {u}. Cap on the integral part: big enough for a few-N steady push without a long wind-up."]
        elif par == "max":
            lines = [f"    max: {vs}".ljust(29) + f"# {MAXNOTE[loop]} Asking for more just saturates."]
        elif par == "lpf_tau_s":
            lines = [f"    lpf_tau_s: {vs}".ljust(29) + f"# s. {WHAT[(loop, 'lpf_tau_s')]}"]
            m = measured(sens, loop, par)
            if m:
                lines.append(" " * 29 + f"#   MEASURED ({TEST[loop]}): " + " | ".join(m))
        else:
            lines = [f"    {par}: {vs}".ljust(29) + "# s. Time constant of a first-order filter on the SETPOINT: a step becomes a smooth ramp, so a big move does not overshoot. Starts from the measurement."]
        out += lines
    return out


def patch_station_keeping(path, gains, sens):
    text = path.read_text().split("\n")
    s = next(i for i, l in enumerate(text) if l.startswith("gains:"))
    e = next(i for i in range(s + 1, len(text)) if text[i].strip() and not text[i].startswith((" ", "#")))
    block = text[s:e]
    def part(name):
        a = next(i for i, l in enumerate(block) if re.match(rf"^  {name}:", l))
        b = next((i for i in range(a + 1, len(block)) if re.match(r"^  [a-z_]+:", block[i])), len(block))
        return block[a:b]
    surge, others = part("surge"), []
    head = block[:1]
    note = ["  # HEAVE, PITCH and YAW were RE-TUNED 2026-10-10 against the offline vehicle with REALISTIC odometry (4.5 Hz, 0.25 s late, 1.5 cm / 0.2 deg noise, freezes) and four plants",
            "  # (nominal, heavy+late, light+fast, stress) by control_code/tuning/loop_tuner.py. The old values (heave 60/60, pitch 6/5) were measured on perfect 100 Hz data: with the",
            "  # real sensing they chattered the thrusters by 2000 RPM peak-to-peak. The measured numbers below are for the NEW values (nominal plant, mean of 2 noise seeds).",
            "  # The SURGE block further down is still the old measurement on the old model."]
    new = head + note
    new += block_style("heave", gains["heave"], sens, "depth hold -> Z force [N, + = down]")
    new += block_style("pitch", gains["pitch"], sens, "pitch hold -> M moment [N*m, + = nose up] by differential heave thrust")
    new += surge
    new += block_style("yaw", gains["yaw"], sens, "heading hold -> N moment [N*m] via the fins. NEEDS FORWARD FLOW (see fin section).")
    # keep any other loops that were after yaw in the old block (cross_track etc.)
    names = [re.match(r"^  ([a-z_]+):", l).group(1) for l in block if re.match(r"^  [a-z_]+:", l)]
    for extra in names:
        if extra not in ("heave", "pitch", "surge", "yaw"):
            new += part(extra)
    if new[-1].strip():
        new.append("")
    path.write_text("\n".join(text[:s] + new + text[e:]))


def patch_dof_testing(path, gains, sens):
    text = path.read_text()
    flow = lambda g: "{" + ", ".join(f"{k}: {v:g}" for k, v in g.items()) + "}"
    for loop in ("heave", "pitch", "speed", "yaw", "roll"):
        pat = re.compile(rf"^(  {loop}:\s*)\{{[^}}]*\}}(.*)$", re.M)
        assert pat.search(text), loop
        text = pat.sub(lambda m: f"{m.group(1)}{flow(gains[loop])}{m.group(2)}", text, count=1)
    notes = ["# RE-TUNED 2026-10-10 (control_code/tuning/loop_tuner.py) against the offline vehicle with REALISTIC odometry (4.5 Hz, 0.25 s late, noise, freezes) on four plants",
             "# (nominal, heavy+late, light+fast, stress), through the pose filter (common/pose_filter.py, see `estimator:` below). Changes: heave/pitch/yaw gains much softer, an",
             "# output low-pass `lpf_tau_s` on heave/pitch/yaw (a first-order filter on the loop output: noise no longer becomes thruster/fin chatter), a setpoint filter `sp_tau_s`",
             "# on the speed loop, and the ROLL gains cut 5x (the old 0.5/0.2 oscillated: fins swung 15-40 deg and roll overshot 50-190% with realistic odometry).",
             "# MEASURED effect of changing each number by 0.5x..2x (nominal plant; overshoot / settle / chatter) is quoted in station_keeping.yaml for heave, pitch and yaw; for the rest:"]
    for loop in ("roll", "speed"):
        for par in ("kp", "kd", "ki"):
            m = measured(sens, loop, par)
            if m:
                notes.append(f"#   {loop}.{par} {gains[loop][par]:g}: " + " | ".join(m))
    notes.append("# Result: yaw, roll and speed pass on all four plants (overshoot < 10%, fin swing < 6 deg). Heave/pitch: steady pushes are rejected (see station_keeping.yaml), chatter 190-250 RPM,")
    notes.append("# a depth step overshoots ~25% (10 cm per 0.4 m). The STRESS plant (thruster lag 0.45 s, odometry 0.5 s late, 1.8x inertia) stays stable but chatters ~430 RPM.")
    marker = "# PID gains. Output units:"
    i = text.index(marker)
    text = text[:i] + "\n".join(notes) + "\n" + text[i:]
    path.write_text(text)


def main(argv):
    gains = yaml.safe_load(open(argv[0]))
    sens = json.load(open(argv[1]))
    patch_station_keeping(CTRL / "station_keeping" / "station_keeping.yaml", gains, sens)
    patch_dof_testing(CTRL / "dof_testing" / "dof_testing.yaml", gains, sens)
    for rel in ("station_keeping/station_keeping.yaml", "dof_testing/dof_testing.yaml"):
        d = yaml.safe_load(open(CTRL / rel))
        print(rel, {k: d["gains"][k]["kp"] for k in ("heave", "pitch", "yaw")}, "ok")


if __name__ == "__main__":
    main(sys.argv[1:])
