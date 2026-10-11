#!/usr/bin/env python3
"""Step 0-2 of the first-real-sim checklist (docs/project_notes.md section 10): is the sim up, is the odometry alive and not frozen, who else publishes actuator_cmd.

  ROS_DOMAIN_ID=42 python3 real_sim_preflight.py [--seconds 8] [--vessel Mako_01]

Exit code 0 = ready, 1 = something is wrong (the reason is printed), 2 = the sim is not there. It only listens; it never commands anything.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Tuple


def assess(samples: List[Tuple[float, tuple]], seconds: float, min_hz: float = 2.0):
    """samples: [(t, (x, y, z, qx, qy, qz, qw, u, v, w, p, q, r))]. -> (ok, [messages])."""
    msgs: List[str] = []
    if not samples:
        return False, ["no odometry messages: the sim is not running/playing, the bridge is down, or the ROS_DOMAIN_ID is wrong"]
    # rate from the first to the last message: a freshly started node needs ~1-2 s to discover the publisher, which must not count as a slow topic
    span = samples[-1][0] - samples[0][0]
    hz = (len(samples) - 1) / span if len(samples) > 1 and span > 0 else 0.0
    ok = True
    msgs.append(f"odometry: {len(samples)} messages in {seconds:.0f} s = {hz:.1f} Hz")
    if hz < min_hz:
        ok = False
        msgs.append(f"  rate below {min_hz} Hz: control loops will be starved (the live sim gave 4.4 Hz)")
    distinct = len({s for _, s in samples})
    if distinct <= 1:
        ok = False
        msgs.append("  every message is IDENTICAL: the odometry is FROZEN (known intermittent fault, docs/project_notes.md 2c item 3). Wait, unpause the sim, or restart it, then retry")
    elif distinct < 0.5 * len(samples):
        msgs.append(f"  only {distinct} distinct states in {len(samples)} messages: partly frozen; a longer test may come out INVALID")
    z = [s[2] for _, s in samples]
    msgs.append(f"  depth z {min(z):.2f}..{max(z):.2f} m")
    if min(z) < 1.0:
        ok = False
        msgs.append("  vehicle is within 1 m of the surface: move it to a safe depth first (tests expect about 3 m)")
    return ok, msgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight for the first real-sim run")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--vessel", default="Mako_01")
    args = ap.parse_args()

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node

    rclpy.init()
    node = Node("real_sim_preflight")
    samples: List[Tuple[float, tuple]] = []

    def on_odom(m):
        p, q, lv, av = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear, m.twist.twist.angular
        samples.append((time.monotonic(), (p.x, p.y, p.z, q.x, q.y, q.z, q.w, lv.x, lv.y, lv.z, av.x, av.y, av.z)))

    node.create_subscription(Odometry, f"/{args.vessel}/odometry_sim", on_odom, 10)
    t_end = time.monotonic() + args.seconds
    while rclpy.ok() and time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)

    pubs = node.get_publishers_info_by_topic(f"/{args.vessel}/actuator_cmd")
    names = sorted({f"{p.node_namespace.rstrip('/')}/{p.node_name}" for p in pubs})
    ok, msgs = assess(samples, args.seconds)
    for m in msgs:
        print(m)
    print(f"actuator_cmd publishers: {names or 'none'}")
    if any("teleop" in n for n in names):
        ok = False
        print("  mavsim_teleop is publishing zero commands on actuator_cmd (20 Hz): it would interleave with any controller. Stop it first "
              "(docker exec <bridge-container> kill -INT <teleop pid>; see docs/project_notes.md 2c item 5).")
    if samples and math.isfinite(samples[-1][1][2]):
        q = samples[-1][1][3:7]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (q[3] * q[1] - q[2] * q[0])))))
        roll = math.degrees(math.atan2(2.0 * (q[3] * q[0] + q[1] * q[2]), 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)))
        print(f"  current pitch {pitch:+.1f} deg, roll {roll:+.1f} deg" + ("   (large: the vehicle is tilted; reset it before testing)" if max(abs(pitch), abs(roll)) > 20 else ""))
        if max(abs(pitch), abs(roll)) > 20:
            ok = False
    node.destroy_node()
    rclpy.shutdown()
    print("READY" if ok else "NOT READY")
    sys.exit(0 if ok else (2 if not samples else 1))


if __name__ == "__main__":
    main()
