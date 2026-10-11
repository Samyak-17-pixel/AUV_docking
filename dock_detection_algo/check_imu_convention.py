#!/usr/bin/env python3
"""Compare the IMU attitude with the odometry attitude on the live sim and say whether the roll / pitch signs agree.

  source /opt/ros/humble/setup.bash; source ../control_code/ws/install/setup.bash; export ROS_DOMAIN_ID=42
  python3 check_imu_convention.py [--seconds 20] [--imu /Mako_01/imu_01/data] [--odom /Mako_01/odometry_sim]

The dock elevation (dock_align_msg.elevation_down_rad) uses the IMU pitch as if it were NED / nose-up positive, like the odometry. If the IMU is in
another convention (ENU / FLU) the sign of pitch (and roll) is flipped. To see it the vehicle has to be TILTED: pitch it by a few degrees (a
dof_testing pitch step, or push it in the viewer / sim UI) while this runs. With the vehicle level every angle is ~0 and the verdict is UNDECIDED.
It never changes anything; it prints the line to put in dock_detection.yaml (camera.imu_pitch_sign / imu_roll_sign).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dock_align_msg import quat_to_roll_pitch  # noqa: E402


def verdict(pairs, min_amp_deg: float = 1.5):
    """pairs: [(imu_angle_rad, odom_angle_rad)] -> ('SAME' | 'FLIPPED' | 'UNDECIDED', corr, odom_amplitude_deg).

    Decided by the sign of the sum of products (the correlation without the mean removal: the sim may start tilted) and only when the odometry
    angle moved by at least min_amp_deg (otherwise there is nothing to compare).
    """
    if len(pairs) < 5:
        return "UNDECIDED", 0.0, 0.0
    odom = [b for _, b in pairs]
    amp = math.degrees(max(odom) - min(odom)) / 2.0
    ma, mb = sum(a for a, _ in pairs) / len(pairs), sum(odom) / len(pairs)
    sxy = sum((a - ma) * (b - mb) for a, b in pairs)
    sxx = sum((a - ma) ** 2 for a, _ in pairs)
    syy = sum((b - mb) ** 2 for b in odom)
    corr = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    if amp < min_amp_deg or abs(corr) < 0.7:
        return "UNDECIDED", corr, amp
    return ("SAME" if corr > 0 else "FLIPPED"), corr, amp


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the IMU roll/pitch sign convention against odometry")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--imu", default="/Mako_01/imu_01/data")
    ap.add_argument("--odom", default="/Mako_01/odometry_sim")
    args = ap.parse_args()

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Imu

    rclpy.init()
    node = Node("check_imu_convention")
    last = {"odom": None, "imu": None}
    roll_pairs, pitch_pairs = [], []

    def on_odom(m):
        q = m.pose.pose.orientation
        last["odom"] = quat_to_roll_pitch(q.x, q.y, q.z, q.w)

    def on_imu(m):
        q = m.orientation
        last["imu"] = quat_to_roll_pitch(q.x, q.y, q.z, q.w)
        if last["odom"] is not None:
            roll_pairs.append((last["imu"][0], last["odom"][0]))
            pitch_pairs.append((last["imu"][1], last["odom"][1]))

    node.create_subscription(Odometry, args.odom, on_odom, 10)
    node.create_subscription(Imu, args.imu, on_imu, 10)
    t_end = time.monotonic() + args.seconds
    print(f"Listening for {args.seconds:.0f} s on {args.imu} and {args.odom}. Tilt the vehicle by a few degrees meanwhile.", flush=True)
    while rclpy.ok() and time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if not pitch_pairs:
        print("No IMU/odometry samples received: check the topic names and ROS_DOMAIN_ID.")
        sys.exit(2)
    for name, pairs in (("pitch", pitch_pairs), ("roll", roll_pairs)):
        v, corr, amp = verdict(pairs)
        off = math.degrees(sum(a - b for a, b in pairs) / len(pairs)) if v == "SAME" else float("nan")
        print(f"{name:5s}: {v:9s} corr {corr:+.2f}, odometry swing +-{amp:.1f} deg" + (f", mean offset IMU-odom {off:+.2f} deg" if v == "SAME" else ""))
        if v == "FLIPPED":
            print(f"   -> set camera.imu_{name}_sign: -1 in dock_detection.yaml")
        elif v == "UNDECIDED":
            print(f"   -> not enough {name} motion (or the signals do not correlate): tilt the vehicle more and run again")


if __name__ == "__main__":
    main()
