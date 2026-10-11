#!/usr/bin/env python3
"""Log the intermittent odometry freezes of the live sim (docs/project_notes.md section 2c item 3) together with what the machine was doing.

  source /opt/ros/humble/setup.bash; source ../ws/install/setup.bash; export ROS_DOMAIN_ID=42
  python3 plots/odom_freeze_monitor.py [--minutes 10] [--out outputs/odom_freeze_log.csv] [--odom /Mako_01/odometry_sim] [--cameras camera_03 camera_04 camera_05]

A freeze = the odometry keeps arriving but pose and twist are bit-identical to the previous message (the real topic did this for 5-25 s, with w = q = 0).
For every freeze it writes one CSV row: start time, duration, odometry rate before / during, 1-minute load average and the top CPU processes when it began, and the
camera message rates (cameras publish through the same bridge). Run it in different conditions and compare the freeze fraction / frequency:
  A. nothing else running (baseline);   B. + dock_detection_algo/run_live.sh (camera subscriber);   C. + a controller publishing actuator_cmd at 20 Hz;
  D. B and C;   E. sim time scale or camera count changed in the session.
If the freezes follow B (or C) the bridge / camera path is the cause; if they are the same in all conditions they come from the simulator. It only listens.
"""

from __future__ import annotations

import argparse
import sys
import csv
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from outdirs import out_dir  # noqa: E402


class FreezeDetector:
    """Feed it (time, state_tuple) for every odometry message. A run of identical tuples longer than `min_s` is a freeze."""

    def __init__(self, min_s: float = 1.0) -> None:
        self.min_s = float(min_s)
        self._last: Optional[Tuple] = None
        self._since: Optional[float] = None
        self._count = 0
        self.episodes: List[dict] = []
        self._open: Optional[dict] = None

    def push(self, t: float, state: Tuple) -> Optional[dict]:
        """-> the finished episode dict when a freeze ends, else None."""
        done = None
        if state == self._last:
            self._count += 1
            if self._since is None:
                self._since = self._t_last
            if self._open is None and t - self._since >= self.min_s:
                self._open = {"start": self._since, "msgs": self._count}
            if self._open is not None:
                self._open["msgs"] = self._count
        else:
            if self._open is not None:
                self._open["end"] = self._t_last
                self._open["duration"] = self._t_last - self._open["start"]
                self.episodes.append(self._open)
                done = self._open
                self._open = None
            self._since = None
            self._count = 0
        self._last = state
        self._t_last = t
        return done

    @property
    def frozen(self) -> bool:
        return self._open is not None

    def fraction_frozen(self, t_total: float) -> float:
        return sum(e["duration"] for e in self.episodes) / max(t_total, 1e-9)


def _top_cpu(n: int = 3) -> str:
    try:
        out = subprocess.run(["ps", "-eo", "pcpu,comm", "--sort=-pcpu"], capture_output=True, text=True, timeout=2).stdout.splitlines()[1:n + 1]
        return "; ".join(" ".join(l.split()) for l in out)
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description="Log odometry freezes on the live sim")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=out_dir("odom_freeze_log.csv"))
    ap.add_argument("--odom", default="/Mako_01/odometry_sim")
    ap.add_argument("--cameras", nargs="*", default=["camera_03", "camera_04", "camera_05"])
    ap.add_argument("--vessel", default="Mako_01")
    ap.add_argument("--min-freeze-s", type=float, default=1.0)
    ap.add_argument("--label", default="", help="condition label written in every row, e.g. 'B detector on'")
    args = ap.parse_args()

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = Node("odom_freeze_monitor")
    det = FreezeDetector(args.min_freeze_s)
    stamps: Deque[float] = deque(maxlen=200)
    cam_counts = {c: deque(maxlen=400) for c in args.cameras}

    def rate(dq, now, window=5.0):
        n = sum(1 for x in dq if now - x <= window)
        return n / window

    new = args.out.exists() is False
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["label", "wall_start", "duration_s", "msgs_frozen", "odom_hz_before", "load1", "top_cpu_at_start"] + [f"{c}_hz" for c in args.cameras])
    pending = {}

    def on_odom(m):
        now = time.monotonic()
        p, q, lv, av = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear, m.twist.twist.angular
        state = (p.x, p.y, p.z, q.x, q.y, q.z, q.w, lv.x, lv.y, lv.z, av.x, av.y, av.z)
        was = det.frozen
        stamps.append(now)
        ep = det.push(now, state)
        if det.frozen and not was:
            pending.update(wall=time.strftime("%H:%M:%S"), hz=rate(stamps, now - 1.0), load=os.getloadavg()[0], top=_top_cpu(),
                           cams=[rate(cam_counts[c], now) for c in args.cameras])
            node.get_logger().warning("odometry FROZEN")
        if ep is not None:
            w.writerow([args.label, pending.get("wall", ""), f"{ep['duration']:.1f}", ep["msgs"], f"{pending.get('hz', 0):.1f}",
                        f"{pending.get('load', 0):.2f}", pending.get("top", "")] + [f"{x:.1f}" for x in pending.get("cams", [])])
            f.flush()
            node.get_logger().info(f"freeze ended after {ep['duration']:.1f} s")

    node.create_subscription(Odometry, args.odom, on_odom, 10)
    for c in args.cameras:
        node.create_subscription(CompressedImage, f"/{args.vessel}/{c}/image/compressed", lambda m, c=c: cam_counts[c].append(time.monotonic()), 1)
    t0 = time.monotonic()
    print(f"Watching {args.odom} for {args.minutes:.0f} min -> {args.out}", flush=True)
    try:
        while rclpy.ok() and time.monotonic() - t0 < args.minutes * 60:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    total = time.monotonic() - t0
    print(f"{len(det.episodes)} freezes, {100 * det.fraction_frozen(total):.1f}% of {total / 60:.1f} min, "
          f"mean {sum(e['duration'] for e in det.episodes) / max(len(det.episodes), 1):.1f} s")
    f.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
