#!/usr/bin/env python3
"""Record a run from the ROS topics to a CSV (for replay in the viewer, the real-vs-model overlay and calibration).

  ROS_DOMAIN_ID=42 ./run_record.sh --label heave_step            # the REAL mavsim: read-only, it only subscribes
  ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ./run_record.sh          # the offline fake vehicle

Records odometry (one row per sample) with the last actuator commands and DockAlign values. Stop with Ctrl-C, or use --duration.
--frames N --frames-every S saves up to N camera_03 JPEGs (with the pose at that moment) next to the CSV: compare them with the synthetic camera to tune the
'look' section of sim_viewer.yaml. Default file: outputs/sim_viewer_runs/run_<time>.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HERE)] + [str(_d) for _d in sorted(_HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(_HERE.parent / "common"))
from outdirs import out_dir  # noqa: E402

from recording import CsvRecorder  # noqa: E402
from state import State  # noqa: E402


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Record a run to CSV")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--vessel", default="Mako_01")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    ap.add_argument("--label", default="")
    ap.add_argument("--frames", type=int, default=0, help="save up to this many camera_03 JPEGs")
    ap.add_argument("--frames-every", type=float, default=5.0)
    args = ap.parse_args(argv)
    out = args.out or out_dir("sim_viewer_runs") / time.strftime("run_%Y%m%d_%H%M%S.csv")

    import rclpy
    from interfaces.msg import Actuator, DockAlign
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    rec = CsvRecorder(out, {"vessel": args.vessel, "label": args.label, "ros_domain": os.environ.get("ROS_DOMAIN_ID", "0"),
                            "note": "one row per odometry sample; u..r are body velocities; z is depth (+down)"})
    frames_dir = out.with_suffix("") .parent / (out.stem + "_frames")
    frames_log = None
    if args.frames > 0:
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames_log = open(frames_dir / "frames.csv", "w", newline="")
        fw = csv.writer(frames_log)
        fw.writerow(["file", "t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"])

    class Rec(Node):
        def __init__(self) -> None:
            super().__init__("record_run")
            self.t0 = time.monotonic()
            self.last_state: Optional[State] = None
            self.frames_saved = 0
            self._next_frame = 0.0
            self.create_subscription(Odometry, f"/{args.vessel}/odometry_sim", self.on_odom, 20)
            self.create_subscription(Actuator, f"/{args.vessel}/actuator_cmd", self.on_cmd, 20)
            self.create_subscription(DockAlign, f"/{args.vessel}/dock_align", self.on_align, 10)
            if args.frames > 0:
                self.create_subscription(CompressedImage, f"/{args.vessel}/camera_03/image/compressed", self.on_image, 2)

        def now(self) -> float:
            return time.monotonic() - self.t0

        def on_odom(self, m: Odometry) -> None:
            s = State.from_odom(m)
            self.last_state = s
            rec.add_state(self.now(), s.pos, s.eul, s.nu)

        def on_cmd(self, m: Actuator) -> None:
            rec.set_cmd(m.actuator_names, m.actuator_values, self.now())

        def on_align(self, m: DockAlign) -> None:
            rec.set_align(m.valid, m.num_lights, m.error_x_px, m.error_y_px, m.radius_px, math.degrees(m.elevation_rad) if m.elevation_valid else 0.0)

        def on_image(self, m: CompressedImage) -> None:
            if self.frames_saved >= args.frames or self.last_state is None or self.now() < self._next_frame:
                return
            name = f"frame_{self.frames_saved:03d}.jpg"
            (frames_dir / name).write_bytes(bytes(m.data))
            s = self.last_state
            fw.writerow([name, f"{self.now():.3f}", *[f"{v:.4f}" for v in s.pos], *[f"{math.degrees(a):.3f}" for a in s.eul]])
            frames_log.flush()
            self.frames_saved += 1
            self._next_frame = self.now() + args.frames_every

    rclpy.init()
    node = Rec()
    print(f"recording to {out}" + (f" (+ up to {args.frames} frames in {frames_dir})" if args.frames else "") + "  - Ctrl-C to stop", flush=True)
    try:
        while rclpy.ok() and (args.duration <= 0 or node.now() < args.duration):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        path = rec.close()
        if frames_log:
            frames_log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    rate = rec.rows / rec.duration if rec.duration > 0 else 0.0
    print(f"saved {rec.rows} rows, {rec.duration:.1f} s ({rate:.1f} Hz) to {path}", flush=True)
    return 0 if rec.rows else 1


if __name__ == "__main__":
    sys.exit(main())
