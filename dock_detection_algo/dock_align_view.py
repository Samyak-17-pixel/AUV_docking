#!/usr/bin/env python3
"""A separate window that shows the /<vessel>/dock_align topic as steering hints: which way to yaw / pitch, how far off, and every number the detector publishes.

  python3 dock_align_view.py                          # subscribes to /Mako_01/dock_align (ROS_DOMAIN_ID as usual)
  python3 dock_align_view.py --topic /Mako_01/dock_align --save-png out.png   # write one picture after 3 s and exit (for a screenshot / headless check)
  ./run_live.sh --align-window                        # the same window inside the live detector

Needs the workspace sourced (interfaces/DockAlign) and a display (OpenCV window). Does NOT run the detector: start run_live.sh (or the viewer's offline stack) first.
Keys: q / Esc quit.  Drawing code: dock_hud.draw_align_view (testable without ROS).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import rclpy
from rclpy.node import Node

import dock_hud
from interfaces.msg import DockAlign

WIN = "Dock align"


class AlignView(Node):
    def __init__(self, topic: str, vfov_deg: float) -> None:
        super().__init__("dock_align_view")
        self.vfov = vfov_deg
        self.hist = dock_hud.History(12.0)
        self.info = None
        self.t_msg = 0.0
        self.count = 0
        self.create_subscription(DockAlign, topic, self._on_msg, 10)

    def _on_msg(self, m) -> None:
        now = time.time()
        self.info = dock_hud.info_from_msg(m, self.vfov, now)
        self.hist.add(self.info)
        self.t_msg = now
        self.count += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topic", default="/Mako_01/dock_align")
    ap.add_argument("--vfov", type=float, default=60.0)
    ap.add_argument("--save-png", type=Path, default=None, help="write one picture after 3 s and exit")
    a = ap.parse_args(argv)
    rclpy.init()
    node = AlignView(a.topic, a.vfov)
    waiting = dict(dock_hud.info_from_msg(DockAlign(), a.vfov, 0.0), status=f"waiting for {a.topic}"[:34])
    if a.save_png is None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 900, 612)
    t0 = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            info = node.info if node.info is not None else waiting
            age = (time.time() - node.t_msg) if node.count else None
            if age is not None and age > 1.5:                                  # the detector stopped: say so instead of showing stale numbers as live
                info = dict(info, valid=False, status=f"no message for {age:.0f} s")
            img = dock_hud.draw_align_view(info, node.hist, age)
            if a.save_png is not None:
                if time.time() - t0 > 3.0:
                    cv2.imwrite(str(a.save_png), img)
                    print(f"wrote {a.save_png} ({node.count} messages)")
                    return 0 if node.count else 2
                continue
            cv2.imshow(WIN, img)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
