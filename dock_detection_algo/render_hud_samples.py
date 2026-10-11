#!/usr/bin/env python3
"""Render the detector pop-ups for a few synthetic scenes to PNG files (no display, no ROS topics; needs the workspace sourced for the DockAlign message).

  python3 render_hud_samples.py [OUT_DIR]       default: outputs/detection_reports/hud_samples
Writes camera_<n>.png (the 'Dock camera' window), mask.png, align_<n>.png (the 'Dock align' window) for: a good lock, an oblique view, a close view and a hidden dock.
This is how the pop-up layout is checked on a machine without a screen."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import dock_hud  # noqa: E402
import reliability_study as rs  # noqa: E402
from dock_detector import DockDetector  # noqa: E402

SCENES = [("good", dict(range=7.0, lat=1.0, head=-6.0)), ("oblique", dict(range=5.0, view_deg=40.0, lat=0.0)), ("close", dict(range=3.0, lat=0.5, head=3.0)),
          ("hidden", dict(range=6.0, lat=0.0, head=60.0))]


def render(out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    cfg0 = rs.load_cam_cfg()
    det = DockDetector()
    rng = np.random.default_rng(2)
    written = []
    for i, (name, cs) in enumerate(SCENES):
        hist = dock_hud.History(12.0)
        c = rs.base_case(rng)
        c.update(cs)
        c["roll"] = 3.0
        fr = rs.render_case(c, cfg0)
        bgr = cv2.imdecode(np.frombuffer(fr["jpeg"], np.uint8), 1)
        h, w = bgr.shape[:2]
        for k in range(40):
            res = det.process(bgr, fr["roll"], fr["pitch"], i * 10 + k * 0.1)
            info = dock_hud.info_from_result(res, 60.0, i * 10 + k * 0.1)
            hist.add(info)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(fr["roll"]), 1.0)
        level = cv2.warpAffine(bgr, M, (w, h))
        for fn, img in ((f"camera_{name}.png", dock_hud.draw_camera_view(level, info, hist)), (f"mask_{name}.png", dock_hud.draw_mask_view(cv2.warpAffine(res.mask, M, (w, h)), info)),
                        (f"align_{name}.png", dock_hud.draw_align_view(info, hist, 0.05))):
            cv2.imwrite(str(out / fn), img)
            written.append(out / fn)
    return written


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE.parent / "outputs" / "detection_reports" / "hud_samples"
    for p in render(d):
        print(p)
