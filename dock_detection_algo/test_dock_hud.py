"""The pop-up drawings (dock_hud.py): sizes, no crash on any input, and a few pixel checks. Needs the workspace for the DockAlign message."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("interfaces.msg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dock_hud as H  # noqa: E402
from interfaces.msg import DockAlign  # noqa: E402


def msg(valid=True, aligned=True, ex=0.0, ey=0.0, radius=60.0, num=4, conf=0.9):
    m = DockAlign()
    m.valid, m.aligned, m.num_lights, m.confidence = bool(valid), bool(aligned), int(num), float(conf)
    m.image_width, m.image_height = 640, 480
    m.error_x_px, m.error_y_px, m.error_x_norm, m.error_y_norm = float(ex), float(ey), float(ex / 320.0), float(ey / 240.0)
    m.radius_px = float(radius)
    cx, cy = 320 + ex, 240 + ey
    for n, (dx, dy) in {"top": (0, -radius), "bottom": (0, radius), "left": (-0.7071 * radius, -0.7071 * radius), "right": (0.7071 * radius, -0.7071 * radius)}.items():
        p = getattr(m, n)
        p.x, p.y = float(cx + dx), float(cy + dy)
    m.center.x, m.center.y = float(cx), float(cy)
    m.elevation_valid, m.elevation_rad = True, math.radians(2.0)
    m.status = "ok" if aligned else "ok_not_aligned"
    return m


def test_range_bearing_and_view_angle_from_the_message():
    info = H.info_from_msg(msg(ex=40.0, radius=83.14))
    assert abs(H.range_m(info) - 415.69 / 83.14) < 0.01
    assert abs(H.bearing_deg(info) - math.degrees(math.atan(40.0 / 415.69))) < 0.01
    assert H.view_angle_deg(info) < 5.0                                  # square-on ring: side lights at +-0.7 R -> cos ~ 1
    assert H.range_m(H.info_from_msg(msg(valid=False, num=0))) is None


def test_windows_have_the_requested_size_and_do_not_crash_on_any_state():
    hist = H.History(12.0)
    states = [msg(), msg(aligned=False, ex=-120, ey=70), msg(valid=False, num=2, conf=0.1), msg(valid=False, num=0, conf=0.0)]
    for i, m in enumerate(states):
        info = H.info_from_msg(m, 60.0, float(i))
        hist.add(info)
        frame = np.full((480, 640, 3), 20, np.uint8)
        assert H.draw_camera_view(frame, info, hist).shape == (736, 1120, 3)
        assert H.draw_camera_view(None, info, None).shape == (736, 1120, 3)
        assert H.draw_mask_view(np.zeros((480, 640), np.uint8), info).shape == (600, 960, 3)
        assert H.draw_mask_view(None, info).shape == (600, 960, 3)
        assert H.draw_align_view(info, hist, 0.05).shape == (680, 1000, 3)
        assert H.draw_align_view(info, None, None).shape == (680, 1000, 3)


def test_banner_colour_tells_locked_from_lost():
    def banner_pixel(m):
        img = H.draw_align_view(H.info_from_msg(m, 60.0, 0.0), None, 0.0)
        return img[30, 36]                                               # the status dot in the banner (BGR)
    assert tuple(banner_pixel(msg())) == H.GOOD
    assert tuple(banner_pixel(msg(aligned=False))) == H.WARN
    assert tuple(banner_pixel(msg(valid=False, num=0, conf=0))) == H.BAD
    assert tuple(banner_pixel(msg(valid=False, num=2, conf=0.1))) == H.INFO


def test_direction_lamps_light_up_on_the_right_side():
    def lamp_colours(m):
        img = H.draw_align_view(H.info_from_msg(m, 60.0, 0.0), None, 0.0)
        return {"left": tuple(img[262, 22]), "right": tuple(img[262, 330]), "up": tuple(img[74, 112]), "down": tuple(img[62 + 400 - 40 - 12, 112])}    # lamp corners, clear of the text
    c = lamp_colours(msg(ex=80.0, ey=-60.0, aligned=False))              # dock right of centre and above it
    assert c["right"] == H.WARN and c["up"] == H.WARN and c["left"] != H.WARN and c["down"] != H.WARN
    c = lamp_colours(msg(ex=-80.0, ey=60.0, aligned=False))
    assert c["left"] == H.WARN and c["down"] == H.WARN and c["right"] != H.WARN and c["up"] != H.WARN


def test_history_keeps_only_the_window():
    h = H.History(5.0)
    for t in range(0, 20):
        h.add(H.info_from_msg(msg(ex=float(t)), 60.0, float(t)))
    t, v = h.series("err_x_px")
    assert t.min() >= -5.0 - 1e-9 and v[-1] == 19.0 and len(v) == 6
