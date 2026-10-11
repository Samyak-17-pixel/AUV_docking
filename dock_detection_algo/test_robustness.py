"""Detector robustness on rendered frames with the image degradations of synthetic_camera.py (needs the workspace sourced for DockAlign).
Each test has a negative control: the OLD behaviour (ring check / background subtraction / plateau centres off) fails the same frames."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("interfaces.msg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reliability_study as rs  # noqa: E402
from dock_detection_config import load_config  # noqa: E402
from dock_detector import DockDetector  # noqa: E402

CFG0 = rs.load_cam_cfg()


def _run(det: DockDetector, n=20, seed=5, **kw):
    """-> dict(good, bad, miss, err) over n frames of random modest poses with the case changes in kw."""
    import cv2
    rng = np.random.default_rng(seed)
    kinds = []
    errs = []
    for _ in range(n):
        c = rs.base_case(rng)
        c["range"] = float(rng.uniform(3.5, 7.0))
        c["lat"] = float(rng.uniform(-1.0, 1.0))
        for k, v in kw.items():
            if k == "stress":
                c["stress"].update(v)
            elif k == "look_scale":
                c["look_scale"] = v
            else:
                c[k] = v
        fr = rs.render_case(c, CFG0)
        if not fr["in_view"]:
            continue
        m = det.process(cv2.imdecode(np.frombuffer(fr["jpeg"], np.uint8), 1), fr["roll"], fr["pitch"], 0.0).msg
        if not m.valid:
            kinds.append("miss")
            continue
        pix = np.array([[m.top.x, m.top.y], [m.bottom.x, m.bottom.y], [m.right.x, m.right.y], [m.left.x, m.left.y]])
        e = np.linalg.norm(pix - fr["truth"], axis=1)
        kinds.append("good" if (m.num_lights == 4 and e.max() <= rs.TOL_PX) else "bad")
        errs.append(float(e.mean()))
    k = np.array(kinds)
    return {"good": float(np.mean(k == "good")), "bad": float(np.mean(k == "bad")), "miss": float(np.mean(k == "miss")), "err": float(np.mean(errs)) if errs else float("nan"), "n": len(k)}


def _old(**off):
    cfg = copy.deepcopy(load_config())
    cfg["ring"]["enabled"] = off.get("ring", True) and cfg["ring"]["enabled"]
    if off.get("bg") is False:
        cfg["mask"]["bg_subtract"] = False
    if off.get("plateau") is False:
        cfg["peaks"]["plateau_min_px"] = 0
    return DockDetector(cfg=cfg)


def test_false_bright_lights_do_not_steal_a_place():
    new = _run(DockDetector(), stress={"distractors": 2})
    old = _run(_old(ring=False), stress={"distractors": 2})
    assert new["good"] >= 0.9 and new["bad"] <= 0.1, new
    assert old["bad"] >= 0.5, old                                       # negative control: the old 'four strongest peaks' rule is fooled


def test_surface_glint_and_reflections():
    r = _run(DockDetector(), stress={"glint": 0.8, "reflections": 2})
    assert r["good"] >= 0.9 and r["bad"] <= 0.1, r


def test_a_hidden_light_is_a_miss_not_a_false_lock():
    for light in ("top", "bottom", "left", "right"):
        r = _run(DockDetector(), n=10, stress={"occlude": light})
        assert r["bad"] == 0.0 and r["good"] == 0.0, (light, r)


def test_murky_water_and_a_bright_veil():
    murky = _run(DockDetector(), look_scale={"background_bgr": 3.5})
    assert murky["good"] >= 0.9 and murky["bad"] <= 0.1, murky
    old = _run(_old(bg=False, plateau=False), look_scale={"background_bgr": 3.5})
    assert old["good"] <= 0.5, old                                      # negative control: without the new steps most frames are lost or wrong
    veil = _run(DockDetector(), stress={"backscatter": 0.5})
    assert veil["good"] >= 0.9, veil


def test_blur_noise_jpeg_and_bubbles_keep_working():
    assert _run(DockDetector(), stress={"blur_sigma_px": 3.0})["good"] >= 0.9
    assert _run(DockDetector(), jpeg_q=25)["good"] >= 0.9
    assert _run(DockDetector(), stress={"bubbles": 15, "snow": 150})["good"] >= 0.9


def test_saturated_lights_are_centred_on_their_plateau():
    new = _run(DockDetector(), n=20)
    old = _run(_old(plateau=False), n=20)
    assert new["good"] == 1.0 and new["err"] < 0.3, new
    assert old["err"] > new["err"] * 1.5, (old, new)                     # negative control: response-weighted centres are about 3x less precise
