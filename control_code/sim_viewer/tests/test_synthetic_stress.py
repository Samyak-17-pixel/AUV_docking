"""The optional image degradations of the synthetic camera (sim_viewer.yaml `stress:`): off by default, deterministic, and each one does what it says."""
import copy

import numpy as np
import pytest
import yaml

from synthetic_camera import DEFAULT_CONFIG, SyntheticCamera

POS, EUL = [5.0, 0.0, 3.0], [0.0, 0.0, 0.0]


def _sharp(im):
    return float(np.abs(np.diff(im[:, :, 1], axis=1)).max())             # steepest horizontal edge (the lights' cores)


def cam(**stress):
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    cfg["stress"] = dict(cfg.get("stress") or {}, **stress)
    return SyntheticCamera(cfg)


def frame(**stress):
    return cam(**stress).render(POS, EUL).astype(int)


def test_everything_is_off_by_default_and_the_picture_is_unchanged():
    cfg = yaml.safe_load(open(DEFAULT_CONFIG))
    no_stress = copy.deepcopy(cfg)
    no_stress.pop("stress", None)
    a = SyntheticCamera(cfg).render(POS, EUL)
    b = SyntheticCamera(no_stress).render(POS, EUL)
    assert np.array_equal(a, b)                                       # the shipped `stress:` block changes nothing


def test_same_seed_same_picture():
    assert np.array_equal(frame(bubbles=10, snow=50, seed=3), frame(bubbles=10, snow=50, seed=3))
    assert not np.array_equal(frame(bubbles=10, seed=3), frame(bubbles=10, seed=4))


@pytest.mark.parametrize("name,kw,check", [
    ("blur", dict(blur_sigma_px=3.0), lambda base, im: _sharp(im) < 0.6 * _sharp(base)),             # blurred lights have softer edges
    ("motion", dict(motion_blur_px=15), lambda base, im: (im[:, :, 1] > 150).sum() != (base[:, :, 1] > 150).sum()),
    ("backscatter", dict(backscatter=0.5), lambda base, im: im[:40].mean() > base[:40].mean() + 20 and abs(im[-40:].mean() - base[-40:].mean()) < 5),   # bright top, untouched bottom
    ("bubbles", dict(bubbles=20), lambda base, im: (np.abs(im - base).sum(axis=2) > 40).sum() > 100),
    ("snow", dict(snow=100), lambda base, im: (np.abs(im - base).sum(axis=2) > 40).sum() > 50),
    ("distractors", dict(distractors=3), lambda base, im: (im[:, :, 1] > 200).sum() > (base[:, :, 1] > 200).sum()),
    ("glint", dict(glint=1.0), lambda base, im: im[:70].mean() > base[:70].mean() + 5),
    ("reflections", dict(reflections=2), lambda base, im: (np.abs(im - base).sum(axis=2) > 40).sum() > 20),
])
def test_each_degradation_changes_the_picture_as_described(name, kw, check):
    base = frame()
    assert check(base, frame(**kw)), name


def test_occlusion_hides_exactly_that_light():
    c = cam(occlude="left")
    t = c.truth(POS, EUL)
    im = c.render(POS, EUL).astype(int)
    base = frame()
    def lit(img, d):
        u, v = int(d["u"]), int(d["v"])
        return img[v - 3:v + 4, u - 3:u + 4, 1].max()
    assert lit(base, t["left"]) > 150 and lit(im, t["left"]) < 120
    assert lit(im, t["right"]) > 150 and lit(im, t["top"]) > 150
