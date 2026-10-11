"""The ring check (dock_ring.py): pure numpy, no ROS."""
import math

import numpy as np

from dock_ring import fit_ring, label_four, select_ring

S = math.sqrt(0.5)


def ring(cx=320.0, cy=240.0, R=80.0, k=1.0, shear=0.0):
    """Four lights of a perfect ring seen with side-to-side scale k: top, bottom, right, left."""
    return [(cx, cy - R), (cx, cy + R), (cx + k * R * S + shear, cy - R * S), (cx - k * R * S + shear, cy - R * S)]


def test_perfect_ring_has_zero_residual_at_any_squeeze():
    for k in (1.0, 0.7, 0.3):
        f = fit_ring(*ring(k=k))
        assert f.ok and f.residual < 1e-9 and abs(f.squeeze_x - k) < 1e-6


def test_labels_follow_image_position():
    top, bottom, right, left = label_four([(100, 300), (100, 100), (150, 160), (50, 160)])
    assert top == (100, 100) and bottom == (100, 300) and right == (150, 160) and left == (50, 160)


def test_ring_is_picked_out_of_false_lights():
    true = ring()
    rng = np.random.default_rng(3)
    for _ in range(40):
        false = [(float(rng.uniform(20, 620)), float(rng.uniform(20, 460))) for _ in range(4)]
        cands = false + true                                           # the false lights are 'stronger' (listed first)
        got, fit = select_ring(cands, [1.0 - 0.05 * i for i in range(len(cands))], fit_tol=0.08)
        assert fit is not None and sorted(got) == sorted(true), (false, got)


def test_three_real_lights_plus_one_false_is_not_accepted():
    true = ring()
    imposter = true[:3] + [(500.0, 400.0)]                           # the left light is hidden, a bubble sits elsewhere
    got, fit = select_ring(imposter, fit_tol=0.08)
    assert fit is None and len(got) == 3


def test_noise_within_a_pixel_is_accepted_but_a_misplaced_side_light_is_not():
    rng = np.random.default_rng(1)
    noisy = [(x + rng.normal(0, 0.7), y + rng.normal(0, 0.7)) for x, y in ring(R=60.0, k=0.8)]
    assert fit_ring(*label_four(noisy)).residual < 0.04
    bad = ring(R=60.0)
    bad[2] = (bad[2][0] + 18.0, bad[2][1] + 12.0)
    assert select_ring(bad, fit_tol=0.08)[1] is None


def test_sideways_slid_side_lights_and_absurd_squeeze_are_rejected():
    assert fit_ring(*ring(shear=60.0)).residual > 0.2                    # both side lights slid 60 px: no affine view of a circle looks like that
    assert not fit_ring(*ring(k=1.8)).ok
