"""The detector tuning tool: dataset generation, scoring, and the shipped (tuned) thresholds beating the old ones. Needs ROS sourced (DockAlign)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_detection as td  # noqa: E402

OLD = dict(v_thresh=180, close_k=7, open_k=3, min_area=20.0, peak_sep=28, core_pct=92, tight_floor=200.0, response_min=0.05, peak_abs_v_floor=160.0, peak_abs_v_frac=0.7,
           dog_sigma_small=1.2, dog_sigma_large=4.0, tight_erode_k=3, search_dilate_k=9)


@pytest.fixture(scope="module")
def data():
    td.MURKY_SHARE = 0.3
    return td.make_dataset(120, seed=21)


def test_dataset_is_reproducible_and_every_frame_has_the_whole_ring_inside_the_image(data):
    again = td.make_dataset(20, seed=21)
    assert all(a["jpeg"] == b["jpeg"] for a, b in zip(data[:20], again))
    for d in data:
        assert d["truth"].shape == (4, 2) and (d["truth"][:, 0] > -80).all() and (d["truth"][:, 0] < 720).all()      # (roll-levelled coordinates) and 2.4 < d["range"] < 10.6
    r = np.array([d["range"] for d in data])
    assert r.min() < 4 and r.max() > 8                                        # covers the range band


def test_scoring_adds_up_and_the_shipped_thresholds_beat_the_old_ones(data):
    new = td.evaluate(data, workers=4)                                         # the shipped dock_detection.yaml
    old = td.evaluate(data, OLD, workers=4)
    for r in (new, old):
        assert r["good"] + r["bad"] + r["miss"] == pytest.approx(1.0)
    assert new["good"] > old["good"] + 0.15 and new["bad"] < old["bad"]       # the 2026-10-10 tuning, on frames it did not see
    assert new["good"] > 0.75 and new["bad"] < 0.12


def test_cost_punishes_a_bad_lock_more_than_a_miss():
    miss = {"good": 0.5, "bad": 0.0, "miss": 0.5, "err_px": 0.5}
    bad = {"good": 0.5, "bad": 0.5, "miss": 0.0, "err_px": 0.5}
    assert td.cost(bad) > 5 * td.cost(miss) - 1
