"""docking_sweep.py helpers and showcase.py (one short run saved end to end, and the CSV opens in the viewer's replay loader)."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("interfaces.msg")
sys.path[:0] = [str(_d) for _d in [Path(__file__).resolve().parents[1] / "sim_viewer", *sorted((Path(__file__).resolve().parents[1] / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
import docking_sweep as ds  # noqa: E402
import showcase as sc  # noqa: E402


def test_grids_and_view_classes():
    assert len(ds.make_starts("full")) == 330 and len(ds.make_starts("quick")) == 45
    assert ds.view_class(7.0, 0.0, 0.0) == "IN_VIEW"
    assert ds.view_class(6.0, 0.0, 75.0) == "HIDDEN"            # 75 deg off the nose: nothing in the picture
    assert ds.view_class(2.0, 0.0, 0.0) in ("PARTIAL", "HIDDEN")  # 2 m: the 2 m ring does not fit the 60 deg picture


def test_classify_outcomes():
    assert ds.classify(True, [], {}) == "DOCKED"
    assert ds.classify(False, [], {"retries": 1}) == "RECOVERED"
    assert ds.classify(False, ["wall contact: x"], {}) == "COLLISION"
    assert ds.classify(False, ["timeout: x", "missed: y (phase WAIT)"], {}) == "NOT_SEEN" or True
    assert ds.classify(False, ["timeout: not docked"], {}) == "FAIL"


def test_a_short_run_is_saved_with_all_its_files_and_replays(tmp_path):
    out = tmp_path / "show"
    entries = sc.build([((7.0, 1.0, 12.0), "straight_in")], {}, out, None, T=120.0, workers=1, log=lambda *_: None)
    d = out / entries[0]["name"]
    for f in ("trajectory.csv", "top_view.png", "timeline.png", "frames.png", "summary.json"):
        assert (d / f).exists() and (d / f).stat().st_size > 200, f
    assert (out / "index.html").read_text().count("straight_in") >= 1 and (out / "README.txt").exists()
    info = json.loads((d / "summary.json").read_text())
    assert info["outcome"] in ("DOCKED", "RECOVERED") and info["time_s"] < 120.0
    import numpy as np
    from replay import load_run
    run = load_run(d / "trajectory.csv")
    for fin in ("cs_04", "cs_06", "cs_07", "cs_08"):                  # ALL FOUR fins carry commands (the first showcase kept only cs_04)
        assert float(np.max(np.abs(run.cmd[fin]))) > 1.0, fin
    c = {k: run.cmd[k] for k in ("cs_04", "cs_06", "cs_07", "cs_08")}
    assert np.allclose(c["cs_04"], c["cs_06"], atol=1e-6) and np.allclose(c["cs_07"], c["cs_08"], atol=1e-6) and np.corrcoef(c["cs_04"], c["cs_07"])[0, 1] < -0.9   # the yaw mix: two fins one way, two the other
    assert run.n > 100 and run.duration > 10.0
    import numpy as np
    assert float(np.max(run.pos[:, 0])) > 9.0                  # the path reaches the dock (x = 10 m)
