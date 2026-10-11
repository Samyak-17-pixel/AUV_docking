"""mission_showcase.py: flies missions with realistic odometry, judges them on the true path, saves the trajectories. A bad mission must be judged FAILED."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mission_showcase as M  # noqa: E402


def test_a_good_mission_passes_and_is_saved_with_all_four_fins(tmp_path):
    rc = M.main(["--only", "goto_hold", "--out", str(tmp_path / "ms"), "--no-copy", "--T", "200"])
    assert rc == 0
    d = tmp_path / "ms" / "goto_hold"
    for f in ("trajectory.csv", "top_view.png", "timeline.png", "summary.json"):
        assert (d / f).exists() and (d / f).stat().st_size > 200, f
    info = json.loads((d / "summary.json").read_text())
    assert info["pass"] and info["min_dock_distance_m"] > 6.0 and info["end_from_home_m"] < 1.5
    sys.path[:0] = [str(_d) for _d in [Path(__file__).resolve().parents[1] / "sim_viewer", *sorted((Path(__file__).resolve().parents[1] / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
    from replay import load_run
    run = load_run(d / "trajectory.csv")
    assert run.n > 200 and float(np.max(np.abs(run.cmd["cs_04"]))) > 0.5 and float(np.max(np.abs(run.cmd["cs_08"]))) > 0.5      # fins recorded (not just one)


def test_a_mission_that_drives_at_the_dock_is_judged_failed(tmp_path):
    M.MISSIONS["bad_toward_dock"] = [{"type": "goto", "x": 8.0, "y": 0.0, "z": 3.0}, {"type": "return_home"}]
    try:
        mr, L, warn, cfg = M.fly(M.MISSIONS["bad_toward_dock"], T=150.0)
        ok, fails, met = M.judge("bad", mr, L, cfg)
    finally:
        M.MISSIONS.pop("bad_toward_dock", None)
    assert not ok and any("dock" in f or "aborted" in f for f in fails), fails       # negative control: the keep-out aborts it and the judge says so


def test_the_lane_judge_sees_a_longer_runin_matters():
    """With realistic odometry the default 5 m run-in left the first lane ~1.9 m off; the shipped 8 m run-in passes (the reason for leadin_m: 8)."""
    import copy
    short = copy.deepcopy(M.MISSIONS["lawnmower"])
    short[0].pop("leadin_m")
    mr, L, _, cfg = M.fly(short, T=300.0)
    ok_short, f_short, m_short = M.judge("x", mr, L, cfg)
    mr2, L2, _, cfg2 = M.fly(M.MISSIONS["lawnmower"], T=300.0)
    ok_long, f_long, m_long = M.judge("x", mr2, L2, cfg2)
    assert ok_long and m_long["lane_error_p95_m"] < 1.0, f_long
    assert m_short["lane_error_p95_m"] > m_long["lane_error_p95_m"] * 2.0
