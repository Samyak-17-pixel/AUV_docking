"""Terrain following (fixed altitude from the simulated altimeter) and the side-scan lawnmower survey harness (flown closed loop, data saved, replayable)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path[:0] = [str(_d) for _d in [_HERE.parent / "sim_viewer", *sorted((_HERE.parent / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
import sonar_survey as SS  # noqa: E402
from mission_core import MissionRunner  # noqa: E402
from state import State  # noqa: E402
from synthetic_sidescan import SideScanConfig, SideScanSim  # noqa: E402
from terrain import Terrain  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

CFG = yaml.safe_load((_HERE / "mission.yaml").read_text())
RAMP = {"size_m": 240.0, "dx_m": 0.5, "boulders": 0, "trenches": 0, "relief_m": 0.0, "ripples_m": 0.0, "objects": False, "center_m": [0.0, 60.0], "dock_xy_m": [0.0, 0.0],
        "calm_radius_m": 1000.0, "dock_floor_depth_m": 8.0, "calm_slope": [0.0, 0.12]}          # the floor sinks 0.12 m per m to the east: 8 m at y = 0, 17.6 m at y = 80


def fly_over_ramp(follow: bool, T=110.0):
    t = Terrain(RAMP)
    sim = SideScanSim(t, SideScanConfig())
    cfg = copy.deepcopy(CFG)
    cfg["legs"] = [{"type": "goto", "x": 0.0, "y": 90.0, "z": 2.0}]
    cfg["safety"].update({"dock_keepout_m": 0.0, "max_depth_m": 22.0, "geofence": {"x_min": -30, "x_max": 30, "y_min": -20, "y_max": 140, "z_min": 0.4, "z_max": 22.0}})
    cfg["terrain_follow"] = {"enabled": follow, "altitude_m": 5.0, "rate_mps": 0.6, "max_depth_m": 20.0}
    mr = MissionRunner(cfg)
    m = VehicleModel(pos=(0.0, 0.0, 2.0), eul_deg=(0, 0, 90.0))
    alts, depths = [], []
    alt = None
    tt = 0.0
    while tt < T and not mr.done:
        a = sim.altimeter(m.pos, m.eul)
        alt = a if a is not None else alt
        out, s = mr.update(State.from_model(m), 0.05, alt)
        m.set_command(out)
        for _ in range(5):
            m.step(0.01)
        tt += 0.05
        alts.append(t.altitude(m.pos[0], m.pos[1], m.pos[2]))
        depths.append(m.pos[2])
    return np.array(alts), np.array(depths), tt


def test_terrain_following_holds_the_altitude_over_a_ramp_and_without_it_the_altitude_grows():
    a, d, tt = fly_over_ramp(True)
    n = len(a)
    assert abs(float(np.mean(a[int(0.5 * n):])) - 5.0) < 1.6 and float(np.max(np.abs(a[int(0.5 * n):] - 5.0))) < 3.0, (a[int(0.5 * n):].min(), a[int(0.5 * n):].max())
    assert d[-1] > d[0] + 6.0                                          # it dived to follow the floor
    a0, d0, _ = fly_over_ramp(False)
    assert float(np.max(a0)) > 12.0 and abs(d0[-1] - 2.0) < 0.6          # negative control: the fixed-depth flight ends 15 m above the floor


def test_a_survey_is_flown_judged_saved_and_can_be_replayed(tmp_path):
    out = tmp_path / "ss"
    rc = SS.main(["--only", "terrain_following", "--lanes", "2", "--length", "30", "--T", "160", "--out", str(out), "--no-copy"])
    assert rc == 0
    d = out / "terrain_following"
    for f in ("trajectory.csv", "sidescan.npz", "waterfall.png", "mosaic.png", "mosaic_vs_truth.png", "summary.json"):
        assert (d / f).exists() and (d / f).stat().st_size > 500, f
    s = json.loads((d / "summary.json").read_text())
    assert s["pass"] and s["coverage_of_survey_area"] >= 0.8 and s["min_altitude_m"] >= 1.0 and s["pings"] > 3000
    from recording import SonarLog, sonar_sidecar_path
    assert sonar_sidecar_path(d / "trajectory.csv") == d / "sidescan.npz"
    log = SonarLog(d / "sidescan.npz")
    assert len(log) == s["pings"] and log.intensity.shape[1] == 600
    from replay import load_run
    assert load_run(d / "trajectory.csv").duration > 60.0


def test_a_survey_with_lanes_too_far_apart_is_judged_failed(tmp_path):
    out = tmp_path / "wide"
    SS.main(["--only", "terrain_following", "--lanes", "2", "--spacing", "70", "--length", "30", "--T", "260", "--out", str(out), "--no-copy"])
    assert (out / "FAILED_terrain_following").exists() and not (out / "terrain_following").exists()       # negative control: 70 m lanes leave big gaps
    s = json.loads((out / "FAILED_terrain_following" / "summary.json").read_text())
    assert s["coverage_of_survey_area"] < 0.8
