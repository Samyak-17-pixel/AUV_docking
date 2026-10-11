"""Mosaic georeferencing (objects land where they are, shadows fall away from the vehicle on the right side), side-scan recording / sidecar / replay."""
import math
from pathlib import Path

import numpy as np
import pytest

from recording import SonarLog, SonarRecorder, sonar_sidecar_path
from sidescan_mosaic import Mosaic
from sidescan_view import mosaic_image, waterfall_image
from synthetic_sidescan import SideScanConfig, SideScanSim
from telemetry import Telemetry
from terrain import Terrain

FLAT = {"size_m": 120.0, "dx_m": 0.25, "boulders": 0, "trenches": 0, "relief_m": 0.0, "ripples_m": 0.0, "center_m": [0.0, 0.0], "dock_xy_m": [0.0, 0.0], "calm_radius_m": 1000.0,
        "dock_floor_depth_m": 10.0, "calm_slope": [0.0, 0.0], "objects": True}
OBJ = [{"kind": "box", "xy": [0.0, 8.0], "size": [2.0, 2.0, 1.5]}, {"kind": "box", "xy": [10.0, -9.0], "size": [2.0, 2.0, 1.5]}]


@pytest.fixture(scope="module")
def survey():
    tr = Terrain(dict(FLAT, object_list=OBJ))
    sim = SideScanSim(tr, SideScanConfig(gain_index=-1, range_m=30.0))
    mo = Mosaic(tr.extent, 0.5)
    rec = SonarRecorder()
    for k in range(400):                                                    # 40 m north at 1 m/s, 10 Hz
        pos = [-20.0 + k * 0.1, 0.0, 3.0]
        for side in ("port", "starboard"):
            r = sim.ping(pos, [0.0, 0.0, 0.0], side)
            meta = {"start_range_m": 0.0, "range_m": r.range_m, "gain_db": r.gain_db, "gain_index": -1, "altitude_m": r.altitude_m, "ping_hz": 10.0, "sound_speed_mps": 1500.0, "pos": pos, "eul": [0.0, 0.0, 0.0]}
            mo.add_ping(side, r.intensity, meta)
            rec.add(k * 0.1, side, r.intensity, meta)
    return tr, mo, rec


def window(mo, cx, cy, r):
    xs = mo.x0 + (np.arange(mo.nx) + 0.5) * mo.cell
    ys = mo.y0 + (np.arange(mo.ny) + 0.5) * mo.cell
    m = (np.abs(xs - cx) < r)[:, None] & (np.abs(ys - cy) < r)[None, :]
    v = mo.image()[m]
    return v[np.isfinite(v)]


def test_shadows_fall_behind_the_objects_on_the_correct_side(survey):
    tr, mo, _ = survey
    assert window(mo, 0.0, 11.0, 1.5).mean() < 0.5 * window(mo, 0.0, 5.0, 1.0).mean()          # starboard (east) box: dark just beyond it
    assert window(mo, 10.0, -12.0, 1.5).mean() < 0.5 * window(mo, 10.0, -6.0, 1.0).mean()      # port (west) box: dark beyond it, on the other side
    assert window(mo, 0.0, -11.0, 1.5).mean() > 0.4 * window(mo, 0.0, 5.0, 1.0).mean()         # negative control: no shadow where there is no object (port side at the first box's x)


def test_nadir_gap_and_coverage(survey):
    tr, mo, _ = survey
    assert np.isnan(mo.image()[int((0.0 - mo.x0) / mo.cell), int((0.0 - mo.y0) / mo.cell)])   # straight under the track (the nadir gap): nothing
    assert 0.0 < mo.coverage((-10.0, 10.0, -14.0, 14.0)) < 1.0
    ix = int((0.0 - mo.x0) / mo.cell)
    seen = np.isfinite(mo.image()[ix])
    ys = mo.y0 + np.arange(mo.ny) * mo.cell
    assert seen[(ys > 3.0) & (ys < 10.5)].mean() > 0.9 and seen[(ys > -10.5) & (ys < -3.0)].mean() > 0.9     # both sides reach out to ~11 m at 7 m altitude (1.9 x altitude slant)
    assert not seen[ys > 14.0].any() and not seen[ys < -14.0].any()                                  # and not beyond the beam (the bins there are noise only)


def test_recording_roundtrip_and_sidecar_and_replay(survey, tmp_path):
    tr, mo, rec = survey
    f = rec.save(tmp_path / "run_sidescan.npz")
    log = SonarLog(f)
    assert len(log) == 800 and log.intensity.dtype == np.uint16
    side, inten, meta = log.ping(1)
    assert side == "starboard" and np.array_equal(inten, rec.inten[1]) and abs(meta["altitude_m"] - 7.0) < 0.05 and meta["pos"][0] == pytest.approx(-20.0)
    (tmp_path / "run.csv").write_text("x")
    assert sonar_sidecar_path(tmp_path / "run.csv") == f
    assert sonar_sidecar_path(tmp_path / "other.csv") is None
    got = log.until(1.0)                                                                  # pings up to t = 1 s: 11 per side
    assert len(got) == 22 and got[0][1] == "port"
    assert log.until(1.0) == []                                                           # only the new ones are given
    log.seek(0.5)
    assert len(log.until(1.0)) == 10                                                      # seeking back replays from there


def test_pictures_have_the_right_shape_and_content(survey):
    tr, mo, rec = survey
    port = [rec.inten[i] for i in range(0, 200, 2)]
    stbd = [rec.inten[i] for i in range(1, 200, 2)]
    img = waterfall_image(port, stbd, (800, 300), 30.0)
    assert img.shape == (300, 800, 3) and img[:100, :400].std() > 5 and img[:100, 400:].std() > 5
    m = mosaic_image(mo, 2, np.array([[-20.0, 0.0], [20.0, 0.0]]), (0.0, 0.0), (-25.0, 25.0, -20.0, 20.0))
    assert m.shape[0] > 50 and m.std() > 5


def test_a_replay_with_a_sidecar_feeds_the_telemetry(tmp_path, survey):
    tr, mo, rec = survey
    from recording import CsvRecorder
    from replay import ReplaySource, load_run
    import time
    p = tmp_path / "run.csv"
    r = CsvRecorder(p, {"controller": "test"})
    for k in range(60):
        r.add_state(k * 0.1, [-20.0 + k * 0.1, 0.0, 3.0], [0, 0, 0], [1.0, 0, 0, 0, 0, 0])
    r.close()
    rec.save(tmp_path / "run_sidescan.npz")
    tel = Telemetry()
    src = ReplaySource(tel, load_run(p), speed=6.0, camera=False)
    src.start()
    src.ready.wait(5.0)
    t0 = time.time()
    while time.time() - t0 < 4.0 and len(tel.sonar_rows("starboard", 400)) < 20:
        time.sleep(0.05)
    src.stop()
    src.join(3.0)
    rows = tel.sonar_rows("starboard", 400)
    assert len(rows) >= 20 and len(tel.sonar_rows("port", 400)) >= 20
    assert abs(rows[0][2]["altitude_m"] - 7.0) < 0.05
