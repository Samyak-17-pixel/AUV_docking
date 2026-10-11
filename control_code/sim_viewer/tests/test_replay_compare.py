"""Run loader (three log formats), interpolation, ReplaySource, compare (model vs recording) and calibrate (parameter recovery)."""
import csv
import math
import time

import numpy as np
import pytest

from calibrate import calibrate, main as calibrate_main
from compare import build_model, main as compare_main, rms_table, simulate, write_model_csv
from recording import ACTUATORS, COLUMNS, CsvRecorder
from replay import ReplaySource, UnsupportedLog, load_run
from runs_helper import write_model_run
from telemetry import Telemetry


@pytest.fixture(scope="module")
def own(tmp_path_factory):
    return write_model_run(tmp_path_factory.mktemp("runs") / "own.csv", T=30.0)


# ------------------------------------------------------------------ loader
def test_recorder_csv_round_trip(own):
    r = load_run(own)
    assert r.fmt == "recorder" and r.n == 301 and r.t[0] == 0.0 and r.duration == pytest.approx(30.0)
    assert set(r.cmd) == set(ACTUATORS) and r.meta["label"] == "synthetic"
    assert r.pos[0, 2] == pytest.approx(3.0) and r.nu.shape == (301, 6)
    assert np.all(np.diff(r.t) > 0)


def test_interpolation_and_zero_order_hold(tmp_path):
    rec = CsvRecorder(tmp_path / "a.csv")
    for t, x, th in [(0.0, 0.0, 0.0), (1.0, 2.0, 500.0), (2.0, 2.0, 0.0)]:
        rec.set_cmd(["th_01"], [th], t)
        rec.add_state(t, [x, 0, 3.0], [0, 0, 0], [0, 0, 0, 0, 0, 0])
    rec.close()
    r = load_run(tmp_path / "a.csv")
    pos, _, _, cmd = r.at(0.5)
    assert pos[0] == pytest.approx(1.0) and cmd["th_01"] == 0.0              # position interpolated, command held from the previous row
    assert r.at(1.5)[3]["th_01"] == 500.0
    assert r.at(-5.0)[0][0] == 0.0 and r.at(99.0)[0][0] == 2.0               # clamped at both ends
    assert r.slice(1.0, 2.0).n == 2 and r.slice(1.0, 2.0).t[0] == 0.0


def test_yaw_is_unwrapped_and_bad_rows_are_dropped(tmp_path):
    rec = CsvRecorder(tmp_path / "w.csv")
    for i, yaw in enumerate([170.0, 178.0, -176.0, -168.0]):                 # crosses +-180
        rec.set_cmd(["th_01"], [0.0], float(i))
        rec.add_state(float(i), [0, 0, 3], [0, 0, math.radians(yaw)], [0] * 6)
    rec.close()
    text = (tmp_path / "w.csv").read_text().splitlines()
    text.insert(5, text[5])                                                   # a duplicated timestamp
    text.append("5," + ",".join(["nan"] * (len(COLUMNS) - 1)))                # a row of NaNs
    (tmp_path / "w2.csv").write_text("\n".join(text) + "\n")
    r = load_run(tmp_path / "w2.csv")
    assert r.n == 4 and np.all(np.diff(r.t) > 0)
    assert np.all(np.diff(np.degrees(r.eul[:, 2])) > 0) and np.degrees(r.eul[-1, 2]) == pytest.approx(192.0)


def test_dof_testing_and_station_keeping_formats(tmp_path, own):
    base = load_run(own)
    # dof_testing: t, phase, x, y, z, roll_deg, ..., r, X, Z, K, M, N, th_01, ...
    fields = ["t", "phase", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg", "u", "v", "w", "p", "q", "r", "X", "Z", "K", "M", "N", "th_01", "th_02", "th_03", "cs_04"]
    with open(tmp_path / "dof.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(0, 100):
            row = dict(zip(["t", "x", "y", "z"], [base.t[i], *base.pos[i]]), phase="run", roll_deg=0, pitch_deg=0, yaw_deg=0, u=base.nu[i, 0], v=0, w=0, p=0, q=0, r=0, X=0, Z=0, K=0, M=0, N=0,
                       th_01=base.cmd["th_01"][i], th_02=0, th_03=0, cs_04=1.0)
            w.writerow(row)
    r = load_run(tmp_path / "dof.csv")
    assert r.fmt == "dof_testing" and r.n == 100 and "th_01" in r.cmd and "cs_06" not in r.cmd
    # station_keeping: depth_err_m ... depth, x, y, z, attitude, v..r
    sk = ["t", "depth_err_m", "u", "depth", "th_01", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg", "v", "w", "p", "q", "r"]
    with open(tmp_path / "sk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sk, restval=0.0)
        w.writeheader()
        for i in range(50):
            w.writerow({"t": i * 0.05, "depth_err_m": 0.1, "u": 0.0, "depth": 3.0, "th_01": 0.0, "x": 0, "y": 0, "z": 3.0, "roll_deg": 0, "pitch_deg": 0, "yaw_deg": 0})
    assert load_run(tmp_path / "sk.csv").fmt == "station_keeping"


def test_unusable_logs_say_why(tmp_path):
    (tmp_path / "old_sk.csv").write_text("t,depth_err_m,u,depth\n0,0.1,0,3\n0.05,0.1,0,3\n0.1,0.1,0,3\n")
    with pytest.raises(UnsupportedLog, match="older version"):
        load_run(tmp_path / "old_sk.csv")
    (tmp_path / "junk.csv").write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
    with pytest.raises(UnsupportedLog, match="not a recorder"):
        load_run(tmp_path / "junk.csv")
    (tmp_path / "short.csv").write_text("t,x\n0,0\n")
    with pytest.raises(UnsupportedLog, match="at least two"):
        load_run(tmp_path / "short.csv")


# ------------------------------------------------------------------ ReplaySource
def _drain(src, timeout=10.0):
    t0 = time.time()
    while src.playing and time.time() - t0 < timeout:
        time.sleep(0.05)


def test_replay_source_plays_seeks_pauses_and_pushes_model(own):
    run = load_run(own)
    model = simulate(run)
    tel = Telemetry()
    src = ReplaySource(tel, run, speed=30.0, model=model)
    src.start()
    assert src.ready.wait(3.0) and src.error is None and "REPLAY" in tel.source_name and "model overlay" in tel.source_name
    _drain(src)
    assert not src.playing and src.position == pytest.approx(run.duration)
    last = tel.latest()
    assert last.pos[0] == pytest.approx(run.pos[-1, 0], abs=1e-6) and tel.latest_model() is not None
    assert tel.latest_model().pos[0] == pytest.approx(model.pos[-1, 0], abs=1e-6)
    w = tel.window(1000.0)
    assert "m_depth" in w and len(w["m_depth"]) == len(w["m_t"]) > 10
    # seek back: telemetry restarts at that row; pause holds it
    src.seek(10.0)
    src.play()
    time.sleep(0.2)
    src.pause()
    time.sleep(0.2)
    p = src.position
    time.sleep(0.3)
    assert src.position == pytest.approx(p) and p >= 10.0
    first = tel.latest().pos.copy()
    time.sleep(0.2)
    assert np.allclose(tel.latest().pos, first)
    src.stop()
    src.join(3.0)
    assert not src.is_alive()


def test_replay_source_loops(own):
    run = load_run(own).slice(0.0, 3.0)
    tel = Telemetry()
    src = ReplaySource(tel, run, speed=40.0, loop=True)
    src.start()
    time.sleep(1.0)
    assert src.playing and src.counts["samples"] > run.n          # went around at least once
    src.stop()
    src.join(3.0)


def test_replay_source_error_is_reported_not_raised(own):
    run = load_run(own)
    run.pos = run.pos[:5]                                          # inconsistent arrays -> index error inside the thread
    src = ReplaySource(Telemetry(), run, speed=50.0)
    src.start()
    src.ready.wait(3.0)
    time.sleep(0.5)
    assert src.error is not None
    src.stop()
    src.join(3.0)


# ------------------------------------------------------------------ compare
def test_model_reproduces_its_own_run(own):
    run = load_run(own)
    for mode in ("free", "segments"):
        rows = {r["channel"]: r for r in rms_table(run, simulate(run, mode=mode, segment_s=3.0))}
        assert rows["x"]["rms"] < 0.01 and rows["depth"]["rms"] < 0.005 and rows["u"]["rms"] < 0.001 and rows["yaw"]["rms"] < 0.1, mode


def test_wrong_parameters_show_up_and_segments_localise_the_error(tmp_path):
    truth = load_run(write_model_run(tmp_path / "t.csv", {"rpm_to_rps": 0.0233, "drag_quad_X": 9.0}, T=30.0))
    wrong = {r["channel"]: r for r in rms_table(truth, simulate(truth, mode="free"))}
    seg = {r["channel"]: r for r in rms_table(truth, simulate(truth, mode="segments", segment_s=3.0))}
    assert wrong["x"]["rms"] > 0.5 and wrong["u"]["rms"] > 0.05            # the nominal model is clearly off in a free run
    assert seg["x"]["rms"] < 0.5 * wrong["x"]["rms"]                       # restarting every 3 s keeps the drift small
    fixed = {r["channel"]: r for r in rms_table(truth, simulate(truth, {"rpm_to_rps": 0.0233, "drag_quad_X": 9.0}))}
    assert fixed["x"]["rms"] < 0.01                                        # with the right numbers it matches again


def test_compare_rejects_bad_input(own, tmp_path):
    run = load_run(own)
    with pytest.raises(KeyError):
        build_model({"nonsense": 1.0})
    run.cmd = {}
    with pytest.raises(UnsupportedLog, match="no actuator commands"):
        simulate(run)


def test_compare_cli_writes_model_csv_and_plot(own, tmp_path, capsys):
    out, png = tmp_path / "model.csv", tmp_path / "o.png"
    assert compare_main([str(own), "--mode", "segments", "--segment-s", "4", "--out", str(out), "--plot", str(png), "--set", "rpm_to_rps=0.0166667"]) == 0
    text = capsys.readouterr().out
    assert "RMS err" in text and "depth" in text
    assert png.stat().st_size > 5000
    m = load_run(out)                                                    # the model prediction is itself a loadable run
    assert m.n == load_run(own).n and "MODEL" in m.meta["recorder"]
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
    assert compare_main([str(tmp_path / "bad.csv")]) == 2


# ------------------------------------------------------------------ calibrate
def test_calibration_recovers_perturbed_parameters(tmp_path):
    truth = {"thruster_tau_s": 0.35, "drag_quad_Z": 200.0, "drag_quad_M": 11.0, "net_up_n": 1.0}
    run = load_run(write_model_run(tmp_path / "p.csv", truth, T=40.0))
    res = calibrate([run], list(truth), segment_s=4.0)
    got = {r["name"]: r for r in res["params"]}
    assert res["rms_after"] < 0.25 * res["rms_before"]
    assert got["thruster_tau_s"]["fitted"] == pytest.approx(0.35, rel=0.05)
    assert got["drag_quad_Z"]["fitted"] == pytest.approx(200.0, rel=0.08)
    assert got["drag_quad_M"]["fitted"] == pytest.approx(11.0, rel=0.1)
    assert got["net_up_n"]["fitted"] == pytest.approx(1.0, abs=0.1)


def test_calibration_flags_what_the_data_cannot_tell(tmp_path):
    run = load_run(write_model_run(tmp_path / "h.csv", {"net_up_n": 0.5}, T=20.0, heave_only=True))     # no forward speed, no fin commands
    res = calibrate([run], ["cl_alpha_per_rad", "net_up_n"], segment_s=4.0)
    st = {r["name"]: r["status"] for r in res["params"]}
    assert st["cl_alpha_per_rad"] == "NOT IDENTIFIABLE"
    assert st["net_up_n"] == "ok"
    # thrust scale and surge drag trade off in a surge-only run: the tool must say they are entangled, not report both as 'ok'
    full = load_run(write_model_run(tmp_path / "f.csv", {"rpm_to_rps": 0.0233, "drag_quad_X": 9.0}, T=30.0))
    res2 = calibrate([full], ["rpm_to_rps", "drag_quad_X"], segment_s=4.0)
    assert all(r["status"] != "ok" for r in res2["params"]) and all(r["max_corr"] > 0.9 for r in res2["params"])


def test_calibration_cli_prints_suggestions_and_never_edits_the_yaml(tmp_path, capsys):
    import hashlib
    from pathlib import Path
    yaml_path = Path(__file__).resolve().parents[2] / "common" / "mako_geometry.yaml"
    before = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    run_path = write_model_run(tmp_path / "c.csv", {"net_up_n": 0.8, "drag_quad_M": 11.0}, T=24.0)
    js = tmp_path / "fit.json"
    assert calibrate_main([str(run_path), "--params", "net_up_n,drag_quad_M", "--json", str(js)]) == 0
    out = capsys.readouterr().out
    assert "Suggested edits" in out and "buoyancy_mass_kg" in out and "Nothing was written" in out
    assert js.exists() and hashlib.sha256(yaml_path.read_bytes()).hexdigest() == before
    assert calibrate_main([str(run_path), "--params", "bogus"]) == 2
