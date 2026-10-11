"""Scenario judge, scenario tab (live + batch), run history recorder/panel, two-run comparison."""
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

import compare_runs
from history import HistoryPanel, HistoryRecorder, list_runs, read_meta
from replay import load_run
from runs_helper import write_model_run
from scenario_judge import LiveJudge
from telemetry import Telemetry

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
NOSE = 0.66


def _feed_approach(j, lateral=0.0, heading_deg=0.0, speed=0.4, stop_depth=0.5, retries=0, phase_end="DOCKED", cmd=None, t0=0.0):
    """Fly a straight line towards the dock at `speed` and feed the judge at 5 Hz."""
    t, x = t0, 6.0
    eul = np.array([0.0, 0.0, math.radians(heading_deg)])
    verdict = ""
    while x + NOSE < 10.0 + stop_depth:
        verdict = j.feed(t, [x, lateral, 3.0], eul, speed, "TERMINAL", retries, cmd)
        x += speed * 0.2
        t += 0.2
    for _ in range(30):                                                    # stopped inside, phase DOCKED
        verdict = j.feed(t, [10.0 + stop_depth - NOSE, lateral, 3.0], eul, 0.0, phase_end, retries, cmd)
        t += 0.2
    return verdict


def test_judge_passes_a_clean_entry_and_reports_the_numbers():
    j = LiveJudge()
    assert _feed_approach(j) == "PASS"
    assert j.cross["speed"] == pytest.approx(0.4, abs=0.05) and abs(j.cross["lat"]) < 0.01 and j.min_clear > 0.1
    assert "PASS" in j.summary() and "entry lat" in j.summary()


@pytest.mark.parametrize("kw,needle", [
    (dict(lateral=0.3), "bad entry"), (dict(heading_deg=9.0), "bad entry"), (dict(speed=0.7), "bad entry"), (dict(retries=1), "retry"),
])
def test_judge_fails_each_entry_rule(kw, needle):
    j = LiveJudge()
    v = _feed_approach(j, **kw)
    assert v == "FAIL" or (needle == "retry" and any(needle in f for f in j.failures)), (v, j.failures)
    assert any(needle in f for f in j.failures)


def test_judge_detects_wall_contact_overshoot_and_pinned_thruster():
    j = LiveJudge()
    j.feed(0.0, [9.0, 0.0, 3.0], [0, 0, 0], 0.4, "TERMINAL")
    j.feed(0.2, [9.6, 0.6, 3.0], [0, 0, 0], 0.4, "TERMINAL")           # 0.5 m off the axis, hull inside the funnel mouth region
    assert j.verdict == "FAIL" and any("wall contact" in f for f in j.failures)
    j2 = LiveJudge()
    assert _feed_approach(j2, stop_depth=1.6, phase_end="TERMINAL") == "FAIL" and any("overshoot" in f for f in j2.failures)
    j3 = LiveJudge()
    for k in range(20):
        j3.feed(0.2 * k, [5.0, 0.0, 3.0], [0, 0, 0], 0.3, "APPROACH", 0, {"th_01": 1800.0})
    assert any("actuator pinned" in f for f in j3.failures)


def test_judge_times_out_and_checks_safety_limits_and_stays_final():
    j = LiveJudge(timeout_s=5.0)
    for k in range(40):
        j.feed(0.2 * k, [3.0, 0.0, 3.0], [0, 0, 0], 0.1, "APPROACH")
    assert j.verdict == "FAIL" and any("timeout" in f for f in j.failures)
    n = len(j.failures)
    j.feed(99.0, [3.0, 0.0, 3.0], [0, 0, 0], 0.1, "APPROACH")
    assert len(j.failures) == n                                               # a finished verdict does not change
    k = LiveJudge()
    k.feed(0.0, [3.0, 0.0, 3.0], [math.radians(40), 0, 0], 0.0, "APPROACH")
    k.feed(0.2, [3.0, 0.0, 9.0], [0, 0, 0], 0.0, "APPROACH")
    assert any("attitude" in f for f in k.failures) or any("depth" in f for f in k.failures)


# ----------------------------------------------------------------------------------------------------------------- scenario tab
class FakeSource:
    def __init__(self, available=True):
        self.sim_controls_available = available
        self.sim_status = {}
        self.sent = []

    def send_sim_command(self, d):
        self.sent.append(d)
        return True


class FakeControls:
    """Just the parts of ControlsPanel the scenario tab calls."""

    def __init__(self):
        self.started, self.stopped = [], 0
        self.ctrl = QtWidgets.QComboBox()
        self.ctrl.addItems(["dock_test", "terminal_docking"])
        self._pending_start = None
        self._running = []

    def running_controllers(self): return list(self._running)
    def external_controllers(self): return {}
    def _stop_all_controllers(self): self.stopped += 1; self._running.clear()
    def _on_stop(self): self.stopped += 1; self._running.clear()
    def _start_now(self, name): self.started.append(name); self._running.append(name)


def test_scenario_tab_refuses_without_a_fake_vehicle_and_resets_then_starts_with_one(monkeypatch):
    from scenario_panel import ScenarioPanel
    tel = Telemetry()
    p = ScenarioPanel(FakeSource(False), tel, FakeControls())
    p.run_live()
    assert "Not available" in p.verdict.text() and not p._running
    src, ctl = FakeSource(True), FakeControls()
    p = ScenarioPanel(src, tel, ctl)
    p.range.setValue(8.0); p.lateral.setValue(1.5); p.heading.setValue(-20.0)
    p.run_live()
    reset = [d for d in src.sent if d.get("cmd") == "reset"][0]
    assert reset["pos"] == [2.0, 1.5, 3.0] and reset["eul_deg"] == [0.0, 0.0, -20.0]
    t0 = time.time()
    while not ctl.started and time.time() - t0 < 3:
        app.processEvents(); time.sleep(0.05)
    assert ctl.started == ["terminal_docking"] and p._running and p.btn_abort.isEnabled() and not p.btn_run.isEnabled()
    # feed truth through the status like the fake vehicle does: a clean docking
    tel.push_ctrl({"mode": "TERMINAL"})
    clock = {"t": 0.0}
    p.clock = lambda: clock["t"]
    x = 2.0
    for k in range(80):
        src.sim_status = {"truth": [x, 1.5 * max(0.0, 1 - (x - 2.0) / 5.5), 3.0, 0.0, 0.0, 0.0, 0.4]}
        clock["t"] += 0.2
        p._tick()
        x = min(x + 0.1, 9.5 - NOSE + 0.66 * 0)
    tel.push_ctrl({"mode": "DOCKED"})
    for k in range(40):
        src.sim_status = {"truth": [10.5 - NOSE + 1e-4 * k, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]}
        clock["t"] += 0.2
        p._tick()
    assert p.judge.verdict in ("PASS", "FAIL") and not p._running and ctl.stopped >= 1
    assert "truth" in p.verdict.text()


def test_scenario_batch_runs_the_evaluator_and_fills_the_scoreboard(tmp_path):
    from scenario_panel import ScenarioPanel
    p = ScenarioPanel(FakeSource(True), Telemetry(), FakeControls())
    p.grid_box.setCurrentText("smoke"); p.plants_box.setCurrentText("nominal"); p.vision_box.setCurrentText("perfect")
    p.run_batch()
    t0 = time.time()
    while p.batch_proc.state() != QtCore.QProcess.NotRunning and time.time() - t0 < 120:
        app.processEvents(); time.sleep(0.1)
    app.processEvents()
    assert p.table.rowCount() == 4                                          # smoke grid: 2 laterals x 2 headings, 1 plant
    assert "first-try passes" in p.summary.text()
    assert {p.table.item(r, 4).text() for r in range(4)} <= {"PASS", "FAIL"}
    p._batch_csv.unlink()


# ----------------------------------------------------------------------------------------------------------------- history
def _tel_with(n=40, controller=None):
    t = Telemetry()
    for k in range(n):
        t.push_odom([0.1 * k, 0, 3.0], [0, 0, 0], [0.3, 0, 0, 0, 0, 0], t=0.2 * k)
        t.push_cmd(["th_01"], [100.0 + k], t=0.2 * k)
    return t


def test_recorder_starts_with_a_controller_follows_new_samples_and_closes_after_idle(tmp_path):
    rec = HistoryRecorder(tmp_path, idle_close_s=1.0, min_rows=5)
    tel = Telemetry()
    now = 0.0
    rec.observe(None, tel, now)
    assert not rec.recording                                                  # nothing running: no file
    for k in range(30):
        tel.push_odom([0.1 * k, 0, 3.0], [0, 0, 0], [0.3, 0, 0, 0, 0, 0], t=0.2 * k)
        tel.push_cmd(["th_01"], [500.0], t=0.2 * k)
        now += 0.2
        rec.observe("dock_test", tel, now)
        rec.observe("dock_test", tel, now)                                    # the same sample twice must not duplicate the row
    assert rec.recording and rec.path.name.startswith("auto_dock_test_")
    for _ in range(8):
        now += 0.2
        rec.observe(None, tel, now)
    assert not rec.recording and len(rec.finished) == 1
    run = load_run(rec.finished[0])
    assert run.n == 30 and run.cmd["th_01"][0] == 500.0 and read_meta(rec.finished[0])["controller"] == "dock_test"


def test_recorder_discards_runs_that_stop_at_once(tmp_path):
    rec = HistoryRecorder(tmp_path, idle_close_s=0.1, min_rows=10)
    tel = Telemetry()
    tel.push_odom([0, 0, 3], [0, 0, 0], [0] * 6, t=0.0)
    rec.observe("dof_testing", tel, 0.0)
    rec.observe(None, tel, 0.5)
    rec.observe(None, tel, 1.0)
    assert not rec.recording and list_runs(tmp_path) == []


def test_history_panel_lists_replays_compares_and_deletes(tmp_path, monkeypatch):
    a = write_model_run(tmp_path / "auto_dock_test_a.csv", T=20.0)
    b = write_model_run(tmp_path / "auto_dock_test_b.csv", {"drag_quad_X": 9.0}, T=20.0)
    (tmp_path / "scoreboard_x.csv").write_text("plant,range\n")
    panel = HistoryPanel(tmp_path, confirm=lambda t, m: True)
    assert panel.table.rowCount() == 2                                        # the scoreboard file is not a run
    launched = []
    monkeypatch.setattr(panel, "_launch", lambda args: launched.append(args))
    panel.table.selectRow(0)
    panel.btn_replay.click()
    panel.btn_overlay.click()
    assert launched[0][-2] == "--replay" or "--replay" in launched[0]
    assert "--overlay" in launched[1]
    panel.table.selectAll()
    panel.btn_compare.click()
    assert "RMS diff" in panel.info.toPlainText() and panel.image.pixmap() is not None and not panel.image.pixmap().isNull()
    panel.table.selectRow(0)
    panel.btn_delete.click()
    assert panel.table.rowCount() == 1
    panel.table.selectAll()
    panel.btn_compare.click()
    assert "exactly two" in panel.info.toPlainText()


def test_compare_runs_numbers_and_cli(tmp_path, capsys):
    a = write_model_run(tmp_path / "a.csv", T=20.0)
    b = write_model_run(tmp_path / "b.csv", {"drag_quad_X": 9.0}, T=20.0)
    rows = compare_runs.difference_table(load_run(a), load_run(b))
    by = {r["channel"]: r for r in rows}
    assert by["north x [m]"]["rms_diff"] > 0.2 and compare_runs.difference_table(load_run(a), load_run(a))[0]["rms_diff"] == pytest.approx(0.0, abs=1e-9)
    png = tmp_path / "ab.png"
    assert compare_runs.main([str(a), str(b), "--plot", str(png)]) == 0 and png.stat().st_size > 5000
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
    assert compare_runs.main([str(a), str(tmp_path / "bad.csv")]) == 2
