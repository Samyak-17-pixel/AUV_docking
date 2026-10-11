"""ControlsPanel with a fake data source and a recording ProcessManager. No ROS, no real processes."""
import os
import sys

import pytest
import yaml
from PyQt5 import QtWidgets

import gains_editor
from controls_panel import CONTROLLERS, STACK, ControlsPanel
from process_manager import ProcessManager

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeSource:
    def __init__(self, available=True):
        self.sim_controls_available = available
        self.sim_status = {"paused": False, "time_scale": 1.0, "push_left": 0.0, "error": ""}
        self.sent = []

    def send_sim_command(self, d):
        self.sent.append(d)
        return True

    def send_gain(self, msg):
        self.gains_sent = getattr(self, "gains_sent", []) + [msg]
        return True


class RecordingProcs(ProcessManager):
    def __init__(self):
        super().__init__()
        self.started = []
        self.fake_running = set()

    def start(self, name, argv, cwd=None, env=None):
        self.started.append((name, argv))
        self.fake_running.add(name)
        self.state_changed.emit(name, True)                    # like the real manager, so the panel refreshes its buttons
        return True

    def is_running(self, name):
        return name in self.fake_running

    def stop(self, name, grace_s=3.0):
        if name in self.fake_running:
            self.fake_running.discard(name)
            self.state_changed.emit(name, False)


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "77")

    def _make(available=True, confirm=None):
        src = FakeSource(available)
        procs = RecordingProcs()
        panel = ControlsPanel(src, procs, confirm=confirm or (lambda t, m: True))
        panel._ext_finder = lambda: {}                       # do not look at the real /proc in tests
        return panel, src, procs
    return _make


def test_buttons_send_the_right_commands(make):
    p, src, _ = make()
    p.btn_pause.click()
    assert src.sent[-1] == {"cmd": "pause", "value": True}
    src.sim_status["paused"] = True
    p.btn_pause.click()
    assert src.sent[-1] == {"cmd": "pause", "value": False}                    # the button resumes when paused
    p.btn_step.click()
    assert src.sent[-2:] == [{"cmd": "pause", "value": True}, {"cmd": "step", "seconds": 0.5}]
    p.scale.setCurrentIndex(p.scale.findData(4.0))
    assert src.sent[-1] == {"cmd": "time_scale", "value": 4.0}
    p.reset_spins["x"].setValue(1.5); p.reset_spins["y"].setValue(-0.4); p.reset_spins["z"].setValue(3.5); p.reset_spins["yaw"].setValue(90)
    p.btn_reset.click()
    assert src.sent[-1] == {"cmd": "reset", "pos": [1.5, -0.4, 3.5], "eul_deg": [0.0, 0.0, 90.0]}
    p.btn_clear.click()
    assert src.sent[-1] == {"cmd": "clear_push"}


def test_push_presets_fill_the_fields_and_the_push_command_carries_them(make):
    p, src, _ = make()
    p.preset.setCurrentText("sideways shove (Y 6 N, 1.5 s)")
    assert [w.value() for w in p.wrench] == [0, 6, 0, 0, 0, 0] and p.duration.value() == 1.5
    p.btn_push.click()
    assert src.sent[-1] == {"cmd": "push", "wrench": [0.0, 6.0, 0.0, 0.0, 0.0, 0.0], "duration": 1.5}


def test_simulation_controls_are_disabled_without_a_fake_vehicle(make):
    p, src, _ = make(available=False)
    p._refresh()
    assert not p.sim_box.isEnabled() and not p.btn_push.isEnabled() and "Not available" in p.sim_status.text()
    p.btn_push.click()                                                          # a disabled button ignores clicks
    assert src.sent == []
    p._on_push()                                                                # and the handler itself refuses too, and says why
    assert src.sent == [] and "nothing sent" in p.log.toPlainText()


def test_status_text_and_time_scale_follow_the_source(make):
    p, src, _ = make()
    src.sim_status.update(paused=True, time_scale=2.0, push_left=1.2)
    p._refresh()
    assert "PAUSED" in p.sim_status.text() and "2x" in p.sim_status.text() and "pushing" in p.sim_status.text()
    assert p.btn_pause.text() == "Resume" and p.scale.currentData() == 2.0 and src.sent == []     # syncing must not echo a command back


def test_starting_a_controller_uses_the_original_yaml_unless_gains_were_edited(make):
    p, _, procs = make()
    p.ctrl.setCurrentText("station_keeping")
    p.btn_start.click()
    name, argv = procs.started[-1]
    assert name == "station_keeping" and argv[argv.index("--config") + 1] == str(CONTROLLERS["station_keeping"]["config"])
    procs.fake_running.clear()
    p._refresh()
    base = gains_editor.load_yaml(CONTROLLERS["station_keeping"]["config"])
    p._edited["station_keeping"] = gains_editor.apply_edits(base, {("gains", "heave", "kp"): "77"})
    p.btn_start.click()
    cfg_path = procs.started[-1][1][procs.started[-1][1].index("--config") + 1]
    assert cfg_path != str(CONTROLLERS["station_keeping"]["config"])
    assert yaml.safe_load(open(cfg_path))["gains"]["heave"]["kp"] == 77.0
    assert yaml.safe_load(open(CONTROLLERS["station_keeping"]["config"]))["gains"]["heave"]["kp"] != 77.0       # the original is untouched


def test_dof_testing_gets_its_dof_and_mode_arguments(make):
    p, _, procs = make()
    p.ctrl.setCurrentText("dof_testing")
    p.dof.setCurrentText("pitch"); p.mode.setCurrentText("hold")
    p.btn_start.click()
    argv = procs.started[-1][1]
    assert argv[argv.index("--dof") + 1] == "pitch" and argv[argv.index("--mode") + 1] == "hold"
    assert argv[0] == sys.executable and "-u" in argv


def test_switching_controller_stops_the_running_one_then_starts_the_new_one(make):
    p, _, procs = make()
    p.ctrl.setCurrentText("dock_test"); p.btn_start.click()
    assert procs.fake_running == {"dock_test"}
    p.ctrl.setCurrentText("station_keeping")                  # the combo now shows a controller that is NOT running ...
    assert p.btn_stop.isEnabled() and p.btn_start.isEnabled()  # ... but Stop still works (this was the bug) and Start is available
    p.btn_start.click()
    assert [n for n, _ in procs.started] == ["dock_test", "station_keeping"] and procs.fake_running == {"station_keeping"}
    assert "stopping dock_test" in p.log.toPlainText()


def test_switch_can_be_cancelled_and_leaves_the_first_controller_running(make):
    p, _, procs = make(confirm=lambda t, m: False)
    p.ctrl.setCurrentText("dock_test"); p.btn_start.click()
    p.ctrl.setCurrentText("station_keeping"); p.btn_start.click()
    assert procs.fake_running == {"dock_test"} and "cancelled" in p.log.toPlainText()


def test_stop_button_stops_whatever_runs_even_when_another_is_selected(make):
    p, _, procs = make()
    p.ctrl.setCurrentText("dock_test"); p.btn_start.click()
    p.ctrl.setCurrentText("dof_testing")
    p.btn_stop.click()
    assert procs.fake_running == set() and not p.btn_stop.isEnabled()
    assert "no controller running" in p.ctrl_state.text()


def test_controller_started_in_a_terminal_is_seen_and_can_be_stopped(make, monkeypatch):
    import subprocess
    p, _, procs = make()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "station_keeping.py"])      # stands in for a terminal-started controller
    try:
        p._ext_finder = lambda: {"station_keeping": [child.pid]}
        p._refresh()
        assert "station_keeping" in p.ctrl_state.text() and "terminal" in p.ctrl_state.text() and p.btn_stop.isEnabled()
        p.btn_stop.click()
        child.wait(5)
        assert child.returncode is not None                         # SIGINT delivered
    finally:
        if child.poll() is None:
            child.kill()


def test_emergency_stop_stops_everything_and_asks_the_source_for_neutral(make):
    p, src, procs = make()
    zeros = []
    src.publish_zero = lambda: zeros.append(1)
    p.ctrl.setCurrentText("dock_test"); p.btn_start.click()
    p.btn_estop.click()
    assert procs.fake_running == set() and zeros == [1] and "EMERGENCY STOP" in p.log.toPlainText()


def test_finder_matches_only_known_controller_scripts(tmp_path):
    from controls_panel import find_external_controllers
    import subprocess
    script = CONTROLLERS["dock_test"]["script"]
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", str(script)])
    try:
        import time; time.sleep(0.3)
        found = find_external_controllers()
        assert child.pid in found.get("dock_test", [])
        assert not any(child.pid in v for k, v in found.items() if k != "dock_test")
    finally:
        child.kill()


def test_real_mavsim_domain_asks_for_confirmation_and_can_be_cancelled(make, monkeypatch):
    asked = []
    p, _, procs = make(available=False, confirm=lambda t, m: asked.append(m) or False)
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    p.ctrl.setCurrentText("dock_test"); p.btn_start.click()
    assert procs.started == [] and asked and "REAL mavsim" in asked[0]
    p2, _, procs2 = make(available=False, confirm=lambda t, m: True)
    p2.ctrl.setCurrentText("dock_test"); p2.btn_start.click()
    assert [n for n, _ in procs2.started] == ["dock_test"]                     # confirmed: it starts


def test_offline_stack_is_refused_on_the_real_domain_and_starts_three_programs_otherwise(make, monkeypatch):
    p, _, procs = make()
    p.reset_spins["y"].setValue(0.8)
    p.btn_stack.click()
    assert sorted(n for n, _ in procs.started) == sorted(STACK) and "sidescan" in STACK       # the sonar node is part of the stack
    fake = procs.started[0][1]
    assert fake[fake.index("--y") + 1] == "0.8" and "--no-gui" in dict(procs.started)["detector"]
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    p3, _, procs3 = make(available=False)
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    p3._refresh()
    assert not p3.btn_stack.isEnabled()                                         # the button is disabled on the real domain
    p3._on_stack_start()                                                        # and the handler refuses if it is called anyway
    assert procs3.started == [] and "refused" in p3.log.toPlainText()
    p.btn_stack_stop.click()
    assert procs.fake_running == set()


def test_live_gains_send_the_right_message_and_follow_the_selected_controller(make):
    import json
    p, src, _ = make()
    p.ctrl.setCurrentText("terminal_docking")
    assert "heave" in [p.lg_loop.itemText(i) for i in range(p.lg_loop.count())]
    p.lg_loop.setCurrentText("heave"); p.lg_key.setCurrentText("kp")
    base = p.lg_value.value()
    assert base > 0
    p.lg_up.click()                                                  # x 1.25, applied while changing
    msg = json.loads(src.gains_sent[-1])
    assert msg["controller"] == "terminal_docking" and msg["loop"] == "heave" and msg["key"] == "kp" and msg["value"] == pytest.approx(base * 1.25, rel=1e-3)
    assert "sent terminal_docking" in p.lg_info.text()
    p.lg_auto.setChecked(False)
    n = len(src.gains_sent)
    p.lg_value.setValue(base * 3)
    assert len(src.gains_sent) == n                                  # auto-apply off: nothing until 'Apply now'
    p.lg_apply.click()
    assert json.loads(src.gains_sent[-1])["value"] == pytest.approx(base * 3, rel=1e-3)
    p.ctrl.setCurrentText("station_keeping")                         # other controller: its own loops and values
    assert p.lg_loop.count() >= 1 and p.lg_value.value() != pytest.approx(base * 3)


def test_live_gains_report_when_nothing_can_receive_them(make):
    p, src, _ = make()
    src.send_gain = lambda m: False
    p.ctrl.setCurrentText("terminal_docking")
    p.lg_apply.click()
    assert "not sent" in p.lg_info.text()


def test_save_as_default_edits_a_copy_of_the_yaml_and_can_be_cancelled(make, tmp_path, monkeypatch):
    import shutil
    from controls_panel import CONTROLLERS as C
    copy = tmp_path / "terminal_docking.yaml"
    shutil.copy(C["terminal_docking"]["config"], copy)
    monkeypatch.setitem(C["terminal_docking"], "config", copy)
    before = copy.read_text()
    p, _, _ = make(confirm=lambda t, m: False)
    p.ctrl.setCurrentText("terminal_docking"); p.lg_loop.setCurrentText("heave"); p.lg_key.setCurrentText("kd")
    p.lg_auto.setChecked(False)
    p.lg_value.setValue(31.5)
    p.lg_save.click()
    assert copy.read_text() == before and "not saved" in p.lg_info.text()
    p2, _, _ = make(confirm=lambda t, m: True)
    p2.ctrl.setCurrentText("terminal_docking"); p2.lg_loop.setCurrentText("heave"); p2.lg_key.setCurrentText("kd")
    p2.lg_auto.setChecked(False)
    p2.lg_value.setValue(31.5)
    p2.lg_save.click()
    import yaml
    assert yaml.safe_load(copy.read_text())["gains"]["heave"]["kd"] == 31.5 and copy.read_text().count("\n") == before.count("\n")


def _fake_controller(tmp_path, name):
    script = tmp_path / f"{name}.py"
    script.write_text("import signal, sys, time\n"
                      "def bye(*a):\n    print('NEUTRAL SENT', flush=True); sys.exit(0)\n"
                      "signal.signal(signal.SIGINT, bye)\n"
                      f"print('{name} up', flush=True)\n"
                      "while True:\n    time.sleep(0.05)\n")
    return script


def test_real_processes_switch_stop_and_estop_end_to_end(tmp_path, monkeypatch):
    """Two real child processes standing in for controllers (they print NEUTRAL SENT on Ctrl-C like the real ones publish zeros)."""
    import time
    from PyQt5 import QtCore
    import controls_panel as cp
    a, b = _fake_controller(tmp_path, "ctrl_a"), _fake_controller(tmp_path, "ctrl_b")
    monkeypatch.setitem(cp.CONTROLLERS, "dock_test", dict(cp.CONTROLLERS["dock_test"], script=a))
    monkeypatch.setitem(cp.CONTROLLERS, "station_keeping", dict(cp.CONTROLLERS["station_keeping"], script=b))
    monkeypatch.setenv("ROS_DOMAIN_ID", "77")
    procs = ProcessManager()
    zeros = []
    src = FakeSource()
    src.publish_zero = lambda: zeros.append(1)
    panel = ControlsPanel(src, procs, confirm=lambda t, m: True)
    panel._ext_finder = lambda: {}

    def pump(cond, seconds=8.0):
        t0 = time.time()
        while not cond() and time.time() - t0 < seconds:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.03)
        return cond()

    try:
        panel.ctrl.setCurrentText("dock_test"); panel.btn_start.click()
        assert pump(lambda: "ctrl_a up" in panel.log.toPlainText()) and procs.is_running("dock_test")
        panel.ctrl.setCurrentText("station_keeping")                         # the situation that used to leave the user stuck
        assert panel.btn_stop.isEnabled()
        panel.btn_start.click()                                              # confirm -> stop A (Ctrl-C), start B when A has exited
        assert pump(lambda: "ctrl_b up" in panel.log.toPlainText()), panel.log.toPlainText()
        assert "NEUTRAL SENT" in panel.log.toPlainText() and not procs.is_running("dock_test") and procs.is_running("station_keeping")
        panel.btn_estop.click()
        assert pump(lambda: not procs.is_running("station_keeping")) and zeros == [1]
        assert "EMERGENCY STOP" in panel.log.toPlainText()
    finally:
        procs.stop_all()
