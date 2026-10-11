"""ConfigPanel with the real controller YAMLs copied to a temp folder. No ROS, no real processes."""
import copy
import shutil

from yaml_fixture import copy_config
import sys
from pathlib import Path

import pytest
import yaml
from PyQt5 import QtCore, QtWidgets

import controls_panel
from config_panel import ConfigPanel
from controls_panel import ControlsPanel
from process_manager import ProcessManager
from test_controls_panel import FakeSource, RecordingProcs

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
CTRL = Path(__file__).resolve().parents[2]


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "77")

    def _make(confirm=True, ask=None):
        for name in ("waypoint_tracking", "mission", "dock_test"):
            src = Path(controls_panel.CONTROLLERS[name]["config"]) if not (tmp_path / f"{name}.yaml").exists() else tmp_path / f"{name}.yaml"
            if src.parent != tmp_path:
                copy_config(CTRL / name / f"{name}.yaml", tmp_path / f"{name}.yaml")
            monkeypatch.setitem(controls_panel.CONTROLLERS[name], "config", tmp_path / f"{name}.yaml")
        procs = RecordingProcs()
        ctl = ControlsPanel(FakeSource(), procs, confirm=lambda t, m: True)
        ctl._ext_finder = lambda: {}
        panel = ConfigPanel(ctl, None, confirm=lambda t, m: confirm, ask_path=ask)
        return panel, ctl, procs, tmp_path

    return _make


def select(panel, name):
    panel.ctrl.setCurrentText(name)


def test_waypoint_table_shows_the_waypoints_and_a_cell_edit_reaches_the_controls_tab(make):
    panel, ctl, procs, d = make()
    select(panel, "waypoint_tracking")
    y1 = yaml.safe_load((d / "waypoint_tracking.yaml").read_text())["waypoints"][1]["y"]               # the shipped list is the user's: read it, do not hard-code it
    assert panel.table.rowCount() == len(yaml.safe_load((d / "waypoint_tracking.yaml").read_text())["waypoints"])
    panel.table.item(1, 2).setText("6.5")                                    # y of waypoint 1
    assert panel.doc.items()[1]["y"] == 6.5
    assert ctl._edited["waypoint_tracking"]["waypoints"][1]["y"] == 6.5      # "apply to next Start"
    ctl.ctrl.setCurrentText("waypoint_tracking")
    ctl._on_start()
    cfg_arg = procs.started[-1][1][procs.started[-1][1].index("--config") + 1]
    assert yaml.safe_load(Path(cfg_arg).read_text())["waypoints"][1]["y"] == 6.5
    assert yaml.safe_load((d / "waypoint_tracking.yaml").read_text())["waypoints"][1]["y"] == y1          # the file is untouched


def test_map_click_adds_a_waypoint_and_drag_moves_it_and_right_click_deletes_it(make):
    panel, ctl, procs, d = make()
    select(panel, "waypoint_tracking")
    panel.map.pointAdded.emit(3.0, 4.0)
    assert len(panel.doc.items()) == 5 and panel.doc.items()[-1]["x"] == 3.0 and panel.table.rowCount() == 5
    panel.map.pointMoved.emit(4, 3.5, 4.5)
    assert panel.doc.items()[4]["x"] == 3.5 and panel.doc.items()[4]["y"] == 4.5
    panel.map.pointRemoved.emit(4)
    assert len(panel.doc.items()) == 4


def test_a_tight_corner_is_warned_about_and_drawn_red(make):
    panel, ctl, procs, d = make()
    select(panel, "waypoint_tracking")
    panel.doc.items()[1]["y"] = 3.0                                          # the old 5 x 3 rectangle: unreachable after the first corner
    panel._refresh_all()
    assert "turning circle" in panel.warn.toPlainText()
    assert panel.map.bad_points


def test_parameter_tree_edits_numbers_switches_and_text(make):
    panel, ctl, procs, d = make()
    select(panel, "waypoint_tracking")

    def find(path):
        stack = [panel.tree.topLevelItem(i) for i in range(panel.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            if it.data(0, QtCore.Qt.UserRole) == path:
                return it
            stack += [it.child(i) for i in range(it.childCount())]

    find(("speed", "cruise_mps")).setText(1, "0.8")
    find(("mission", "zero_on_exit")).setText(1, "false")
    find(("node", "vessel")).setText(1, "Mako_02")
    assert panel.doc.data["speed"]["cruise_mps"] == 0.8 and panel.doc.data["mission"]["zero_on_exit"] is False and panel.doc.data["node"]["vessel"] == "Mako_02"
    find(("speed", "cruise_mps")).setText(1, "fast")                           # not a number: refused, value unchanged
    assert panel.doc.data["speed"]["cruise_mps"] == 0.8
    assert "waypoints" not in {panel.tree.topLevelItem(i).text(0) for i in range(panel.tree.topLevelItemCount())}


def test_save_to_file_keeps_comments_makes_a_backup_and_asks_first(make):
    panel, ctl, procs, d = make(confirm=False)
    select(panel, "waypoint_tracking")
    before = (d / "waypoint_tracking.yaml").read_text()
    panel.map.pointAdded.emit(2.0, 2.0)
    panel._save()                                                            # cancelled in the dialog
    assert (d / "waypoint_tracking.yaml").read_text() == before and not (d / "waypoint_tracking.yaml.bak").exists()
    panel._confirm = lambda t, m: True
    panel._save()
    new = (d / "waypoint_tracking.yaml").read_text()
    assert (d / "waypoint_tracking.yaml.bak").read_text() == before
    assert new.count("#") == before.count("#") and "{x: 2.0, y: 2.0, z: 3.0}" in new
    assert len(yaml.safe_load(new)["waypoints"]) == 5
    assert "waypoint_tracking" not in ctl._edited                            # nothing left to apply: the file now holds the values


def test_save_as_new_file_leaves_the_original_alone(make, tmp_path):
    out = tmp_path / "survey_a.yaml"
    panel, ctl, procs, d = make(ask=lambda s: str(out))
    select(panel, "waypoint_tracking")
    before = (d / "waypoint_tracking.yaml").read_text()
    panel.map.pointAdded.emit(2.0, 2.0)
    panel._save_as()
    assert (d / "waypoint_tracking.yaml").read_text() == before
    assert len(yaml.safe_load(out.read_text())["waypoints"]) == 5


def test_revert_and_reload(make):
    panel, ctl, procs, d = make()
    select(panel, "waypoint_tracking")
    panel.map.pointAdded.emit(2.0, 2.0)
    assert "waypoint_tracking" in ctl._edited
    panel._revert()
    assert len(panel.doc.items()) == 4 and "waypoint_tracking" not in ctl._edited and not panel.doc.dirty


def test_mission_legs_can_be_added_moved_and_edited_and_the_preview_follows(make):
    panel, ctl, procs, d = make()
    select(panel, "mission")
    n0 = len(panel.doc.items())
    before = list(panel.map.points)
    panel.leg_type.setCurrentText("hold")
    panel.table.selectRow(0)
    panel._add()
    assert len(panel.doc.items()) == n0 + 1 and panel.doc.items()[1]["type"] == "hold"
    panel.table.selectRow(0)
    w = panel._leg_widgets["spacing_m"]
    w.setValue(10.0)
    assert panel.doc.items()[0]["spacing_m"] == 10.0
    assert panel.map.points != before                                           # the map preview was rebuilt from the edited leg
    panel._move(1)
    assert panel.doc.items()[1]["type"] == "lawnmower"
    panel._remove()
    assert len(panel.doc.items()) == n0


def test_map_click_sets_the_selected_legs_position(make):
    panel, ctl, procs, d = make()
    select(panel, "mission")
    panel.table.selectRow(0)
    panel.chk_place.setChecked(True)
    panel.map.clicked.emit(1.5, 6.5)
    assert panel.doc.items()[0]["origin"] == [1.5, 6.5]
    panel.table.selectRow(2)                                                 # the yo-yo: its length and direction are kept
    old_to = copy.deepcopy(panel.doc.items()[2]["to"])
    old_from = copy.deepcopy(panel.doc.items()[2]["from"])
    panel.map.clicked.emit(-3.0, 4.0)
    leg = panel.doc.items()[2]
    assert leg["from"] == [-3.0, 4.0] and leg["to"][0] - leg["from"][0] == pytest.approx(old_to[0] - old_from[0])


def test_mission_validation_names_the_problem(make):
    panel, ctl, procs, d = make()
    select(panel, "mission")
    panel.doc.items()[0]["spacing_m"] = 1.0
    panel.doc.items()[0]["width_m"] = 2.0                                    # 3 lanes 1 m apart
    panel._after_change()
    assert "lawnmower" in panel.warn.toPlainText() and "no problems" not in panel.warn.toPlainText()


def test_negative_control_bad_edits_do_not_touch_the_file(make):
    panel, ctl, procs, d = make(confirm=True)
    select(panel, "dock_test")
    before = (d / "dock_test.yaml").read_text()
    panel.doc.data["no_such_section"] = {"a": 1}                             # a change that has no place in the text: the comment-preserving save refuses it
    panel._save()
    assert (d / "dock_test.yaml").read_text() == before
    assert "not saved" in panel.info.text()


def test_edits_made_in_the_controls_tab_dialog_carry_over(make):
    panel, ctl, procs, d = make()
    ed = yaml.safe_load((d / "dock_test.yaml").read_text())
    ed["speed"]["creep_mps"] = 0.4
    ctl._edited["dock_test"] = ed
    select(panel, "dock_test")
    assert panel.doc.data["speed"]["creep_mps"] == 0.4
