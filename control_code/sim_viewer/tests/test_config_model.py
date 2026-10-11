"""ConfigDoc (no Qt): comment-preserving edits of the real controller YAMLs."""
import shutil

from yaml_fixture import copy_config
from pathlib import Path

import pytest
import yaml

from config_model import LEG_TEMPLATES, ConfigDoc

CTRL = Path(__file__).resolve().parents[2]


@pytest.fixture
def doc(tmp_path):
    def _doc(name):
        copy_config(CTRL / name / f"{name}.yaml", tmp_path / f"{name}.yaml")
        return ConfigDoc(tmp_path / f"{name}.yaml", name)
    return _doc


def test_scalar_edits_are_made_in_place_and_every_comment_is_kept(doc):
    d = doc("waypoint_tracking")
    d.set(("speed", "cruise_mps"), 0.8)
    d.set(("mission", "zero_on_exit"), False)
    d.set(("node", "vessel"), "Mako_02")
    text = d.render()
    assert text.count("#") == d.text.count("#")
    new = yaml.safe_load(text)
    assert new["speed"]["cruise_mps"] == 0.8 and new["mission"]["zero_on_exit"] is False and new["node"]["vessel"] == "Mako_02"
    other = yaml.safe_load(d.text)
    other["speed"]["cruise_mps"], other["mission"]["zero_on_exit"], other["node"]["vessel"] = 0.8, False, "Mako_02"
    assert new == other                                                      # nothing else changed


def test_add_remove_move_waypoints_rewrite_only_the_list(doc):
    d = doc("waypoint_tracking")
    n0 = len(d.items())                                                      # the shipped list changes (the user edits it): do not hard-code its length
    d.add_item(d.waypoint(2.0, 2.0))
    d.remove_item(0)
    d.add_item(d.waypoint(1.0, 1.0), 0)
    text = d.render()
    assert text.count("#") == d.text.count("#")
    wps = yaml.safe_load(text)["waypoints"]
    assert wps[0] == {"x": 1.0, "y": 1.0, "z": 3.0} and wps[-1] == {"x": 2.0, "y": 2.0, "z": 3.0} and len(wps) == n0 + 1
    assert d.move_item(0, 1) == 1 and d.items()[1]["x"] == 1.0 and d.move_item(0, -1) == 0          # moving up from the top does nothing


def test_save_makes_a_backup_and_reloads(doc, tmp_path):
    d = doc("waypoint_tracking")
    before = d.path.read_text()
    d.set(("speed", "cruise_mps"), 0.9)
    d.save()
    assert (tmp_path / "waypoint_tracking.yaml.bak").read_text() == before
    assert yaml.safe_load(d.path.read_text())["speed"]["cruise_mps"] == 0.9 and not d.dirty and d.orig["speed"]["cruise_mps"] == 0.9


def test_mission_legs_change_type_and_length(doc):
    d = doc("mission")
    d.add_item(LEG_TEMPLATES["hold"], 1)
    d.items()[0]["origin"] = [1.0, 6.0]
    d.items()[0]["spacing_m"] = 9.0
    text = d.render()
    assert text.count("#") == d.text.count("#")
    legs = yaml.safe_load(text)["legs"]
    assert legs[1] == {"type": "hold", "seconds": 10.0} and legs[0]["origin"] == [1.0, 6.0] and legs[0]["spacing_m"] == 9.0
    d.items()[2] = dict(LEG_TEMPLATES["goto"])                                # a leg that changes TYPE has other keys: the whole list is rewritten
    assert yaml.safe_load(d.render())["legs"][2]["type"] == "goto"


def test_a_change_without_a_place_in_the_text_is_refused_and_nothing_is_written(doc):
    d = doc("dock_test")
    before = d.path.read_text()
    d.data["brand_new_section"] = {"a": 1}
    with pytest.raises(ValueError):
        d.save()
    assert d.path.read_text() == before and not d.path.with_suffix(".yaml.bak").exists()


def test_emptying_the_list_is_refused(doc):
    d = doc("waypoint_tracking")
    d.data["waypoints"] = []
    with pytest.raises(ValueError):
        d.render()


def test_to_temp_never_touches_the_file(doc):
    d = doc("waypoint_tracking")
    d.set(("speed", "cruise_mps"), 0.7)
    p = d.to_temp()
    assert yaml.safe_load(p.read_text())["speed"]["cruise_mps"] == 0.7 and yaml.safe_load(d.path.read_text())["speed"]["cruise_mps"] == 1.0


def test_changes_lists_what_differs(doc):
    d = doc("waypoint_tracking")
    assert d.changes() == [] and not d.dirty
    d.set(("speed", "cruise_mps"), 0.8)
    assert [c[0] for c in d.changes()] == [("speed", "cruise_mps")]
    d.add_item(d.waypoint(2, 2))
    assert ("waypoints",) in [c[0] for c in d.changes()]


def test_validate_flags_the_turning_circle_and_the_dock(doc):
    d = doc("waypoint_tracking")
    assert d.validate() == []
    d.items()[1]["y"] = 3.0
    assert any("turning circle" in m for m in d.validate())
    d.revert()
    d.add_item({"x": 10.0, "y": 0.0, "z": 3.0})
    assert any("dock" in m for m in d.validate())
    m = doc("mission")
    assert m.validate() == []
    m.items()[0].pop("length_m")
    assert any("missing parameter" in x for x in m.validate())


def test_path_points_for_the_map(doc):
    assert len(doc("waypoint_tracking").path_points()) == 4
    pts = doc("mission").path_points()
    assert len(pts) > 30 and {p[3] for p in pts} >= {"orbit", "yoyo", "return_home"}


def test_every_shipped_yaml_survives_a_noop_render(doc):
    for n in ("waypoint_tracking", "mission", "dock_test"):
        d = doc(n)
        assert d.render() == d.text
