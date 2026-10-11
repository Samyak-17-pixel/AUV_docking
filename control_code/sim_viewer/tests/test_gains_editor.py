import yaml
import pytest

import gains_editor as G

CFG = {
    "node": {"name": "dock_test", "rate_hz": 20.0},
    "gains": {"pitch_nm_per_px": 0.008, "pitch_kd": 8.0, "flag": True, "count": 3},
    "waypoints": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
    "fins": [1.0, -1.0, 2.0],
    "text_only": "hello",
}


def test_leaves_lists_numeric_bool_and_number_lists_but_not_text():
    paths = {p for p, _ in G.leaves(CFG)}
    assert ("gains", "pitch_kd") in paths and ("gains", "flag") in paths and ("fins",) in paths
    assert ("waypoints", 1, "x") in paths                                   # dicts inside lists are reachable
    assert ("node", "name") not in paths and ("text_only",) not in paths


@pytest.mark.parametrize("text,orig,expected", [("2.5", 1.0, 2.5), ("4", 3, 4), ("true", False, True), ("off", True, False), ("1, 2, 3", [0.0, 0.0, 0.0], [1.0, 2.0, 3.0])])
def test_parse_value_follows_the_original_type(text, orig, expected):
    assert G.parse_value(text, orig) == expected and type(G.parse_value(text, orig)) is type(expected)


@pytest.mark.parametrize("text,orig", [("abc", 1.0), ("maybe", True), ("1, 2", [0.0, 0.0, 0.0])])
def test_parse_value_rejects_bad_text(text, orig):
    with pytest.raises(ValueError):
        G.parse_value(text, orig)


def test_apply_edits_changes_only_the_named_values_and_keeps_the_original():
    out = G.apply_edits(CFG, {("gains", "pitch_kd"): "12", ("waypoints", 0, "x"): "9.5"})
    assert out["gains"]["pitch_kd"] == 12.0 and out["waypoints"][0]["x"] == 9.5 and out["gains"]["pitch_nm_per_px"] == 0.008
    assert CFG["gains"]["pitch_kd"] == 8.0 and CFG["waypoints"][0]["x"] == 1.0
    assert G.changed_leaves(CFG, out) == [(("gains", "pitch_kd"), 8.0, 12.0), (("waypoints", 0, "x"), 1.0, 9.5)]


def test_temp_yaml_roundtrip_is_loadable_and_does_not_touch_the_source(tmp_path):
    src = tmp_path / "orig.yaml"
    src.write_text("# comment\ngains:\n  kd: 1.0\n")
    cfg = G.load_yaml(src)
    out = G.write_temp_yaml(G.apply_edits(cfg, {("gains", "kd"): "3"}), "unit", tmp_path)
    assert yaml.safe_load(out.read_text())["gains"]["kd"] == 3.0 and src.read_text() == "# comment\ngains:\n  kd: 1.0\n"


def test_real_controller_yamls_can_be_edited_end_to_end(tmp_path):
    from controls_panel import CONTROLLERS
    for name, spec in CONTROLLERS.items():
        cfg = G.load_yaml(spec["config"])
        lv = G.leaves(cfg)
        assert len(lv) > 5, name
        path, val = next((p, v) for p, v in lv if isinstance(v, float) and not isinstance(v, bool) and v > 0)
        edited = G.apply_edits(cfg, {path: repr(val * 2)})
        assert G.get_path(edited, path) == pytest.approx(val * 2)
        assert yaml.safe_load(G.write_temp_yaml(edited, name, tmp_path).read_text())


def test_dialog_edits_validates_and_resets():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dlg = G.make_dialog(CFG, "t")
    dlg.set_value(("gains", "pitch_kd"), "11")
    assert dlg.edited_config()["gains"]["pitch_kd"] == 11.0
    dlg.set_value(("gains", "pitch_kd"), "not a number")
    with pytest.raises(ValueError):
        dlg.edited_config()
    dlg._reset()
    assert dlg.edited_config() == CFG
    dlg.filter.setText("kd")
    shown = [p for p, it in dlg._items.items() if not it.isHidden()]
    assert ("gains", "pitch_kd") in shown and ("gains", "flag") not in shown


def test_changed_leaves_survives_structural_edits_from_the_config_tab():
    # a mission leg replaced by a leg of another type: the old keys no longer exist in the edited copy (this raised KeyError and broke the Controls tab label)
    orig = {"legs": [{"type": "lawnmower", "origin": [0.0, 5.0], "spacing_m": 8.0}], "speed": {"cruise_mps": 1.0}}
    edited = {"legs": [{"type": "hold", "seconds": 5.0}], "speed": {"cruise_mps": 0.8}}
    ch = G.changed_leaves(orig, edited)
    assert (("speed", "cruise_mps"), 1.0, 0.8) in ch
    assert any(p == ("legs", 0, "origin") and new is None for p, _, new in ch)
    assert G.changed_leaves(orig, orig) == []
