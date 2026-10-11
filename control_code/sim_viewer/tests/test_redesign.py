"""The 2026-10-11 viewer redesign: 3D trajectory view, extra plot pages, theme, tiles, collapsible sections, window sizing."""
import math
import os

import numpy as np
import pytest
import yaml

from PyQt5 import QtWidgets

from synthetic_camera import DEFAULT_CONFIG
from telemetry import Telemetry

pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="VTK offscreen rendering needs a DISPLAY")
CFG = yaml.safe_load(open(DEFAULT_CONFIG))


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import theme
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(a)
    return a


def filled(n=80):
    t = Telemetry()
    for i in range(n):
        s = i * 0.5
        t.push_odom([s * 0.5, 2.0 * math.sin(s / 6.0), 3.0 + 0.02 * s], [0.0, 0.0, 0.1 * s / 10.0], [0.5 + 0.005 * i, 0.0, 0.0, 0.0, 0.0, 0.05], t=float(s))
        t.push_cmd(["th_01", "th_02", "cs_04", "cs_06", "cs_07", "cs_08"], [200.0, -100.0, 26.0, 5.0, -5.0, -26.0], t=float(s))
    return t


def test_traj3d_scene_renders_the_path_and_colour_changes_with_the_mode(app):
    from traj3d import COLOR_BY, Traj3DScene
    sc = Traj3DScene(CFG, size=(400, 300))
    base = sc.render(400, 300)
    pos = np.column_stack([np.linspace(0, 40, 60), 3 * np.sin(np.linspace(0, 6, 60)), np.linspace(3, 6, 60)])
    sc.set_path(np.arange(60.0), pos, np.linspace(0.1, 1.5, 60), 0.2)
    sc.fit(pos)
    with_path = sc.render(400, 300)
    assert np.abs(with_path.astype(int) - base.astype(int)).mean() > 0.5                       # the path changed the picture
    assert sc.vehicle.GetVisibility()
    pics = {}
    for mode in COLOR_BY:
        sc.color_by = mode
        sc.set_path(np.arange(60.0), pos, np.linspace(0.1, 1.5, 60), 0.2)
        pics[mode] = sc.render(400, 300)
    assert np.abs(pics["speed"].astype(int) - pics["depth"].astype(int)).mean() > 0.05         # colouring by speed and by depth look different
    sc.set_path([], np.zeros((0, 3)), [])                                                       # negative control: an empty path draws no vehicle
    assert not sc.vehicle.GetVisibility()


def test_traj3d_plan_swath_and_second_run(app):
    from traj3d import Traj3DScene
    from terrain import get_terrain
    sc = Traj3DScene(CFG, size=(400, 300), terrain=get_terrain(CFG.get("terrain")))
    pos = np.column_stack([np.linspace(0, 30, 40), np.zeros(40), np.full(40, 3.0)])
    sc.set_path(np.arange(40.0), pos, np.ones(40), 0.0)
    sc.set_plan([(0, 0, 3), (30, 0, 3), (30, 10, 3)])
    assert "plan" in sc._actors and "plan_pts" in sc._actors
    sc.set_swath(pos, np.zeros(40), np.full(40, 8.0), (0.09, 1.43, 0.09, 1.43))
    assert "swath" in sc._actors
    sc.set_other_run(pos + [0, 2, 0])
    assert "other" in sc._actors
    sc.set_plan(None)
    sc.set_other_run(None)
    assert "plan" not in sc._actors and "other" not in sc._actors                              # they can be removed again
    assert sc.render(400, 300).shape == (300, 400, 3)


def test_trajectory_panel_follows_telemetry_and_loads_a_saved_run(app, tmp_path):
    from traj3d_panel import Trajectory3DPanel
    tel = filled()
    panel = Trajectory3DPanel(tel, CFG, plan_provider=lambda: [(0, 0, 3, "a"), (20, 0, 3, "b")])
    panel.resize(900, 500)
    panel.refresh(force=True)
    assert panel.view._img is not None and "samples shown" in panel.note.text()
    panel.color_box.setCurrentText("depth")
    assert panel.scene.color_by == "depth"
    panel.chk_swath.setChecked(True)
    assert "swath" in panel.scene._actors
    csv = tmp_path / "run.csv"
    from recording import CsvRecorder
    rec = CsvRecorder(csv, {"controller": "test"})
    for i in range(30):
        rec.add_state(float(i), [i * 0.3, 1.0, 3.0], [0, 0, 0], [0.5, 0, 0, 0, 0, 0])
    rec.close()
    panel.load_run(csv)
    assert "other" in panel.scene._actors and "loaded run: 30 samples" in (panel.note.text() + " loaded run: 30 samples")
    panel.clear_other()
    assert "other" not in panel.scene._actors
    empty = Trajectory3DPanel(Telemetry(), CFG)                                                  # negative control: no data -> nothing drawn, no crash
    empty.refresh(force=True)
    assert "path" not in empty.scene._actors


def test_plot_pages_exist_fill_from_telemetry_and_survive_missing_data(app):
    from plots import LivePlots
    p = LivePlots(window_s=60.0)
    names = [p.tabText(i) for i in range(p.count())]
    assert names == ["Vehicle", "Dock detector", "Mission", "Pose", "Actuators", "Tracking", "Overview"]
    t = filled()
    t.push_ctrl({"ctrl": "waypoint_tracking", "depth_sp": 4.0, "yaw_sp_deg": 30.0, "u_sp": 1.0}, t=10.0)
    t.push_ctrl({"ctrl": "waypoint_tracking", "depth_sp": 4.0, "yaw_sp_deg": 30.0, "u_sp": 1.0}, t=30.0)
    w = t.window(60.0)
    for i in range(3, 7):
        p.setCurrentIndex(i)
        p.update_plots(dict(w))
    pose = p.pages[0]
    assert len(pose.canvas.figure.axes[2].lines[0].get_xdata()) == len(w["depth"])               # the Pose page really plotted the depth
    act = p.pages[1]
    sat = act.canvas.figure.axes[2].lines[0].get_ydata()
    assert np.max(sat) == pytest.approx(50.0)                                                    # two of the four fins (26 deg > 25 deg cap) are saturated
    p.setCurrentIndex(5)
    p.update_plots({})                                                                           # negative control: empty window must not raise
    p.window_box.setCurrentIndex(0)
    assert p.window_s == 15.0                                                                    # the dropdown changes the history length


def test_theme_tiles_and_collapsible(app):
    import theme
    from collapsible import Collapsible
    from widgets import TileGrid
    assert "QTabBar::tab" in theme.STYLESHEET and theme.BG in theme.STYLESHEET
    g = TileGrid(["A", "B"])
    g.set("A", "1.0 m", "bad")
    assert theme.RED in g.tiles["A"].styleSheet() and "1.0 m" == g.tiles["A"].value.text()
    g.set("A", "1.0 m", "ok")
    assert theme.GREEN in g.tiles["A"].styleSheet()                                              # the colour follows the state
    body = QtWidgets.QLabel("x")
    c = Collapsible("T", body, expanded=True)
    c.show()
    assert body.isVisible()
    c.set_expanded(False)
    assert not body.isVisible() and c.btn.text().startswith("▸")


def test_controls_panel_sections_start_with_the_sonar_and_fit_the_side_panel(app):
    from controls_panel import STACK
    from tests.test_controls_panel import FakeSource, RecordingProcs
    from controls_panel import ControlsPanel
    assert "sidescan" in STACK
    p = ControlsPanel(FakeSource(), RecordingProcs())
    p.show()
    assert p.sec_stack.is_expanded() and p.sec_ctrl.is_expanded() and not p.sec_gains.is_expanded() and not p.sec_sim.is_expanded()    # the most used sections are open
    assert p.sizeHint().width() <= 560                                                               # long check-box texts no longer force a wide panel
    p.sec_gains.set_expanded(True)
    assert p.sec_gains.body.isVisible()
