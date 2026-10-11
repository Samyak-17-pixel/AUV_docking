import numpy as np
from PyQt5 import QtWidgets

from plots import LivePlots
from telemetry import Telemetry

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_setpoints_are_drawn_dashed_on_their_own_time_base():
    t = Telemetry()
    for i in range(20):
        t.push_odom([i * 0.1, 0, 3.0 + 0.01 * i], [0, 0.01 * i, 0.02 * i], [0.3, 0, 0, 0, 0, 0], t=float(i) * 0.5)
    t.push_ctrl({"depth_sp": 3.5, "pitch_sp_deg": 2.0, "yaw_sp_deg": 10.0}, t=2.0)
    t.push_ctrl({"depth_sp": 3.5, "pitch_sp_deg": 2.0, "yaw_sp_deg": 10.0}, t=6.0)
    plots = LivePlots(window_s=60.0)
    plots.update_plots(t.window(60.0))
    sp = plots.p_depth.lines["ctrl_depth_sp"]
    assert sp.get_linestyle() == "--" and len(sp.get_xdata()) == 2 and list(sp.get_ydata()) == [3.5, 3.5]
    assert len(plots.p_depth.lines["depth"].get_xdata()) == 20                          # the measured line keeps its own samples
    assert len(plots.p_att.lines["ctrl_yaw_sp_deg"].get_xdata()) == 2


def test_plots_work_with_no_data_and_with_missing_setpoints():
    plots = LivePlots()
    plots.update_plots({})
    t = Telemetry()
    t.push_odom([0, 0, 3], [0, 0, 0], [0] * 6, t=0.0)
    t.push_odom([1, 0, 3], [0, 0, 0], [0] * 6, t=1.0)
    plots.update_plots(t.window(60.0))                                                  # no ctrl_* keys at all: must not raise
    assert len(plots.p_depth.lines["ctrl_depth_sp"].get_xdata()) == 0


def test_mission_tab_shows_cross_track_speed_progress_and_the_plan():
    t = Telemetry()
    for i in range(20):
        t.push_odom([0.0, i * 0.5, 3.0], [0, 0, np.radians(90)], [0.9, 0, 0, 0, 0, 0], t=float(i) * 0.5)
    for k in range(5):
        t.push_ctrl({"ctrl": "mission", "cross_track_m": 0.2 * k, "heading_err_deg": 3.0, "u_sp": 1.0, "depth_err_m": 0.05, "progress": 0.1 * k, "eta_s": 100.0 - 10 * k}, t=2.0 * k)
    plots = LivePlots(window_s=60.0)
    plots.set_plan([(0.0, 0.0, 3.0, "a"), (0.0, 10.0, 3.0, "a"), (5.0, 10.0, 3.0, "b")])
    plots.setCurrentIndex(2)                                                              # the Mission tab
    plots.update_plots(t.window(60.0))
    assert list(plots.m_ct.lines["ctrl_cross_track_m"].get_ydata()) == [0.0, 0.2, 0.4, 0.6000000000000001, 0.8]
    assert len(plots.m_sp.lines["u"].get_xdata()) == 20 and list(plots.m_sp.lines["ctrl_u_sp"].get_ydata()) == [1.0] * 5
    assert abs(plots.m_pr.lines["ctrl_progress_pct"].get_ydata()[-1] - 40.0) < 1e-9
    assert list(plots.mx_plan.get_xdata()) == [0.0, 10.0, 10.0] and list(plots.mx_plan.get_ydata()) == [0.0, 0.0, 5.0]       # east on x, north on y
    assert len(plots.mx_path.get_xdata()) == 20
    plots.set_plan([])
    assert len(plots.mx_plan.get_xdata()) == 0
