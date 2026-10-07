"""Sign checks for the standoff wrench. No ROS.

    python3 test_dock_test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dock_test_core import DockTestCore, DockView, VehicleSnap

CFG = yaml.safe_load((Path(__file__).resolve().parent / "dock_test.yaml").read_text())
FF = float(CFG["feedforward"]["heave_n"])


def _core() -> DockTestCore:
    return DockTestCore(CFG)


def _still() -> VehicleSnap:
    return VehicleSnap()


def test_positive_elevation_dives():
    view = DockView(fresh=True, valid=True, elevation_valid=True, elevation_rad=0.1)
    wrench, status = _core().update(view, _still())
    assert wrench[2] > FF
    assert status["mode"] == "standoff"
    assert wrench[0] == 0.0


def test_dock_below_pitches_nose_down():
    view = DockView(fresh=True, valid=True, elevation_valid=True, error_y_px=40.0)
    wrench, _ = _core().update(view, _still())
    assert wrench[4] < 0.0


def test_dock_right_yaws_right():
    view = DockView(fresh=True, valid=True, elevation_valid=True, error_x_px=40.0)
    wrench, status = _core().update(view, _still())
    assert wrench[5] > 0.0
    assert wrench[0] > 0.0
    assert status["mode"] == "creep"


def test_negative_lateral_yaws_left():
    view = DockView(fresh=True, valid=True, elevation_valid=True, lateral_px=-20.0)
    wrench, _ = _core().update(view, _still())
    assert wrench[5] < 0.0


def test_squared_pose_has_zero_surge():
    view = DockView(
        fresh=True, valid=True, elevation_valid=True,
        elevation_rad=0.005, error_x_px=3.0, error_y_px=-4.0, lateral_px=2.0,
        radius_px=80.0,
    )
    wrench, status = _core().update(view, _still())
    assert wrench[0] == 0.0
    assert status["mode"] == "standoff"


def test_too_close_reverses():
    view = DockView(fresh=True, valid=True, elevation_valid=True, radius_px=250.0)
    wrench, status = _core().update(view, _still())
    assert wrench[0] < 0.0
    assert status["mode"] == "backup"


def test_partial_detection_does_not_heave_on_elevation():
    view = DockView(
        fresh=True, valid=False, elevation_valid=True, elevation_rad=0.4,
        search_surge_norm=-1.0, search_yaw_norm=0.0, search_pitch_norm=1.0,
    )
    wrench, status = _core().update(view, _still())
    assert status["mode"] == "search"
    assert abs(wrench[2] - FF) < 1e-6
    assert wrench[0] < 0.0
    assert wrench[4] < 0.0


def test_stale_message_is_neutral():
    wrench, status = _core().update(DockView(fresh=False, valid=True, elevation_rad=0.5), _still())
    assert status["mode"] == "stale"
    assert wrench.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def main() -> None:
    tests = [
        test_positive_elevation_dives,
        test_dock_below_pitches_nose_down,
        test_dock_right_yaws_right,
        test_negative_lateral_yaws_left,
        test_squared_pose_has_zero_surge,
        test_too_close_reverses,
        test_partial_detection_does_not_heave_on_elevation,
        test_stale_message_is_neutral,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
