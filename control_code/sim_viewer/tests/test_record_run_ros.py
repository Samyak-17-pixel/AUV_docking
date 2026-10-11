"""record_run.py against a real fake_vehicle (+ camera_node) over ROS, then the recording goes through load -> compare -> calibrate. Skipped without rclpy / the built interfaces."""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

from compare import rms_table, simulate  # noqa: E402
from replay import load_run  # noqa: E402

SV = Path(__file__).resolve().parents[1]
FAKE = SV.parent / "sim_offline" / "fake_vehicle.py"


def _stop(p):
    p.send_signal(2)
    try:
        p.wait(8)
    except subprocess.TimeoutExpired:
        p.kill()


def test_recorder_then_replay_compare_calibrate(tmp_path):
    os.environ["ROS_DOMAIN_ID"] = "76"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    from interfaces.msg import Actuator
    from rclpy.node import Node

    env = dict(os.environ)
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs = [subprocess.Popen([sys.executable, str(FAKE), "--rate", "50"], env=env, **quiet),
             subprocess.Popen([sys.executable, str(SV / "camera" / "camera_node.py")], env=env, **quiet)]
    out = tmp_path / "run.csv"
    rec = None
    rclpy.init()
    node = Node("rec_test_pub")
    pub = node.create_publisher(Actuator, "/Mako_01/actuator_cmd", 10)
    try:
        time.sleep(2.0)
        rec = subprocess.Popen([sys.executable, str(SV / "data" / "record_run.py"), "--out", str(out), "--duration", "14", "--label", "ros_test", "--frames", "2", "--frames-every", "1.0"],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        while time.time() - t0 < 13.0:
            tt = time.time() - t0
            a = Actuator()
            a.actuator_names = ["th_01", "th_02", "th_03", "cs_04", "cs_06", "cs_07", "cs_08"]
            heave = -350.0 if 4 < tt < 7 else 0.0
            fin = 6.0 * np.sin(tt)
            a.actuator_values = [700.0 if 1 < tt < 9 else 300.0, heave, heave, fin, fin, -fin, -fin]
            a.covariance = [0.0] * 7
            pub.publish(a)
            time.sleep(0.05)
        text = rec.communicate(timeout=20)[0]
    finally:
        if rec and rec.poll() is None:
            rec.kill()
        for p in procs:
            _stop(p)
        node.destroy_node()
        rclpy.shutdown()
    assert rec.returncode == 0, text
    assert "saved" in text and str(out) in text

    run = load_run(out)
    assert run.fmt == "recorder" and run.meta["label"] == "ros_test" and run.meta["vessel"] == "Mako_01"
    rate = run.n / run.duration
    assert 20.0 < rate < 70.0, f"odometry rate {rate:.1f} Hz"
    assert run.cmd["th_01"].max() == pytest.approx(700.0) and run.cmd["th_02"].min() == pytest.approx(-350.0)
    assert run.pos[-1, 0] > run.pos[0, 0] + 1.0 and run.pos[:, 2].max() > run.pos[0, 2] + 0.01        # it surged and dived
    assert float(np.min(run.cmd_age)) < 0.2                                                           # commands were arriving while recording

    # the recording was made by the very model compare.py uses, so a segment-wise comparison must agree closely (timing jitter only)
    rows = {r["channel"]: r for r in rms_table(run, simulate(run, mode="segments", segment_s=3.0))}
    assert rows["u"]["rms"] < 0.08 and rows["depth"]["rms"] < 0.05 and rows["x"]["rms"] < 0.25, rows

    # frames: JPEGs + a pose log next to the CSV
    fdir = tmp_path / "run_frames"
    with open(fdir / "frames.csv") as f:
        fr = list(csv.DictReader(f))
    assert 1 <= len(fr) <= 2
    import cv2
    img = cv2.imread(str(fdir / fr[0]["file"]))
    assert img is not None and img.shape[:2] == (480, 640)
