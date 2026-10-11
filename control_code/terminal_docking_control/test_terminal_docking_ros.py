"""The whole docking stack over ROS in a private domain: fake vehicle (realistic odometry) + synthetic camera + REAL detector node + terminal_docking node.
Skipped without rclpy / the built interfaces package. Takes about a minute per run (real time)."""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent
FAKE = CTRL / "sim_offline" / "fake_vehicle.py"
CAMERA = CTRL / "sim_viewer" / "camera" / "camera_node.py"
DETECTOR = CTRL.parent / "dock_detection_algo" / "live_dock_lights.py"
NODE = HERE / "terminal_docking.py"


def test_full_stack_docks_from_an_offset_start(tmp_path):
    """Start 7 m out, 0.6 m off the axis, 8 deg crooked, realistic odometry (4.5 Hz, 0.25 s late, noisy), real detector. The node is stopped once it reports DOCKED."""
    os.environ["ROS_DOMAIN_ID"] = "73"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    env = dict(os.environ)
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cfg = yaml.safe_load(open(HERE / "terminal_docking.yaml"))
    cfg["logging"]["dir"] = str(tmp_path)
    cfg["node"]["ros_domain_id"] = 73
    cfgp = tmp_path / "td.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    procs = [subprocess.Popen([sys.executable, str(FAKE), "--realistic", "--odom-freeze-interval", "0", "--x", "3", "--y", "0.6", "--z", "3", "--yaw", "8"], env=env, **quiet),
             subprocess.Popen([sys.executable, str(CAMERA)], env=env, **quiet),
             subprocess.Popen([sys.executable, str(DETECTOR), "--no-gui"], env=env, **quiet)]
    node = None
    log_path = None
    try:
        time.sleep(4.0)
        node = subprocess.Popen([sys.executable, str(NODE), "--config", str(cfgp)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        import threading
        lines = []
        phase = {"docked_t": None}

        def reader():
            for ln in node.stdout:
                lines.append(ln)
                if "DOCKED" in ln and phase["docked_t"] is None:
                    phase["docked_t"] = time.time()
        th = threading.Thread(target=reader, daemon=True)
        th.start()
        t0 = time.time()
        while time.time() - t0 < 150 and node.poll() is None:
            time.sleep(0.5)
            if phase["docked_t"] and time.time() - phase["docked_t"] > 6.0:
                break
        node.send_signal(2)
        node.wait(15)
        th.join(3)
        text = "".join(lines)
    finally:
        if node and node.poll() is None:
            node.kill()
        for p in procs:
            p.send_signal(2)
        for p in procs:
            try:
                p.wait(8)
            except subprocess.TimeoutExpired:
                p.kill()
    logs = sorted(tmp_path.glob("terminal_docking_*.csv"))
    assert logs, text[-1500:]
    rows = list(csv.DictReader(open(logs[-1])))
    phases = [r["phase"] for r in rows]
    assert "DOCKED" in phases, f"never docked; phases seen: {sorted(set(phases))}\n{text[-1500:]}"
    assert "RETRY" not in phases and "SAFE_STOP" not in phases
    last = rows[-1]
    x_nose = float(last["x"]) + 0.66
    assert 10.1 < x_nose < 11.3, x_nose                                           # nose inside the funnel
    assert abs(float(last["y"])) < 0.2 and abs(float(last["z"]) - 3.0) < 0.2
    assert abs(float(last["yaw_deg"])) < 10.0
