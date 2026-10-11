#!/usr/bin/env python3
"""Confirm terminal_docking over REAL ROS on a private domain: fake vehicle (realistic odometry) + synthetic camera + the REAL detector node + the controller node,
judged live with the same failure list as the offline evaluation (ground truth from the fake vehicle's /sim/status).

  python3 ros_dock_confirm.py [--domain 70] [--starts 7,0,0:9,1,15:...] [--freeze 0]
Each start is range[m],lateral[m],heading[deg]. Default: a spread of 10 starts. One run takes 30-60 s real time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
from outdirs import tmp_dir  # noqa: E402
import time
from pathlib import Path

import yaml

CTRL = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_d) for _d in [CTRL / "sim_viewer", *sorted((CTRL / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
sys.path.insert(0, str(CTRL / "common"))
from scenario_judge import LiveJudge  # noqa: E402

FAKE = CTRL / "sim_offline" / "fake_vehicle.py"
CAMERA = CTRL / "sim_viewer" / "camera" / "camera_node.py"
DETECTOR = CTRL.parent / "dock_detection_algo" / "live_dock_lights.py"
NODE = CTRL / "terminal_docking_control" / "terminal_docking.py"
DEFAULT_STARTS = "9,0,0:9,1.5,-20:9,-1.5,20:8,1,0:8,-1,15:7,0,0:7,0.8,-10:7,-0.8,10:6,0,-15:6,0.5,10"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", type=int, default=70)
    ap.add_argument("--starts", default=DEFAULT_STARTS)
    ap.add_argument("--freeze", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=150.0)
    a = ap.parse_args(argv)
    if a.domain == 42:
        print("refusing domain 42 (the real bridge)")
        return 2
    env = dict(os.environ, ROS_DOMAIN_ID=str(a.domain), ROS_LOCALHOST_ONLY="1")
    os.environ.update(env)
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp = Path(tempfile.mkdtemp(prefix="dock_confirm_", dir=tmp_dir()))
    cfg = yaml.safe_load(open(CTRL / "terminal_docking_control" / "terminal_docking.yaml"))
    cfg["logging"]["dir"] = str(tmp)
    cfg["node"]["ros_domain_id"] = a.domain
    cfgp = tmp / "td.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    procs = [subprocess.Popen([sys.executable, str(FAKE), "--realistic", "--odom-freeze-interval", str(a.freeze), "--x", "3", "--z", "3"], env=env, **quiet),
             subprocess.Popen([sys.executable, str(CAMERA)], env=env, **quiet),
             subprocess.Popen([sys.executable, str(DETECTOR), "--no-gui"], env=env, **quiet)]
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    results = []
    try:
        time.sleep(4.0)
        rclpy.init()
        node = Node("dock_confirm")
        state = {"truth": None, "ctrl": {}, "stamp": 0}
        node.create_subscription(String, "/Mako_01/sim/status", lambda m: state.update(truth=json.loads(m.data).get("truth"), stamp=state["stamp"] + 1), 10)
        node.create_subscription(String, "/Mako_01/ctrl_debug", lambda m: state.update(ctrl=json.loads(m.data)), 10)
        pub = node.create_publisher(String, "/Mako_01/sim/cmd", 10)
        for spec in a.starts.split(":"):
            rng, lat, head = (float(x) for x in spec.split(","))
            for _ in range(5):
                pub.publish(String(data=json.dumps({"cmd": "reset", "pos": [10.0 - rng, lat, 3.0], "eul_deg": [0, 0, head]})))
                rclpy.spin_once(node, timeout_sec=0.1)
            t0 = time.time()
            while time.time() - t0 < 2.5:
                rclpy.spin_once(node, timeout_sec=0.1)
            judge = LiveJudge(timeout_s=a.timeout)
            nodep = subprocess.Popen([sys.executable, str(NODE), "--config", str(cfgp)], env=env, **quiet)
            t0, last = time.time(), -1
            try:
                while time.time() - t0 < a.timeout + 5 and judge.verdict not in ("PASS", "FAIL") and nodep.poll() is None:
                    rclpy.spin_once(node, timeout_sec=0.1)
                    if state["truth"] and state["stamp"] != last:
                        last = state["stamp"]
                        tr = state["truth"]
                        judge.feed(time.time(), tr[:3], tr[3:6], tr[6], str(state["ctrl"].get("mode", "")), state["ctrl"].get("retries") or 0)
            finally:
                nodep.send_signal(2)
                try:
                    nodep.wait(10)
                except subprocess.TimeoutExpired:
                    nodep.kill()
            results.append((spec, judge.verdict, judge))
            print(f"start range {rng:4.1f} lat {lat:+5.2f} head {head:+5.0f}: {judge.verdict:7s} {time.time() - t0:5.0f} s | " + (" ; ".join(judge.failures) or ("entry lat %+.2f vert %+.2f head %+.1f spd %.2f clearance %.2f" % (judge.cross.get("lat", 0), judge.cross.get("vert", 0), judge.cross.get("heading_deg", 0), judge.cross.get("speed", 0), judge.min_clear))), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        for p in procs:
            p.send_signal(2)
        for p in procs:
            try:
                p.wait(8)
            except subprocess.TimeoutExpired:
                p.kill()
    ok = sum(1 for _, v, _ in results if v == "PASS")
    print(f"\n{ok}/{len(results)} first-try passes over ROS")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
