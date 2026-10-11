#!/usr/bin/env python3
"""Confirm the tuned controllers over REAL ROS on a private domain: fake vehicle with realistic odometry + dof_testing (6 DOF x step/hold) + station_keeping.

  python3 ros_confirm.py [--domain 71] [--only heave:hold,yaw:step] [--freeze 0]
Prints one line per test. Takes about 10 minutes for everything (the fake vehicle runs in real time). Never touches domain 42.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CTRL = Path(__file__).resolve().parents[1]
FAKE = CTRL / "sim_offline" / "fake_vehicle.py"
DOF = CTRL / "dof_testing" / "dof_testing.py"
SK = CTRL / "station_keeping" / "station_keeping.py"
DOFS = ["heave", "surge", "pitch", "yaw", "roll", "sway"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", type=int, default=71)
    ap.add_argument("--only", default="")
    ap.add_argument("--freeze", type=float, default=0.0, help="mean seconds between odometry freezes (0 = none)")
    ap.add_argument("--no-sk", action="store_true")
    a = ap.parse_args(argv)
    if a.domain == 42:
        print("refusing domain 42 (the real bridge)")
        return 2
    env = dict(os.environ, ROS_DOMAIN_ID=str(a.domain), ROS_LOCALHOST_ONLY="1")
    tests = [(d, m) for m in ("step", "hold") for d in DOFS]
    if a.only:
        want = {tuple(x.split(":")) for x in a.only.split(",")}
        tests = [t for t in tests if t in want]
    fake = subprocess.Popen([sys.executable, str(FAKE), "--realistic", "--odom-freeze-interval", str(a.freeze), "--z", "3"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    try:
        time.sleep(3.0)
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        os.environ.update(env)
        rclpy.init()
        node = Node("confirm_reset")
        pub = node.create_publisher(String, "/Mako_01/sim/cmd", 10)

        def reset():
            for _ in range(5):
                pub.publish(String(data=json.dumps({"cmd": "reset", "pos": [0.0, 0.0, 3.0], "eul_deg": [0.0, 0.0, 0.0]})))
                time.sleep(0.1)
            time.sleep(2.0)

        for dof, mode in tests:
            reset()
            t0 = time.time()
            p = subprocess.run([sys.executable, str(DOF), "--config", str(CTRL / "dof_testing" / "dof_testing.yaml"), "--dof", dof, "--mode", mode], env=env, capture_output=True, text=True, timeout=300)
            m = re.search(r"=== RESULT ===\s*(\{.*\})", p.stdout, re.S)
            res = ast.literal_eval(m.group(1).splitlines()[0]) if m else {"verdict": "NO RESULT", "tail": p.stdout[-300:]}
            results.append((f"dof_testing {dof} {mode}", res))
            print(f"{results[-1][0]:28s} {res.get('verdict'):8s} {time.time() - t0:5.0f} s  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items() if k != "verdict"), flush=True)
        if not a.no_sk:
            reset()
            sk = subprocess.Popen([sys.executable, str(SK), "--config", str(CTRL / "station_keeping" / "station_keeping.yaml")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            time.sleep(45.0)
            sk.send_signal(2)
            out = sk.communicate(timeout=20)[0]
            last = [l for l in out.splitlines() if "depth_err" in l][-1:] or ["(no status line)"]
            print("station_keeping 45 s".ljust(28), last[0].split("]: ")[-1], flush=True)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        fake.send_signal(2)
        try:
            fake.wait(8)
        except subprocess.TimeoutExpired:
            fake.kill()
    bad = [n for n, r in results if r.get("verdict") != "PASS"]
    print(f"\n{len(results) - len(bad)}/{len(results)} dof tests PASS" + (f"; not PASS: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
