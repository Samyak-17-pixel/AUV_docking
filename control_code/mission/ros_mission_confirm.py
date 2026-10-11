#!/usr/bin/env python3
"""Fly a mission over REAL ROS on a PRIVATE domain: the offline fake vehicle (realistic odometry: 4.5 Hz, 0.25 s late, noisy, optional freezes) + the mission node.
Judged on the fake vehicle's ground truth (/Mako_01/sim/status truth = [x y z roll pitch yaw u], angles in rad).

  python3 ros_mission_confirm.py [--domain 77] [--preset quick|full] [--seconds 300] [--freezes]
quick = the lawnmower (2 lanes) then return home; full = the shipped mission.yaml. PASS = the node exits 0 (mission complete), the vehicle ends within 1.5 m of home,
never gets closer than 5 m to the dock (10, 0) and never leaves the geofence. Refuses domain 42. NOT a real-sim result.
"""

from __future__ import annotations

import argparse
import json
import math
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

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent
FAKE = CTRL / "sim_offline" / "fake_vehicle.py"
NODE = HERE / "mission.py"
QUICK_LEGS = [
    {"type": "lawnmower", "origin": [0.0, 5.0], "heading_deg": 90.0, "length_m": 10.0, "width_m": 8.0, "spacing_m": 8.0, "z": 3.0},
    {"type": "return_home"},
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", type=int, default=77)
    ap.add_argument("--preset", choices=("quick", "full"), default="quick")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--freezes", action="store_true", help="let the fake odometry freeze (real-sim like) instead of --odom-freeze-interval 0")
    ap.add_argument("--config", type=Path, default=None, help="a mission yaml to fly instead of the preset")
    a = ap.parse_args(argv)
    if a.domain == 42:
        print("refusing domain 42 (the real bridge)")
        return 2
    env = dict(os.environ, ROS_DOMAIN_ID=str(a.domain), ROS_LOCALHOST_ONLY="1")
    os.environ.update(env)
    cfg_path = a.config
    if cfg_path is None:
        cfg = yaml.safe_load((HERE / "mission.yaml").read_text())
        if a.preset == "quick":
            cfg["legs"] = QUICK_LEGS
        cfg["node"]["ros_domain_id"] = a.domain
        tmp = tempfile.NamedTemporaryFile("w", suffix="_mission.yaml", delete=False, dir=tmp_dir())
        yaml.safe_dump(cfg, tmp)
        tmp.close()
        cfg_path = Path(tmp.name)
    cfg = yaml.safe_load(cfg_path.read_text())
    gf = cfg.get("safety", {}).get("geofence") or {}
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init()
    node = Node("ros_mission_confirm")
    last = {}
    dbg = {}
    node.create_subscription(String, "/Mako_01/sim/status", lambda m: last.update(json.loads(m.data)), 10)
    node.create_subscription(String, "/Mako_01/ctrl_debug", lambda m: dbg.update(json.loads(m.data)), 10)
    fake_cmd = [sys.executable, str(FAKE), "--realistic", "--z", "3"]
    if not a.freezes:
        fake_cmd += ["--odom-freeze-interval", "0"]
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fake = subprocess.Popen(fake_cmd, env=env, **quiet)
    time.sleep(4.0)
    ctl = subprocess.Popen([sys.executable, str(NODE), "--config", str(cfg_path)], env=env, **quiet)
    t0 = time.monotonic()
    dmin, max_out, legs_seen, tr, aborted = 1e9, 0.0, [], None, ""
    try:
        while time.monotonic() - t0 < a.seconds and ctl.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            tr = last.get("truth") or tr
            if tr:
                dmin = min(dmin, math.hypot(tr[0] - 10.0, tr[1]))
                for ax, v in (("x", tr[0]), ("y", tr[1]), ("z", tr[2])):
                    lo, hi = gf.get(f"{ax}_min"), gf.get(f"{ax}_max")
                    if lo is not None:
                        max_out = max(max_out, lo - v)
                    if hi is not None:
                        max_out = max(max_out, v - hi)
            lg = dbg.get("leg")
            if lg and (not legs_seen or legs_seen[-1] != lg):
                legs_seen.append(lg)
            aborted = dbg.get("aborted") or aborted
        rc = ctl.poll()
    finally:
        for p in (ctl, fake):
            p.terminate()
        for p in (ctl, fake):
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
    took = time.monotonic() - t0
    end_home = math.hypot(tr[0], tr[1]) if tr else 99.0
    ok = rc == 0 and end_home < 1.5 and dmin > 5.0 and max_out <= 0.0 and not aborted
    print(f"{a.preset}: node exit {rc} after {took:.0f} s, legs {legs_seen}, ends {end_home:.2f} m from home, closest dock {dmin:.1f} m, "
          f"max geofence overshoot {max(max_out, 0):.2f} m, aborted '{aborted}' -> {'PASS' if ok else 'FAIL'}")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
