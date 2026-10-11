#!/usr/bin/env python3
"""Confirm the dock_test heading-error fix (mode 'realign') over REAL ROS on a private domain: fake vehicle (realistic odometry) + synthetic camera +
the REAL detector node + the dock_test node. The vehicle starts near the standoff with a yaw error; the verdict uses the ground truth from /sim/status.

  python3 ros_standoff_confirm.py [--domain 71] [--yaws 15,-20] [--seconds 90] [--standoff 3.0] [--set speed.blind_max_forward_m=-1]
Close starts: at 3 m a yaw of about 24 deg or more puts the side lights out of view, the detector then asks for a forward 'search' surge, and the OLD dock_test (no
distance estimate before it has seen four lights) drove into the dock. Since 2026-10-10 `speed.blind_max_forward_m` limits that blind travel and the node backs away
(mode 'search_back'): `--yaws 30,-30 --standoff 3` passes; add `--set speed.blind_max_forward_m=-1` to see the old behaviour fail (closest 0.75 m).
PASS = the controller is in mode 'standoff' (dock centred, square, stopped) for at least half of the last 20 s, and the nose never gets closer than 2.0 m to the dock plane.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

CTRL = Path(__file__).resolve().parents[1]
FAKE = CTRL / "sim_offline" / "fake_vehicle.py"
CAMERA = CTRL / "sim_viewer" / "camera" / "camera_node.py"
DETECTOR = CTRL.parent / "dock_detection_algo" / "live_dock_lights.py"
NODE = CTRL / "dock_test" / "dock_test.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", type=int, default=71)
    ap.add_argument("--yaws", default="15,-20")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--standoff", type=float, default=3.0, help="start distance of the nose from the dock plane [m]")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=value", help="override a dock_test.yaml value for this run (repeatable), for example speed.blind_max_forward_m=-1 for the old behaviour")
    a = ap.parse_args(argv)
    if a.domain == 42:
        print("refusing domain 42 (the real bridge)")
        return 2
    env = dict(os.environ, ROS_DOMAIN_ID=str(a.domain), ROS_LOCALHOST_ONLY="1")
    os.environ.update(env)
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init()
    node = Node("ros_standoff_confirm")
    last = {}
    node.create_subscription(String, "/Mako_01/sim/status", lambda m: last.update(json.loads(m.data)), 10)
    modes = set()
    cur = {"mode": None}
    cfg_path = None
    if a.set:
        import tempfile
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
        from outdirs import tmp_dir

        import yaml
        cfg = yaml.safe_load((NODE.parent / "dock_test.yaml").read_text())
        for item in a.set:
            key, _, val = item.partition("=")
            d = cfg
            parts = key.split(".")
            for k in parts[:-1]:
                d = d[k]
            d[parts[-1]] = yaml.safe_load(val)
        tmp = tempfile.NamedTemporaryFile("w", suffix="_dock_test.yaml", delete=False, dir=tmp_dir())
        yaml.safe_dump(cfg, tmp)
        tmp.close()
        cfg_path = tmp.name

    def on_dbg(m):
        d = json.loads(m.data)
        if d.get("ctrl") == "dock_test":
            cur["mode"] = d.get("mode")
            modes.add(cur["mode"])

    node.create_subscription(String, "/Mako_01/ctrl_debug", on_dbg, 10)
    all_ok = True
    for yaw in [float(x) for x in a.yaws.split(",")]:
        x0 = 10.0 - 0.575 - a.standoff
        procs = [subprocess.Popen([sys.executable, str(FAKE), "--realistic", "--odom-freeze-interval", "0", "--x", f"{x0}", "--z", "3", "--yaw", f"{yaw}"], env=env, **quiet),
                 subprocess.Popen([sys.executable, str(CAMERA)], env=env, **quiet),
                 subprocess.Popen([sys.executable, str(DETECTOR), "--no-gui"], env=env, **quiet)]
        time.sleep(6.0)
        ctl = subprocess.Popen([sys.executable, str(NODE)] + (["--config", str(cfg_path)] if cfg_path else []), env=env, **quiet)
        t0 = time.monotonic()
        dmin, series = 1e9, []
        modes.clear()
        last.clear()
        try:
            while time.monotonic() - t0 < a.seconds:
                rclpy.spin_once(node, timeout_sec=0.2)
                tr = last.get("truth")                               # x y z roll pitch yaw u (angles in rad)
                if tr:
                    d = 10.0 - (tr[0] + 0.66)
                    dmin = min(dmin, d)
                    series.append((time.monotonic() - t0, math.degrees(tr[5]), d, cur["mode"]))
        finally:
            for p in (ctl, *procs):
                p.terminate()
            for p in (ctl, *procs):
                try:
                    p.wait(timeout=5)
                except Exception:
                    p.kill()
            time.sleep(1.5)
        if not series:
            print(f"yaw0 {yaw:+.0f}: no ground truth on /Mako_01/sim/status (keys: {sorted(last)})")
            all_ok = False
            continue
        if a.trace:
            print("   t[s]  yaw[deg]  dist[m]  mode")
            for r in series[:: max(1, len(series) // 30)]:
                print(f"   {r[0]:5.1f} {r[1]:8.1f} {r[2]:8.2f}  {r[3]}")
        late = [s for s in series if s[0] > a.seconds - 20]
        yaw_end = sum(s[1] for s in late) / max(len(late), 1)
        rest = sum(1 for s in late if s[3] == "standoff") / max(len(late), 1)
        ok = rest >= 0.5 and dmin > 2.0
        all_ok &= ok
        print(f"yaw0 {yaw:+5.0f} deg: at rest {100 * rest:3.0f} % of the last 20 s, final yaw {yaw_end:+5.1f} deg, closest {dmin:.2f} m, modes {sorted(m for m in modes if m)}  -> {'PASS' if ok else 'FAIL'}")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
