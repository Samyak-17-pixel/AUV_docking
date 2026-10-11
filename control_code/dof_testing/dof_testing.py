#!/usr/bin/env python3
"""Per-DOF actuator test + tuning harness for Mako_01 (ROS 2).

  ./run_dof_testing.sh --dof heave --mode step      # open loop: verify allocation / signs
  ./run_dof_testing.sh --dof heave --mode hold      # closed loop: verify / tune gains

The engine (DofTest) is pure Python (no ROS) so tests can drive it against sim_offline/vehicle_model.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "common"))
from outdirs import out_dir  # noqa: E402

from allocation import Allocator  # noqa: E402
from live_gains import attach as attach_live_gains  # noqa: E402
from pose_filter import PoseFilterConfig, PoseTracker  # noqa: E402
from loops import HoldLoops  # noqa: E402
from state import State, wrap_pi  # noqa: E402

DEFAULT_CONFIG = _HERE / "dof_testing.yaml"
DOFS = ("surge", "heave", "pitch", "yaw", "roll", "sway")
FIN_DOFS = ("yaw", "roll", "sway")


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_allocator(cfg: dict) -> Allocator:
    lim = cfg["limits"]
    return Allocator(
        rpm_cap=lim["rpm_cap"], fin_deg_cap=lim["fin_deg_cap"],
        u_fin_min=lim["u_fin_min_mps"], u_fin_off=lim["u_fin_off_mps"], u_fin_full=lim["u_fin_full_mps"], small_force_n=lim.get("small_force_n", 0.0),
    )


class DofTest:
    """Phases: baseline -> (spinup, fin DOFs only) -> run -> done."""

    def __init__(self, cfg: dict, dof: str, mode: str, alloc: Optional[Allocator] = None) -> None:
        if dof not in DOFS:
            raise ValueError(f"dof must be one of {DOFS}")
        if mode not in ("step", "hold"):
            raise ValueError("mode must be step|hold")
        self.cfg, self.dof, self.mode = cfg, dof, mode
        self.alloc = alloc or make_allocator(cfg)
        self.loops = HoldLoops(cfg["gains"], heave_ff_n=cfg["background"]["heave_ff_n"])
        self.t = 0.0
        self.phase = "baseline"
        self._phase_t = 0.0
        self.done = False
        self.aborted: Optional[str] = None
        self.result: Dict[str, Any] = {}
        self.log: List[dict] = []
        self._start: Optional[State] = None
        self._baseline: List[float] = []
        self._run_meas: List[float] = []
        self._run_err: List[float] = []
        self._run_times: List[float] = []
        self._run_sig: List[tuple] = []   # full state per tick, to detect a frozen simulation
        self.uses_fins = dof in FIN_DOFS

    # ------------------------------------------------------------- measurements
    def _meas(self, st: State) -> float:
        """Measured rate used for the STEP direction check."""
        if self.dof == "surge":
            return float(st.nu[0])
        if self.dof == "heave":
            return float(st.vel_ned()[2])
        if self.dof == "pitch":
            return float(st.nu[4])
        if self.dof == "yaw":
            return float(st.nu[5])
        if self.dof == "roll":
            return float(st.nu[3])
        right = np.array([-math.sin(self._start.yaw), math.cos(self._start.yaw)])
        return float(st.vel_ned()[:2] @ right)

    def _axes(self):
        psi = self._start.yaw
        fwd = np.array([math.cos(psi), math.sin(psi)])
        right = np.array([-math.sin(psi), math.cos(psi)])
        return fwd, right

    def _hold_error(self, st: State) -> float:
        """Error in the unit of the tolerance block (m or deg), setpoint - measurement."""
        h, s0 = self.cfg["hold"], self._start
        fwd, right = self._axes()
        d = (st.pos[:2] - s0.pos[:2])
        if self.dof == "surge":
            return h["surge_ahead_m"] - float(d @ fwd)
        if self.dof == "heave":
            return (s0.depth + h["heave_delta_m"]) - st.depth
        if self.dof == "pitch":
            return h["pitch_deg"] - math.degrees(st.pitch)
        if self.dof == "yaw":
            return math.degrees(wrap_pi(s0.yaw + math.radians(h["yaw_delta_deg"]) - st.yaw))
        if self.dof == "roll":
            return h["roll_deg"] - math.degrees(st.roll)
        return h["sway_offset_m"] - float(d @ right)

    # ----------------------------------------------------------------- control
    def _safety(self, st: State) -> Optional[str]:
        s = self.cfg["safety"]
        if st.depth < s["min_depth_m"] and self.phase != "baseline":
            return f"too shallow ({st.depth:.2f} m < {s['min_depth_m']} m)"
        if st.depth > s["max_depth_m"]:
            return f"too deep ({st.depth:.2f} m)"
        if abs(math.degrees(st.pitch)) > s["max_pitch_deg"]:
            return f"pitch {math.degrees(st.pitch):.0f} deg"
        if abs(math.degrees(st.roll)) > s["max_roll_deg"]:
            return f"roll {math.degrees(st.roll):.0f} deg"
        return None

    def _run_wrench(self, st: State, dt: float) -> np.ndarray:
        cfg, s0, dof = self.cfg, self._start, self.dof
        w = np.zeros(6)  # X Y Z K M N
        bg = cfg["background"]["hold_depth_and_pitch"]
        fwd, right = self._axes()

        # background: keep the rest of the vehicle still
        if bg and dof != "heave":
            w[2] = self.loops.heave(st, s0.depth, dt)
        if bg and dof != "pitch":
            w[4] = self.loops.pitch(st, 0.0, dt)

        if self.uses_fins:
            w[0] = self.loops.speed(st, cfg["fin_dofs"]["cruise_speed_mps"], dt)

        if self.mode == "step":
            sp = cfg["step"]
            if dof == "surge":
                w[0] = sp["surge_n"]
            elif dof == "heave":
                w[2] = sp["heave_n"]
            elif dof == "pitch":
                w[4] = sp["pitch_nm"]
            elif dof in ("yaw", "sway"):
                w[5] = sp["yaw_nm"]
            elif dof == "roll":
                w[3] = sp["roll_nm"]
                w[5] = self.loops.yaw(st, s0.yaw, dt)      # keep heading while rolling
        else:
            h = cfg["hold"]
            d = st.pos[:2] - s0.pos[:2]
            if dof == "surge":
                w[0] = self.loops.surge(st, h["surge_ahead_m"] - float(d @ fwd), fwd, dt)
            elif dof == "heave":
                w[2] = self.loops.heave(st, s0.depth + h["heave_delta_m"], dt)
            elif dof == "pitch":
                w[4] = self.loops.pitch(st, math.radians(h["pitch_deg"]), dt)
            elif dof == "yaw":
                w[5] = self.loops.yaw(st, s0.yaw + math.radians(h["yaw_delta_deg"]), dt)
            elif dof == "roll":
                w[3] = self.loops.roll(st, math.radians(h["roll_deg"]), dt)
                w[5] = self.loops.yaw(st, s0.yaw, dt)
            elif dof == "sway":
                off = self.loops.cross_track(h["sway_offset_m"] - float(d @ right), dt)
                w[5] = self.loops.yaw(st, s0.yaw + off, dt)
        return w

    def update(self, st: State, dt: float):
        """One control tick -> (actuator dict {name: value}, wrench). Sets self.done when finished."""
        t_cfg = self.cfg["timing"]
        self.t += dt
        self._phase_t += dt
        w = np.zeros(6)
        why = self._safety(st)
        if why and not self.done:
            self.aborted, self.done = why, True
            self.result = {"verdict": "ABORT", "reason": why}
            return self._neutral(), w

        if self.phase == "baseline":
            self._baseline.append(self._meas_nostart(st))
            if self._phase_t >= t_cfg["baseline_s"]:
                self._start = State(st.pos.copy(), st.eul.copy(), st.nu.copy())
                self._baseline_mean = float(np.mean(self._baseline[-max(1, int(1.0 / dt)):]))
                self._next("spinup" if self.uses_fins else "run")
        elif self.phase == "spinup":
            w[0] = self.loops.speed(st, self.cfg["fin_dofs"]["cruise_speed_mps"], dt)
            w[2] = self.loops.heave(st, self._start.depth, dt)
            w[4] = self.loops.pitch(st, 0.0, dt)
            if st.speed_u >= 0.9 * self.cfg["fin_dofs"]["cruise_speed_mps"] and self._phase_t > 1.0:
                self._start = State(st.pos.copy(), st.eul.copy(), st.nu.copy())   # capture at test speed
                self._baseline_mean = self._meas(st)
                self._next("run")
            elif self._phase_t > self.cfg["fin_dofs"]["spinup_timeout_s"]:
                self.aborted, self.done = "speed spin-up timeout", True
                self.result = {"verdict": "ABORT", "reason": self.aborted}
                return self._neutral(), w
        elif self.phase == "run":
            dof = self.dof
            w = self._run_wrench(st, dt)
            self._run_times.append(self._phase_t)
            self._run_sig.append((st.depth, st.roll, st.pitch, st.yaw, *[float(v) for v in st.nu]))
            self._run_meas.append(self._meas(st))
            self._run_err.append(self._hold_error(st) if self.mode == "hold" else 0.0)
            dur = self.cfg["step"]["duration_s"][dof] if self.mode == "step" else t_cfg["hold_duration_s"]
            if self._phase_t >= dur:
                self._finish()
        out = self.alloc.allocate(w, st.speed_u) if not self.done else self._neutral()
        self.log.append({
            "t": round(self.t, 3), "phase": self.phase, "x": st.pos[0], "y": st.pos[1], "z": st.pos[2],
            "roll_deg": math.degrees(st.roll), "pitch_deg": math.degrees(st.pitch), "yaw_deg": math.degrees(st.yaw),
            "u": st.nu[0], "v": st.nu[1], "w": st.nu[2], "p": st.nu[3], "q": st.nu[4], "r": st.nu[5],
            "X": w[0], "Z": w[2], "K": w[3], "M": w[4], "N": w[5], **{k: round(v, 2) for k, v in out.items()},
        })
        return out, w

    def _meas_nostart(self, st: State) -> float:
        """Same quantity as _meas() but usable before the start state exists (sway: body lateral speed v)."""
        return {"surge": float(st.nu[0]), "heave": float(st.vel_ned()[2]), "pitch": float(st.nu[4]),
                "yaw": float(st.nu[5]), "roll": float(st.nu[3]), "sway": float(st.nu[1])}[self.dof]

    def _next(self, phase: str) -> None:
        self.phase, self._phase_t = phase, 0.0

    def _neutral(self) -> Dict[str, float]:
        return {n: 0.0 for n in self.alloc.th_ids + self.alloc.fin_ids}

    def debug(self) -> dict:
        """Setpoints of the current phase for the viewer's plots (published on ctrl_debug). Empty before the start state is captured."""
        s0 = self._start
        if s0 is None:
            return {"ctrl": "dof_testing", "mode": f"{self.dof} {self.mode}: {self.phase}"}
        h = self.cfg["hold"]
        out = {"ctrl": "dof_testing", "mode": f"{self.dof} {self.mode}: {self.phase}", "depth_sp": float(s0.depth), "pitch_sp_deg": 0.0, "yaw_sp_deg": math.degrees(s0.yaw)}
        if self.mode == "hold" and self.phase == "run":
            if self.dof == "heave":
                out["depth_sp"] = float(s0.depth + h["heave_delta_m"])
            elif self.dof == "pitch":
                out["pitch_sp_deg"] = float(h["pitch_deg"])
            elif self.dof == "yaw":
                out["yaw_sp_deg"] = math.degrees(s0.yaw) + float(h["yaw_delta_deg"])
        return out

    def _frozen(self) -> bool:
        """True if NOTHING in the state changed during the whole run (identical to 1e-9): the sim is paused or the
        odometry is frozen. Seen on the live sim 2026-10-01: a PASS/FAIL from such a run means nothing."""
        if len(self._run_sig) < 5:
            return False
        a = np.asarray(self._run_sig)
        return bool(np.all(np.ptp(a, axis=0) < 1e-9))

    def _finish(self) -> None:
        self.done = True
        n = len(self._run_meas)
        if self._frozen():
            self.result = {
                "verdict": "INVALID", "mode": self.mode, "dof": self.dof,
                "reason": "the state did not change at all during the run: the simulation is paused or the odometry is "
                          "frozen. No verdict is possible. Check that the sim is playing and run again.",
            }
            return
        if self.mode == "step":
            tail = self._run_meas[int(0.6 * n):]
            resp = float(np.mean(tail)) - self._baseline_mean
            expect = {"surge": 1, "heave": 1, "pitch": 1, "yaw": 1, "roll": 1, "sway": 1}[self.dof]
            need = float(self.cfg["step"]["min_response"][self.dof])
            ok = resp * expect >= need
            self.result = {
                "verdict": "PASS" if ok else "FAIL", "mode": "step", "dof": self.dof,
                "response_rate": resp, "required": need,
                "meaning": "positive demand should give a POSITIVE rate "
                           "(surge: u>0, heave: depth increasing=down, pitch: nose up, yaw: to starboard, "
                           "roll: starboard down, sway: moves right)",
            }
            if not ok and abs(resp) < 0.1 * need:
                self.result["hint"] = ("almost no response at all: this DOF may be LOCKED in the session (check active_dof in "
                                       "the vessel file: [surge, sway, heave, roll, pitch, yaw]), the thruster may be at its cap, "
                                       "or the sim may be intermittently freezing")
        else:
            tail = self._run_err[int((1.0 - self.cfg["timing"]["settle_fraction"]) * n):]
            err = float(np.mean(np.abs(tail)))
            tol = float(self.cfg["tolerance"][{"surge": "surge_m", "heave": "heave_m", "pitch": "pitch_deg",
                                               "yaw": "yaw_deg", "roll": "roll_deg", "sway": "sway_m"}[self.dof]])
            self.result = {"verdict": "PASS" if err <= tol else "FAIL", "mode": "hold", "dof": self.dof,
                           "final_abs_error": err, "tolerance": tol}

    def write_csv(self, directory: str) -> Optional[Path]:
        if not self.log:
            return None
        d = out_dir(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"dof_{self.dof}_{self.mode}.csv"
        fields: List[str] = []
        for r in self.log:
            fields += [k for k in r if k not in fields]
        with open(p, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, restval=0.0)
            wr.writeheader()
            wr.writerows(self.log)
        return p


def run_offline(cfg: dict, dof: str, mode: str, model, dt: float = 0.05, max_s: float = 200.0) -> DofTest:
    """Drive a DofTest against sim_offline.VehicleModel (no ROS). Used by tests and for quick checks."""
    test = DofTest(cfg, dof, mode)
    inner = int(round(dt / 0.01))
    while not test.done and test.t < max_s:
        out, _ = test.update(State.from_model(model), dt)
        model.set_command(out)
        for _ in range(inner):
            model.step(0.01)
    return test


# ------------------------------------------------------------------------ ROS
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Per-DOF test harness (mavsim)")
    ap.add_argument("--dof", required=True, choices=DOFS)
    ap.add_argument("--mode", required=True, choices=("step", "hold"))
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    domain = cfg["node"].get("ros_domain_id")
    if domain is not None and "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(domain)

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from interfaces.msg import Actuator
    from std_msgs.msg import String
    import json

    class Node_(Node):
        def __init__(self) -> None:
            super().__init__("dof_testing")
            self.test = DofTest(cfg, args.dof, args.mode)
            self.names = self.test.alloc.th_ids + self.test.alloc.fin_ids
            self.st: Optional[State] = None
            self.dt = 1.0 / float(cfg["node"]["rate_hz"])
            self._since_status = 0.0
            self.pub = self.create_publisher(Actuator, cfg["topics"]["actuator_cmd"], 10)
            self.dbg_pub = self.create_publisher(String, f"/{cfg['node'].get('vessel', 'Mako_01')}/ctrl_debug", 10)   # setpoints for the viewer's plots
            self._dbg_t = 0.0
            est = cfg.get("estimator", {})
            self.use_filter = bool(est.get("filter", True))
            self.tracker = PoseTracker(float(est.get("odom_latency_s", 0.25)), 1.0, PoseFilterConfig(**{k: v for k, v in est.items() if k not in ("filter", "odom_latency_s")}))
            self._t0 = time.monotonic()
            self.frozen_s = 0.0
            self.create_subscription(Odometry, cfg["topics"]["odometry"], self._on_odom, 10)
            attach_live_gains(self, cfg["node"].get("vessel", "Mako_01"), "dof_testing", self.test.loops, self.get_logger().info)
            self.create_timer(self.dt, self._tick)
            self.get_logger().info(f"dof_testing: dof={args.dof} mode={args.mode}; waiting for odometry")

        def _on_odom(self, msg) -> None:
            raw = State.from_odom(msg)
            self.tracker.push(raw, time.monotonic() - self._t0)
            if not self.use_filter:
                self.st = raw

        def publish(self, values: Dict[str, float]) -> None:
            m = Actuator()
            m.actuator_names = list(self.names)
            m.actuator_values = [float(values.get(n, 0.0)) for n in self.names]
            m.covariance = [0.0] * len(self.names)
            self._cmd_active = any(abs(float(values.get(n, 0.0))) > (50.0 if n.startswith("th_") else 1.0) for n in self.names)
            self.pub.publish(m)

        def _tick(self) -> None:
            if self.use_filter:                                   # smooth state from the slow, late, noisy odometry (common/pose_filter.py)
                now = time.monotonic() - self._t0
                f = self.tracker.at(now)
                if f is not None:
                    self.st = f
                if f is not None and self.tracker.frozen(now, expect_motion=getattr(self, "_cmd_active", True)):
                    # The odometry stopped changing (the real sim does this for 5-25 s). Controlling on stale data winds the integrators up and, with no righting moment,
                    # the pitch ran away to 77 deg in a test: command NEUTRAL, pause the test clock, and give up with INVALID if it lasts.
                    self.frozen_s += self.dt
                    self.publish({})
                    self.get_logger().warning("odometry frozen -> neutral", throttle_duration_sec=2.0)
                    if self.frozen_s > 3.0 and not self.test.done:
                        self.test.done = True
                        self.test.result = {"verdict": "INVALID", "mode": self.test.mode, "dof": self.test.dof,
                                            "reason": f"the odometry froze for more than 3 s during the test (total {self.frozen_s:.0f} s): no verdict is possible. Run again."}
                        raise SystemExit(0)
                    return
            if self.st is None:
                self.publish({})
                return
            out, w = self.test.update(self.st, self.dt)
            self._dbg_t += self.dt
            if self._dbg_t >= 0.5:
                self._dbg_t = 0.0
                self.dbg_pub.publish(String(data=json.dumps(self.test.debug())))
            self.publish(out)
            self._since_status += self.dt
            if self._since_status >= float(cfg["logging"]["status_period_s"]):
                self._since_status = 0.0
                self.get_logger().info(
                    f"[{self.test.phase}] t={self.test.t:.1f} z={self.st.depth:.2f} "
                    f"pitch={math.degrees(self.st.pitch):.1f} yaw={math.degrees(self.st.yaw):.1f} "
                    f"u={self.st.speed_u:.2f} | X={w[0]:.1f} Z={w[2]:.1f} K={w[3]:.2f} M={w[4]:.2f} N={w[5]:.2f}"
                )
            if self.test.done:
                self.publish({})
                raise SystemExit(0)

    rclpy.init()
    node = Node_()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.publish({})
        except Exception:
            pass
        t = node.test
        print("\n=== RESULT ===")
        print(t.result or {"verdict": "INTERRUPTED"})
        p = t.write_csv(cfg["logging"]["dir"])
        if p:
            print(f"log: {p}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
