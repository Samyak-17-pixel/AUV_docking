"""In-process demo data source: the offline vehicle flies scripted manoeuvres towards the dock while the synthetic camera renders its view.

Needs NO ROS and no other terminals. It is only a visual demonstration of the viewer: the commands come from simple scripted loops, not from
the controllers under test. When the vehicle gets to 2.4 m from the dock it is put back at the start.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parents[1]
ROOT = _HERE.parent
sys.path[:0] = [str(_HERE)] + [str(_d) for _d in sorted(_HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders stay flat-importable
for p in (_HERE, ROOT / "common", ROOT / "sim_offline"):
    sys.path.insert(0, str(p))

from allocation import Allocator  # noqa: E402
from loops import HoldLoops  # noqa: E402
from state import State  # noqa: E402
from synthetic_camera import SyntheticCamera  # noqa: E402
from telemetry import Telemetry  # noqa: E402
from sim_control import SimControl  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402


class DemoSource(threading.Thread):
    START = (0.0, 0.8, 3.0)

    def __init__(self, telemetry: Telemetry, cfg: dict, time_scale: float = 1.0) -> None:
        super().__init__(daemon=True, name="demo_source")
        self.tel, self.cfg = telemetry, cfg
        self.ctl = SimControl(start_pos=self.START)
        self.ctl.time_scale = float(time_scale)
        self.sim_status: dict = {}
        self._t_sim = 0.0
        self._stop_evt = threading.Event()
        self.ready = threading.Event()
        self.error: Optional[str] = None

    def stop(self) -> None:
        self._stop_evt.set()

    # the viewer's panel talks to every source through these two members
    @property
    def sim_controls_available(self) -> bool:
        return True

    def send_sim_command(self, d: dict) -> bool:
        ok = self.ctl.handle(d)
        self.sim_status = self.ctl.status()
        return ok

    def run(self) -> None:
        try:
            gains = yaml.safe_load(open(ROOT / "dof_testing" / "dof_testing.yaml"))["gains"]
            loops = HoldLoops(gains, heave_ff_n=0.0)
            alloc = Allocator(rpm_cap=1800.0, fin_deg_cap=25.0)
            cam = SyntheticCamera(self.cfg)
            try:
                from synthetic_sidescan import SideScanSim
                from terrain import get_terrain
                from sidescan_node import sonar_config_from_yaml
                sonar = SideScanSim(get_terrain(self.cfg.get("terrain")), sonar_config_from_yaml(self.cfg))
            except Exception:                                                # noqa: BLE001
                sonar = None
            model = VehicleModel(pos=self.START)
            self.tel.source_name = "DEMO (scripted manoeuvres, in-process model)"
            self.sim_status = self.ctl.status()
            self.ready.set()
            dt = 0.05                                          # one control cycle = 50 ms of SIMULATED time, at every time scale
            budget, next_img = 0.0, 0.0
            while not self._stop_evt.is_set():
                reset = self.ctl.take_reset()
                if reset is not None:
                    model.reset(reset["pos"], reset["eul_deg"])
                    loops.reset()
                    self.tel.debug["restart"] = time.time()
                budget += self.ctl.sim_dt(dt)                  # 0 when paused; `time_scale` x dt otherwise; step budget while paused
                ran = False
                while budget >= dt - 1e-9:
                    budget -= dt
                    ran = True
                    st = State.from_model(model)
                    self._t_sim += dt
                    model_t = self._t_sim
                    heading = math.radians(18.0) * math.sin(2 * math.pi * model_t / 24.0)
                    depth_sp = 3.0 + 0.5 * math.sin(2 * math.pi * model_t / 20.0)
                    w = np.zeros(6)
                    w[0] = loops.speed(st, 0.6, dt)
                    w[2] = loops.heave(st, depth_sp, dt)
                    w[4] = loops.pitch(st, 0.0, dt)
                    w[5] = loops.yaw(st, heading, dt)
                    cmd = alloc.allocate(w, st.speed_u)
                    model.set_command(cmd)
                    for _ in range(5):
                        model.ext[:] = self.ctl.external_wrench(dt / 5)
                        model.step(dt / 5)
                    if model.pos[0] > 7.4:                       # near the standoff: start again
                        model.reset(self.START, (0.0, 0.0, 0.0))
                        loops.reset()
                        self.tel.debug["restart"] = time.time()
                self.tel.push_odom(model.pos, model.eul, model.nu)           # also while paused, so the display stays alive
                if ran:
                    names = list(cmd)
                    self.tel.push_cmd(names, [cmd[n] for n in names])
                now = time.monotonic()
                if now >= next_img:                                           # camera at most ~10 Hz of WALL time
                    self.tel.set_image(cam.encode_jpeg(cam.render(model.pos, np.degrees(model.eul))))
                    next_img = now + 0.1
                    if sonar is not None:                                    # side-scan pings (both sides) with the same pose, at the camera's wall-clock rate
                        for side in ("port", "starboard"):
                            r = sonar.ping(model.pos, model.eul, side)
                            self.tel.push_sonar(side, r.intensity, {"start_range_m": 0.0, "range_m": r.range_m, "gain_db": r.gain_db, "gain_index": r.gain_index, "altitude_m": r.altitude_m,
                                                                   "pos": model.pos.tolist(), "eul": model.eul.tolist(), "ping_hz": r.ping_hz, "sound_speed_mps": r.sound_speed_mps})
                self.sim_status = self.ctl.status()
                time.sleep(dt if not self.ctl.paused else 0.05)
        except Exception as exc:                                             # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
