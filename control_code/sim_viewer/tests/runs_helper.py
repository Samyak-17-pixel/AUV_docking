"""Builds recorder-format runs by driving the offline VehicleModel with a scripted command profile (the 'recorded' truth for the replay/compare/calibrate tests)."""
import numpy as np

from compare import build_model
from recording import CsvRecorder


def profile(tt: float, heave_only: bool = False) -> dict:
    heave = -400.0 * (5 < tt < 8) + 300.0 * (18 < tt < 20)
    cmd = {"th_01": 0.0 if heave_only else 900.0 * (1.0 if 2 < tt < 14 else 0.5 if tt >= 14 else 0.0), "th_02": heave, "th_03": -400.0 * (5 < tt < 8) - 300.0 * (18 < tt < 20)}
    s = 0.0 if heave_only else 8.0 * np.sin(tt / 3.0)
    cmd.update({"cs_04": s, "cs_06": s, "cs_07": -s, "cs_08": -s})
    return cmd


def write_model_run(path, overrides=None, T=30.0, row_dt=0.1, sub=0.01, noise=0.0, seed=0, heave_only=False, start_yaw_deg=0.0):
    m = build_model(overrides)
    m.reset((0.0, 0.0, 3.0), (0.0, 0.0, start_yaw_deg))
    rec = CsvRecorder(path, {"label": "synthetic"})
    rng = np.random.default_rng(seed)
    for i in range(int(round(T / row_dt)) + 1):
        tt = i * row_dt
        cmd = profile(tt, heave_only)
        rec.set_cmd(list(cmd), list(cmd.values()), tt)
        pos = m.pos + (rng.normal(0, noise, 3) if noise else 0.0)
        nu = m.nu + (rng.normal(0, noise, 6) if noise else 0.0)
        rec.add_state(tt, pos, m.eul, nu)
        m.set_command(cmd)
        for _ in range(int(round(row_dt / sub))):
            m.step(sub)
    rec.close()
    return path
