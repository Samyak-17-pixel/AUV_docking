#!/usr/bin/env python3
"""Measure and tune the dock-light detector on synthetic frames. No GUI; needs the workspace sourced (DockAlign message).

  python3 tune_detection.py report                    # how the shipped dock_detection.yaml scores (recall by range, false locks)
  python3 tune_detection.py tune --rounds 3 --out tuned.yaml     # coordinate search on a training split, checked on a hold-out split

Dataset: random poses with the whole dock ring inside the image (range 2.5-10 m, bearing, pitch, roll, depth offset) x random 'looks' of the lights (brightness,
glow size, noise, water clarity, how dim the side lights are). The looks are deliberately varied: the real renderer is not available, so a setting is only trusted if it
works across many looks, not on one. Frames go through JPEG like the real camera topic.
Per frame: GOOD = valid, four lights, every light within `tol_px` of its true pixel position (labels right). BAD LOCK = valid but a light is wrong: the dangerous case,
a controller would steer on it. Otherwise MISS (not valid). Tuning maximises GOOD and punishes BAD LOCK heavily.
This is SYNTHETIC-ONLY evidence: re-check on real camera_03 frames (control_code/sim_viewer/data/record_run.py --frames) before trusting the numbers.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_SV = _HERE.parent / "control_code" / "sim_viewer"
sys.path[:0] = [str(_d) for _d in [_SV, *sorted(_SV.iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]
for p in (_HERE, _HERE.parent / "control_code" / "common"):
    sys.path.insert(0, str(p))

from dock_detector import DockDetector  # noqa: E402
from dock_detection_config import DEFAULT_CONFIG, load_config  # noqa: E402
from synthetic_camera import SyntheticCamera, load_config as load_cam_cfg  # noqa: E402

TOL_PX = 6.0                   # a light counts as correct within this many pixels (6 px = 0.8 deg of bearing; 5 cm sideways at 3 m, 12 cm at 8 m)
BANDS = [(2.5, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.5)]
DOCK = np.array([10.0, 0.0, 3.0])
MURKY_SHARE = 0.3                # fraction of the frames made with a bright, noisy background (set 0 for clear water only)
NAMES = ("top", "bottom", "right", "left")


def sample_look(rng: np.random.Generator, base: dict) -> dict:
    lk = copy.deepcopy(base)
    lk["core_gain"] = float(base["core_gain"]) * float(np.exp(rng.uniform(math.log(0.45), math.log(2.2))))
    lk["bloom_sigma_px"] = float(base["bloom_sigma_px"]) * float(rng.uniform(0.6, 1.8))
    lk["bloom_sigma_gain"] = float(base["bloom_sigma_gain"]) * float(rng.uniform(0.6, 1.8))
    lk["bloom_amplitude"] = float(np.clip(float(base["bloom_amplitude"]) * rng.uniform(0.6, 1.4), 0.2, 1.0))
    lk["water_attenuation_per_m"] = float(rng.uniform(0.02, 0.16))
    lk["noise_sigma"] = float(rng.uniform(0.004, 0.04))
    lk["light_radius_m"] = float(base["light_radius_m"]) * float(rng.uniform(0.7, 1.5))
    if rng.random() < MURKY_SHARE:                                          # murky water: a much brighter, noisier background (the real water colour is unknown)
        lk["background_bgr"] = [float(min(0.5, c * rng.uniform(2.0, 5.0))) for c in base["background_bgr"]]
        lk["noise_sigma"] = float(rng.uniform(0.025, 0.06))
    return lk


def make_dataset(n: int, seed: int = 11) -> List[dict]:
    """Frames as JPEG bytes + truth. Poses are rejected unless the 4 ring lights are all inside the image with a 12 px margin."""
    rng = np.random.default_rng(seed)
    cfg0 = load_cam_cfg()
    out: List[dict] = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        rng_range = float(np.exp(rng.uniform(math.log(2.5), math.log(10.5))))
        lat = float(rng.uniform(-2.2, 2.2))
        head = float(rng.uniform(-30, 30))
        pitch, roll = float(rng.uniform(-10, 10)), float(rng.uniform(-12, 12))
        dz = float(rng.uniform(-0.6, 0.6))
        pos = np.array([DOCK[0] - rng_range, lat, DOCK[2] + dz])
        eul = np.array([roll, pitch, head])
        cfg = copy.deepcopy(cfg0)
        cfg["look"] = sample_look(rng, cfg0["look"])
        cfg["look"]["seed"] = int(rng.integers(0, 10 ** 6))
        side = float(np.exp(rng.uniform(math.log(0.4), math.log(4.0))))          # the 2000 lm side lights: 0.4x .. 4x what the vessel file says
        for L in cfg["dock"]["ring_lights"]:
            if L["name"] in ("left", "right"):
                L["lumens"] = float(L["lumens"]) * side
        cam = SyntheticCamera(cfg)
        pts = {}
        ok = True
        for L in cam.lights:
            if not L["ring"]:
                continue
            pr = cam.project(pos, eul, L["p"])
            if pr is None or not (12 < pr[0] < cam.W - 12 and 12 < pr[1] < cam.H - 12):
                ok = False
                break
            pts[L["name"]] = pr
        if not ok:
            continue
        # truth in the roll-levelled image the detector works on (rotate about the image centre by +roll, like unrotate_cores)
        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        lev = {}
        for k, (u, v, _) in pts.items():
            dx, dy = u - cam.W / 2, v - cam.H / 2
            lev[k] = (cam.W / 2 + c * dx - s * dy, cam.H / 2 + s * dx + c * dy)
        side_pts = sorted([lev["right"], lev["left"]], key=lambda q: q[0])
        truth = np.array([lev["top"], lev["bottom"], side_pts[1], side_pts[0]])     # the detector labels the side lights by IMAGE position
        jpeg = cam.encode_jpeg(cam.render(pos, eul))
        out.append({"jpeg": jpeg, "truth": truth, "range": rng_range, "roll": math.radians(roll), "pitch": math.radians(pitch), "side": side,
                    "gain": cfg["look"]["core_gain"], "noise": cfg["look"]["noise_sigma"]})
    return out


_DET: Optional[DockDetector] = None
_OVER: dict = {}
_DATA: List[dict] = []


def _init_worker(data, overrides, cfg_path):
    global _DET, _DATA
    import cv2
    cv2.setNumThreads(1)
    _DATA = data
    cfg = load_config(cfg_path)
    _DET = DockDetector(cfg=cfg, overrides=overrides)


def _score_idx(i: int) -> Tuple[str, float]:
    import cv2
    d = _DATA[i]
    bgr = cv2.imdecode(np.frombuffer(d["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    m = _DET.process(bgr, d["roll"], d["pitch"], 0.0).msg
    if not m.valid:
        return "miss", float("nan")
    pix = np.array([[m.top.x, m.top.y], [m.bottom.x, m.bottom.y], [m.right.x, m.right.y], [m.left.x, m.left.y]])
    err = np.linalg.norm(pix - d["truth"], axis=1)
    if m.num_lights == 4 and float(err.max()) <= TOL_PX:
        return "good", float(err.mean())
    return "bad", float(err.max())


def evaluate(data: List[dict], overrides: Optional[dict] = None, cfg_path: Path = DEFAULT_CONFIG, workers: int = 24) -> dict:
    # 'spawn', not fork: forking a process that already has Qt/VTK/ROS/OpenCV threads (e.g. inside a pytest run) can deadlock the workers
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(data, overrides or {}, cfg_path)) as pool:
        res = pool.map(_score_idx, range(len(data)), chunksize=4)
    kinds = np.array([r[0] for r in res])
    out = {"n": len(data), "good": float(np.mean(kinds == "good")), "bad": float(np.mean(kinds == "bad")), "miss": float(np.mean(kinds == "miss")),
           "err_px": float(np.nanmean([r[1] for r in res if r[0] == "good"])) if np.any(kinds == "good") else float("nan"), "bands": {}}
    rng_ = np.array([d["range"] for d in data])
    for lo, hi in BANDS:
        m = (rng_ >= lo) & (rng_ < hi)
        if m.any():
            out["bands"][f"{lo:g}-{hi:g} m"] = {"n": int(m.sum()), "good": float(np.mean(kinds[m] == "good")), "bad": float(np.mean(kinds[m] == "bad"))}
    out["kinds"] = kinds
    return out


def cost(r: dict) -> float:
    """A bad lock costs 8x a miss; good frames count for nothing; a little pull towards low pixel error."""
    return 100.0 * (r["miss"] + 8.0 * r["bad"]) + 2.0 * (0.0 if math.isnan(r["err_px"]) else r["err_px"])


SPACE: Dict[str, Tuple[float, float, bool]] = {          # name -> (low, high, integer)
    "v_thresh": (110, 235, True), "close_k": (3, 17, True), "open_k": (1, 7, True), "min_area": (5, 80, False),
    "peak_sep": (8, 60, True), "core_pct": (82, 98, True), "tight_floor": (140, 235, False), "response_min": (0.01, 0.30, False),
    "peak_abs_v_floor": (80, 210, False), "peak_abs_v_frac": (0.4, 0.95, False), "dog_sigma_small": (0.8, 3.5, False), "dog_sigma_large": (2.5, 10.0, False),
    "tight_erode_k": (1, 9, True), "search_dilate_k": (3, 15, True),
}
SECTION = {"v_thresh": "mask", "close_k": "mask", "open_k": "mask", "min_area": "mask"}


def current_values(cfg: dict) -> Dict[str, float]:
    return {k: float(cfg["mask" if SECTION.get(k) == "mask" else "peaks"].get(k, (lo + hi) / 2)) for k, (lo, hi, _) in SPACE.items()}


def tune(train: List[dict], hold: List[dict], rounds: int, workers: int, log=print) -> Dict[str, float]:
    cfg = load_config()
    vals = current_values(cfg)
    ov = {k: (int(v) if SPACE[k][2] else v) for k, v in vals.items()}
    best = evaluate(train, ov, workers=workers)
    log(f"start: good {best['good']:.3f} bad {best['bad']:.3f} miss {best['miss']:.3f} cost {cost(best):.1f}")
    step = 0.25
    for r in range(rounds):
        improved = False
        for k, (lo, hi, isint) in SPACE.items():
            for sign in (+1, -1):
                v = float(np.clip(vals[k] * (1 + sign * step) if vals[k] != 0 else sign * step, lo, hi))
                v = float(round(v)) if isint else v
                if v == vals[k]:
                    continue
                cand = dict(ov)
                cand[k] = int(v) if isint else v
                res = evaluate(train, cand, workers=workers)
                if cost(res) < cost(best) - 1e-6:
                    best, ov, vals[k], improved = res, cand, v, True
                    log(f"  round {r} {k} -> {cand[k]}: good {best['good']:.3f} bad {best['bad']:.3f} miss {best['miss']:.3f} cost {cost(best):.1f}")
        if not improved:
            step *= 0.5
    h = evaluate(hold, ov, workers=workers)
    h0 = evaluate(hold, {k: (int(v) if SPACE[k][2] else v) for k, v in current_values(cfg).items()}, workers=workers)
    log(f"hold-out: shipped good {h0['good']:.3f} bad {h0['bad']:.3f}  ->  tuned good {h['good']:.3f} bad {h['bad']:.3f}")
    return ov


def print_report(r: dict, title: str) -> None:
    print(f"{title}: {r['n']} frames  GOOD {r['good'] * 100:.1f}%  BAD LOCK {r['bad'] * 100:.1f}%  MISS {r['miss'] * 100:.1f}%  mean error {r['err_px']:.2f} px")
    for b, v in r["bands"].items():
        print(f"    {b:10s} n={v['n']:3d}  good {v['good'] * 100:5.1f}%  bad {v['bad'] * 100:5.1f}%")


def main(argv: Optional[list] = None) -> int:
    global TOL_PX, MURKY_SHARE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["report", "tune"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--murky-share", type=float, default=MURKY_SHARE)
    ap.add_argument("--tol-px", type=float, default=TOL_PX)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    TOL_PX = a.tol_px
    MURKY_SHARE = a.murky_share
    data = make_dataset(a.n, a.seed)
    cut = int(0.7 * len(data))
    train, hold = data[:cut], data[cut:]
    print(f"{len(data)} frames ({len(train)} train, {len(hold)} hold-out)")
    if a.mode == "report":
        print_report(evaluate(data, cfg_path=a.config, workers=a.workers), "config " + a.config.name)
        return 0
    ov = tune(train, hold, a.rounds, a.workers)
    print_report(evaluate(hold, ov, cfg_path=a.config, workers=a.workers), "tuned, hold-out")
    fresh = make_dataset(max(300, a.n // 2), a.seed + 1000)                 # a third, independent set that the search never saw
    print_report(evaluate(fresh, {k: (int(v) if SPACE[k][2] else v) for k, v in current_values(load_config(a.config)).items()}, cfg_path=a.config, workers=a.workers), "shipped, fresh set")
    print_report(evaluate(fresh, ov, cfg_path=a.config, workers=a.workers), "tuned, fresh set")
    print(yaml.safe_dump({"tuned": ov}, sort_keys=False))
    if a.out:
        a.out.write_text(yaml.safe_dump(ov, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
