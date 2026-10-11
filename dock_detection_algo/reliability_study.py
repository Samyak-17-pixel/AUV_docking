#!/usr/bin/env python3
"""Reliability study of the dock detector: how does it score when ONE thing at a time gets harder? No GUI, needs the workspace sourced (DockAlign message).

  python3 reliability_study.py                        # every axis, 40 frames per value, report in outputs/detection_reports/
  python3 reliability_study.py --axes range,view_deg  # only these axes
  python3 reliability_study.py --n 80 --config other.yaml --tag after_fix --workers 24
  python3 reliability_study.py --list                 # axis names and their values

Each axis keeps everything else at a modest base case (range 3-7 m, small random offsets) and changes one thing: range, how far the AUV is off the dock
axis (view_deg: it sits on a circle round the dock and always points at it = the dock is seen obliquely), heading error, lateral offset, pitch, roll,
sensor noise, JPEG quality, light brightness, side-light dimness, water clarity, background brightness and the stress options of synthetic_camera.py
(blur, motion blur, bubbles, marine snow, backscatter, surface glint, false lights, reflections, one light hidden).
Per frame: GOOD  = valid, four lights, every light within 6 px of its true pixel;  BAD LOCK = valid but a light is wrong (the dangerous case: the controller would
steer on it);  MISS = not valid;  OUT = the dock is not fully inside the image, so no detector could succeed (counted apart, NOT as a miss).
Also measured: the pixel error of the centre, the range estimate fy*1m/radius_px against the true distance (the controller's quick range formula), and the time per frame.
Outputs (outputs/detection_reports/<tag>/): results.csv (one row per frame), summary.txt (one table per axis), one PNG per group of axes.
SYNTHETIC-ONLY evidence: the real renderer's look is not known (see sim_viewer.yaml `look:` and `stress:`).
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_CTRL = _HERE.parent / "control_code"
sys.path[:0] = [str(_d) for _d in [_CTRL / "sim_viewer", *sorted((_CTRL / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # sim_viewer and its sub-folders
for p in (_HERE, _CTRL / "sim_viewer", _CTRL / "common"):
    sys.path.insert(0, str(p))

from dock_detection_config import DEFAULT_CONFIG, load_config  # noqa: E402
from dock_detector import DockDetector  # noqa: E402
from synthetic_camera import SyntheticCamera, load_config as load_cam_cfg  # noqa: E402

TOL_PX = 6.0
DOCK = np.array([10.0, 0.0, 3.0])
MARGIN_PX = 12


# ---------------------------------------------------------------------------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------------------------------------------------------------------------
def base_case(rng: np.random.Generator) -> dict:
    """A modest random start: the dock ahead, 3-7 m away, small offsets and attitudes."""
    return {"range": float(rng.uniform(3.0, 7.0)), "lat": float(rng.uniform(-1.0, 1.0)), "head": float(rng.uniform(-10.0, 10.0)), "view_deg": None,
            "pitch": float(rng.uniform(-4, 4)), "roll": float(rng.uniform(-4, 4)), "dz": float(rng.uniform(-0.3, 0.3)),
            "look": {}, "stress": {}, "jpeg_q": None, "side": 1.0, "seed": int(rng.integers(0, 10 ** 6))}


def _set(key: str) -> Callable[[dict, float], None]:
    def f(c: dict, v) -> None:
        c[key] = v
    return f


def _look(key: str) -> Callable[[dict, float], None]:
    def f(c: dict, v) -> None:
        c["look"][key] = v
    return f


def _look_scale(key: str) -> Callable[[dict, float], None]:
    def f(c: dict, v) -> None:
        c.setdefault("look_scale", {})[key] = v
    return f


def _stress(key: str) -> Callable[[dict, float], None]:
    def f(c: dict, v) -> None:
        c["stress"][key] = v
    return f


def _range(c: dict, v) -> None:
    c["range"] = v
    c["lat"] = float(np.clip(c["lat"], -0.4 * v, 0.4 * v))


def _view(c: dict, v) -> None:
    c["view_deg"] = v
    c["lat"] = 0.0


# axis -> (group, values, setter, label)
AXES: Dict[str, Tuple[str, list, Callable, str]] = {
    "range": ("geometry", [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0], _range, "range to the dock plane [m]"),
    "view_deg": ("geometry", [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75], _view, "position angle off the dock axis at 5 m, always pointing at the dock [deg]"),
    "head": ("geometry", [-40, -30, -20, -10, 0, 10, 20, 30, 40], _set("head"), "heading error vs the dock axis at 5 m [deg]"),
    "lat": ("geometry", [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], _set("lat"), "lateral offset at 6 m [m]"),
    "pitch": ("geometry", [-25, -20, -10, -5, 0, 5, 10, 20, 25], _set("pitch"), "vehicle pitch [deg]"),
    "roll": ("geometry", [-40, -30, -20, -10, 0, 10, 20, 30, 40], _set("roll"), "vehicle roll [deg]"),
    "noise": ("camera", [0.004, 0.012, 0.025, 0.04, 0.06, 0.09], _look("noise_sigma"), "sensor noise sigma (fraction of full scale)"),
    "jpeg": ("camera", [95, 80, 60, 40, 25, 12], _set("jpeg_q"), "JPEG quality"),
    "gain": ("camera", [0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.0], _look_scale("core_gain"), "light brightness (x core_gain)"),
    "side": ("camera", [0.2, 0.4, 0.7, 1.0, 2.0, 4.0], _set("side"), "side-light brightness (x the vessel-file lumens)"),
    "water": ("camera", [0.0, 0.04, 0.08, 0.12, 0.16, 0.22], _look("water_attenuation_per_m"), "water attenuation [1/m]"),
    "background": ("camera", [1.0, 2.0, 3.0, 4.0, 5.0], _look_scale("background_bgr"), "background brightness (x the default water colour)"),
    "blur": ("stress", [0.0, 1.0, 2.0, 3.0, 4.0, 6.0], _stress("blur_sigma_px"), "Gaussian blur sigma [px]"),
    "motion": ("stress", [0, 5, 9, 15, 25], _stress("motion_blur_px"), "motion blur length [px] (horizontal)"),
    "bubbles": ("stress", [0, 3, 8, 15, 30, 60], _stress("bubbles"), "bubbles per frame"),
    "snow": ("stress", [0, 20, 60, 150, 400], _stress("snow"), "marine-snow specks per frame"),
    "backscatter": ("stress", [0.0, 0.1, 0.2, 0.3, 0.5, 0.8], _stress("backscatter"), "backscatter veil strength"),
    "glint": ("stress", [0.0, 0.3, 0.6, 1.0], _stress("glint"), "surface glint strength"),
    "distractors": ("stress", [0, 1, 2, 4, 8], _stress("distractors"), "false bright lights per frame"),
    "reflections": ("stress", [0, 1, 2, 4], _stress("reflections"), "mirrored ring lights"),
    "occlude": ("stress", ["", "top", "bottom", "left", "right"], _stress("occlude"), "one ring light hidden"),
}


def build_camera(case: dict, cfg0: dict) -> SyntheticCamera:
    cfg = copy.deepcopy(cfg0)
    lk = cfg["look"]
    lk.update(case.get("look", {}))
    for k, s in case.get("look_scale", {}).items():
        lk[k] = [float(x) * s for x in lk[k]] if isinstance(lk[k], list) else float(lk[k]) * s
    lk["seed"] = case["seed"]
    if case.get("jpeg_q"):
        cfg["camera"]["jpeg_quality"] = int(case["jpeg_q"])
    cfg["stress"] = {**(cfg.get("stress") or {}), "seed": case["seed"], **case.get("stress", {})}
    if abs(case.get("side", 1.0) - 1.0) > 1e-9:
        for L in cfg["dock"]["ring_lights"]:
            if L["name"] in ("left", "right"):
                L["lumens"] = float(L["lumens"]) * case["side"]
    return SyntheticCamera(cfg)


def case_pose(case: dict) -> Tuple[np.ndarray, np.ndarray]:
    if case.get("view_deg") is not None:
        a = math.radians(case["view_deg"])
        r = case["range"]
        pos = np.array([DOCK[0] - r * math.cos(a), r * math.sin(a), DOCK[2] + case["dz"]])
        eul = np.array([case["roll"], case["pitch"], -case["view_deg"]])
    else:
        pos = np.array([DOCK[0] - case["range"], case["lat"], DOCK[2] + case["dz"]])
        eul = np.array([case["roll"], case["pitch"], case["head"]])
    return pos, eul


def render_case(case: dict, cfg0: dict) -> dict:
    """-> dict(jpeg, truth (4x2 roll-levelled px in detector order top,bottom,right,left), centre_truth, dist, in_view, roll, pitch)."""
    cam = build_camera(case, cfg0)
    pos, eul = case_pose(case)
    pts, ok = {}, True
    for L in cam.lights:
        if not L["ring"]:
            continue
        pr = cam.project(pos, eul, L["p"])
        if pr is None or not (MARGIN_PX < pr[0] < cam.W - MARGIN_PX and MARGIN_PX < pr[1] < cam.H - MARGIN_PX):
            ok = False
        pts[L["name"]] = pr
    c, s = math.cos(math.radians(eul[0])), math.sin(math.radians(eul[0]))

    def lev(pr):
        dx, dy = pr[0] - cam.W / 2, pr[1] - cam.H / 2
        return (cam.W / 2 + c * dx - s * dy, cam.H / 2 + s * dx + c * dy)
    truth = None
    ctr = None
    if all(v is not None for v in pts.values()):
        lv = {k: lev(v) for k, v in pts.items()}
        sp = sorted([lv["right"], lv["left"]], key=lambda q: q[0])
        truth = np.array([lv["top"], lv["bottom"], sp[1], sp[0]])
        pc = cam.project(pos, eul, cam.dock_pos)
        ctr = lev(pc) if pc is not None else None
    cam_pos = cam.camera_position(pos, eul)
    return {"jpeg": cam.encode_jpeg(cam.render(pos, eul)), "truth": truth, "ctr": ctr, "dist": float(np.linalg.norm(cam_pos - cam.dock_pos)), "in_view": ok,
            "roll": math.radians(eul[0]), "pitch": math.radians(eul[1]), "f": cam.f}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------------------------------------------------------------------------
_DET: Optional[DockDetector] = None
_CFG0: dict = {}


def _init(cfg_path, overrides) -> None:
    global _DET, _CFG0
    import cv2
    cv2.setNumThreads(1)
    _DET = DockDetector(cfg=load_config(Path(cfg_path)), overrides=overrides or {})
    _CFG0 = load_cam_cfg()


def score_case(case: dict) -> dict:
    import cv2
    fr = render_case(case, _CFG0)
    row = {"kind": "out", "pix_err": float("nan"), "ctr_err": float("nan"), "range_rel_err": float("nan"), "ms": float("nan"), "conf": float("nan"), "num": 0}
    bgr = cv2.imdecode(np.frombuffer(fr["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    t0 = time.perf_counter()
    m = _DET.process(bgr, fr["roll"], fr["pitch"], 0.0).msg
    row["ms"] = 1000.0 * (time.perf_counter() - t0)
    row["num"] = int(m.num_lights)
    row["conf"] = float(m.confidence)
    if not fr["in_view"] or fr["truth"] is None:
        # the dock is (partly) outside the image: success is impossible; a valid claim is still a false lock if the lights it reports are wrong
        row["kind"] = "out" if not m.valid else "out_valid"
        return row
    if not m.valid:
        row["kind"] = "miss"
        return row
    pix = np.array([[m.top.x, m.top.y], [m.bottom.x, m.bottom.y], [m.right.x, m.right.y], [m.left.x, m.left.y]])
    err = np.linalg.norm(pix - fr["truth"], axis=1)
    good = m.num_lights == 4 and float(err.max()) <= TOL_PX
    row["kind"] = "good" if good else "bad"
    row["pix_err"] = float(err.mean())
    if fr["ctr"] is not None:
        row["ctr_err"] = float(math.hypot(m.center.x - fr["ctr"][0], m.center.y - fr["ctr"][1]))
    if m.radius_px > 1.0:
        row["range_rel_err"] = float((fr["f"] * 1.0 / m.radius_px - fr["dist"]) / fr["dist"])
    return row


# ---------------------------------------------------------------------------------------------------------------------------------------------
# running and reporting
# ---------------------------------------------------------------------------------------------------------------------------------------------
def make_cases(axis: str, n: int, seed: int) -> List[Tuple[object, dict]]:
    group, values, setter, _ = AXES[axis]
    out = []
    for vi, v in enumerate(values):
        rng = np.random.default_rng(seed + 1000 * vi + (hash(axis) % 997 if False else 0))
        for _ in range(n):
            c = base_case(rng)
            if axis == "view_deg":
                c["range"] = 5.0
            elif axis == "head":
                c["range"] = 5.0
            elif axis == "lat":
                c["range"] = 6.0
            setter(c, v)
            out.append((v, c))
    return out


def summarise(rows: List[dict]) -> dict:
    k = np.array([r["kind"] for r in rows])
    inview = np.isin(k, ["good", "bad", "miss"])
    n_in = int(inview.sum())
    def frac(name):
        return float((k == name).sum() / n_in) if n_in else float("nan")
    def stat(key, fn):
        v = np.array([r[key] for r in rows if not math.isnan(r[key])])
        return float(fn(v)) if v.size else float("nan")
    return {"n": len(rows), "out": float(np.mean(np.isin(k, ["out", "out_valid"]))), "good": frac("good"), "bad": frac("bad"), "miss": frac("miss"),
            "out_valid": float((k == "out_valid").sum()), "pix_err": stat("pix_err", np.mean), "ctr_p95": stat("ctr_err", lambda v: np.percentile(v, 95)),
            "range_rel_mean": stat("range_rel_err", np.mean), "range_rel_p95": stat("range_rel_err", lambda v: np.percentile(np.abs(v), 95)),
            "ms": stat("ms", np.mean)}


def run(axes: List[str], n: int, seed: int, cfg_path: Path, overrides: dict, workers: int, log=print) -> Dict[str, List[Tuple[object, dict]]]:
    results: Dict[str, List[Tuple[object, dict]]] = {}
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(cfg_path), overrides)) as pool:
        for ax in axes:
            cases = make_cases(ax, n, seed)
            rows = pool.map(score_case, [c for _, c in cases], chunksize=4)
            by_v: Dict[object, List[dict]] = {}
            for (v, _), r in zip(cases, rows):
                by_v.setdefault(v, []).append(r)
            results[ax] = [(v, summarise(rs)) for v, rs in by_v.items()]
            for v, rs in by_v.items():
                for r in rs:
                    r["axis"], r["value"] = ax, v
            results[ax + "#rows"] = [(v, r) for v, rs in by_v.items() for r in rs]
            log(f"  {ax:12s} done")
    return results


def format_table(ax: str, series: List[Tuple[object, dict]]) -> str:
    lines = [f"{ax}  ({AXES[ax][3]})", f"  {'value':>9s} {'good%':>6s} {'bad%':>6s} {'miss%':>6s} {'out%':>6s} {'pix px':>7s} {'ctr95':>6s} {'rng err%':>9s} {'rng95%':>7s} {'ms':>5s}"]
    for v, s in series:
        f = lambda x, m=100.0: ("   -  " if math.isnan(x) else f"{x * m:6.1f}")
        lines.append(f"  {str(v):>9s} {f(s['good'])} {f(s['bad'])} {f(s['miss'])} {f(s['out'])} {s['pix_err']:7.2f} {s['ctr_p95']:6.1f} {f(s['range_rel_mean'], 100):>9s} {f(s['range_rel_p95'], 100):>7s} {s['ms']:5.1f}")
    return "\n".join(lines)


def save_report(results: dict, out: Path, tag: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    axes = [a for a in results if "#" not in a]
    with open(out / "summary.txt", "w") as f:
        f.write(f"dock detector reliability study '{tag}'  (GOOD/BAD/MISS are fractions of the frames where the whole dock is inside the image; OUT = share of frames where it is not)\n\n")
        for ax in axes:
            f.write(format_table(ax, results[ax]) + "\n\n")
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "value", "kind", "num_lights", "pix_err", "ctr_err", "range_rel_err", "conf", "ms"])
        for ax in axes:
            for v, r in results[ax + "#rows"]:
                w.writerow([ax, v, r["kind"], r["num"], r["pix_err"], r["ctr_err"], r["range_rel_err"], r["conf"], r["ms"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                                                # noqa: BLE001
        return
    for group in ("geometry", "camera", "stress"):
        ga = [a for a in axes if AXES[a][0] == group]
        if not ga:
            continue
        cols = 3
        rows_n = int(math.ceil(len(ga) / cols))
        fig, axs = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 3.4 * rows_n), squeeze=False)
        for i, a in enumerate(ga):
            ax = axs[i // cols][i % cols]
            vals = [str(v) if isinstance(v, str) else v for v, _ in results[a]]
            x = np.arange(len(vals))
            good = np.array([s["good"] for _, s in results[a]]) * 100
            bad = np.array([s["bad"] for _, s in results[a]]) * 100
            miss = np.array([s["miss"] for _, s in results[a]]) * 100
            out_ = np.array([s["out"] for _, s in results[a]]) * 100
            ax.bar(x, np.nan_to_num(good), color="#2e9d5b", label="good")
            ax.bar(x, np.nan_to_num(miss), bottom=np.nan_to_num(good), color="#e0a526", label="miss")
            ax.bar(x, np.nan_to_num(bad), bottom=np.nan_to_num(good) + np.nan_to_num(miss), color="#d63b3b", label="bad lock")
            ax.plot(x, out_, "k.--", lw=0.8, ms=4, label="out of view")
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in vals], fontsize=7, rotation=45)
            ax.set_ylim(0, 105)
            ax.set_title(AXES[a][3], fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=6, loc="lower left")
        for j in range(len(ga), rows_n * cols):
            axs[j // cols][j % cols].axis("off")
        fig.suptitle(f"dock detector reliability - {group} ({tag}); % of frames with the dock in view", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"reliability_{group}.png", dpi=110)
        plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="all")
    ap.add_argument("--n", type=int, default=40, help="frames per value")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--set", action="append", default=[], help="detector override key=value (mask / peaks keys), repeatable")
    ap.add_argument("--tag", default="study")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "outputs" / "detection_reports")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for k, (g, v, _, lab) in AXES.items():
            print(f"{k:12s} [{g}] {lab}: {v}")
        return 0
    axes = list(AXES) if a.axes == "all" else a.axes.split(",")
    ov = {}
    for it in a.set:
        k, v = it.split("=", 1)
        ov[k] = float(v) if "." in v else int(v)
    t0 = time.time()
    res = run(axes, a.n, a.seed, a.config, ov, a.workers)
    save_report(res, a.out / a.tag, a.tag)
    for ax in axes:
        print(format_table(ax, res[ax]) + "\n")
    print(f"{sum(len(res[ax + '#rows']) for ax in axes)} frames in {time.time() - t0:.0f} s -> {a.out / a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
