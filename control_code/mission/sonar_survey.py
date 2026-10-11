#!/usr/bin/env python3
"""A side-scan SURVEY of the rugged sea floor: the lawnmower mission flown closed loop (offline vehicle, realistic odometry) while the two simulated Omniscan 450 SS sonars ping at their
physical rate; everything is recorded so it can be looked at again.

  python3 sonar_survey.py                          # both flying modes -> outputs/sim_viewer_runs/sonar_showcase/
  python3 sonar_survey.py --only terrain_following --range 30 --lanes 4
Per run (<out>/<mode>/): trajectory.csv + sidescan.npz (replay: ./run_sim_viewer.sh --replay <mode>/trajectory.csv --bottom-tab Sonar : the stored pings play back in step with the
vehicle), waterfall.png, mosaic.png, mosaic_vs_truth.png (backscatter mosaic next to the TRUE backscatter map and the depth map with the path), summary.json (coverage, gaps, correlation,
altitude, object contrast), plus index.html. Modes: constant_depth (the lawnmower at a fixed depth: altitude changes with the terrain) and terrain_following (fixed altitude above the floor
from the simulated altimeter). OFFLINE and on ASSUMED sonar levels / sea floor (README section 9): nothing here was compared with a real Omniscan.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_CTRL = _HERE.parent
sys.path[:0] = [str(_d) for _d in [_CTRL / "sim_viewer", *sorted((_CTRL / "sim_viewer").iterdir())] if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # sim_viewer and its sub-folders
for p in (_HERE, _CTRL / "common", _CTRL / "sim_offline", _CTRL / "sim_viewer"):
    sys.path.insert(0, str(p))

from mission_core import MissionRunner  # noqa: E402
from outdirs import out_dir  # noqa: E402
from pose_filter import PoseFilterConfig, PoseTracker  # noqa: E402
from recording import CsvRecorder, SonarRecorder  # noqa: E402
from sensors import OdometryConfig, OdometrySensor  # noqa: E402
from sidescan_mosaic import Mosaic  # noqa: E402
from sidescan_view import mosaic_image, waterfall_image  # noqa: E402
from state import State  # noqa: E402
from synthetic_sidescan import SideScanSim, flat_swath_geometry  # noqa: E402
from synthetic_camera import load_config as load_view_cfg  # noqa: E402
from terrain import get_terrain  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

MISSION = yaml.safe_load((_HERE / "mission.yaml").read_text())


def build_mission(mode: str, lanes: int, spacing_m: float, length_m: float, depth_m: float, altitude_m: float, origin=(-40.0, -60.0)) -> dict:
    cfg = copy.deepcopy(MISSION)
    width = spacing_m * (lanes - 1)
    cfg["legs"] = [{"type": "lawnmower", "origin": [origin[0], origin[1]], "heading_deg": 90.0, "length_m": length_m, "width_m": width, "spacing_m": spacing_m, "z": depth_m, "leadin_m": 10.0}, {"type": "return_home"}]
    cfg["mission"]["home"] = [origin[0], origin[1] - 14.0, depth_m]
    cfg["safety"].update({"dock_keepout_m": 6.0, "min_depth_m": 0.4, "max_depth_m": 22.0, "geofence": {"x_min": origin[0] - width - 40.0, "x_max": origin[0] + 40.0, "y_min": origin[1] - 40.0, "y_max": origin[1] + length_m + 40.0, "z_min": 0.4, "z_max": 22.0}})
    cfg["terrain_follow"] = {"enabled": mode == "terrain_following", "altitude_m": altitude_m, "rate_mps": 0.8, "max_depth_m": 20.0}
    cfg["survey"] = {"swath_width_m": 2 * flat_swath_geometry(altitude_m)["ground_far_m"]}
    return cfg


def fly_survey(mode: str, terrain, sonar_cfg: dict, lanes: int = 4, spacing_m: float = 14.0, length_m: float = 90.0, depth_m: float = 2.0, altitude_m: float = 6.0, T: float = 900.0,
               seed: int = 3, dt: float = 0.05, progress=None):
    from synthetic_sidescan import SideScanConfig
    scfg = SideScanConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in sonar_cfg.items() if k in SideScanConfig.__dataclass_fields__})
    sim = SideScanSim(terrain, scfg)
    cfg = build_mission(mode, lanes, spacing_m, length_m, depth_m, altitude_m)
    mr = MissionRunner(cfg)
    origin = cfg["legs"][0]["origin"]
    m = VehicleModel(pos=(origin[0], origin[1] - 14.0, depth_m), eul_deg=(0.0, 0.0, 90.0))
    sensor = OdometrySensor(OdometryConfig(rate_hz=4.5, latency_s=0.25, freeze_mean_interval_s=0.0, seed=seed))
    trk = PoseTracker(0.25, 1.0, PoseFilterConfig())
    csv_rows: List[list] = []
    srec = SonarRecorder()
    nxt = {"port": 0.0, "starboard": 0.0}
    cnt = {"port": 0, "starboard": 0}
    alt_meas: Optional[float] = None
    alt_next = 0.0
    t = 0.0
    out: Dict[str, float] = {}
    min_alt = 1e9
    t_wall = time.time()
    while t < T and not mr.done:
        for _ in range(int(round(dt / 0.01))):
            m.step(0.01)
            for o in sensor.update(t, m.pos, m.eul, m.nu):
                trk.push(State(pos=np.array(o.pos), eul=np.array(o.eul), nu=np.array(o.nu)), t)
            t += 0.01
        for side in ("port", "starboard"):                                       # the sonars ping at the physical rate, with the TRUE pose (a survey-grade navigation solution)
            if t >= nxt[side]:
                r = sim.ping(m.pos, m.eul, side)
                cnt[side] += 1
                srec.add(t, side, r.intensity, {"start_range_m": r.start_range_m, "range_m": r.range_m, "gain_db": r.gain_db, "gain_index": r.gain_index, "altitude_m": r.altitude_m,
                                                 "ping_hz": r.ping_hz, "sound_speed_mps": r.sound_speed_mps, "pos": m.pos.tolist(), "eul": m.eul.tolist()})
                nxt[side] += 1.0 / r.ping_hz
                min_alt = min(min_alt, r.altitude_m)
        if t >= alt_next:
            alt_next = t + 0.1
            a = sim.altimeter(m.pos, m.eul)
            alt_meas = a if a is not None else alt_meas
        st = trk.at(t)
        if st is None:
            continue
        out, s = mr.update(st, dt, alt_meas)
        m.set_command(out)
        csv_rows.append([t, *m.pos, *m.eul, *m.nu, out.get("th_01", 0.0), out.get("th_02", 0.0), out.get("th_03", 0.0), out.get("cs_04", 0.0), out.get("cs_06", 0.0), out.get("cs_07", 0.0), out.get("cs_08", 0.0)])
        if progress and int(t * 10) % 300 == 0 and abs(t - round(t, 1)) < 1e-6:
            progress(t, time.time() - t_wall)
    return mr, cfg, np.array(csv_rows), srec, min_alt, sim


def write_csv(rows: np.ndarray, path: Path, meta: dict) -> None:
    rec = CsvRecorder(path, meta)
    nxt = 0.0
    for r in rows:
        if r[0] + 1e-9 < nxt:
            continue
        nxt += 0.1
        rec.set_cmd(["th_01", "th_02", "th_03", "cs_04", "cs_06", "cs_07", "cs_08"], list(r[13:20]), r[0])
        rec.add_state(r[0], r[1:4], r[4:7], r[7:13])
    rec.close()


def analyse(terrain, srec: SonarRecorder, rows: np.ndarray, cfg: dict, mr: MissionRunner, min_alt: float, mode: str) -> dict:
    ext = terrain.extent
    mos = Mosaic(ext, 0.5)
    for i in range(len(srec)):
        mos.add_ping("port" if srec.side[i] == 0 else "starboard", srec.inten[i], srec.meta[i])
    lg = cfg["legs"][0]
    o = lg["origin"]
    region = (o[0] - lg["width_m"] - 8.0, o[0] + 8.0, o[1], o[1] + lg["length_m"])
    cov = mos.coverage(region)
    ix0, ix1 = int((region[0] - mos.x0) / mos.cell), int((region[1] - mos.x0) / mos.cell)
    iy0, iy1 = int((region[2] - mos.y0) / mos.cell), int((region[3] - mos.y0) / mos.cell)
    im = mos.image()[ix0:ix1, iy0:iy1]
    # truth: the true backscatter map sampled at the mosaic cells
    xs = mos.x0 + (np.arange(ix0, ix1) + 0.5) * mos.cell
    ys = mos.y0 + (np.arange(iy0, iy1) + 0.5) * mos.cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    refl = terrain.reflectivity_db(X, Y)
    ok = np.isfinite(im)
    # compare after removing the range trend: the correlation of the mosaic with the true backscatter on insonified cells
    lit = ok & (im > 0.03)                                                                      # leave the shadows out: they are geometry, not backscatter
    r_corr = float(np.corrcoef(im[lit], refl[lit])[0, 1]) if lit.sum() > 100 else float("nan")
    # shadows: fraction of insonified cells that are almost black
    shadow = float(np.mean(im[ok] < 0.03)) if ok.any() else 0.0
    # objects: contrast of the object footprint against the surroundings in the mosaic
    objs = []
    for ob in terrain.objects:
        c = np.array(ob["xy"] if ob["kind"] == "box" else [(ob["from"][0] + ob["to"][0]) / 2, (ob["from"][1] + ob["to"][1]) / 2], float)
        if not (region[0] <= c[0] <= region[1] and region[2] <= c[1] <= region[3]):
            continue
        d = np.hypot(X - c[0], Y - c[1])
        inner, ring = ok & (d < 3.0), ok & (d >= 3.0) & (d < 12.0)
        if inner.sum() > 5 and ring.sum() > 20:
            objs.append({"kind": ob["kind"], "xy": c.tolist(), "inside_mean": float(im[inner].mean()), "around_mean": float(im[ring].mean()), "dark_fraction_inside": float(np.mean(im[inner] < 0.05))})
    z = rows[:, 3]
    return {"mode": mode, "finished": bool(mr.done), "aborted": mr.aborted or "", "time_s": float(rows[-1, 0]), "pings": int(len(srec)), "coverage_of_survey_area": cov,
            "gap_area_m2": float((1.0 - cov) * (region[1] - region[0]) * (region[3] - region[2])), "mosaic_truth_corr": r_corr, "shadow_fraction": shadow,
            "min_altitude_m": float(min_alt), "depth_min_m": float(z.min()), "depth_max_m": float(z.max()), "objects": objs, "swath_m_per_side_flat_at_6m": flat_swath_geometry(6.0)["ground_far_m"],
            "region": list(region)}, mos, region


def figures(terrain, mos: Mosaic, rows: np.ndarray, srec: SonarRecorder, region, out: Path, title: str, rng_m: float) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path_xy = rows[:, 1:3]
    pad = 20.0
    crop = (region[0] - pad, region[1] + pad, region[2] - pad, region[3] + pad)
    cv2.imwrite(str(out / "mosaic.png"), mosaic_image(mos, 3, path_xy, None, crop))
    # waterfall: a slice from the middle of the survey
    mid = len(srec) // 2
    sel = [i for i in range(max(0, mid - 600), min(len(srec), mid + 600))]
    port = [srec.inten[i] for i in sel if srec.side[i] == 0][-400:]
    stbd = [srec.inten[i] for i in sel if srec.side[i] == 1][-400:]
    cv2.imwrite(str(out / "waterfall.png"), waterfall_image(port, stbd, (1200, 640), rng_m, title="waterfall (a slice from the middle of the survey)"))
    # mosaic vs truth
    x0, x1, y0, y1 = crop
    gx = np.arange(x0, x1, 0.5)
    gy = np.arange(y0, y1, 0.5)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    refl = terrain.reflectivity_db(X, Y)
    depth = terrain.height(X, Y)
    im = mos.image()
    ix0, iy0 = int((x0 - mos.x0) / mos.cell), int((y0 - mos.y0) / mos.cell)
    sub = im[ix0:ix0 + len(gx), iy0:iy0 + len(gy)]
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.2))
    ex = [y0, y1, x0, x1]
    a = ax[0].imshow(np.ma.masked_invalid(sub[::-1]), extent=ex, cmap="afmhot", vmin=0, vmax=1)
    ax[0].set_title("side-scan mosaic (what the sonar saw)")
    b = ax[1].imshow(refl[::-1], extent=ex, cmap="gray")
    ax[1].set_title("TRUE backscatter map [dB] (terrain.py)")
    plt.colorbar(b, ax=ax[1], shrink=0.7)
    c = ax[2].imshow(depth[::-1], extent=ex, cmap="viridis_r")
    ax[2].set_title("TRUE depth [m] with the vehicle path")
    plt.colorbar(c, ax=ax[2], shrink=0.7)
    for k in range(3):
        ax[k].plot(path_xy[:, 1], path_xy[:, 0], "-", color="#ffcc33" if k else "#33aaff", lw=0.8)
        ax[k].set_xlabel("east [m]")
    ax[0].set_ylabel("north [m]")
    for ob in terrain.objects:
        cc = ob["xy"] if ob["kind"] == "box" else [(ob["from"][0] + ob["to"][0]) / 2, (ob["from"][1] + ob["to"][1]) / 2]
        if x0 <= cc[0] <= x1 and y0 <= cc[1] <= y1:
            for k in (1, 2):
                ax[k].plot([cc[1]], [cc[0]], "r+", ms=10)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "mosaic_vs_truth.png", dpi=85)
    plt.close(fig)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=14.0)
    ap.add_argument("--length", type=float, default=90.0)
    ap.add_argument("--range", type=float, default=30.0, dest="range_m")
    ap.add_argument("--gain", type=int, default=-1)
    ap.add_argument("--altitude", type=float, default=6.0)
    ap.add_argument("--depth", type=float, default=2.0)
    ap.add_argument("--T", type=float, default=900.0)
    ap.add_argument("--out", type=Path, default=out_dir("sim_viewer_runs") / "sonar_showcase")
    ap.add_argument("--no-copy", action="store_true", help="do not copy the CSV/npz into outputs/sim_viewer_runs (History tab)")
    a = ap.parse_args(argv)
    vcfg = load_view_cfg()
    terrain = get_terrain(vcfg.get("terrain"))
    sonar_cfg = dict(vcfg["sonar"]["sensor"])
    sonar_cfg.update({"range_m": a.range_m, "gain_index": a.gain})
    modes = [m for m in ("constant_depth", "terrain_following") if not a.only or m in a.only.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for mode in modes:
        print(f"flying {mode} ...", flush=True)
        mr, cfg, rows, srec, min_alt, sim = fly_survey(mode, terrain, sonar_cfg, a.lanes, a.spacing, a.length, a.depth, a.altitude, a.T, progress=lambda t, w: print(f"  t={t:5.0f} s  ({w:.0f} s wall)", flush=True))
        met, mos, region = analyse(terrain, srec, rows, cfg, mr, min_alt, mode)
        ok = met["finished"] and not met["aborted"] and met["coverage_of_survey_area"] >= 0.80 and met["min_altitude_m"] >= 1.0
        d = a.out / (mode if ok else "FAILED_" + mode)
        d.mkdir(exist_ok=True)
        title = f"{mode}: {'PASS' if ok else 'FAILED'}  coverage {100 * met['coverage_of_survey_area']:.0f}%  min altitude {met['min_altitude_m']:.1f} m  depth {met['depth_min_m']:.1f}-{met['depth_max_m']:.1f} m  {met['pings']} pings"
        write_csv(rows, d / "trajectory.csv", {"recorder": "mission/sonar_survey.py", "controller": "mission", "scenario": title})
        srec.save(d / "sidescan.npz")
        figures(terrain, mos, rows, srec, region, d, title, a.range_m)
        met["pass"] = bool(ok)
        (d / "summary.json").write_text(json.dumps(met, indent=2))
        entries.append((d.name, met, title))
        if not a.no_copy:
            shutil.copy(d / "trajectory.csv", out_dir("sim_viewer_runs") / f"sonar_{d.name}.csv")
            shutil.copy(d / "sidescan.npz", out_dir("sim_viewer_runs") / f"sonar_{d.name}_sidescan.npz")
        print(f"  {title}   corr {met['mosaic_truth_corr']:.2f}  shadow {100 * met['shadow_fraction']:.0f}%", flush=True)
    rows_html = "".join(f"<tr><td><a href='{n}/mosaic_vs_truth.png'><img src='{n}/mosaic_vs_truth.png' width='520'></a></td><td><b>{html.escape(n)}</b><br>{html.escape(t)}<br>mosaic vs true backscatter correlation {m['mosaic_truth_corr']:.2f}, "
                        f"shadows {100 * m['shadow_fraction']:.0f}% of the insonified cells, gaps {m['gap_area_m2']:.0f} m2</td><td><a href='{n}/waterfall.png'>waterfall</a><br><a href='{n}/mosaic.png'>mosaic</a><br>"
                        f"<a href='{n}/trajectory.csv'>trajectory.csv</a><br><a href='{n}/sidescan.npz'>sidescan.npz</a><br><small>replay: ./run_sim_viewer.sh --replay {n}/trajectory.csv --bottom-tab Sonar</small></td></tr>" for n, m, t in entries)
    (a.out / "index.html").write_text("<html><head><meta charset='utf-8'><title>Side-scan survey showcase</title><style>body{font-family:sans-serif;margin:20px} td{border-bottom:1px solid #ddd;padding:8px;vertical-align:top}</style></head>"
                                      f"<body><h2>Side-scan survey of the rugged sea floor (offline simulation)</h2><p>Simulated Omniscan 450 SS pair, mount 30&deg; off the nadir, 50&deg; x 0.5&deg; beams, {a.range_m:g} m range. "
                                      f"Sonar levels and the sea floor are ASSUMPTIONS, not real data.</p><table>{rows_html}</table></body></html>")
    (a.out / "README.txt").write_text("Side-scan survey showcase (offline). Each folder: trajectory.csv + sidescan.npz (replay in the viewer: run_sim_viewer.sh --replay <folder>/trajectory.csv --bottom-tab Sonar), waterfall.png, mosaic.png, mosaic_vs_truth.png, summary.json.\n")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
