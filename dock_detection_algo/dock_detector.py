"""The whole detection pipeline for ONE frame as a plain object (no ROS node, no windows): image + attitude -> DockAlign message.

live_dock_lights.py (the ROS node, with and without GUI), the closed-loop docking harness and the threshold tuner all call this, so they are
guaranteed to run the very same code. Needs `interfaces.msg.DockAlign` (source the workspace) because the output is that message.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dock_acquire import PartialAcquire  # noqa: E402
from dock_align_msg import build_dock_align_msg  # noqa: E402
from dock_detection_config import DEFAULT_CONFIG, detector_kwargs, load_config  # noqa: E402
from dock_geometry import evaluate_dock_geometry, unrotate_cores  # noqa: E402
from dock_light_mask import bloom_mask_and_cores  # noqa: E402
from dock_ring import select_ring  # noqa: E402


def resplit_cores(bgr: np.ndarray, cores: list, kw: dict) -> list:
    """If peaks merged, try again at half the separation before moving the vehicle."""
    if len(cores) >= 4 or not kw.get("peak_mode", True):
        return cores
    tighter = dict(kw)
    tighter["peak_sep"] = max(5, int(kw.get("peak_sep", 28)) // 2)
    _mask, retry = bloom_mask_and_cores(bgr, **tighter)
    return retry if len(retry) > len(cores) else cores


@dataclass
class DetectorResult:
    msg: Any                    # interfaces.msg.DockAlign
    mask: np.ndarray
    cores: list                 # roll-levelled cores (what the geometry saw)
    geo: Any                    # DockGeometry
    ring_residual: float = float("nan")      # how well the 4 chosen lights form the dock ring (0 = perfect; nan = ring check off or fewer than 4 lights)
    n_candidates: int = 0       # bright spots that competed for the 4 places


class DockDetector:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg if cfg is not None else load_config(config_path or DEFAULT_CONFIG)
        self.base_kw = detector_kwargs(self.cfg)
        self.base_kw.update(overrides or {})
        rg = self.cfg.get("ring", {})
        self.ring_on = bool(rg.get("enabled", False))
        self.ring_tol = float(rg.get("fit_tol", 0.08))
        self.ring_cands = int(rg.get("candidates", 8))
        if self.ring_on:
            self.base_kw["candidates"] = max(int(self.base_kw.get("candidates", 0)), self.ring_cands)
        a, aq = self.cfg["alignment"], self.cfg.get("acquire", {})
        self.frac = float(a.get("spread_align_frac", 0.08))
        self.min_px = float(a.get("spread_align_min_px", 8.0))
        self.conf_base = float(a.get("confidence_base", 0.4))
        self.lat_px = float(a.get("lateral_exact_px", 8.0))
        self.lat_frac = float(a.get("lateral_exact_frac", 0.06))
        self.vfov_deg = float(self.cfg["camera"].get("vfov_deg", 60.0))
        self.flow_surge_norm = float(aq.get("flow_surge_norm", 0.35))
        self.acquire = PartialAcquire(
            edge_frac=float(aq.get("edge_frac", 0.10)), nod_pitch_deg=float(aq.get("nod_pitch_deg", 20.0)), yaw_wiggle_deg=float(aq.get("yaw_wiggle_deg", 8.0)),
            nod_period_s=float(aq.get("nod_period_s", 8.0)), backup_radius_frac=float(aq.get("backup_radius_frac", 0.22)),
            backup_after_s=float(aq.get("backup_after_s", 8.0)), vfov_deg=self.vfov_deg, flow_surge_norm=self.flow_surge_norm,
        )

    def process(self, bgr: np.ndarray, roll_rad: float = 0.0, pitch_rad: Optional[float] = 0.0, t: Optional[float] = None, header=None,
                kw_override: Optional[Dict[str, Any]] = None) -> DetectorResult:
        kw = dict(self.base_kw)
        kw.update(kw_override or {})
        if kw.get("core_pct", 92) < 80:
            kw["core_pct"] = 80
        kw["peak_sep"] = max(5, int(kw.get("peak_sep", 28)))
        mask, cores = bloom_mask_and_cores(bgr, **kw)
        cores = resplit_cores(bgr, cores, kw)
        h, w = bgr.shape[:2]
        leveled = unrotate_cores(cores, roll_rad, 0.5 * w, 0.5 * h)
        n_cand = len(leveled)
        resid = float("nan")
        no_ring = False
        if self.ring_on and n_cand >= 4:
            leveled, fit = select_ring(leveled, [1.0 - 0.05 * i for i in range(n_cand)], fit_tol=self.ring_tol, max_candidates=self.ring_cands)
            if fit is not None:
                resid = fit.residual
            else:
                no_ring = True
        elif len(leveled) > 4:
            leveled = leveled[:4]
        geo = evaluate_dock_geometry(leveled, lateral_exact_px=self.lat_px, lateral_exact_frac=self.lat_frac)
        if no_ring and not geo.ok:
            geo.message = f"no ring among {n_cand} bright spots"
        search = None
        if not geo.ok and len(leveled) < 4:
            search = self.acquire.update(leveled, w, h, time.time() if t is None else t)
        else:
            self.acquire.reset()
        if header is None:
            from std_msgs.msg import Header
            header = Header()
        msg = build_dock_align_msg(
            header=header, geo=geo, cores=leveled, image_width=w, image_height=h, spread_align_frac=self.frac, spread_align_min_px=self.min_px,
            confidence_base=self.conf_base, pitch_rad=pitch_rad, vfov_deg=self.vfov_deg, acquire=search, flow_surge_norm=self.flow_surge_norm,
        )
        if msg.valid and not math.isnan(resid):
            msg.confidence = float(msg.confidence * (1.0 - 0.4 * min(resid / max(self.ring_tol, 1e-6), 1.0)))      # a worse-fitting ring is trusted less
        return DetectorResult(msg, mask, leveled, geo, resid, n_cand)
