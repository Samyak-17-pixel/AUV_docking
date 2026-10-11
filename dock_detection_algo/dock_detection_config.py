"""Load dock_detection.yaml into keyword dicts for the detection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "dock_detection.yaml"

_MASK_KEYS = (
    "v_thresh", "use_cyan_assist", "cyan_h_lo", "cyan_h_hi", "cyan_s_lo",
    "cyan_v_lo", "open_k", "close_k", "min_area", "max_blobs", "bg_subtract", "bg_ref_v", "bg_row_pct",
)
_PEAK_KEYS = (
    "peak_mode", "peak_sep", "core_pct", "tight_floor", "tight_default",
    "tight_erode_k", "search_dilate_k", "dog_sigma_small", "dog_sigma_large",
    "response_min", "peak_abs_v_floor", "peak_abs_v_frac", "refine_half_window",
    "fallback_dist_min", "candidates", "plateau_min_px",
)


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Return the parsed YAML (empty sections if a file/section is missing)."""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("camera", "mask", "peaks", "alignment", "hud", "acquire", "ring"):
        cfg.setdefault(section, {})
    return cfg


def detector_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword args for dock_light_mask.bloom_mask_and_cores()."""
    kw: Dict[str, Any] = {}
    for k in _MASK_KEYS:
        if k in cfg["mask"]:
            kw[k] = cfg["mask"][k]
    for k in _PEAK_KEYS:
        if k in cfg["peaks"]:
            kw[k] = cfg["peaks"][k]
    return kw
