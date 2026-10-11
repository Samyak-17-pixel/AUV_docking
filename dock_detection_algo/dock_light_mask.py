"""Bloom light mask + core estimation for funnel dock lights.

Near the dock, bloom merges into one white blob. Contour centroids then fail
(cores=1). Cores are therefore found as local intensity peaks (DoG + NMS),
while the bloom mask is kept for visualization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Core = Tuple[float, float]  # (cx, cy) in image pixels


def _build_bloom_mask(
    hsv: np.ndarray,
    *,
    v_thresh: int,
    use_cyan_assist: bool,
    cyan_h_lo: int,
    cyan_h_hi: int,
    cyan_s_lo: int,
    cyan_v_lo: int,
    open_k: int,
    close_k: int,
) -> np.ndarray:
    bright = cv2.inRange(hsv, (0, 0, int(v_thresh)), (180, 255, 255))
    if use_cyan_assist:
        cyan = cv2.inRange(
            hsv,
            (int(cyan_h_lo), int(cyan_s_lo), int(cyan_v_lo)),
            (int(cyan_h_hi), 255, 255),
        )
        mask = cv2.bitwise_or(bright, cyan)
    else:
        mask = bright

    if open_k and open_k > 0:
        k = max(1, int(open_k) | 1)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
    if close_k and close_k > 0:
        k = max(1, int(close_k) | 1)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
    return mask


def background_excess(v: np.ndarray, bg_ref_v: float, row_pct: float = 10.0, smooth_rows: float = 15.0) -> np.ndarray:
    """Per-row brightness of the water ABOVE the nominal dark level `bg_ref_v` (0-255), as a column vector H x 1 (float32).

    The water is estimated per image row as a low percentile of V (a row is never mostly light), then smoothed over rows, so a vertical brightness
    ramp (backscatter, sun) and a uniformly brighter, murkier water are both measured. In clear water the estimate is below `bg_ref_v` and the excess is
    ZERO: the image is not touched at all. Subtracting it from V lets the same thresholds work in murky water."""
    pct = np.percentile(v, row_pct, axis=1).astype(np.float32)
    k = max(3, int(smooth_rows * 4) | 1)
    pct = cv2.GaussianBlur(pct.reshape(-1, 1), (1, k), float(smooth_rows)).reshape(-1)
    return np.maximum(pct - float(bg_ref_v), 0.0)[:, None]


def _plateau_centre(v_f: np.ndarray, x: float, y: float, min_px: int, half: int = 45, tol: float = 2.0) -> Optional[Core]:
    """Centre of the saturated plateau around a peak. A light brighter than the sensor range (or on a bright background) clips to a flat top; the DoG
    peak then lands anywhere on it (up to ~9 px off in murky water). If the connected region within `tol` grey levels of the local maximum has at least `min_px`
    pixels, its centroid is returned, else None (the caller keeps the response-weighted centre, which is smoother for unclipped lights)."""
    h, w = v_f.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1, y0, y1 = max(0, xi - half), min(w, xi + half + 1), max(0, yi - half), min(h, yi + half + 1)
    win = v_f[y0:y1, x0:x1]
    if win.size == 0:
        return None
    px, py = xi - x0, yi - y0
    vmax = float(win[max(0, py - 2):py + 3, max(0, px - 2):px + 3].max())
    reg = (win >= vmax - tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(reg, connectivity=8)
    if n <= 1:
        return None
    l = lab[min(max(py, 0), lab.shape[0] - 1), min(max(px, 0), lab.shape[1] - 1)]
    if l == 0:
        ys, xs = np.nonzero(reg)
        if xs.size == 0:
            return None
        k = int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))
        l = lab[ys[k], xs[k]]
    ys, xs = np.nonzero(lab == l)
    if xs.size < min_px or xs.size > 0.35 * win.size:
        return None
    return (float(xs.mean() + x0), float(ys.mean() + y0))


def _cores_from_contours(mask: np.ndarray, min_area: float, max_blobs: int) -> List[Core]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: List[Tuple[float, Core]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 1e-6:
            continue
        blobs.append((area, (m["m10"] / m["m00"], m["m01"] / m["m00"])))
    blobs.sort(key=lambda t: t[0], reverse=True)
    if max_blobs and max_blobs > 0:
        blobs = blobs[: int(max_blobs)]
    return [core for _, core in blobs]


def _nms_peaks(
    ys: np.ndarray,
    xs: np.ndarray,
    scores: np.ndarray,
    *,
    min_sep: float,
    max_peaks: int,
) -> List[Core]:
    """Greedy non-maximum suppression on peak coordinates."""
    if len(scores) == 0:
        return []
    order = np.argsort(scores)[::-1]
    picked: List[Core] = []
    for i in order:
        y, x = float(ys[i]), float(xs[i])
        if any((x - px) ** 2 + (y - py) ** 2 < min_sep * min_sep for px, py in picked):
            continue
        picked.append((x, y))
        if max_peaks > 0 and len(picked) >= max_peaks:
            break
    return picked


def _cores_from_peaks(
    v: np.ndarray,
    bloom: np.ndarray,
    *,
    peak_sep: int,
    core_pct: int,
    max_blobs: int,
    min_area: float,
    search_dilate_k: int = 9,
    dog_sigma_small: float = 1.2,
    dog_sigma_large: float = 4.0,
    tight_floor: float = 200.0,
    tight_default: float = 220.0,
    tight_erode_k: int = 3,
    response_min: float = 0.05,
    peak_abs_v_floor: float = 160.0,
    peak_abs_v_frac: float = 0.7,
    refine_half_window: int = 2,
    fallback_dist_min: float = 1.0,
    candidates: int = 0,
    plateau_min_px: int = 0,
) -> List[Core]:
    """Find light cores as local maxima even when bloom blobs merge.

    Uses:
      1) Difference-of-Gaussians on V (handles flat saturation better than raw V)
      2) Distance-transform peaks on an adaptive high threshold
    Then NMS with min separation.
    """
    v_f = v.astype(np.float32)
    h, w = v.shape

    # Restrict search to bloom (slightly dilated so peaks near edges survive)
    search = bloom.copy()
    if cv2.countNonZero(search) < 5:
        return []
    search = cv2.dilate(
        search, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (search_dilate_k, search_dilate_k)),
        iterations=1,
    )

    # --- Response A: DoG (positive lobes ≈ bright cores) ---
    g_small = cv2.GaussianBlur(v_f, (0, 0), float(dog_sigma_small))
    g_large = cv2.GaussianBlur(v_f, (0, 0), float(dog_sigma_large))
    dog = cv2.subtract(g_small, g_large)
    dog = np.maximum(dog, 0.0)
    dog[search == 0] = 0.0

    # --- Response B: distance transform on adaptive tight core mask ---
    inside = v_f[bloom > 0]
    if inside.size > 0:
        pct = float(np.clip(core_pct, 80, 99))
        tight_thr = float(np.percentile(inside, pct))
        # Never go below a floor so dark frames still work
        tight_thr = max(tight_thr, float(tight_floor))
    else:
        tight_thr = float(tight_default)
    tight = np.zeros_like(bloom)
    tight[(v_f >= tight_thr) & (search > 0)] = 255
    # Split merged lobes a bit
    tight = cv2.erode(
        tight, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tight_erode_k, tight_erode_k)),
        iterations=1,
    )
    dist = cv2.distanceTransform(tight, cv2.DIST_L2, 5)

    # Fuse responses (normalize then add)
    def _norm(x: np.ndarray) -> np.ndarray:
        m = float(x.max()) if x.size else 0.0
        return x / m if m > 1e-6 else x

    response = _norm(dog) + _norm(dist.astype(np.float32))
    response[search == 0] = 0.0

    # Local maxima via dilate equality
    sep = max(3, int(peak_sep))
    k = sep if sep % 2 == 1 else sep + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dil = cv2.dilate(response, kernel)
    # Strict local max + enough strength
    peak_map = (response >= dil - 1e-6) & (response > response_min)
    # Also require some absolute brightness so murk peaks die
    peak_map &= v_f > max(peak_abs_v_floor, tight_thr * peak_abs_v_frac)
    peak_map &= search > 0

    ys, xs = np.where(peak_map)
    if len(xs) == 0:
        # Fallback: peaks on distance alone
        dil_d = cv2.dilate(dist, kernel)
        peak_map = (dist >= dil_d - 1e-6) & (dist > fallback_dist_min) & (search > 0)
        ys, xs = np.where(peak_map)

    if len(xs) == 0:
        return _cores_from_contours(bloom, min_area, max_blobs)

    scores = response[ys, xs] if response.max() > 0 else dist[ys, xs].astype(np.float32)
    max_peaks = int(max_blobs) if max_blobs and max_blobs > 0 else 4
    max_peaks = max(max_peaks, int(candidates))
    cores = _nms_peaks(ys, xs, scores.astype(np.float32), min_sep=float(sep), max_peaks=max_peaks)

    # Sub-pixel refine on response with a tiny window
    refined: List[Core] = []
    for x, y in cores:
        if plateau_min_px > 0:
            pc = _plateau_centre(v_f, x, y, int(plateau_min_px))
            if pc is not None:
                refined.append(pc)
                continue
        xi, yi = int(round(x)), int(round(y))
        r = int(refine_half_window)
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        patch = response[y0:y1, x0:x1]
        if patch.size == 0 or float(patch.sum()) <= 1e-6:
            refined.append((x, y))
            continue
        # Intensity-weighted centroid in patch
        yy, xx = np.mgrid[y0:y1, x0:x1]
        s = float(patch.sum())
        refined.append((float((xx * patch).sum() / s), float((yy * patch).sum() / s)))
    return refined


def bloom_mask_and_cores(
    bgr: np.ndarray,
    *,
    v_thresh: int = 180,
    use_cyan_assist: bool = True,
    cyan_h_lo: int = 80,
    cyan_h_hi: int = 110,
    cyan_s_lo: int = 20,
    cyan_v_lo: int = 60,
    open_k: int = 3,
    close_k: int = 7,
    min_area: float = 20.0,
    max_blobs: int = 4,
    peak_mode: bool = True,
    peak_sep: int = 28,
    core_pct: int = 92,
    search_dilate_k: int = 9,
    dog_sigma_small: float = 1.2,
    dog_sigma_large: float = 4.0,
    tight_floor: float = 200.0,
    tight_default: float = 220.0,
    tight_erode_k: int = 3,
    response_min: float = 0.05,
    peak_abs_v_floor: float = 160.0,
    peak_abs_v_frac: float = 0.7,
    refine_half_window: int = 2,
    fallback_dist_min: float = 1.0,
    candidates: int = 0,
    plateau_min_px: int = 0,
    bg_subtract: bool = False,
    bg_ref_v: float = 50.0,
    bg_row_pct: float = 10.0,
) -> Tuple[np.ndarray, List[Core]]:
    """Bloom mask for display + robust cores (peaks when lights merge).

    Args:
        v_thresh: Bloom mask brightness floor (HSV V).
        peak_mode: If True, find cores as local peaks (needed when close/bright).
        peak_sep: Minimum pixel separation between cores (NMS).
        core_pct: Percentile of bloom V used for tight core mask (80–99).
        max_blobs: Keep at most this many cores (4 for the dock).
        candidates: if larger than max_blobs, up to this many peaks are returned (strongest first) so the caller can pick the ring among them.
        bg_subtract: remove the water's brightness above `bg_ref_v` (per image row) before thresholding. No effect when the water is dark (clear).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if bg_subtract:
        ex = background_excess(hsv[:, :, 2], bg_ref_v, bg_row_pct)
        if float(ex.max()) > 0.0:
            hsv = hsv.copy()
            hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) - ex, 0, 255).astype(np.uint8)
    v = hsv[:, :, 2]
    mask = _build_bloom_mask(
        hsv,
        v_thresh=v_thresh,
        use_cyan_assist=use_cyan_assist,
        cyan_h_lo=cyan_h_lo,
        cyan_h_hi=cyan_h_hi,
        cyan_s_lo=cyan_s_lo,
        cyan_v_lo=cyan_v_lo,
        open_k=open_k,
        close_k=close_k,
    )

    if peak_mode:
        cores = _cores_from_peaks(
            v,
            mask,
            peak_sep=peak_sep,
            core_pct=core_pct,
            max_blobs=max_blobs if max_blobs > 0 else 4,
            min_area=min_area,
            search_dilate_k=search_dilate_k,
            dog_sigma_small=dog_sigma_small,
            dog_sigma_large=dog_sigma_large,
            tight_floor=tight_floor,
            tight_default=tight_default,
            tight_erode_k=tight_erode_k,
            response_min=response_min,
            peak_abs_v_floor=peak_abs_v_floor,
            peak_abs_v_frac=peak_abs_v_frac,
            refine_half_window=refine_half_window,
            fallback_dist_min=fallback_dist_min,
            candidates=candidates,
            plateau_min_px=plateau_min_px,
        )
    else:
        cores = _cores_from_contours(mask, min_area, max_blobs)
    return mask, cores


def draw_cores(bgr: np.ndarray, cores: List[Core]) -> np.ndarray:
    """Overlay core markers on a BGR copy."""
    vis = bgr.copy()
    for i, (x, y) in enumerate(cores):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(vis, pt, 8, (0, 0, 255), 2)
        cv2.drawMarker(vis, pt, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            vis,
            str(i),
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        f"cores={len(cores)}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis
