"""Bloom light mask + core estimation for funnel dock lights.

Near the dock, bloom merges into one white blob. Contour centroids then fail
(cores=1). Cores are therefore found as local intensity peaks (DoG + NMS),
while the bloom mask is kept for visualization.
"""

from __future__ import annotations

from typing import List, Tuple

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
        search, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )

    # --- Response A: DoG (positive lobes ≈ bright cores) ---
    g_small = cv2.GaussianBlur(v_f, (0, 0), 1.2)
    g_large = cv2.GaussianBlur(v_f, (0, 0), 4.0)
    dog = cv2.subtract(g_small, g_large)
    dog = np.maximum(dog, 0.0)
    dog[search == 0] = 0.0

    # --- Response B: distance transform on adaptive tight core mask ---
    inside = v_f[bloom > 0]
    if inside.size > 0:
        pct = float(np.clip(core_pct, 80, 99))
        tight_thr = float(np.percentile(inside, pct))
        # Never go below a floor so dark frames still work
        tight_thr = max(tight_thr, 200.0)
    else:
        tight_thr = 220.0
    tight = np.zeros_like(bloom)
    tight[(v_f >= tight_thr) & (search > 0)] = 255
    # Split merged lobes a bit
    tight = cv2.erode(
        tight, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
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
    peak_map = (response >= dil - 1e-6) & (response > 0.05)
    # Also require some absolute brightness so murk peaks die
    peak_map &= v_f > max(160.0, tight_thr * 0.7)
    peak_map &= search > 0

    ys, xs = np.where(peak_map)
    if len(xs) == 0:
        # Fallback: peaks on distance alone
        dil_d = cv2.dilate(dist, kernel)
        peak_map = (dist >= dil_d - 1e-6) & (dist > 1.0) & (search > 0)
        ys, xs = np.where(peak_map)

    if len(xs) == 0:
        return _cores_from_contours(bloom, min_area, max_blobs)

    scores = response[ys, xs] if response.max() > 0 else dist[ys, xs].astype(np.float32)
    max_peaks = int(max_blobs) if max_blobs and max_blobs > 0 else 4
    cores = _nms_peaks(ys, xs, scores.astype(np.float32), min_sep=float(sep), max_peaks=max_peaks)

    # Sub-pixel refine on response with a tiny window
    refined: List[Core] = []
    for x, y in cores:
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - 2), min(w, xi + 3)
        y0, y1 = max(0, yi - 2), min(h, yi + 3)
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
) -> Tuple[np.ndarray, List[Core]]:
    """Bloom mask for display + robust cores (peaks when lights merge).

    Args:
        v_thresh: Bloom mask brightness floor (HSV V).
        peak_mode: If True, find cores as local peaks (needed when close/bright).
        peak_sep: Minimum pixel separation between cores (NMS).
        core_pct: Percentile of bloom V used for tight core mask (80–99).
        max_blobs: Keep at most this many cores (4 for the dock).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
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
