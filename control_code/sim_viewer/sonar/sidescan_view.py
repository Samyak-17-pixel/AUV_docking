"""Drawing of side-scan data: the WATERFALL (time scrolling down, port left | starboard right, range growing outward from the nadir gap in the middle) and the MOSAIC (north up). Pure
numpy + cv2, so the viewer's Sonar tab, the survey script and the tests use the same pictures."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_X = np.linspace(0.0, 1.0, 256)
_LUT = np.stack([np.clip(60 * _X ** 2.2, 0, 255), np.clip(215 * _X ** 1.1, 0, 255), np.clip(255 * _X ** 0.8, 0, 255)], axis=1).astype(np.uint8)   # B, G, R: black -> amber -> pale yellow


def amber(v: np.ndarray) -> np.ndarray:
    """0..1 floats (NaN = nothing) -> BGR uint8; NaN becomes a dark blue-grey."""
    v = np.asarray(v, np.float32)
    nan = ~np.isfinite(v)
    idx = np.clip(np.nan_to_num(v) * 255.0, 0, 255).astype(np.uint8)
    out = _LUT[idx]
    out[nan] = (46, 38, 32)
    return out


def _resample(row: np.ndarray, n: int) -> np.ndarray:
    if len(row) == n:
        return row.astype(np.float32)
    return np.interp(np.linspace(0, len(row) - 1, n), np.arange(len(row)), row.astype(np.float32)).astype(np.float32)


def waterfall_image(port: Sequence[np.ndarray], stbd: Sequence[np.ndarray], size: Tuple[int, int], range_m: float = 30.0, gap: int = 6, title: str = "") -> np.ndarray:
    """port / stbd: lists of uint16 pings, NEWEST LAST. Returns BGR (h, w): newest ping at the top. A missing ping (one side ahead of the other) is shown as black."""
    w, h = size
    half = (w - gap) // 2
    n = min(max(len(port), len(stbd)), h)
    img = np.full((h, w, 3), (32, 28, 24), np.uint8)
    if n:
        P = np.zeros((n, half), np.float32)
        S = np.zeros((n, half), np.float32)
        for k in range(n):                                            # row k = k-th newest
            if k < len(port):
                P[k] = _resample(port[-1 - k], half) / 65535.0
            if k < len(stbd):
                S[k] = _resample(stbd[-1 - k], half) / 65535.0
        img[:n, :half] = amber(P[:, ::-1])                            # port: far range on the left
        img[:n, half + gap:half + gap + half] = amber(S)
    cv2.line(img, (half + gap // 2, 0), (half + gap // 2, h), (90, 80, 70), 1)
    for frac in (0.25, 0.5, 0.75, 1.0):                               # range ticks
        r = frac * range_m
        xs = int(half * (1 - frac)), half + gap + int(half * frac) - 1
        for x in xs:
            cv2.line(img, (x, h - 8), (x, h - 1), (200, 200, 200), 1)
        cv2.putText(img, f"{r:.0f}m", (max(xs[0] - 4, 0), h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r:.0f}m", (min(xs[1] - 16, w - 30), h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(img, "PORT", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, "STARBOARD", (w - 92, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (w // 2 - 70, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def mosaic_image(m, scale: int = 1, path_xy: Optional[np.ndarray] = None, vehicle_xy: Optional[Tuple[float, float]] = None, crop: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """BGR picture of a Mosaic, north up / east right, optional vehicle path and position. crop = (x0, x1, y0, y1) world window."""
    im = m.image()
    if crop is not None:
        ix0, ix1 = int((crop[0] - m.x0) / m.cell), int((crop[1] - m.x0) / m.cell)
        iy0, iy1 = int((crop[2] - m.y0) / m.cell), int((crop[3] - m.y0) / m.cell)
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        sub = im[ix0:ix1, iy0:iy1]
        ox, oy = m.x0 + ix0 * m.cell, m.y0 + iy0 * m.cell
    else:
        sub, ox, oy = im, m.x0, m.y0
    nxs = sub.shape[0]
    img = amber(sub[::-1, :])
    if scale != 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    def px(x, y):
        return int((y - oy) / m.cell * scale), int((nxs - 1 - (x - ox) / m.cell) * scale)
    if path_xy is not None and len(path_xy) > 1:
        pts = np.array([px(x, y) for x, y in path_xy[::max(1, len(path_xy) // 800)]], np.int32)
        cv2.polylines(img, [pts], False, (255, 200, 90), 1, cv2.LINE_AA)
    if vehicle_xy is not None:
        cv2.circle(img, px(*vehicle_xy), 4, (60, 60, 255), -1, cv2.LINE_AA)
    return img
