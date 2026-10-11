"""Georeferenced backscatter MOSAIC from side-scan pings + vehicle poses (pure numpy). Like a real survey mosaic: each range bin is placed on the ground with the SLANT-RANGE correction
for a flat bottom (ground range = sqrt(R^2 - altitude^2), measured with the altimeter) at the vehicle's heading +- 90 deg, and overlapping bins are averaged. On uneven terrain this flat-bottom
assumption distorts features (as it does in real mosaics), which is part of what the simulator is for. The nadir gap (R < altitude) is left empty, and so is everything beyond max_slant_factor x altitude (outside the beam: noise only)."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


class Mosaic:
    def __init__(self, extent: Tuple[float, float, float, float], cell_m: float = 0.5, max_slant_factor: float = 1.9) -> None:
        self.max_slant_factor = float(max_slant_factor)            # bins beyond this x altitude are outside the beam (noise only) and are not used; 1.9 = 1/cos(58 deg): 30 deg mount + 25 deg half beam + 3 deg margin
        self.x0, self.x1, self.y0, self.y1 = extent
        self.cell = float(cell_m)
        self.nx = int(math.ceil((self.x1 - self.x0) / self.cell)) + 1
        self.ny = int(math.ceil((self.y1 - self.y0) / self.cell)) + 1
        self.sum = np.zeros((self.nx, self.ny), np.float64)
        self.cnt = np.zeros((self.nx, self.ny), np.float32)
        self.pings = 0

    def add_ping(self, side: str, intensity, meta: Dict[str, float]) -> None:
        nb = len(intensity)
        if nb == 0:
            return
        pos = np.asarray(meta.get("pos", [0.0, 0.0, 0.0]), float)
        eul = np.asarray(meta.get("eul", [0.0, 0.0, 0.0]), float)
        alt = float(meta.get("altitude_m", float("nan")))
        if not math.isfinite(alt) or alt <= 0.0:
            return
        rng = float(meta.get("range_m", 30.0))
        r0 = float(meta.get("start_range_m", 0.0))
        R = r0 + (np.arange(nb) + 0.5) * (rng - r0) / nb
        ok = (R > alt * 1.0005) & (R < alt * self.max_slant_factor)
        g = np.sqrt(np.maximum(R[ok] ** 2 - alt ** 2, 0.0))
        psi = eul[2]
        right = np.array([-math.sin(psi), math.cos(psi)])
        sgn = 1.0 if side == "starboard" else -1.0
        px = pos[0] + sgn * g * right[0]
        py = pos[1] + sgn * g * right[1]
        ix = np.floor((px - self.x0) / self.cell).astype(int)
        iy = np.floor((py - self.y0) / self.cell).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        v = np.asarray(intensity, np.float64)[ok] / 65535.0
        np.add.at(self.sum, (ix[inside], iy[inside]), v[inside])
        np.add.at(self.cnt, (ix[inside], iy[inside]), 1.0)
        self.pings += 1

    def image(self) -> np.ndarray:
        """Mean intensity 0..1, NaN where nothing was seen. Indexed [ix (north), iy (east)]."""
        with np.errstate(invalid="ignore", divide="ignore"):
            im = self.sum / self.cnt
        im[self.cnt == 0] = np.nan
        return im.astype(np.float32)

    def covered_mask(self) -> np.ndarray:
        return self.cnt > 0

    def coverage(self, region: Optional[Tuple[float, float, float, float]] = None) -> float:
        """Fraction of the cells in `region` (x0, x1, y0, y1; default all) that were insonified."""
        if region is None:
            return float(self.covered_mask().mean())
        ix0, ix1 = int((region[0] - self.x0) / self.cell), int((region[1] - self.x0) / self.cell)
        iy0, iy1 = int((region[2] - self.y0) / self.cell), int((region[3] - self.y0) / self.cell)
        sub = self.covered_mask()[max(ix0, 0):ix1, max(iy0, 0):iy1]
        return float(sub.mean()) if sub.size else 0.0
