"""Survey coverage on a grid (pure Python, no Qt): which cells a sensor swath of a given width covers along a path, and how much of a planned path a travelled trail covered."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set, Tuple

Cell = Tuple[int, int]


def _disk(radius_cells: float) -> List[Cell]:
    r = int(math.ceil(radius_cells))
    return [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1) if i * i + j * j <= radius_cells * radius_cells + 1e-9]


def covered_cells(points: Sequence[Tuple[float, float]], swath_m: float, cell_m: float = 0.5) -> Set[Cell]:
    """Cells within swath_m / 2 of the polyline through `points` (x north, y east). A single point covers one disc."""
    if swath_m <= 0 or cell_m <= 0 or not points:
        return set()
    disk = _disk(0.5 * swath_m / cell_m)
    out: Set[Cell] = set()

    def stamp(x: float, y: float) -> None:
        ci, cj = int(math.floor(x / cell_m)), int(math.floor(y / cell_m))
        for di, dj in disk:
            out.add((ci + di, cj + dj))

    stamp(points[0][0], points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / (0.5 * cell_m))))
        for k in range(1, n + 1):
            f = k / n
            stamp(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
    return out


def coverage_fraction(plan: Sequence[Tuple[float, float]], trail: Sequence[Tuple[float, float]], swath_m: float, cell_m: float = 0.5) -> float:
    """Share of the planned swath that the travelled trail's swath covered (0..1). 1.0 for an empty plan."""
    want = covered_cells(plan, swath_m, cell_m)
    if not want:
        return 1.0
    got = covered_cells(trail, swath_m, cell_m)
    return len(want & got) / len(want)


def area_m2(cells: Iterable[Cell], cell_m: float = 0.5) -> float:
    return len(set(cells)) * cell_m * cell_m
