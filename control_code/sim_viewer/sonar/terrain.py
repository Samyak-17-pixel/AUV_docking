"""A very uneven seabed for the offline simulator, shared by the 3D view, the side-scan sonar and the tests. Pure numpy (+ numba when present), no ROS, no VTK.

World frame = NED like everything else: x north, y east, depth z POSITIVE DOWN. `Terrain` holds, on a regular grid (default 300 x 300 m at 0.25 m cells):
  z      depth of the sea floor [m] (bigger = deeper)
  refl   backscatter strength of the surface at 450 kHz [dB] (Lambert coefficient: rock > gravel > sand > mud, man-made objects highest)
  cls    surface class 0 mud, 1 sand, 2 gravel, 3 rock, 4 object
and answers `height(x, y)`, `normal(x, y)`, `reflectivity_db(x, y)` (bilinear, vectorised, clamped at the edge). Built from a seed, so the same seed gives the same seabed in every process
(viewer, sonar node, tests, replays). Ingredients: fractal (power-law spectrum) relief, ridges, trenches, boulder fields, sand ripples, and a few man-made objects (boxes, a pipe)
that stand on the bottom and cast acoustic shadows. A calm, gently sloping patch around the dock keeps the docking world as it was (dock at 3 m depth, floor ~11 m under it).
ALL numbers (relief, backscatter dB per class) are ASSUMPTIONS; the real sea floor is not known.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys as _sys

_cov_saved = _sys.modules.get("coverage", False)
_sys.modules["coverage"] = None                          # numba 0.63 crashes on import next to the old system 'coverage' package (no coverage.types): hide it while importing
try:                                                    # numba makes the sonar ray marching ~100x faster; everything also works (slowly) without it
    import numba as _nb
    njit = _nb.njit
    HAVE_NUMBA = True
except Exception:                                        # noqa: BLE001
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if (len(a) == 1 and callable(a[0])) else deco
finally:
    if _cov_saved is False:
        _sys.modules.pop("coverage", None)
    else:
        _sys.modules["coverage"] = _cov_saved

DEFAULTS: Dict[str, object] = dict(
    seed=5,
    size_m=300.0,                  # side of the square area [m]
    center_m=[0.0, 0.0],           # (north, east) of the area centre
    dx_m=0.25,                     # grid cell [m]
    depth_m=14.0,                  # mean floor depth [m]
    relief_m=5.0,                  # fractal relief amplitude [m] (peak to peak is about 2.5 x this)
    min_depth_m=5.0,               # the floor never gets shallower than this [m]
    max_depth_m=26.0,
    ridges=0.35,                   # weight of the ridged (rock crest) component
    trenches=7,                    # number of trenches
    trench_depth_m=3.5,
    boulders=320,                  # boulders (in rocky areas)
    boulder_radius_m=[0.4, 1.7],
    ripples_m=0.04,                # sand ripple amplitude [m] (wavelength ~1.5 m)
    objects=True,                  # stamp the man-made objects below
    dock_xy_m=[10.0, 0.0],
    dock_floor_depth_m=11.0,       # the floor under the dock
    calm_slope=[0.012, 0.02],      # gentle slope of the calm patch (north, east) [m per m]
    calm_radius_m=24.0,            # around the dock: gentle floor, no boulders
    refl_db=dict(mud=-33.0, sand=-27.0, gravel=-22.0, rock=-17.0, object=-9.0),
)

# man-made objects: (kind, params) in world metres. boxes: centre x, y, size x, y, height (and yaw deg); pipe: from (x,y) to (x,y), radius
DEFAULT_OBJECTS: List[dict] = [
    {"kind": "box", "xy": [-38.0, 14.0], "size": [4.0, 2.4, 2.2], "yaw_deg": 20.0},         # a container
    {"kind": "box", "xy": [-52.0, 4.0], "size": [1.6, 1.6, 1.6], "yaw_deg": 0.0},           # a block
    {"kind": "box", "xy": [-20.0, 24.0], "size": [6.0, 1.2, 0.9], "yaw_deg": 75.0},         # a slab (wreck-like)
    {"kind": "pipe", "from": [-65.0, -20.0], "to": [-15.0, -6.0], "radius_m": 0.35},        # a pipeline lying on the bottom
]


def _fft_noise(n: int, rng: np.random.Generator, beta: float = 2.8, kmin: int = 1) -> np.ndarray:
    """Fractal field: white noise filtered with a k^(-beta/2) amplitude spectrum, zero mean, unit std (n x n)."""
    w = rng.normal(size=(n, n))
    F = np.fft.rfft2(w)
    ky = np.fft.fftfreq(n)[:, None] * n
    kx = np.fft.rfftfreq(n)[None, :] * n
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0
    F *= (np.maximum(k, kmin) ** (-beta / 2.0))
    F[0, 0] = 0.0
    f = np.fft.irfft2(F, s=(n, n))
    return (f - f.mean()) / (f.std() + 1e-12)


def _band_noise(n: int, rng: np.random.Generator, k_lo: float, k_hi: float) -> np.ndarray:
    """Band-limited noise: structure only at wavelengths between n/k_hi and n/k_lo cells (smooth band edges), zero mean, unit std."""
    F = np.fft.rfft2(rng.normal(size=(n, n)))
    ky = np.fft.fftfreq(n)[:, None] * n
    kx = np.fft.rfftfreq(n)[None, :] * n
    k = np.hypot(kx, ky)
    band = np.exp(-0.5 * (np.log(np.maximum(k, 1e-6) / math.sqrt(k_lo * k_hi)) / (0.5 * math.log(k_hi / k_lo))) ** 2)
    F *= band
    F[0, 0] = 0.0
    f = np.fft.irfft2(F, s=(n, n))
    return (f - f.mean()) / (f.std() + 1e-12)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@njit(cache=True)
def _bilinear(z, x0, y0, inv_dx, x, y):
    """Bilinear sample of grid z[ix, iy] (cell (0,0) at x0, y0), clamped to the grid."""
    nx, ny = z.shape
    fx = (x - x0) * inv_dx
    fy = (y - y0) * inv_dx
    if fx < 0.0:
        fx = 0.0
    if fy < 0.0:
        fy = 0.0
    if fx > nx - 1.001:
        fx = nx - 1.001
    if fy > ny - 1.001:
        fy = ny - 1.001
    i = int(fx)
    j = int(fy)
    tx = fx - i
    ty = fy - j
    return (z[i, j] * (1 - tx) * (1 - ty) + z[i + 1, j] * tx * (1 - ty) + z[i, j + 1] * (1 - tx) * ty + z[i + 1, j + 1] * tx * ty)


class Terrain:
    def __init__(self, cfg: Optional[dict] = None) -> None:
        c = dict(DEFAULTS)
        c.update({k: v for k, v in (cfg or {}).items() if v is not None})
        self.cfg = c
        self.seed = int(c["seed"])
        self.dx = float(c["dx_m"])
        size = float(c["size_m"])
        self.n = int(round(size / self.dx))
        cx, cy = c["center_m"]
        self.x0 = float(cx) - 0.5 * self.n * self.dx
        self.y0 = float(cy) - 0.5 * self.n * self.dx
        self.inv_dx = 1.0 / self.dx
        self.objects: List[dict] = list(c.get("object_list", DEFAULT_OBJECTS)) if c.get("objects", True) else []
        self.z, self.refl, self.cls = self._build()

    # ------------------------------------------------------------------ construction
    def _grid(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + self.dx * np.arange(self.n, dtype=np.float32)
        ys = self.y0 + self.dx * np.arange(self.n, dtype=np.float32)
        return np.meshgrid(xs, ys, indexing="ij")

    def _build(self):
        c = self.cfg
        rng = np.random.default_rng(self.seed)
        n = self.n
        X, Y = self._grid()
        relief = float(c["relief_m"])
        # relief at several scales in METRES (a single power-law spectrum gave 70 deg slopes at 0.25 m cells): broad hills, 20-40 m ridges and valleys, 3-8 m lumps, rock texture
        base = _fft_noise(n, rng, 3.8)
        wl = lambda lam: n * self.dx / lam                                                  # wavenumber (cycles per area) for a wavelength lam [m]
        mid = _band_noise(n, rng, wl(60.0), wl(18.0))
        ridge = 1.0 - np.abs(_band_noise(n, rng, wl(45.0), wl(14.0)))                       # ridged noise: sharp crests
        ridge = (ridge - ridge.mean()) / (ridge.std() + 1e-9)
        fine = _band_noise(n, rng, wl(8.0), wl(2.5))
        h = 0.62 * base + 0.16 * mid + float(c["ridges"]) * 0.26 * ridge + 0.012 * fine      # height above the mean (relief units): + = shallower
        z = float(c["depth_m"]) - relief * h
        # trenches: smooth random curves, a deep groove with a rounded profile
        pts = []
        for _ in range(int(c["trenches"])):
            p0 = np.array([self.x0, self.y0]) + rng.uniform(0.1, 0.9, 2) * self.n * self.dx
            ang = rng.uniform(0, 2 * math.pi)
            curv = rng.normal(0, 0.012)
            t = np.arange(0, 140, 0.5)
            a = ang + curv * t
            xs = p0[0] + np.cumsum(np.cos(a)) * 0.5
            ys = p0[1] + np.cumsum(np.sin(a)) * 0.5
            wid = rng.uniform(4.0, 11.0)
            dep = float(c["trench_depth_m"]) * rng.uniform(0.5, 1.2)
            pts.append((np.column_stack([xs, ys]), wid, dep))
        try:
            from scipy.spatial import cKDTree
            flat = np.column_stack([X.ravel(), Y.ravel()])
            for line, wid, dep in pts:
                d, _ = cKDTree(line).query(flat, k=1)
                z += (dep * np.exp(-0.5 * (d / wid) ** 2)).reshape(n, n)
        except Exception:                                                                   # noqa: BLE001
            pass
        # a calm, gently sloping patch around the dock (the docking world stays as it was)
        dxy = np.array(c["dock_xy_m"], float)
        r = np.hypot(X - dxy[0], Y - dxy[1])
        calm = 1.0 - _smoothstep((r - float(c["calm_radius_m"])) / 18.0)
        plane = float(c["dock_floor_depth_m"]) + float(c["calm_slope"][0]) * (X - dxy[0]) + float(c["calm_slope"][1]) * (Y - dxy[1])
        z = (1 - calm) * z + calm * (plane + 0.15 * relief * 0.1 * base)
        z = np.clip(z, float(c["min_depth_m"]), float(c["max_depth_m"])).astype(np.float32)
        # classes from depth, slope and noise
        gx, gy = np.gradient(z, self.dx, self.dx)
        slope = np.hypot(gx, gy)
        tex = _fft_noise(n, rng, 1.6, kmin=8)
        patch = _fft_noise(n, rng, 3.2, kmin=1)
        cls = np.ones((n, n), np.int8)                                                      # sand
        cls[(patch < -0.55) & (z > np.percentile(z, 55))] = 0                               # mud in the deeper basins
        cls[(patch > 0.65)] = 2                                                             # gravel
        cls[(slope > 0.55) | (h > np.percentile(h, 92))] = 3                                # rock on steep slopes and high crests
        cls[calm > 0.5] = np.where(cls[calm > 0.5] == 3, 1, cls[calm > 0.5])                # no rock in the calm patch
        # boulders in the rocky and gravel areas, away from the dock
        rmin, rmax = c["boulder_radius_m"]
        cand = np.argwhere(((cls == 3) | (cls == 2)) & (calm < 0.05))
        if len(cand):
            sel = cand[rng.integers(0, len(cand), size=int(c["boulders"]))]
            for (i, j) in sel:
                rad = rng.uniform(rmin, rmax)
                hh = rad * rng.uniform(0.6, 1.4)
                self._stamp_dome(z, cls, i, j, rad, hh)
        # sand ripples
        sandy = cls == 1
        k = 2 * math.pi / 1.5
        ang = rng.uniform(0, math.pi)
        phase = k * (X * math.cos(ang) + Y * math.sin(ang)) + 3.0 * _fft_noise(n, rng, 3.0, kmin=2)
        z = z - (float(c["ripples_m"]) * np.sin(phase) * sandy).astype(np.float32)
        # man-made objects
        for ob in self.objects:
            self._stamp_object(z, cls, X, Y, ob)
        rd = c["refl_db"]
        table = np.array([rd["mud"], rd["sand"], rd["gravel"], rd["rock"], rd["object"]], np.float32)
        refl = table[cls] + 1.6 * tex.astype(np.float32)                                    # a little texture so the mosaic has detail
        return z.astype(np.float32), refl.astype(np.float32), cls

    def _stamp_dome(self, z, cls, i, j, rad, hh) -> None:
        w = int(math.ceil(rad / self.dx)) + 1
        i0, i1, j0, j1 = max(0, i - w), min(self.n, i + w + 1), max(0, j - w), min(self.n, j + w + 1)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        d = np.hypot(ii - i, jj - j) * self.dx
        dome = hh * np.sqrt(np.clip(1.0 - (d / rad) ** 2, 0.0, 1.0))
        ref = float(z[i, j])                                                              # a dome standing on the floor: the depth gets SMALLER
        sub = z[i0:i1, j0:j1]
        z[i0:i1, j0:j1] = np.where(dome > 0, np.minimum(sub, ref - dome), sub).astype(np.float32)
        cls[i0:i1, j0:j1][d < rad] = 3

    def _stamp_object(self, z, cls, X, Y, ob) -> None:
        if ob["kind"] == "box":
            cx, cy = ob["xy"]
            sx, sy, sh = ob["size"]
            a = math.radians(ob.get("yaw_deg", 0.0))
            dxm, dym = X - cx, Y - cy
            u = dxm * math.cos(a) + dym * math.sin(a)
            v = -dxm * math.sin(a) + dym * math.cos(a)
            m = (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2)
        else:                                                                                 # pipe lying on the bottom
            (x0, y0), (x1, y1) = ob["from"], ob["to"]
            rad = float(ob["radius_m"])
            L = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            t = np.clip((X - x0) * ux + (Y - y0) * uy, 0.0, L)
            d = np.hypot(X - (x0 + t * ux), Y - (y0 + t * uy))
            m = d <= rad
            sh = rad * 2.0
            hh = np.sqrt(np.clip(1 - (d / rad) ** 2, 0, 1)) * sh
            z[m] = (z[m] - hh[m]).astype(np.float32)
            cls[m] = 4
            return
        base = z[m].mean() if m.any() else 0.0
        z[m] = np.minimum(z[m], base - sh).astype(np.float32)
        cls[m] = 4

    # ------------------------------------------------------------------ queries
    def height(self, x, y):
        """Depth of the sea floor at world (x, y) [m, positive down]. Scalars or arrays (vectorised)."""
        xa, ya = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
        out = np.empty(xa.shape, np.float64)
        flat_x, flat_y, flat_o = xa.ravel(), ya.ravel(), out.ravel()
        _height_many(self.z, self.x0, self.y0, self.inv_dx, flat_x, flat_y, flat_o)
        return out if out.ndim else float(out)

    def reflectivity_db(self, x, y):
        xa, ya = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
        out = np.empty(xa.shape, np.float64)
        _height_many(self.refl, self.x0, self.y0, self.inv_dx, xa.ravel(), ya.ravel(), out.ravel())
        return out if out.ndim else float(out)

    def normal(self, x, y, eps: Optional[float] = None):
        """Unit surface normal pointing up out of the sea floor (NED: -z is up), shape (..., 3)."""
        e = self.dx if eps is None else eps
        zx = (self.height(np.asarray(x) + e, y) - self.height(np.asarray(x) - e, y)) / (2 * e)
        zy = (self.height(x, np.asarray(y) + e) - self.height(x, np.asarray(y) - e)) / (2 * e)
        n = np.stack([zx, zy, -np.ones_like(zx)], axis=-1)          # depth increases with z; the up-normal is (dz/dx, dz/dy, -1) normalised
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.x0 + (self.n - 1) * self.dx, self.y0, self.y0 + (self.n - 1) * self.dx)

    def altitude(self, x: float, y: float, z: float) -> float:
        """Straight-down distance from a point at depth z to the floor [m]."""
        return float(self.height(x, y)) - float(z)

    def decimated(self, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """xs, ys, z, refl on every `step`-th cell (for the 3D mesh)."""
        sl = slice(0, self.n, int(step))
        xs = self.x0 + self.dx * np.arange(self.n)[sl]
        ys = self.y0 + self.dx * np.arange(self.n)[sl]
        return xs, ys, self.z[sl, sl], self.refl[sl, sl]


@njit(cache=True)
def _height_many(z, x0, y0, inv_dx, xs, ys, out):
    for k in range(xs.shape[0]):
        out[k] = _bilinear(z, x0, y0, inv_dx, xs[k], ys[k])


_CACHE: Dict[tuple, Terrain] = {}


def get_terrain(cfg: Optional[dict] = None) -> Terrain:
    """One Terrain per distinct config in this process (building takes a few seconds)."""
    key = tuple(sorted((k, repr(v)) for k, v in (cfg or {}).items()))
    if key not in _CACHE:
        _CACHE[key] = Terrain(cfg)
    return _CACHE[key]
