"""Underwater environment for the 3D scene (pure VTK + numpy, no Qt): sea floor with procedural terrain, rippling water surface, light shafts, drifting
particles ('marine snow'), cones of light from the dock lights, and a soft shadow under the vehicle.

Everything is generated from a seed, so the picture is the same on every start. All sizes and colours come from `viewer.look3d` in sim_viewer.yaml.
The sea floor is only decoration (the dock and vehicle do not interact with it); the real mavsim scene has its own floor, whose depth is not known here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support as ns


def _fbm(n: int, rng: np.random.Generator, octaves: int = 5, persistence: float = 0.5) -> np.ndarray:
    """Fractal noise on an n x n grid in [0, 1] (sum of upsampled random grids; no external dependency)."""
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        k = 2 ** (o + 2)
        coarse = rng.random((k + 1, k + 1))
        xs = np.linspace(0, k, n)
        i0 = np.minimum(xs.astype(int), k - 1)
        f = xs - i0
        f = f * f * (3 - 2 * f)                                              # smoothstep
        a = coarse[i0][:, i0] * (1 - f)[None, :] + coarse[i0][:, i0 + 1] * f[None, :]
        b = coarse[i0 + 1][:, i0] * (1 - f)[None, :] + coarse[i0 + 1][:, i0 + 1] * f[None, :]
        out += amp * (a * (1 - f)[:, None] + b * f[:, None])
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / max(np.ptp(out), 1e-9)


def _poly_from_grid(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, colors: np.ndarray) -> vtk.vtkPolyData:
    ny, nx = z.shape
    X, Y = np.meshgrid(xs, ys)
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.column_stack([X.ravel(), Y.ravel(), z.ravel()]).astype(np.float32), deep=True))
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.concatenate([np.column_stack([np.full(len(a), 3), a, b, c]), np.column_stack([np.full(len(a), 3), a, c, d])]).astype(np.int64)
    cells = vtk.vtkCellArray()
    cells.SetCells(len(tris), ns.numpy_to_vtkIdTypeArray(tris.ravel(), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    col = ns.numpy_to_vtk(colors.astype(np.uint8), deep=True)
    col.SetName("Colors")
    pd.GetPointData().SetScalars(col)
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(pd)
    nrm.SetFeatureAngle(80.0)
    nrm.SplittingOff()
    nrm.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(nrm.GetOutput())
    return out


class Environment:
    def __init__(self, ren: vtk.vtkRenderer, look: dict, dock_pos: np.ndarray, mouth_normal: np.ndarray, light_positions: List[np.ndarray], terrain=None) -> None:
        self.ren = ren
        self.terrain = terrain                       # terrain.Terrain: the SAME sea floor the side-scan sonar maps (None = the old decorative floor)
        self.cfg = look
        self.rng = np.random.default_rng(int(look.get("seed", 3)))
        self.dock_pos = np.asarray(dock_pos, float)
        self.t = 0.0
        self.floor_z = float(look.get("floor_depth_m", 11.0))
        self.actors: Dict[str, vtk.vtkActor] = {}
        if look.get("floor", True):
            self._build_floor()
        if look.get("surface_waves", True):
            self._build_surface()
        if look.get("light_shafts", True):
            self._build_shafts()
        if look.get("dock_light_cones", True):
            self._build_cones(light_positions, np.asarray(mouth_normal, float))
        if look.get("particles", True):
            self._build_particles()
        if look.get("shadow", True):
            self._build_shadow()

    # ------------------------------------------------------------------ construction
    def _build_terrain_floor(self) -> None:
        """The shared rugged floor (terrain.py) as a 1 m mesh, coloured by surface class and backscatter, lit by a hill shade."""
        t = self.terrain
        step = max(1, int(round(1.0 / t.dx)))
        xs, ys, z, refl = t.decimated(step)                                  # z[ix, iy]
        cls = t.cls[::step, ::step]
        pal = np.array([[92, 78, 60], [190, 170, 125], [150, 140, 120], [104, 106, 108], [170, 60, 50]], float)     # mud, sand, gravel, rock, object
        base = pal[cls]
        gx, gy = np.gradient(z, float(xs[1] - xs[0]), float(ys[1] - ys[0]))
        n = np.stack([gx, gy, -np.ones_like(gx)], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        light = np.array([-0.4, 0.3, -0.85])
        light /= np.linalg.norm(light)
        shade = 0.45 + 0.55 * np.clip(np.einsum("ijk,k->ij", n, light), 0.0, 1.0)
        tone = 0.85 + 0.03 * (refl + 27.0)                                    # louder surfaces a bit lighter
        col = base * (shade * np.clip(tone, 0.6, 1.3))[..., None]
        poly = _poly_from_grid(xs, ys, np.ascontiguousarray(z.T), np.clip(col.transpose(1, 0, 2), 0, 255).reshape(-1, 3))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarModeToUsePointData()
        mapper.ScalarVisibilityOn()
        a = vtk.vtkActor()
        a.SetMapper(mapper)
        a.GetProperty().SetSpecular(0.05)
        a.GetProperty().SetAmbient(0.5)
        self.ren.AddActor(a)
        self.actors["floor"] = a
        self._floor_xyz = None

    def _build_floor(self) -> None:
        if self.terrain is not None:
            self._build_terrain_floor()
            return
        c = self.cfg
        x0, x1 = c.get("floor_x_m", [-14.0, 30.0])
        y0, y1 = c.get("floor_y_m", [-22.0, 22.0])
        n = int(c.get("floor_resolution", 160))
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        h = _fbm(n, self.rng, 6, 0.55)
        ridge = 1.0 - np.abs(2.0 * _fbm(n, self.rng, 4, 0.5) - 1.0)           # ridged noise: rock crests
        X, Y = np.meshgrid(xs, ys)
        # a calmer sandy patch under the approach corridor and dock, rougher further out
        calm = np.exp(-(((X - 8.0) / 12.0) ** 2 + (Y / 6.0) ** 2))
        amp = float(c.get("floor_relief_m", 1.2))
        z = self.floor_z - amp * (0.35 + 0.65 * (1 - calm)) * (0.6 * h + 0.4 * ridge)
        ripples = 0.04 * np.sin(2.4 * X + 1.7 * np.sin(0.35 * Y)) * calm                # sand ripples
        z = z - ripples
        sand = np.array(c.get("sand_color", [201, 180, 130]), float)
        rock = np.array(c.get("rock_color", [96, 98, 100]), float)
        t = np.clip((ridge - 0.62) * 4.0, 0.0, 1.0) * (1 - 0.7 * calm)
        grain = 0.88 + 0.24 * self.rng.random((n, n))
        col = (sand[None, None, :] * (1 - t[..., None]) + rock[None, None, :] * t[..., None]) * grain[..., None] * (0.55 + 0.45 * h[..., None])
        poly = _poly_from_grid(xs, ys, z, np.clip(col, 0, 255).reshape(-1, 3))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarModeToUsePointData()
        mapper.ScalarVisibilityOn()
        a = vtk.vtkActor()
        a.SetMapper(mapper)
        a.GetProperty().SetSpecular(0.05)
        a.GetProperty().SetAmbient(0.35)
        self.ren.AddActor(a)
        self.actors["floor"] = a
        self._floor_xyz = (xs, ys, z)

    def floor_height(self, x: float, y: float) -> float:
        if self.terrain is not None:
            return float(self.terrain.height(x, y))
        if "floor" not in self.actors:
            return self.floor_z
        xs, ys, z = self._floor_xyz
        i = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
        j = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
        return float(z[i, j])

    def _build_surface(self) -> None:
        n = int(self.cfg.get("surface_resolution", 48))
        xs, ys = np.linspace(-14.0, 30.0, n), np.linspace(-22.0, 22.0, n)
        self._surf_xy = np.meshgrid(xs, ys)
        z = np.zeros((n, n))
        poly = _poly_from_grid(xs, ys, z, np.full((n * n, 3), [130, 190, 215]))
        self._surf_poly = poly
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOn()
        a = vtk.vtkActor()
        a.SetMapper(mapper)
        p = a.GetProperty()
        p.SetOpacity(float(self.cfg.get("surface_opacity", 0.32)))
        p.SetSpecular(0.6)
        p.SetSpecularPower(40.0)
        p.SetAmbient(0.6)
        self.ren.AddActor(a)
        self.actors["surface"] = a

    def _build_shafts(self) -> None:
        """Slanted translucent sun-ray planes from the surface down to the floor."""
        n = int(self.cfg.get("shaft_count", 9))
        pts, polys = vtk.vtkPoints(), vtk.vtkCellArray()
        for k in range(n):
            x = self.rng.uniform(-6.0, 22.0)
            y = self.rng.uniform(-9.0, 9.0)
            w = self.rng.uniform(0.35, 1.1)
            slant = np.array([0.35, 0.15])                                        # horizontal drift per metre of depth (sun not overhead)
            zt, zb = 0.0, self.floor_z
            top = np.array([x, y, zt])
            bot = np.array([x + slant[0] * (zb - zt), y + slant[1] * (zb - zt), zb])
            d = np.array([0.0, 1.0, 0.0]) * w
            q = [top - d, top + d, bot + 1.8 * d, bot - 1.8 * d]
            ids = [pts.InsertNextPoint(*p) for p in q]
            polys.InsertNextCell(4)
            for i in ids:
                polys.InsertCellPoint(i)
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(polys)
        a = vtk.vtkActor()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        a.SetMapper(m)
        p = a.GetProperty()
        p.SetColor(0.75, 0.92, 1.0)
        p.SetOpacity(float(self.cfg.get("shaft_opacity", 0.045)))
        p.SetLighting(False)
        self.ren.AddActor(a)
        self.actors["shafts"] = a

    def _build_cones(self, lights: List[np.ndarray], normal: np.ndarray) -> None:
        length = float(self.cfg.get("cone_length_m", 4.5))
        half = math.radians(float(self.cfg.get("cone_half_angle_deg", 41.0)))
        app = vtk.vtkAppendPolyData()
        for p in lights:
            cone = vtk.vtkConeSource()
            cone.SetHeight(length)
            cone.SetRadius(length * math.tan(half))
            cone.SetResolution(28)
            cone.SetCapping(False)
            cone.Update()
            # vtkConeSource: apex at +x of its centre, axis along x. We want the apex at the light and the axis along the mouth normal.
            tr = vtk.vtkTransform()
            tr.Translate(*(p + normal * length / 2.0))
            axis = normal / max(np.linalg.norm(normal), 1e-9)
            x = np.array([1.0, 0.0, 0.0])
            v = np.cross(x, -axis)
            s, c = float(np.linalg.norm(v)), float(x @ -axis)
            if s > 1e-9:
                tr.RotateWXYZ(math.degrees(math.atan2(s, c)), *(v / s))
            elif c < 0:
                tr.RotateWXYZ(180.0, 0, 0, 1)
            f = vtk.vtkTransformPolyDataFilter()
            f.SetTransform(tr)
            f.SetInputData(cone.GetOutput())
            f.Update()
            app.AddInputData(f.GetOutput())
        app.Update()
        a = vtk.vtkActor()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(app.GetOutput())
        a.SetMapper(m)
        pr = a.GetProperty()
        pr.SetColor(1.0, 0.95, 0.6)
        pr.SetOpacity(float(self.cfg.get("cone_opacity", 0.05)))
        pr.SetLighting(False)
        self.ren.AddActor(a)
        self.actors["cones"] = a

    def _build_particles(self) -> None:
        n = int(self.cfg.get("particle_count", 700))
        self._pbox = np.array(self.cfg.get("particle_box_m", [14.0, 14.0, 7.0]), float)
        self._p = self.rng.random((n, 3)) * self._pbox - self._pbox / 2.0
        self._pv = self.rng.normal(0, 1, (n, 3)) * np.array([0.02, 0.02, 0.01]) + np.array([0.0, 0.0, 0.02])       # slow drift, sinking a little
        self._pts = vtk.vtkPoints()
        self._pts.SetData(ns.numpy_to_vtk(self._p.astype(np.float32), deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(self._pts)
        verts = vtk.vtkCellArray()
        for i in range(n):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        pd.SetVerts(verts)
        self._ppd = pd
        a = vtk.vtkActor()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        a.SetMapper(m)
        pr = a.GetProperty()
        pr.SetColor(0.85, 0.95, 1.0)
        pr.SetOpacity(0.55)
        pr.SetPointSize(float(self.cfg.get("particle_size_px", 2.0)))
        pr.SetLighting(False)
        self.ren.AddActor(a)
        self.actors["particles"] = a
        self._pcentre = np.zeros(3)

    def _build_shadow(self) -> None:
        d = vtk.vtkDiskSource()
        d.SetInnerRadius(0.0)
        d.SetOuterRadius(1.0)
        d.SetCircumferentialResolution(36)
        d.Update()
        a = vtk.vtkActor()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(d.GetOutput())
        a.SetMapper(m)
        pr = a.GetProperty()
        pr.SetColor(0.0, 0.0, 0.0)
        pr.SetOpacity(0.35)
        pr.SetLighting(False)
        self.ren.AddActor(a)
        self.actors["shadow"] = a

    # ------------------------------------------------------------------ per frame
    def update(self, dt: float, vehicle_pos: np.ndarray, vehicle_yaw: float) -> None:
        """dt: wall seconds since the last call (waves and particles move in real time, independent of the simulation)."""
        self.t += dt
        if "surface" in self.actors:
            X, Y = self._surf_xy
            amp = float(self.cfg.get("wave_height_m", 0.08))
            z = amp * (np.sin(0.9 * X + 1.3 * self.t) + 0.6 * np.sin(1.4 * Y - 1.7 * self.t + 1.0) + 0.4 * np.sin(0.5 * (X + Y) + 0.8 * self.t))
            pts = self._surf_poly.GetPoints()
            arr = ns.vtk_to_numpy(pts.GetData())
            arr[:, 2] = z.ravel()
            pts.GetData().Modified()
            self._surf_poly.Modified()
        if "particles" in self.actors:
            self._p += self._pv * dt
            centre = np.asarray(vehicle_pos, float)
            rel = self._p + self._pcentre - centre                               # keep the cloud around the vehicle: wrap points that fall outside the box
            half = self._pbox / 2.0
            wrapped = (rel + half) % self._pbox - half
            self._p = wrapped - self._pcentre + centre
            self._pcentre = np.zeros(3)
            arr = ns.vtk_to_numpy(self._pts.GetData())
            arr[:] = self._p.astype(np.float32)
            self._pts.GetData().Modified()
            self._ppd.Modified()
        if "shadow" in self.actors:
            z = self.floor_height(float(vehicle_pos[0]), float(vehicle_pos[1])) - 0.02
            height = max(z - float(vehicle_pos[2]), 0.1)
            sc = 0.55 + 0.12 * height                                           # grows and fades with height above the floor
            tr = vtk.vtkTransform()
            tr.Translate(float(vehicle_pos[0]), float(vehicle_pos[1]), z)
            tr.RotateZ(math.degrees(vehicle_yaw))
            tr.Scale(0.75 * sc, 0.22 * sc, 1.0)
            self.actors["shadow"].SetUserTransform(tr)
            self.actors["shadow"].GetProperty().SetOpacity(max(0.05, 0.38 - 0.03 * height))

    def set_visible(self, name: str, on: bool) -> None:
        if name in self.actors:
            self.actors[name].SetVisibility(bool(on))
