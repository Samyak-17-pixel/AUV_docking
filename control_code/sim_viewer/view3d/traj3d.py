"""3D trajectory scene (pure VTK, no Qt, renders OFFSCREEN into a numpy image like scene3d.Scene3D, so it is testable and fits `viewer_app.SceneView`).

Shows the seabed (the shared terrain.py heightmap, coloured by depth), the flown path coloured by speed / depth / time, the planned path with numbered waypoints, optional
second (saved) run, the dock, the vehicle marker and the side-scan swath footprint on the floor. World frame NED (x north, y east, z DOWN), camera up = (0, 0, -1).
Needs a DISPLAY (VTK opens an invisible offscreen context on it).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support as ns

COLOR_BY = ("speed", "depth", "time", "altitude")


def _lut(name: str, lo: float, hi: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    ctf = vtk.vtkColorTransferFunction()
    stops = {"speed": [(0, 0.12, 0.35, 0.9), (0.5, 0.15, 0.85, 0.55), (1, 1.0, 0.85, 0.2)],
             "depth": [(0, 0.5, 0.95, 1.0), (0.5, 0.15, 0.45, 0.9), (1, 0.25, 0.05, 0.45)],
             "time": [(0, 0.6, 0.6, 0.7), (0.5, 0.2, 0.7, 0.95), (1, 1.0, 0.6, 0.15)],
             "altitude": [(0, 1.0, 0.3, 0.25), (0.5, 1.0, 0.85, 0.2), (1, 0.2, 0.9, 0.5)],
             "floor": [(0, 0.10, 0.17, 0.25), (0.5, 0.22, 0.38, 0.45), (1, 0.62, 0.58, 0.45)]}[name]
    for p, r, g, b in stops:
        ctf.AddRGBPoint(p, r, g, b)
    for i in range(256):
        r, g, b = ctf.GetColor(i / 255.0)
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.SetRange(float(lo), float(hi))
    return lut


def _polyline(points: np.ndarray, scalars: Optional[np.ndarray] = None) -> vtk.vtkPolyData:
    n = len(points)
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
    cells = vtk.vtkCellArray()
    if n >= 2:
        cells.InsertNextCell(n)
        for i in range(n):
            cells.InsertCellPoint(i)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(cells)
    if scalars is not None and n:
        pd.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(scalars, dtype=np.float32), deep=True))
    return pd


def _text(ren, text: str, x: int, y: int, color=(0.85, 0.9, 0.95), size: int = 14) -> vtk.vtkTextActor:
    a = vtk.vtkTextActor()
    a.SetInput(text)
    a.SetPosition(x, y)
    p = a.GetTextProperty()
    p.SetColor(*color)
    p.SetFontSize(size)
    ren.AddViewProp(a)
    return a


class Traj3DScene:
    def __init__(self, cfg: dict, size=(800, 500), terrain=None) -> None:
        self.cfg = cfg
        self.ren = vtk.vtkRenderer()
        self.ren.GradientBackgroundOn()
        self.ren.SetBackground(0.04, 0.07, 0.11)
        self.ren.SetBackground2(0.10, 0.17, 0.26)
        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(int(size[0]), int(size[1]))
        self.win.SetMultiSamples(4)
        self.terrain = terrain
        self.dock = np.asarray(((cfg.get("dock") or {}).get("position_m")) or [10.0, 0.0, 3.0], float)
        self.color_by = "speed"
        self._actors = {}
        self._floor_actor = None
        self._build_floor()
        self._build_dock()
        self._build_vehicle()
        self._legend = _text(self.ren, "", 10, 10)
        self._title = _text(self.ren, "3D trajectory   (N up-screen = x, E = y, depth down)", 10, size[1] - 24, (0.55, 0.78, 0.95), 14)
        self.show_floor = True
        self.set_view("iso")

    # ----------------------------------------------------------------- static parts
    def _build_floor(self) -> None:
        if self.terrain is None:
            return
        step = max(1, int(round(2.0 / self.terrain.dx)))                          # ~2 m cells: light enough to redraw
        xs, ys, z, _ = self.terrain.decimated(step)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        nx, ny = z.shape
        pts = vtk.vtkPoints()
        pts.SetData(ns.numpy_to_vtk(np.column_stack([X.ravel(), Y.ravel(), z.ravel()]).astype(np.float32), deep=True))
        idx = np.arange(nx * ny).reshape(nx, ny)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        tris = np.concatenate([np.column_stack([np.full(len(a), 3), a, b, c]), np.column_stack([np.full(len(a), 3), a, c, d])]).astype(np.int64)
        cells = vtk.vtkCellArray()
        cells.SetCells(len(tris), ns.numpy_to_vtkIdTypeArray(tris.ravel(), deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        pd.GetPointData().SetScalars(ns.numpy_to_vtk(z.ravel().astype(np.float32), deep=True))
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(pd)
        nrm.SetFeatureAngle(70.0)
        nrm.SplittingOff()
        nrm.Update()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(nrm.GetOutput())
        m.SetLookupTable(_lut("floor", float(z.min()), float(z.max())))
        m.SetScalarRange(float(z.min()), float(z.max()))
        m.ScalarVisibilityOn()
        act = vtk.vtkActor()
        act.SetMapper(m)
        act.GetProperty().SetAmbient(0.45)
        act.GetProperty().SetSpecular(0.05)
        self.ren.AddActor(act)
        self._floor_actor = act

    def _build_dock(self) -> None:
        cyl = vtk.vtkCylinderSource()
        cyl.SetRadius(1.0)
        cyl.SetHeight(0.6)
        cyl.SetResolution(24)
        tf = vtk.vtkTransform()
        tf.RotateZ(90.0)                                                           # the dock mouth faces along the x axis
        f = vtk.vtkTransformPolyDataFilter()
        f.SetInputConnection(cyl.GetOutputPort())
        f.SetTransform(tf)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(f.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.SetPosition(*self.dock)
        a.GetProperty().SetColor(0.95, 0.9, 0.3)
        self.ren.AddActor(a)

    def _build_vehicle(self) -> None:
        cone = vtk.vtkConeSource()
        cone.SetHeight(2.4)
        cone.SetRadius(0.55)
        cone.SetResolution(20)
        cone.SetDirection(1, 0, 0)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(cone.GetOutputPort())
        self.vehicle = vtk.vtkActor()
        self.vehicle.SetMapper(m)
        self.vehicle.GetProperty().SetColor(1.0, 0.62, 0.26)
        self.vehicle.SetVisibility(False)
        self.ren.AddActor(self.vehicle)

    # ----------------------------------------------------------------- data
    def _set_actor(self, key: str, actor: Optional[vtk.vtkActor]) -> None:
        old = self._actors.pop(key, None)
        if old is not None:
            self.ren.RemoveActor(old)
        if actor is not None:
            self.ren.AddActor(actor)
            self._actors[key] = actor

    def _scalar(self, t, pos, u) -> Tuple[np.ndarray, float, float, str]:
        mode = self.color_by
        if mode == "depth":
            s, label = pos[:, 2], "depth [m]"
        elif mode == "time":
            s, label = np.asarray(t, float) - float(t[0]), "time [s]"
        elif mode == "altitude" and self.terrain is not None:
            s = np.array([self.terrain.altitude(p[0], p[1], p[2]) for p in pos])
            label = "altitude above the seabed [m]"
        else:
            s, label = np.asarray(u, float), "speed u [m/s]"
            mode = "speed"
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        return s, lo, hi, label

    def set_path(self, t: Sequence[float], pos: np.ndarray, u: Sequence[float], eul_yaw: Optional[float] = None, key: str = "path", width: float = 4.0) -> None:
        """The flown path (N x 3 NED), coloured by `color_by`; also moves the vehicle marker to the last point (main path only)."""
        pos = np.asarray(pos, float).reshape(-1, 3)
        if len(pos) < 2:
            self._set_actor(key, None)
            if key == "path":
                self.vehicle.SetVisibility(False)
            return
        s, lo, hi, label = self._scalar(np.asarray(t), pos, np.asarray(u))
        pd = _polyline(pos, s)
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        m.SetLookupTable(_lut(self.color_by if self.color_by in COLOR_BY else "speed", lo, hi))
        m.SetScalarRange(lo, hi)
        m.ScalarVisibilityOn()
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetLineWidth(width)
        a.GetProperty().SetRenderLinesAsTubes(True)
        self._set_actor(key, a)
        if key == "path":
            self.vehicle.SetPosition(*pos[-1])
            if eul_yaw is not None:
                self.vehicle.SetOrientation(0, 0, math.degrees(eul_yaw))
            self.vehicle.SetVisibility(True)
            self._legend.SetInput(f"colour = {label}:  {lo:.2f}  ...  {hi:.2f}      points {len(pos)}")

    def set_other_run(self, pos: Optional[np.ndarray]) -> None:
        """A second (saved) run, drawn as a thin light line for comparison."""
        if pos is None or len(pos) < 2:
            self._set_actor("other", None)
            return
        a = vtk.vtkActor()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(_polyline(np.asarray(pos, float)))
        a.SetMapper(m)
        a.GetProperty().SetColor(0.85, 0.85, 0.95)
        a.GetProperty().SetLineWidth(2.0)
        self._set_actor("other", a)

    def set_plan(self, points: Optional[Sequence[Tuple[float, float, float]]]) -> None:
        """The planned path (x, y, z...) as a dashed-looking white line with a sphere at each waypoint."""
        if not points or len(points) < 1:
            self._set_actor("plan", None)
            self._set_actor("plan_pts", None)
            return
        pts = np.array([[p[0], p[1], p[2]] for p in points], float)
        a = vtk.vtkActor()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(_polyline(pts))
        a.SetMapper(m)
        a.GetProperty().SetColor(0.95, 0.95, 1.0)
        a.GetProperty().SetLineWidth(1.5)
        a.GetProperty().SetOpacity(0.7)
        self._set_actor("plan", a)
        vp = vtk.vtkPoints()
        vp.SetData(ns.numpy_to_vtk(pts.astype(np.float32), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vp)
        sph = vtk.vtkSphereSource()
        sph.SetRadius(0.6)
        g = vtk.vtkGlyph3D()
        g.SetInputData(poly)
        g.SetSourceConnection(sph.GetOutputPort())
        g.ScalingOff()
        gm = vtk.vtkPolyDataMapper()
        gm.SetInputConnection(g.GetOutputPort())
        ga = vtk.vtkActor()
        ga.SetMapper(gm)
        ga.GetProperty().SetColor(1.0, 1.0, 1.0)
        self._set_actor("plan_pts", ga)

    def set_swath(self, pos: np.ndarray, yaw: np.ndarray, altitude: np.ndarray, near_far: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)) -> None:
        """The side-scan footprint as two ribbons on the floor: for each sample the ground range [near, far] on each side (lateral offset from the track), from the track yaw."""
        n = len(pos)
        if n < 2:
            self._set_actor("swath", None)
            return
        sel = np.unique(np.linspace(0, n - 1, min(n, 500)).astype(int))
        quads, pts = [], []
        for side in (+1.0, -1.0):
            for k in range(len(sel) - 1):
                row = []
                for j in (sel[k], sel[k + 1]):
                    near, far = altitude[j] * near_far[0], altitude[j] * near_far[1]
                    right = np.array([math.cos(yaw[j] + math.pi / 2), math.sin(yaw[j] + math.pi / 2)])    # starboard unit vector: heading + 90 deg (NED, yaw about down)
                    for g in (near, far):
                        p = pos[j, :2] + side * g * right
                        zf = float(self.terrain.height(p[0], p[1])) if self.terrain is not None else float(pos[j, 2] + altitude[j])
                        row.append((p[0], p[1], zf - 0.15))
                base = len(pts)
                pts += [row[0], row[1], row[3], row[2]]
                quads.append([4, base, base + 1, base + 2, base + 3])
        vp = vtk.vtkPoints()
        vp.SetData(ns.numpy_to_vtk(np.array(pts, np.float32), deep=True))
        cells = vtk.vtkCellArray()
        for q in quads:
            cells.InsertNextCell(4)
            for i in q[1:]:
                cells.InsertCellPoint(i)
        pd = vtk.vtkPolyData()
        pd.SetPoints(vp)
        pd.SetPolys(cells)
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(0.2, 0.9, 0.85)
        a.GetProperty().SetOpacity(0.35)
        a.GetProperty().LightingOff()
        self._set_actor("swath", a)

    def clear(self) -> None:
        for k in list(self._actors):
            if k != "plan" and k != "plan_pts":
                self._set_actor(k, None)
        self.vehicle.SetVisibility(False)
        self._legend.SetInput("")

    def set_floor_visible(self, on: bool) -> None:
        if self._floor_actor is not None:
            self._floor_actor.SetVisibility(bool(on))

    # ----------------------------------------------------------------- camera
    def set_view(self, name: str) -> None:
        cam = self.ren.GetActiveCamera()
        cam.SetViewUp(0, 0, -1)
        c = self.dock.copy()
        c[:2] = [20.0, 0.0]
        span = 60.0
        if name == "top":
            cam.SetFocalPoint(*c)
            cam.SetPosition(c[0], c[1], c[2] - 2.0 * span)
            cam.SetViewUp(1, 0, 0)
        elif name == "side":
            cam.SetFocalPoint(*c)
            cam.SetPosition(c[0], c[1] - 2.0 * span, c[2])
        else:
            cam.SetFocalPoint(*c)
            cam.SetPosition(c[0] - 1.1 * span, c[1] - 1.4 * span, c[2] - 0.9 * span)
        self.ren.ResetCameraClippingRange()

    def fit(self, pos: np.ndarray) -> None:
        """Aim the camera at the middle of the given path and back off far enough to see all of it."""
        pos = np.asarray(pos, float).reshape(-1, 3)
        if len(pos) < 2:
            return
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        c = 0.5 * (lo + hi)
        r = max(float(np.linalg.norm(hi - lo)), 20.0)
        cam = self.ren.GetActiveCamera()
        v = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        v = v / max(np.linalg.norm(v), 1e-9)
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + v * 1.4 * r))
        self.ren.ResetCameraClippingRange()

    def orbit(self, dx_px: float, dy_px: float) -> None:
        cam = self.ren.GetActiveCamera()
        cam.Azimuth(-0.4 * dx_px)
        cam.Elevation(0.4 * dy_px)
        cam.OrthogonalizeViewUp()
        self.ren.ResetCameraClippingRange()

    def pan(self, dx_px: float, dy_px: float) -> None:
        cam = self.ren.GetActiveCamera()
        fp, pos = np.array(cam.GetFocalPoint()), np.array(cam.GetPosition())
        up = np.array(cam.GetViewUp())
        fwd = fp - pos
        dist = float(np.linalg.norm(fwd))
        right = np.cross(fwd / dist, up)
        right /= max(np.linalg.norm(right), 1e-9)
        shift = (-dx_px * right + dy_px * up) * dist * 0.0015
        cam.SetFocalPoint(*(fp + shift))
        cam.SetPosition(*(pos + shift))
        self.ren.ResetCameraClippingRange()

    def zoom(self, factor: float) -> None:
        self.ren.GetActiveCamera().Dolly(float(factor))
        self.ren.ResetCameraClippingRange()

    def render(self, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        if width and height and tuple(self.win.GetSize()) != (int(width), int(height)):
            self.win.SetSize(int(width), int(height))
            self._title.SetPosition(10, int(height) - 24)
        self.win.Render()
        f = vtk.vtkWindowToImageFilter()
        f.SetInput(self.win)
        f.SetInputBufferTypeToRGB()
        f.ReadFrontBufferOff()
        f.Update()
        img = f.GetOutput()
        w, h, _ = img.GetDimensions()
        arr = ns.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[:, :, :3]
        return np.ascontiguousarray(arr[::-1])
