"""The 3D scene (pure VTK, no Qt). Renders OFFSCREEN into a numpy image, so it is testable and embeds in any GUI.

World frame is NED (x North, y East, z DOWN); the camera 'up' vector is (0, 0, -1), so depth increases downward on screen.
Contents: the real Mako mesh, the real dock funnel mesh, the four ring lights, the water surface, a reference grid, a trail, thrust arrows,
and the nose camera's field-of-view pyramid. Needs a DISPLAY (VTK opens an invisible offscreen context on it).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import vtk
from vtk.util import numpy_support as ns

_HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_HERE)] + [str(_d) for _d in sorted(_HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders (app/, data/, view3d/, plots/, sonar/, camera/) stay flat-importable
sys.path.insert(0, str(_HERE.parent / "common"))

import assets  # noqa: E402
from environment import Environment  # noqa: E402
from allocation import eul_to_rotm, load_geometry  # noqa: E402
from sim_viewer_math import pose_matrix, rotation_between  # noqa: E402

VIEWS = ("follow", "free", "top", "side", "dock", "onboard")


def _vtk_matrix(m: np.ndarray) -> vtk.vtkMatrix4x4:
    out = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            out.SetElement(i, j, float(m[i, j]))
    return out


def _actor(poly: vtk.vtkPolyData, color, opacity: float = 1.0, specular: float = 0.3, ambient: float = 0.25) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    a = vtk.vtkActor()
    a.SetMapper(mapper)
    p = a.GetProperty()
    p.SetColor(*[float(c) for c in color])
    p.SetOpacity(float(opacity))
    p.SetSpecular(specular)
    p.SetAmbient(ambient)
    return a


class Scene3D:
    def __init__(self, cfg: dict, size=(900, 600)) -> None:
        self.cfg = cfg
        v = cfg["viewer"]
        col = v["colors"]
        self.ren = vtk.vtkRenderer()
        self.ren.GradientBackgroundOn()
        self.ren.SetBackground(*col["background_bottom"])
        self.ren.SetBackground2(*col["background_top"])
        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(int(size[0]), int(size[1]))
        self.win.SetMultiSamples(4)
        self.geom = load_geometry()
        self.mode = "follow"
        self._last_pos: Optional[np.ndarray] = None
        self._pose = pose_matrix([0, 0, 3], [0, 0, 0])
        self._eul = np.zeros(3)
        self._pos = np.array([0.0, 0.0, 3.0])

        archive = assets.find_vessel_file(cfg)
        self.asset_source = str(archive) if archive else "built-in fallback shapes"
        self._build_vehicle(archive, v, col)
        self._build_dock(archive, cfg, col)
        self._build_environment(v)
        look = v.get("look3d", {})
        self.env: Optional[Environment] = None
        if look.get("enabled", True):
            R_d = eul_to_rotm(cfg["dock"]["orientation_deg"])
            terr = None
            if look.get("use_terrain", True) and cfg.get("terrain") is not None:
                from terrain import get_terrain
                terr = get_terrain(cfg.get("terrain"))
            self.env = Environment(self.ren, look, self.dock_pos, R_d @ np.array([1.0, 0.0, 0.0]), self.light_positions, terr)
            if look.get("hide_reference_grid", True):
                self.grid.SetVisibility(False)
                self.surface.SetVisibility(False) if hasattr(self, "surface") and look.get("surface_waves", True) else None
        self._build_trail(v, col)
        self._build_arrows(v)
        self._build_frustum(cfg, v)
        self.set_pose([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
        self.set_view("free")
        self.reset_camera_overview()

    # ------------------------------------------------------------------ construction
    def _build_vehicle(self, archive, v, col) -> None:
        poly = None
        if archive is not None:
            try:
                poly = assets.load_mesh(archive, "assets/mako_01_geometry.stl", reduction=float(v["assets"]["vehicle_reduction"]))
            except Exception as exc:                                     # noqa: BLE001
                print(f"[scene3d] vehicle mesh failed ({exc}); using the fallback shape", file=sys.stderr)
        poly = poly if poly is not None else assets.fallback_vehicle()
        self.vehicle = _actor(poly, col["vehicle"], specular=0.5)
        self.ren.AddActor(self.vehicle)
        # translucent copy that shows where the vehicle MODEL says the vehicle should be (real-vs-model overlay); hidden until set_ghost() is called
        self.ghost = _actor(poly, col.get("ghost", (1.0, 0.55, 0.1)), opacity=0.45, specular=0.1)
        self.ghost.SetVisibility(False)
        self.ren.AddActor(self.ghost)

    def _build_dock(self, archive, cfg, col) -> None:
        d = cfg["dock"]
        self.dock_pos = np.asarray(d["position_m"], float)
        R = eul_to_rotm(d["orientation_deg"])
        poly = None
        if archive is not None:
            try:
                poly = assets.load_mesh(archive, "assets/dock_02_geometry.stl", reduction=0.0)
            except Exception as exc:                                     # noqa: BLE001
                print(f"[scene3d] dock mesh failed ({exc}); using the fallback shape", file=sys.stderr)
        self.dock = _actor(poly if poly is not None else assets.fallback_dock(), col["dock"], opacity=0.8)
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = R, self.dock_pos
        self.dock.SetUserMatrix(_vtk_matrix(m))
        self.ren.AddActor(self.dock)
        self.light_actors: List[vtk.vtkActor] = []
        self.light_positions: List[np.ndarray] = []
        for L in d["ring_lights"]:
            p = self.dock_pos + R @ np.asarray(L["loc"], float)
            s = vtk.vtkSphereSource()
            s.SetRadius(0.07)
            s.SetThetaResolution(16)
            s.SetPhiResolution(16)
            s.SetCenter(*p)
            s.Update()
            a = _actor(s.GetOutput(), col["light"], ambient=1.0, specular=0.0)
            a.GetProperty().SetLighting(False)
            self.ren.AddActor(a)
            self.light_actors.append(a)
            self.light_positions.append(p)

    def _build_environment(self, v) -> None:
        if v.get("water_surface", True):
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(-10, -12, 0)
            plane.SetPoint1(26, -12, 0)
            plane.SetPoint2(-10, 12, 0)
            plane.Update()
            self.surface = _actor(plane.GetOutput(), (0.25, 0.55, 0.85), opacity=0.18, specular=0.0)
            self.surface.GetProperty().SetLighting(False)
            self.ren.AddActor(self.surface)
        g = v["grid"]
        x0, x1 = g["x_range_m"]
        y0, y1 = g["y_range_m"]
        sp, z = float(g["spacing_m"]), float(g["z_m"])
        pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        def seg(a, b) -> None:
            i = pts.InsertNextPoint(*a)
            j = pts.InsertNextPoint(*b)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i)
            lines.InsertCellPoint(j)
        for x in np.arange(x0, x1 + 1e-6, sp):
            seg((x, y0, z), (x, y1, z))
        for y in np.arange(y0, y1 + 1e-6, sp):
            seg((x0, y, z), (x1, y, z))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        self.grid = _actor(pd, (0.35, 0.50, 0.62), opacity=0.7, specular=0.0)
        self.grid.GetProperty().SetLighting(False)
        self.ren.AddActor(self.grid)
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.AxisLabelsOff()
        self.axes = axes                       # x red (North), y green (East), z blue (Down) at the origin
        self.ren.AddActor(axes)

    def _build_trail(self, v, col) -> None:
        self._trail_n = int(v["trail_points"])
        self._trail_step = float(v["trail_step_m"])
        self._trail: List[np.ndarray] = []
        self.trail_pd = vtk.vtkPolyData()
        self.trail = _actor(self.trail_pd, col["trail"], specular=0.0, ambient=1.0)
        self.trail.GetProperty().SetLineWidth(2.5)
        self.trail.GetProperty().SetLighting(False)
        self.ren.AddActor(self.trail)

    def _build_arrows(self, v) -> None:
        ta = v["thrust_arrows"]
        self._arrow_cfg = ta
        self.arrows: Dict[str, vtk.vtkActor] = {}
        self._thr = {}
        src = vtk.vtkArrowSource()
        src.SetTipResolution(16)
        src.SetShaftResolution(16)
        src.Update()
        for name in [k for k in self.geom["thrusters"] if k.startswith("th_")]:
            t = self.geom["thrusters"][name]
            self._thr[name] = (np.asarray(t["location"], float), eul_to_rotm(t["orientation"]) @ np.array([1.0, 0.0, 0.0]))
            a = _actor(src.GetOutput(), (0.2, 0.9, 1.0), ambient=0.6)
            a.SetVisibility(False)
            self.ren.AddActor(a)
            self.arrows[name] = a

    def _build_frustum(self, cfg, v) -> None:
        c = cfg["camera"]
        depth = 1.2
        hh = math.tan(math.radians(float(c["vfov_deg"])) / 2.0) * depth
        hw = hh * float(c["width"]) / float(c["height"])
        loc = np.asarray(c["location_m"], float)
        R_c = eul_to_rotm(c["orientation_deg"])
        # camera axes in the body frame: forward = R_c@(0,0,-1), right = R_c@(1,0,0), down = -R_c@(0,1,0)
        fwd, right, down = R_c @ np.array([0, 0, -1.0]), R_c @ np.array([1.0, 0, 0]), -(R_c @ np.array([0, 1.0, 0]))
        corners = [loc + depth * fwd + sx * hw * right + sy * hh * down for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        pts.InsertNextPoint(*loc)
        for c3 in corners:
            pts.InsertNextPoint(*c3)
        for i in range(1, 5):
            lines.InsertNextCell(2); lines.InsertCellPoint(0); lines.InsertCellPoint(i)
            lines.InsertNextCell(2); lines.InsertCellPoint(i); lines.InsertCellPoint(1 + (i % 4))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        self.frustum = _actor(pd, (1.0, 0.9, 0.3), opacity=0.8, specular=0.0, ambient=1.0)
        self.frustum.GetProperty().SetLighting(False)
        self.frustum.SetVisibility(bool(v.get("show_camera_frustum", True)))
        self.ren.AddActor(self.frustum)
        self._cam_loc, self._cam_fwd = loc, fwd

    # ------------------------------------------------------------------ state updates
    def set_pose(self, pos, eul_rad) -> None:
        pos = np.asarray(pos, float)
        self._pos, self._eul = pos.copy(), np.asarray(eul_rad, float).copy()
        self._pose = pose_matrix(pos, eul_rad)
        vm = _vtk_matrix(self._pose)
        self.vehicle.SetUserMatrix(vm)
        self.frustum.SetUserMatrix(vm)
        self._follow_camera(pos)

    def update_environment(self, dt: float) -> None:
        """Animate waves and particles (wall-clock dt) and put the soft shadow under the vehicle."""
        if self.env is not None:
            self.env.update(dt, self._pos, float(self._eul[2]))

    def set_ghost(self, pos, eul_rad) -> None:
        self.ghost.SetUserMatrix(_vtk_matrix(pose_matrix(np.asarray(pos, float), np.asarray(eul_rad, float))))
        self.ghost.SetVisibility(True)

    def hide_ghost(self) -> None:
        self.ghost.SetVisibility(False)

    def set_actuators(self, cmd: Dict[str, float]) -> None:
        ta = self._arrow_cfg
        if not ta.get("enabled", True):
            for a in self.arrows.values():
                a.SetVisibility(False)
            return
        for name, a in self.arrows.items():
            rpm = float(cmd.get(name, 0.0))
            if abs(rpm) < float(ta["min_rpm"]):
                a.SetVisibility(False)
                continue
            loc, axis = self._thr[name]
            d = axis * math.copysign(1.0, rpm)                    # force direction (negative RPM reverses it)
            length = max(0.08, min(1.0, abs(rpm) / float(ta["rpm_ref"]))) * float(ta["max_length_m"])
            local = np.eye(4)
            local[:3, :3] = rotation_between([1.0, 0.0, 0.0], d) * length
            local[:3, 3] = loc
            a.SetUserMatrix(_vtk_matrix(self._pose @ local))
            a.SetVisibility(True)

    def add_trail_point(self, pos) -> None:
        pos = np.asarray(pos, float)
        if self._trail and np.linalg.norm(pos - self._trail[-1]) < self._trail_step:
            return
        self._trail.append(pos.copy())
        if len(self._trail) > self._trail_n:
            self._trail.pop(0)
        self._refresh_trail()

    def reset_trail(self) -> None:
        self._trail.clear()
        self._refresh_trail()

    def _refresh_trail(self) -> None:
        n = len(self._trail)
        pts = vtk.vtkPoints()
        if n:
            pts.SetData(ns.numpy_to_vtk(np.asarray(self._trail, float), deep=True))
        lines = vtk.vtkCellArray()
        if n >= 2:
            lines.InsertNextCell(n)
            for i in range(n):
                lines.InsertCellPoint(i)
        self.trail_pd.SetPoints(pts)
        self.trail_pd.SetLines(lines)
        self.trail_pd.Modified()

    # ------------------------------------------------------------------ camera
    def reset_camera_overview(self) -> None:
        """Free view that shows the vehicle start, the dock and the standoff region."""
        cam = self.ren.GetActiveCamera()
        cam.SetViewUp(0, 0, -1)
        cam.SetFocalPoint(5.0, 0.0, 3.0)
        cam.SetPosition(-1.0, -9.0, -2.5)
        cam.SetViewAngle(45.0)
        self.ren.ResetCameraClippingRange()

    def set_view(self, name: str) -> None:
        if name not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        self.mode = name
        cam = self.ren.GetActiveCamera()
        p = self._pos
        if name == "follow":
            cam.SetViewUp(0, 0, -1)
            cam.SetViewAngle(45.0)
            cam.SetFocalPoint(*p)
            cam.SetPosition(*(p + np.array([-3.0, -2.5, -1.5])))
        elif name == "free":
            cam.SetViewAngle(45.0)
        self._last_pos = p.copy()
        self._follow_camera(p)
        self.ren.ResetCameraClippingRange()

    def _follow_camera(self, pos: np.ndarray) -> None:
        cam = self.ren.GetActiveCamera()
        if self.mode == "follow":
            if self._last_pos is not None:
                delta = pos - self._last_pos
                cam.SetPosition(*(np.array(cam.GetPosition()) + delta))
                cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) + delta))
        elif self.mode == "top":
            cam.SetViewAngle(45.0)
            cam.SetFocalPoint(*pos)
            cam.SetPosition(pos[0], pos[1], pos[2] - 14.0)
            cam.SetViewUp(1, 0, 0)
        elif self.mode == "side":
            cam.SetViewAngle(45.0)
            cam.SetFocalPoint(*pos)
            cam.SetPosition(pos[0], pos[1] + 7.0, pos[2] - 0.5)
            cam.SetViewUp(0, 0, -1)
        elif self.mode == "dock":
            mid = 0.5 * (pos + self.dock_pos)
            cam.SetViewAngle(45.0)
            cam.SetFocalPoint(*mid)
            cam.SetPosition(self.dock_pos[0] - 2.0, self.dock_pos[1] + 5.5, self.dock_pos[2] - 2.0)
            cam.SetViewUp(0, 0, -1)
        elif self.mode == "onboard":
            R = self._pose[:3, :3]
            eye = pos + R @ self._cam_loc
            cam.SetViewAngle(float(self.cfg["camera"]["vfov_deg"]))
            cam.SetPosition(*eye)
            cam.SetFocalPoint(*(eye + R @ self._cam_fwd * 5.0))
            cam.SetViewUp(*(R @ np.array([0.0, 0.0, -1.0])))
        self._last_pos = pos.copy()
        self.ren.ResetCameraClippingRange()

    # mouse-driven camera moves (used by the Qt widget)
    def orbit(self, dx_px: float, dy_px: float) -> None:
        cam = self.ren.GetActiveCamera()
        cam.Azimuth(-0.4 * dx_px)
        cam.Elevation(0.4 * dy_px)
        cam.OrthogonalizeViewUp()
        self.ren.ResetCameraClippingRange()

    def pan(self, dx_px: float, dy_px: float) -> None:
        cam = self.ren.GetActiveCamera()
        fp, pos = np.array(cam.GetFocalPoint()), np.array(cam.GetPosition())
        view_up = np.array(cam.GetViewUp())
        fwd = fp - pos
        dist = float(np.linalg.norm(fwd))
        right = np.cross(fwd / dist, view_up)
        right /= max(np.linalg.norm(right), 1e-9)
        scale = dist * 0.0015
        shift = (-dx_px * right + dy_px * view_up) * scale
        cam.SetFocalPoint(*(fp + shift))
        cam.SetPosition(*(pos + shift))
        self.ren.ResetCameraClippingRange()

    def zoom(self, factor: float) -> None:
        self.ren.GetActiveCamera().Dolly(float(factor))
        self.ren.ResetCameraClippingRange()

    # ------------------------------------------------------------------ output
    def render(self, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """-> H x W x 3 uint8 RGB image of the scene."""
        if width and height and tuple(self.win.GetSize()) != (int(width), int(height)):
            self.win.SetSize(int(width), int(height))
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
