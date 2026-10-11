"""Meshes for the 3D scene: the real Mako and dock STL files from the vessel archive (decimated), with a simple fallback.

The vessel file ("Mako (1).mavsim") is a zip holding assets/mako_01_geometry.stl (147k faces) and assets/dock_02_geometry.stl (12k faces).
Both are in the vessel's own frame: Mako in the body frame (x forward), the dock with +x out of the funnel mouth.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import vtk
from vtk.util import numpy_support as ns

_REPO = Path(__file__).resolve().parents[3]            # AUV_docking/


def find_vessel_file(cfg: dict) -> Optional[Path]:
    """The configured vessel archive, or the first '*.mavsim' next to the repo; None if there is none."""
    wanted = cfg.get("viewer", {}).get("assets", {}).get("vessel_file", "")
    candidates = []
    if wanted:
        p = Path(wanted).expanduser()
        candidates.append(p if p.is_absolute() else (_REPO / p).resolve())
    candidates += sorted(_REPO.parent.glob("Mako*.mavsim"))
    for c in candidates:
        if c.is_file() and zipfile.is_zipfile(c):
            return c
    return None


def _to_polydata(vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(vertices, dtype=float), deep=True))
    f = np.asarray(faces, dtype=np.int64)
    conn = np.hstack([np.full((len(f), 1), 3, np.int64), f]).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(len(f), ns.numpy_to_vtkIdTypeArray(conn, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def load_mesh(archive: Path, member: str, reduction: float = 0.0) -> vtk.vtkPolyData:
    """STL from the zip as vtkPolyData with smooth normals, optionally decimated (reduction 0.9 = keep 10% of the faces)."""
    import trimesh

    with zipfile.ZipFile(archive) as z:
        mesh = trimesh.load(io.BytesIO(z.read(member)), file_type="stl")
    pd = _to_polydata(mesh.vertices, mesh.faces)
    if reduction > 0.0:
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(pd)
        dec.SetTargetReduction(float(min(max(reduction, 0.0), 0.98)))
        dec.Update()
        pd = dec.GetOutput()
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(pd)
    nrm.SetFeatureAngle(60.0)
    nrm.SplittingOff()
    nrm.Update()
    return nrm.GetOutput()


def fallback_vehicle() -> vtk.vtkPolyData:
    """A 1.35 m x 0.19 m torpedo (cylinder + nose cone) along +x, used when the vessel file is missing."""
    cyl = vtk.vtkCylinderSource()
    cyl.SetRadius(0.095)
    cyl.SetHeight(1.2)
    cyl.SetResolution(24)
    cyl.Update()
    tf = vtk.vtkTransform()
    tf.RotateZ(-90.0)                                   # cylinder axis is +y by default; make it +x
    tfp = vtk.vtkTransformPolyDataFilter()
    tfp.SetInputData(cyl.GetOutput())
    tfp.SetTransform(tf)
    tfp.Update()
    cone = vtk.vtkConeSource()
    cone.SetRadius(0.095)
    cone.SetHeight(0.2)
    cone.SetResolution(24)
    cone.SetCenter(0.7, 0.0, 0.0)
    cone.Update()
    app = vtk.vtkAppendPolyData()
    app.AddInputData(tfp.GetOutput())
    app.AddInputData(cone.GetOutput())
    app.Update()
    return app.GetOutput()


def fallback_dock() -> vtk.vtkPolyData:
    """A 2 m wide, 2.5 m long open cone along dock-frame -x (mouth at x = 0) when the vessel file is missing."""
    cone = vtk.vtkConeSource()
    cone.SetRadius(1.0)
    cone.SetHeight(2.5)
    cone.SetResolution(32)
    cone.SetCapping(False)
    cone.SetDirection(-1.0, 0.0, 0.0)
    cone.SetCenter(-1.25, 0.0, 0.0)
    cone.Update()
    return cone.GetOutput()
