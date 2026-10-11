"""Small shared maths for the viewer (kept free of Qt, VTK and ROS)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from allocation import eul_to_rotm  # noqa: E402


def body_to_ned(eul_rad) -> np.ndarray:
    """Rotation matrix body -> NED for [roll, pitch, yaw] in radians (intrinsic Z-Y-X, same as the whole project)."""
    return eul_to_rotm(np.degrees(np.asarray(eul_rad, float)))


def pose_matrix(pos, eul_rad) -> np.ndarray:
    """4x4 homogeneous transform body -> NED."""
    m = np.eye(4)
    m[:3, :3] = body_to_ned(eul_rad)
    m[:3, 3] = np.asarray(pos, float)
    return m


def rotation_between(a, b) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else _flip(a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _flip(a: np.ndarray) -> np.ndarray:
    """180 degree rotation about any axis perpendicular to a."""
    axis = np.cross(a, [1.0, 0.0, 0.0])
    if np.linalg.norm(axis) < 1e-6:
        axis = np.cross(a, [0.0, 1.0, 0.0])
    axis /= np.linalg.norm(axis)
    return 2.0 * np.outer(axis, axis) - np.eye(3)
