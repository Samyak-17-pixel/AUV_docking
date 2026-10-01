"""Vehicle state helpers shared by dof_testing, station_keeping and the tests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from allocation import eul_to_rotm


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_euler(x: float, y: float, z: float, w: float):
    """ZYX Euler (roll, pitch, yaw) [rad] from a quaternion."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


@dataclass
class State:
    """pos: NED [m]; eul: (roll, pitch, yaw) [rad]; nu: body [u,v,w,p,q,r] [m/s, rad/s]."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eul: np.ndarray = field(default_factory=lambda: np.zeros(3))
    nu: np.ndarray = field(default_factory=lambda: np.zeros(6))

    @property
    def roll(self) -> float:
        return float(self.eul[0])

    @property
    def pitch(self) -> float:
        return float(self.eul[1])

    @property
    def yaw(self) -> float:
        return float(self.eul[2])

    @property
    def depth(self) -> float:
        return float(self.pos[2])

    @property
    def speed_u(self) -> float:
        """Forward (body-x) speed [m/s] - the flow the fins see."""
        return float(self.nu[0])

    def vel_ned(self) -> np.ndarray:
        return eul_to_rotm(np.degrees(self.eul)) @ self.nu[:3]

    @staticmethod
    def from_odom(msg) -> "State":
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        lv, av = msg.twist.twist.linear, msg.twist.twist.angular
        return State(
            pos=np.array([p.x, p.y, p.z], dtype=float),
            eul=np.array(quat_to_euler(q.x, q.y, q.z, q.w), dtype=float),
            nu=np.array([lv.x, lv.y, lv.z, av.x, av.y, av.z], dtype=float),
        )

    @staticmethod
    def from_model(model) -> "State":
        return State(pos=model.pos.copy(), eul=model.eul.copy(), nu=model.nu.copy())
