"""Synthetic nose camera for the offline simulator (no ROS needed).

Given the vehicle pose it draws what camera_03 would see of the dock's four ring lights, using the exact mounting, resolution, field of
view, dock pose and light data of "Mako (1).mavsim" (see sim_viewer.yaml). How a lamp LOOKS (glow size, brightness vs distance, noise) is an
assumption: the real mavsim renderer is not available here.

Conventions (same as mavsim-controller/core/tests/unit/test_overlay_projection.py, the sim's own camera test):
  * sensor_orientation = [roll, pitch, yaw] deg applied Z-Y-X, body <- camera;
  * the camera looks down its -Z, +X is image-right, +Y is image-up, so  u = W/2 + f*x/(-z),  v = H/2 - f*y/(-z);
  * f = (H/2) / tan(vfov/2): the sim's fov is the VERTICAL one (three.js PerspectiveCamera).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE.parent / "common"))
from allocation import eul_to_rotm  # noqa: E402

DEFAULT_CONFIG = _HERE / "sim_viewer.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


class SyntheticCamera:
    def __init__(self, cfg: Optional[dict] = None) -> None:
        self.cfg = cfg or load_config()
        c = self.cfg["camera"]
        self.W, self.H = int(c["width"]), int(c["height"])
        self.f = 0.5 * self.H / math.tan(0.5 * math.radians(float(c["vfov_deg"])))
        self.t_c = np.asarray(c["location_m"], dtype=float)
        self.R_c = eul_to_rotm(c["orientation_deg"])
        d = self.cfg["dock"]
        self.dock_pos = np.asarray(d["position_m"], dtype=float)
        self.R_d = eul_to_rotm(d["orientation_deg"])
        self.mouth_normal = self.R_d @ np.array([1.0, 0.0, 0.0])      # out of the funnel mouth, toward the vehicle
        self.lights: List[dict] = []
        for L in d["ring_lights"]:
            self.lights.append({"name": L["name"], "p": self.dock_pos + self.R_d @ np.asarray(L["loc"], float),
                                "lumens": float(L["lumens"]), "ring": True})
        if d.get("draw_scene_floods", False):
            for i, L in enumerate(d.get("scene_floods", [])):
                self.lights.append({"name": f"flood{i}", "p": self.dock_pos + self.R_d @ np.asarray(L["loc"], float),
                                    "lumens": float(L["lumens"]), "ring": False})
        self.beam_half = 0.5 * math.radians(float(d.get("beam_angle_deg", 82.0)))
        self.use_beam = bool(d.get("use_beam_pattern", True))
        self.look = self.cfg["look"]
        ref = float(self.look["ref_distance_m"])
        self._i_ref = 20000.0 / (ref * ref)
        seed = int(self.look.get("seed", 7))
        self._rng = np.random.default_rng(None if seed < 0 else seed)
        self.stress = dict(self.cfg.get("stress") or {})          # optional image degradations, all OFF by default (see sim_viewer.yaml `stress:`)
        self._srng = np.random.default_rng(int(self.stress.get("seed", 11)))

    # ------------------------------------------------------------------ geometry
    def camera_position(self, pos: np.ndarray, eul_deg) -> np.ndarray:
        return np.asarray(pos, float) + eul_to_rotm(eul_deg) @ self.t_c

    def project(self, pos, eul_deg, p_world) -> Optional[Tuple[float, float, float]]:
        """-> (u, v, distance_along_axis) or None if the point is behind the camera."""
        R_wb = eul_to_rotm(eul_deg)
        p_body = R_wb.T @ (np.asarray(p_world, float) - np.asarray(pos, float))
        p_cam = self.R_c.T @ (p_body - self.t_c)
        zc = -float(p_cam[2])
        if zc <= 0.05:
            return None
        return (self.W / 2.0 + self.f * float(p_cam[0]) / zc, self.H / 2.0 - self.f * float(p_cam[1]) / zc, zc)

    def brightness(self, pos, eul_deg, light: dict) -> float:
        """Relative brightness of a lamp as seen from the camera (1.0 = a 20000 lm lamp at ref_distance_m, in clear water)."""
        cam = self.camera_position(pos, eul_deg)
        v = cam - light["p"]
        d = float(np.linalg.norm(v))
        if d < 1e-3:
            return 0.0
        b = light["lumens"] / (d * d) / self._i_ref
        b *= math.exp(-float(self.look["water_attenuation_per_m"]) * d)
        if self.use_beam and light["ring"]:
            ang = math.acos(max(-1.0, min(1.0, float(np.dot(self.mouth_normal, v)) / d)))
            b *= 1.0 - _smoothstep((ang - self.beam_half) / math.radians(15.0))
        return b

    def truth(self, pos, eul_deg) -> Dict[str, dict]:
        """Ground truth for tests: pixel position, distance and brightness of every ring light that is in front of the camera."""
        out: Dict[str, dict] = {}
        for L in self.lights:
            pr = self.project(pos, eul_deg, L["p"])
            if pr is None:
                continue
            out[L["name"]] = {"u": pr[0], "v": pr[1], "z": pr[2], "brightness": self.brightness(pos, eul_deg, L)}
        return out

    # ------------------------------------------------------------------ rendering
    def render(self, pos, eul_deg) -> np.ndarray:
        """-> H x W x 3 uint8 BGR image."""
        lk = self.look
        gain = float(lk["core_gain"])
        acc = np.zeros((self.H, self.W, 3), np.float32)
        core_col = np.asarray(lk["color_bgr"], np.float32)
        halo_col = np.asarray(lk["halo_bgr"], np.float32)
        for L in self.lights:
            pr = self.project(pos, eul_deg, L["p"])
            if pr is None:
                continue
            u, v, zc = pr
            b = self.brightness(pos, eul_deg, L)
            if b <= 1e-5:
                continue
            raw = gain * b
            peak = min(1.0, raw)
            r_core = max(1.5, self.f * float(lk["light_radius_m"]) / zc)
            sigma = float(lk["bloom_sigma_px"]) + float(lk["bloom_sigma_gain"]) * math.log2(1.0 + raw)
            half = int(math.ceil(4.0 * sigma + r_core)) + 1
            x0, x1 = int(round(u)) - half, int(round(u)) + half + 1
            y0, y1 = int(round(v)) - half, int(round(v)) + half + 1
            cx0, cx1, cy0, cy1 = max(0, x0), min(self.W, x1), max(0, y0), min(self.H, y1)
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            xs = np.arange(cx0, cx1, dtype=np.float32) - u
            ys = np.arange(cy0, cy1, dtype=np.float32) - v
            r2 = xs[None, :] ** 2 + ys[:, None] ** 2
            r = np.sqrt(r2)
            core = np.clip(r_core + 0.5 - r, 0.0, 1.0) * peak
            halo = float(lk["bloom_amplitude"]) * (1.0 - math.exp(-raw)) * np.exp(-r2 / (2.0 * sigma * sigma))
            acc[cy0:cy1, cx0:cx1, :] += core[..., None] * core_col + halo[..., None] * halo_col
        img = np.asarray(lk["background_bgr"], np.float32)[None, None, :] + acc
        if self.stress:
            img = self._apply_stress(img, pos, eul_deg)
        sig = float(lk["noise_sigma"])
        if sig > 0:
            img = img + self._rng.normal(0.0, sig, img.shape).astype(np.float32)
        return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ optional image degradations (ASSUMED shapes; the real renderer is unknown)
    def _apply_stress(self, img: np.ndarray, pos, eul_deg) -> np.ndarray:
        """Blur, backscatter, bubbles / marine snow, false lights, one light hidden. Every key is optional and 0 / false means off:
        blur_sigma_px, motion_blur_px (+ motion_blur_deg), backscatter (0..1 brightness ramp, bright at the top), bubbles (count per frame),
        bubble_brightness (0..1), snow (specks per frame), glint (0 / 1: a bright surface patch at the top of the image), reflections (count of dim copies
        of the ring lights mirrored in the image), distractors (count of extra bright point lights at random places), occlude (light name hidden
        by drawing background over it)."""
        st, H, W = self.stress, self.H, self.W
        rng = self._srng
        if float(st.get("backscatter", 0.0)) > 0:
            ramp = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None, None]
            img = img + float(st["backscatter"]) * ramp * np.asarray([0.55, 0.45, 0.2], np.float32)[None, None, :]
        def disc(cx, cy, r, val, col=(1.0, 0.95, 0.8)):
            x0, x1, y0, y1 = int(max(0, cx - 4 * r - 2)), int(min(W, cx + 4 * r + 3)), int(max(0, cy - 4 * r - 2)), int(min(H, cy + 4 * r + 3))
            if x0 >= x1 or y0 >= y1:
                return
            xs, ys = np.arange(x0, x1, dtype=np.float32) - cx, np.arange(y0, y1, dtype=np.float32) - cy
            g = val * np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2.0 * max(r, 0.6) ** 2))
            img[y0:y1, x0:x1, :] += g[..., None] * np.asarray(col, np.float32)
        for _ in range(int(st.get("bubbles", 0))):
            disc(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1.5, 5.0), float(st.get("bubble_brightness", 0.6)) * rng.uniform(0.5, 1.0), (0.9, 0.95, 0.9))
        for _ in range(int(st.get("snow", 0))):
            disc(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.6, 1.4), rng.uniform(0.3, 0.9), (0.8, 0.85, 0.8))
        for _ in range(int(st.get("distractors", 0))):
            disc(rng.uniform(0.05 * W, 0.95 * W), rng.uniform(0.05 * H, 0.95 * H), rng.uniform(3.0, 7.0), 1.2)
        if float(st.get("glint", 0.0)) > 0:
            cx = rng.uniform(0.2 * W, 0.8 * W)
            for k in range(6):
                disc(cx + rng.normal(0, 25), 0.06 * H + rng.normal(0, 8), rng.uniform(5.0, 12.0), 0.9 * float(st["glint"]))
        n_ref = int(st.get("reflections", 0))
        if n_ref:
            tr = self.truth(pos, eul_deg)
            for name, d in list(tr.items())[:n_ref]:
                disc(d["u"], d["v"] + 0.18 * self.H, 3.0, 0.55)                     # a dimmer copy below each light (reflection off the hull / floor)
        if st.get("occlude"):
            d = self.truth(pos, eul_deg).get(str(st["occlude"]))
            if d is not None:
                lk = self.look
                raw = float(lk["core_gain"]) * d["brightness"]
                sigma = float(lk["bloom_sigma_px"]) + float(lk["bloom_sigma_gain"]) * math.log2(1.0 + raw)
                r = int(max(10, self.f * 0.35 / max(d["z"], 0.3), 3.0 * sigma))                  # hides the glow as well as the core (a real object does)
                bg = np.asarray(self.look["background_bgr"], np.float32)
                cv2.circle(img, (int(round(d["u"])), int(round(d["v"]))), r, tuple(float(x) for x in bg), -1)
        sg = float(st.get("blur_sigma_px", 0.0))
        if sg > 0:
            img = cv2.GaussianBlur(img, (0, 0), sg)
        mb = int(st.get("motion_blur_px", 0))
        if mb > 1:
            k = np.zeros((mb, mb), np.float32)
            ang = math.radians(float(st.get("motion_blur_deg", 0.0)))
            for i in range(mb):
                c = (i - (mb - 1) / 2.0)
                k[int(round((mb - 1) / 2.0 + c * math.sin(ang))), int(round((mb - 1) / 2.0 + c * math.cos(ang)))] = 1.0
            img = cv2.filter2D(img, -1, k / k.sum())
        return img

    def encode_jpeg(self, bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["camera"]["jpeg_quality"])])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()
