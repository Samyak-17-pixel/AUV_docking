"""Pick the four real ring lights out of a handful of bright candidates by checking that they FORM the dock ring.

The dock's four lights sit on a 1 m circle in a vertical plane: top (0,-1), bottom (0,+1), right (+0.707,-0.707), left (-0.707,-0.707) in (sideways, down).
Seen from a distance and with the camera roll removed, that circle becomes an ellipse and the lights keep their relative places up to a 2x2 matrix (an AFFINE
map: the view angle squeezes x, the pitch squeezes y). Four points have 8 numbers; an affine map has 6 free numbers, so a genuine ring leaves a 2-number
leftover (the residual) near zero, and a bubble, a reflection or a glint in place of one light does not. This removes the false locks the old rule had ('the 4
strongest peaks are the dock'), and a ring with a hidden light (3 real + 1 false) is rejected instead of being steered on.

Pure numpy, no ROS. Used by dock_detector.DockDetector; tuned by the `ring:` section of dock_detection.yaml.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Core = Tuple[float, float]

_S = math.sqrt(0.5)
# model coordinates (sideways to the right, down), in the order  top, bottom, right, left
_Q = np.array([[0.0, -1.0], [0.0, 1.0], [_S, -_S], [-_S, -_S]])
_DESIGN = np.column_stack([np.ones(4), _Q])                # p = [1, qx, qy] @ [c; A^T]
_PINV = np.linalg.pinv(_DESIGN)


@dataclass
class RingFit:
    ok: bool
    residual: float = math.inf            # RMS leftover / half top-bottom distance (0 = a perfect ring)
    squeeze_x: float = 0.0                # sideways scale / half top-bottom distance: cos(view angle) when the view is oblique
    shear: float = 0.0                    # largest off-diagonal term / half top-bottom distance
    reason: str = ""


def label_four(pts: Sequence[Core]) -> Optional[Tuple[Core, Core, Core, Core]]:
    """top = smallest y, bottom = largest y, the other two split by x -> (top, bottom, right, left)."""
    by_y = sorted(pts, key=lambda p: p[1])
    top, bottom = by_y[0], by_y[-1]
    mid = sorted(by_y[1:-1], key=lambda p: p[0])
    if len(mid) != 2:
        return None
    return top, bottom, mid[1], mid[0]


def fit_ring(top: Core, bottom: Core, right: Core, left: Core, *, min_squeeze: float = 0.08, max_squeeze: float = 1.35, max_shear: float = 0.4) -> RingFit:
    P = np.array([top, bottom, right, left], dtype=float)
    half_tb = 0.5 * float(np.hypot(*(P[1] - P[0])))
    if half_tb < 4.0:
        return RingFit(False, reason="top and bottom too close")
    coef = _PINV @ P                                         # rows: [cx cy], [A_xx A_yx]... see below
    # coef[0] = centre; coef[1] = image displacement per unit qx (a_x, a_y); coef[2] = per unit qy
    res = P - _DESIGN @ coef
    resid = float(np.sqrt(np.mean(np.sum(res * res, axis=1)))) / half_tb
    a_x, a_y = coef[1], coef[2]                              # a_x = (x,y) image shift per +1 sideways, a_y = per +1 down
    squeeze = float(a_x[0]) / half_tb
    shear = max(abs(float(a_y[0])), abs(float(a_x[1]))) / half_tb
    if not (min_squeeze <= squeeze <= max_squeeze):
        return RingFit(False, resid, squeeze, shear, "side-to-side scale out of range")
    if shear > max_shear:
        return RingFit(False, resid, squeeze, shear, "ring is sheared")
    return RingFit(True, resid, squeeze, shear, "ok")


def select_ring(cands: Sequence[Core], strength: Optional[Sequence[float]] = None, *, fit_tol: float = 0.10, max_candidates: int = 8,
                prior: Optional[Sequence[Core]] = None, prior_weight: float = 0.0) -> Tuple[List[Core], Optional[RingFit]]:
    """-> (cores, fit). cands are image points (roll already removed), strongest first. If some 4 of them form a ring the best 4 are returned with their fit
    (cores in the order top, bottom, right, left is NOT promised: the detector re-labels them); otherwise the strongest <= 3 candidates, fit None.
    strength: optional per-candidate score (bigger = more trusted); a higher mean strength breaks ties between equally good rings.
    prior: the four cores of the previous frame (optional): a ring that moved little scores better, so a distractor cannot steal a place frame to frame."""
    pts = list(cands)[:max_candidates]
    if len(pts) < 4:
        return pts, None
    st = np.ones(len(pts)) if strength is None else np.asarray(list(strength)[:len(pts)], float)
    st = st / max(float(st.max()), 1e-9)
    best, best_score, best_fit = None, math.inf, None
    for idx in itertools.combinations(range(len(pts)), 4):
        lab = label_four([pts[i] for i in idx])
        if lab is None:
            continue
        fit = fit_ring(*lab)
        if not fit.ok or fit.residual > fit_tol:
            continue
        score = fit.residual - 0.03 * float(np.mean(st[list(idx)]))
        if prior is not None and prior_weight > 0.0 and len(prior) == 4:
            d = np.array([min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in prior) for p in (pts[i] for i in idx)])
            score += prior_weight * float(np.mean(d)) / max(fit_tol * 0.5 * np.hypot(lab[0][0] - lab[1][0], lab[0][1] - lab[1][1]), 1.0)
        if score < best_score:
            best, best_score, best_fit = [pts[i] for i in idx], score, fit
    if best is None:
        return pts[:3], None
    return best, best_fit
