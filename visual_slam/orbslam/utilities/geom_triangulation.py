"""
Triangulation helpers for normalized keypoints.
This module reconstructs world points from two camera poses and matched rays.
"""

from __future__ import annotations

import cv2
import numpy as np


def triangulate_normalized_points(Tcw1, Tcw2, kpn1, kpn2):
    Tcw1 = np.asarray(Tcw1, dtype=np.float64)
    Tcw2 = np.asarray(Tcw2, dtype=np.float64)
    kpn1 = np.asarray(kpn1, dtype=np.float64).reshape(-1, 2)
    kpn2 = np.asarray(kpn2, dtype=np.float64).reshape(-1, 2)

    if len(kpn1) == 0 or len(kpn1) != len(kpn2):
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=bool)

    P1 = np.ascontiguousarray(Tcw1[:3, :], dtype=np.float64)
    P2 = np.ascontiguousarray(Tcw2[:3, :], dtype=np.float64)

    pts4 = cv2.triangulatePoints(P1, P2, kpn1.T, kpn2.T).T

    n = len(pts4)
    pts3 = np.zeros((n, 3), dtype=np.float64)
    mask = np.zeros(n, dtype=bool)

    w = pts4[:, 3]
    valid_w = np.abs(w) >= 1e-12
    if not np.any(valid_w):
        return pts3, mask

    pw = np.zeros((n, 3), dtype=np.float64)
    pw[valid_w] = pts4[valid_w, :3] / w[valid_w, None]

    R1, t1 = Tcw1[:3, :3], Tcw1[:3, 3]
    R2, t2 = Tcw2[:3, :3], Tcw2[:3, 3]
    pc1 = pw @ R1.T + t1
    pc2 = pw @ R2.T + t2

    mask = (
        valid_w
        & np.all(np.isfinite(pw), axis=1)
        & (pc1[:, 2] > 0.0)
        & (pc2[:, 2] > 0.0)
    )
    pts3[mask] = pw[mask]

    return pts3, mask
