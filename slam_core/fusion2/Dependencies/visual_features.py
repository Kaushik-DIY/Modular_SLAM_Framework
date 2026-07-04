"""Visual payload extraction + camera/base frame helpers for fusion v2.

- extract_orb_rgbd(): standalone ORB + depth backprojection for frames that are
  NOT driven through the full ORB-SLAM front-end (lidar_orb mode signatures).
- cam_rel_to_base_se2(): conjugate a relative camera transform into SE(2).

The PnP loop verifier that consumes these lives in
``slam_core/fusion2/Loop_Verifier/pnp.py``.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

# Optical (z-fwd, x-right, y-down) -> base (x-fwd, y-left, z-up); REP-103.
BASE_T_CAM = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

_ORB = None


def extract_orb_rgbd(rgb, depth, K: np.ndarray, depth_factor: float = 1000.0,
                     n_features: int = 1500):
    """ORB keypoints/descriptors + camera-frame 3D points from a raw RGB-D pair.

    Returns (kpts (N,2) f32, des (N,32) u8, pts3d (N,3) f32) with NaN rows in
    pts3d where depth is missing. Returns (None, None, None) when no features.
    """
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
    kps, des = _ORB.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return None, None, None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape[:2]
    kpts = np.array([k.pt for k in kps], dtype=np.float32)
    pts3d = np.full((len(kps), 3), np.nan, dtype=np.float32)
    us = np.clip(kpts[:, 0].astype(int), 0, w - 1)
    vs = np.clip(kpts[:, 1].astype(int), 0, h - 1)
    z = depth[vs, us].astype(np.float32) / float(depth_factor)
    valid = z > 0.05
    # Back-project valid depth samples into the optical camera frame.
    pts3d[valid, 0] = (kpts[valid, 0] - cx) / fx * z[valid]
    pts3d[valid, 1] = (kpts[valid, 1] - cy) / fy * z[valid]
    pts3d[valid, 2] = z[valid]
    return kpts, np.asarray(des, dtype=np.uint8), pts3d


def cam_rel_to_base_se2(T_cam_rel: np.ndarray) -> Tuple[float, float, float]:
    """Conjugate a relative camera transform into the planar base frame."""
    Tb = BASE_T_CAM @ T_cam_rel @ np.linalg.inv(BASE_T_CAM)
    return (float(Tb[0, 3]), float(Tb[1, 3]),
            float(math.atan2(Tb[1, 0], Tb[0, 0])))
