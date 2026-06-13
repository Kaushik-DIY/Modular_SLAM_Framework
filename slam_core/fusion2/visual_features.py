"""Visual payload extraction + PnP verification for fusion v2 cross-modal modes.

- extract_orb_rgbd(): standalone ORB + depth backprojection for frames that are
  NOT driven through the full ORB-SLAM front-end (lidar_orb mode signatures).
- pnp_verify(): ORB ratio-match + PnP RANSAC between two signatures' visual
  payloads -> planar relative pose. Port of the v1 VisualLoopVerifier core
  (slam_core/fusion/visual_verifier.py) operating directly on payload arrays.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

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
    pts3d[valid, 0] = (kpts[valid, 0] - cx) / fx * z[valid]
    pts3d[valid, 1] = (kpts[valid, 1] - cy) / fy * z[valid]
    pts3d[valid, 2] = z[valid]
    return kpts, np.asarray(des, dtype=np.uint8), pts3d


_BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def pnp_verify(q_kpts: np.ndarray, q_des: np.ndarray,
               t_des: np.ndarray, t_pts3d: np.ndarray, K: np.ndarray,
               nndr: float = 0.7, min_inliers: int = 15,
               reproj_error: float = 3.0
               ) -> Tuple[bool, Optional[np.ndarray], int, float]:
    """Match query descriptors to target, solve PnP (target 3D vs query 2D).

    Returns (ok, T_target_query 4x4 camera-frame, n_inliers, inlier_ratio).
    """
    if q_des is None or t_des is None or len(q_des) < 2 or len(t_des) < 2:
        return False, None, 0, 0.0
    knn = _BF.knnMatch(np.asarray(q_des, np.uint8), np.asarray(t_des, np.uint8), k=2)
    obj, img = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < nndr * n.distance:
            p3 = t_pts3d[m.trainIdx]
            if not np.isfinite(p3).all():
                continue
            obj.append(p3)
            img.append(q_kpts[m.queryIdx])
    if len(obj) < min_inliers:
        return False, None, len(obj), 0.0
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(obj, np.float64).reshape(-1, 1, 3),
        np.asarray(img, np.float64).reshape(-1, 1, 2),
        np.asarray(K, np.float64), None,
        reprojectionError=reproj_error, flags=cv2.SOLVEPNP_ITERATIVE)
    n_in = 0 if inliers is None else int(len(inliers))
    if not ok or n_in < min_inliers:
        return False, None, n_in, 0.0
    R, _ = cv2.Rodrigues(rvec)
    T_query_target = np.eye(4)
    T_query_target[:3, :3] = R
    T_query_target[:3, 3] = tvec.ravel()
    return True, np.linalg.inv(T_query_target), n_in, n_in / max(len(obj), 1)


def cam_rel_to_base_se2(T_cam_rel: np.ndarray) -> Tuple[float, float, float]:
    """Conjugate a relative camera transform into the planar base frame."""
    Tb = BASE_T_CAM @ T_cam_rel @ np.linalg.inv(BASE_T_CAM)
    return (float(Tb[0, 3]), float(Tb[1, 3]),
            float(math.atan2(Tb[1, 0], Tb[0, 0])))
