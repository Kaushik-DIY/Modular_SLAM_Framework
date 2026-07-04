"""PnP loop verifier for fusion v2 cross-modal modes.

ORB ratio-match + PnP RANSAC between two signatures' visual payloads ->
planar relative pose. Port of the v1 VisualLoopVerifier core operating
directly on payload arrays. The caller converts the returned camera-frame
transform into the planar base frame via
``Dependencies.visual_features.cam_rel_to_base_se2``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

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
            # Target provides 3D landmarks; query provides their 2D observations.
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
    # Invert OpenCV's query-from-target estimate to return target-from-query.
    return True, np.linalg.inv(T_query_target), n_in, n_in / max(len(obj), 1)
