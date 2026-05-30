"""
VisualLoopVerifier — cross-modal loop verification with ORB + PnP RANSAC.

RTAB_inspired_implementation_plan.md §10.2. Implements the ``LoopVerifier``
Protocol from ``slam_core/loop_closure.py``. Used in Mode D: the LiDAR
front-end proposes loop candidates, this verifier confirms them by matching the
query keyframe's ORB descriptors against the target keyframe's, then solving
PnP RANSAC (query 2D keypoints vs target 3D points) for the relative pose.

The generic ``LoopNode`` carries no ORB payload, so the verifier is given a
signature provider (``get_signature(id) -> Signature``); the query is looked up
by ``node.node_id`` and the target by the keyframe id parsed from
``target.target_id``. Both verifiers stay swappable behind the same Protocol.

PnP returns ``T_query_target`` (camera frame). Its inverse, conjugated by the
camera->base extrinsic (REP-103 optical frame by default), yields the planar
relative pose, from which the query node's corrected global pose is built.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import cv2

from slam_core.common.types import Pose2
from slam_core.common.se2 import pose_compose, wrap_angle
from slam_core.fusion.graph import _parse_target_id
from slam_core.loop_closure import ClosureTarget, LoopMatchResult, LoopNode


def _rep103_base_T_cam() -> np.ndarray:
    """Optical (z-fwd, x-right, y-down) -> base (x-fwd, y-left, z-up)."""
    R = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    return T


class VisualLoopVerifier:
    def __init__(
        self,
        K: np.ndarray,
        get_signature: Callable[[int], object],
        nndr: float = 0.7,
        min_inliers: int = 15,
        reproj_error: float = 3.0,
        base_T_cam: Optional[np.ndarray] = None,
    ):
        self.K = np.asarray(K, dtype=np.float64)
        self._get = get_signature
        self.nndr = float(nndr)
        self.min_inliers = int(min_inliers)
        self.reproj_error = float(reproj_error)
        self.base_T_cam = _rep103_base_T_cam() if base_T_cam is None else np.asarray(base_T_cam, float)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def verify(self, node: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        query = self._get(int(node.node_id))
        tgt = self._get(_parse_target_id(target.target_id))

        if query is None or tgt is None:
            return LoopMatchResult(False, 0.0, None, status="matcher_failed")
        if (getattr(query, "descriptors", None) is None
                or getattr(query, "keypoints", None) is None
                or getattr(tgt, "descriptors", None) is None
                or getattr(tgt, "points3d", None) is None):
            return LoopMatchResult(False, 0.0, None, status="matcher_failed")

        q_desc = np.asarray(query.descriptors, dtype=np.uint8)
        t_desc = np.asarray(tgt.descriptors, dtype=np.uint8)
        if len(q_desc) < 2 or len(t_desc) < 2:
            return LoopMatchResult(False, 0.0, None, status="matcher_failed")

        # ratio-test descriptor matching: query -> target
        knn = self._matcher.knnMatch(q_desc, t_desc, k=2)
        obj_pts, img_pts = [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.nndr * n.distance:
                obj_pts.append(tgt.points3d[m.trainIdx])
                img_pts.append(query.keypoints[m.queryIdx])

        if len(obj_pts) < self.min_inliers:
            return LoopMatchResult(False, float(len(obj_pts)), None, status="score_failed")

        obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
        img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.K, None,
            reprojectionError=self.reproj_error,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        n_inliers = 0 if inliers is None else int(len(inliers))
        if not ok or n_inliers < self.min_inliers:
            return LoopMatchResult(False, float(n_inliers), None, status="score_failed")

        R, _ = cv2.Rodrigues(rvec)
        T_query_target = np.eye(4)
        T_query_target[:3, :3] = R
        T_query_target[:3, 3] = tvec.ravel()
        T_target_query = np.linalg.inv(T_query_target)

        rel_base = self._rel_cam_to_base_se2(T_target_query)
        matched_global = pose_compose(target.pose_global, rel_base)
        score = float(n_inliers) / float(len(obj_pts))

        return LoopMatchResult(
            success=True,
            score=score,
            matched_node_pose_global=matched_global,
            status="accepted",
        )

    def _rel_cam_to_base_se2(self, T_cam_rel: np.ndarray) -> Pose2:
        """Conjugate a relative camera transform into the planar base frame."""
        Tb = self.base_T_cam @ T_cam_rel @ np.linalg.inv(self.base_T_cam)
        return Pose2(
            float(Tb[0, 3]),
            float(Tb[1, 3]),
            float(wrap_angle(math.atan2(Tb[1, 0], Tb[0, 0]))),
        )
