"""
Runnable visual front-end stand-ins for the fusion modes.

The production Mode C front-end is the full ORB-SLAM stack (tracking +
KeyFrameDatabase appearance loop detector). That stack needs a DBoW3 vocabulary
and the full KeyFrame machinery, so this module provides lightweight, dependency-
free stand-ins that plug into the same Phase 6 adapter seams:

- ``OrbRgbdVoBackend``  — minimal ORB RGB-D visual odometry (a
  ``VisualFrontendService`` backend exposing ``track(rgb, depth, t)``).
- ``BruteForceOrbDetector`` — an appearance loop-candidate detector wrapped by
  ``OrbLoopProposer`` (``add`` / ``detect`` returning candidate ids + scores).

Both honour the documented adapter interfaces, so the real ORB-SLAM
``LoopDetector``/tracking can replace them later without touching the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class VoKeyframe:
    id: int
    pose: np.ndarray              # 4x4 T_world_cam
    keypoints: np.ndarray         # (N, 2)
    descriptors: np.ndarray       # (N, 32) uint8
    points3d: np.ndarray          # (N, 3) camera frame (NaN where depth invalid)


def _backproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray, max_depth: float):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = uv[:, 0], uv[:, 1]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    H, W = depth.shape
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    z = np.full(len(uv), np.nan)
    z[inb] = depth[np.clip(vi, 0, H - 1)[inb], np.clip(ui, 0, W - 1)[inb]]
    good = np.isfinite(z) & (z > 0) & (z < max_depth)
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.column_stack([X, Y, z]), good


class OrbRgbdVoBackend:
    """Minimal ORB RGB-D visual odometry. Keyframe-to-keyframe PnP."""

    def __init__(self, K: np.ndarray, n_features: int = 1000,
                 keyframe_every: int = 5, max_depth: float = 8.0, nndr: float = 0.75):
        self.K = np.asarray(K, dtype=np.float64)
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keyframe_every = int(keyframe_every)
        self.max_depth = float(max_depth)
        self.nndr = float(nndr)
        self.frame_idx = 0
        self.kf_id = 0
        self._prev: Optional[VoKeyframe] = None

    def track(self, rgb, depth, t) -> Optional[VoKeyframe]:
        self.frame_idx += 1
        if self.frame_idx % self.keyframe_every != 0:
            return None

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < 8:
            return None
        kpts = np.array([k.pt for k in kp], dtype=np.float64)
        pts3d, good = _backproject(kpts, np.asarray(depth, float), self.K, self.max_depth)

        if self._prev is None:
            world_T_cam = np.eye(4)
        else:
            T_cp = self._estimate_rel(kpts, desc)
            world_T_cam = self._prev.pose if T_cp is None else self._prev.pose @ np.linalg.inv(T_cp)

        kf = VoKeyframe(id=self.kf_id, pose=world_T_cam,
                        keypoints=kpts, descriptors=desc, points3d=pts3d)
        self._prev = kf
        self.kf_id += 1
        return kf

    def _estimate_rel(self, kpts, desc) -> Optional[np.ndarray]:
        prev = self._prev
        knn = self.bf.knnMatch(prev.descriptors, desc, k=2)
        obj, img = [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.nndr * n.distance:
                p3 = prev.points3d[m.queryIdx]
                if np.isfinite(p3).all():
                    obj.append(p3)
                    img.append(kpts[m.trainIdx])
        if len(obj) < 6:
            return None
        obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
        img = np.asarray(img, np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, self.K, None,
                                                     reprojectionError=3.0)
        if not ok or inliers is None or len(inliers) < 6:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        return T  # T_curr_prev


@dataclass
class _DetectorOutput:
    candidate_idxs: List[int]
    candidate_scores: List[float]


class BruteForceOrbDetector:
    """Appearance loop-candidate detector by ORB descriptor vote count.

    Not a BoW: a brute-force ratio-test match against registered keyframes.
    Wrapped by ``OrbLoopProposer``; production swaps in ORB-SLAM's
    ``LoopDetector`` (KeyFrameDatabase) behind the same interface.
    """

    def __init__(self, min_votes: int = 25, top_k: int = 3, nndr: float = 0.75):
        self.min_votes = int(min_votes)
        self.top_k = int(top_k)
        self.nndr = float(nndr)
        self._db: Dict[int, np.ndarray] = {}
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def add(self, keyframe) -> None:
        self._db[int(keyframe.id)] = np.asarray(keyframe.descriptors, dtype=np.uint8)

    def detect(self, keyframe) -> _DetectorOutput:
        qid = int(keyframe.id)
        q = np.asarray(keyframe.descriptors, dtype=np.uint8)
        ids, scores = [], []
        for cid, d in self._db.items():
            if cid == qid:
                continue
            v = self._votes(q, d)
            if v >= self.min_votes:
                ids.append(cid)
                scores.append(float(v))
        if not ids:
            return _DetectorOutput([], [])
        order = np.argsort(scores)[::-1][: self.top_k]
        return _DetectorOutput([ids[i] for i in order], [scores[i] for i in order])

    def _votes(self, q, d) -> int:
        if len(q) < 2 or len(d) < 2:
            return 0
        knn = self.bf.knnMatch(q, d, k=2)
        good = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.nndr * n.distance:
                good += 1
        return good


class ProximityTargetProvider:
    """Spatial-proximity loop-candidate provider (TargetProvider Protocol).

    Runnable stand-in for ``carto.loop_closure_adapter.CartoTargetProvider`` in
    Mode D: proposes past keyframes whose current pose estimate is within a
    radius of the query and far enough apart in index. Wrapped by
    ``LidarLoopProposer``; the real B&B/submap provider replaces it behind the
    same interface.
    """

    def __init__(self, signatures: Dict[int, object], radius: float = 1.0,
                 min_index_separation: int = 10):
        self._sigs = signatures
        self.radius = float(radius)
        self.min_index_separation = int(min_index_separation)

    def get_candidate_targets_for_node(self, node, all_nodes, config):
        from slam_core.loop_closure import ClosureTarget

        qid = int(node.node_id)
        ranked = []
        for cid, sig in self._sigs.items():
            cid = int(cid)
            if cid == qid or abs(cid - qid) < self.min_index_separation:
                continue
            d = float(np.hypot(node.pose_guess_global.x - sig.pose.x,
                               node.pose_guess_global.y - sig.pose.y))
            if d <= self.radius:
                ranked.append((d, cid, sig))
        ranked.sort(key=lambda x: x[0])
        max_t = int(getattr(config, "max_candidate_targets_per_new_node", 4))
        if max_t > 0:
            ranked = ranked[:max_t]
        return [
            ClosureTarget(target_id=str(cid), target_type="keyframe",
                          pose_global=sig.pose, is_finished=True, is_fixed=False,
                          map_view=sig, search_source="proximity")
            for _, cid, sig in ranked
        ]

    def get_finished_target(self, target_id):  # Protocol completeness (unused in v1)
        raise NotImplementedError

    def get_candidate_nodes_for_finished_target(self, target, all_nodes, config):
        return []
