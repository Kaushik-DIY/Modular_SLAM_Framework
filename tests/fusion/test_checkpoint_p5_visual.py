"""
Phase 5 checkpoint — Visual verifier (ORB + PnP).

Per CLAUDE.md §4 Phase 5: ORB feature sets from known-overlapping frames yield a
relative transform within tolerance of ground truth; unrelated frames are
rejected. The positive case is built deterministically: a set of 3D points with
distinct descriptors is observed from two camera poses related by a known planar
(base-frame) motion, so PnP must recover that motion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from slam_core.common.types import Pose2
from slam_core.fusion.signature import Signature
from slam_core.fusion.graph import keyframe_target_id
from slam_core.fusion.visual_verifier import VisualLoopVerifier, _rep103_base_T_cam
from slam_core.loop_closure import ClosureTarget, LoopNode

# fr1 intrinsics
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], dtype=np.float64)


def _se3_from_pose2(p: Pose2) -> np.ndarray:
    c, s = math.cos(p.theta), math.sin(p.theta)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [p.x, p.y, 0.0]
    return T


def _project(P_cam, K):
    z = P_cam[:, 2]
    u = K[0, 0] * P_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * P_cam[:, 1] / z + K[1, 2]
    return np.column_stack([u, v])


def _make_overlapping_signatures(rng, g_base: Pose2, n=120):
    """Two keyframes viewing the same points, related by planar motion g_base."""
    # 3D points in the TARGET camera frame, in front of the camera.
    P_t = np.column_stack([
        rng.uniform(-2, 2, n),
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(1.5, 5.0, n),
    ])
    desc = rng.integers(0, 256, (n, 32), np.uint8)  # distinct -> unambiguous match
    target = Signature(id=0, timestamp=0.0, pose=Pose2(0, 0, 0),
                       descriptors=desc, points3d=P_t, keypoints=_project(P_t, K))

    # query camera relative to target, in camera frame, from the base-frame motion.
    base_T_cam = _rep103_base_T_cam()
    cam_T_base = np.linalg.inv(base_T_cam)
    T_tq_cam = cam_T_base @ _se3_from_pose2(g_base) @ base_T_cam  # target->query (cam)
    T_qt_cam = np.linalg.inv(T_tq_cam)
    P_q = (T_qt_cam[:3, :3] @ P_t.T).T + T_qt_cam[:3, 3]   # same points, query frame
    query = Signature(id=1, timestamp=1.0, pose=Pose2(0, 0, 0),
                      descriptors=desc.copy(), points3d=P_q, keypoints=_project(P_q, K))
    return query, target


def _provider(sigs):
    return lambda i: sigs.get(i)


def test_known_planar_motion_recovered():
    rng = np.random.default_rng(0)
    g_base = Pose2(0.25, 0.10, 0.12)
    query, target = _make_overlapping_signatures(rng, g_base)
    sigs = {0: target, 1: query}

    v = VisualLoopVerifier(K=K, get_signature=_provider(sigs), min_inliers=15)
    node = LoopNode(node_id=1, scan_points=np.zeros((0, 2)),
                    pose_guess_global=Pose2(0, 0, 0), timestamp=1.0)
    tgt = ClosureTarget(target_id=keyframe_target_id(0), target_type="keyframe",
                        pose_global=Pose2(0, 0, 0), is_finished=True, is_fixed=False,
                        map_view=None)

    res = v.verify(node, tgt)
    assert res.success
    p = res.matched_node_pose_global
    assert abs(p.x - g_base.x) < 0.03
    assert abs(p.y - g_base.y) < 0.03
    assert abs(p.theta - g_base.theta) < 0.03


def test_unrelated_frames_rejected():
    rng = np.random.default_rng(1)
    target = Signature(id=0, timestamp=0.0, pose=Pose2(0, 0, 0),
                       descriptors=rng.integers(0, 256, (120, 32), np.uint8),
                       points3d=rng.uniform(-2, 2, (120, 3)) + [0, 0, 4],
                       keypoints=rng.uniform(0, 640, (120, 2)))
    # query has entirely different descriptors -> no ratio-test matches
    query = Signature(id=1, timestamp=1.0, pose=Pose2(0, 0, 0),
                      descriptors=rng.integers(0, 256, (120, 32), np.uint8),
                      keypoints=rng.uniform(0, 640, (120, 2)),
                      points3d=rng.uniform(-2, 2, (120, 3)) + [0, 0, 4])
    sigs = {0: target, 1: query}
    v = VisualLoopVerifier(K=K, get_signature=_provider(sigs), min_inliers=15)
    node = LoopNode(node_id=1, scan_points=np.zeros((0, 2)),
                    pose_guess_global=Pose2(0, 0, 0), timestamp=1.0)
    tgt = ClosureTarget(target_id=keyframe_target_id(0), target_type="keyframe",
                        pose_global=Pose2(0, 0, 0), is_finished=True, is_fixed=False,
                        map_view=None)
    res = v.verify(node, tgt)
    assert not res.success
    assert res.status in ("matcher_failed", "score_failed")


def test_missing_payload_matcher_failed():
    target = Signature(id=0, timestamp=0.0, pose=Pose2(0, 0, 0))  # no ORB payload
    query = Signature(id=1, timestamp=1.0, pose=Pose2(0, 0, 0))
    sigs = {0: target, 1: query}
    v = VisualLoopVerifier(K=K, get_signature=_provider(sigs))
    node = LoopNode(node_id=1, scan_points=np.zeros((0, 2)),
                    pose_guess_global=Pose2(0, 0, 0), timestamp=1.0)
    tgt = ClosureTarget(target_id=keyframe_target_id(0), target_type="keyframe",
                        pose_global=Pose2(0, 0, 0), is_finished=True, is_fixed=False,
                        map_view=None)
    res = v.verify(node, tgt)
    assert not res.success
    assert res.status == "matcher_failed"
