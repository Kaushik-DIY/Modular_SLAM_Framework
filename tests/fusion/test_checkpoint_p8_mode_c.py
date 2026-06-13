"""
Phase 8 checkpoint — Mode C end-to-end (Visual-main + LiDAR ICP verifier).

Per CLAUDE.md §4 Phase 8: a full Mode C run wires the visual front-end +
memory + graph + ORB proposer + ICP verifier. Checks: >= 1 ICP-accepted
cross-modal loop; memory tier counts within configured caps; ATE no worse than
the front-end (pre-optimization) baseline (loop closure must not degrade it).

The run is scripted over REAL fr1_room frames along a synthetic looping
trajectory with injected drift: the last keyframes revisit the first (identical
ORB descriptors + identical synthesized scans), guaranteeing a detectable,
ICP-verifiable cross-modal loop. ORB-SLAM's own Sim(3) verification never runs
(the OrbLoopProposer only proposes).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import cv2

from slam_core.common.types import Pose2
from slam_core.common.se2 import pose_compose, pose_inverse, wrap_angle
from slam_core.fusion.config import FusionConfig, Mode
from slam_core.fusion.dataset import FusionDataset
from slam_core.fusion.frontends import _backproject, BruteForceOrbDetector
from slam_core.fusion.runner import run_mode_c

DATASET = Path("datasets/tum/rgbd_dataset_freiburg1_room")
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], dtype=np.float64)

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="TUM fr1_room not present")


def _se2_to_se3(p: Pose2) -> np.ndarray:
    c, s = math.cos(p.theta), math.sin(p.theta)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [p.x, p.y, 0.0]
    return T


class _ScriptedKeyframe:
    def __init__(self, kid, pose_se2, desc, kpts, pts3d):
        self.id = kid
        self.pose = _se2_to_se3(pose_se2)
        self.descriptors = desc
        self.keypoints = kpts
        self.points3d = pts3d


class _ScriptedBackend:
    """Returns one prescribed keyframe per track() call."""

    def __init__(self, keyframes):
        self.keyframes = keyframes
        self.i = 0

    def track(self, rgb, depth, t):
        if self.i >= len(self.keyframes):
            return None
        kf = self.keyframes[self.i]
        self.i += 1
        return kf


class _Frame:
    def __init__(self, rgb, depth, rgb_t, scan):
        self.rgb, self.depth, self.rgb_t, self.scan = rgb, depth, rgb_t, scan


def _build_run(n_distinct=8, n_revisit=3, seed=0):
    ds = FusionDataset(DATASET, num_beams=360, noise_sigma=0.0)
    raw = list(ds.iter_frames(max_frames=n_distinct))
    orb = cv2.ORB_create(1000)

    # ORB payload per distinct frame
    payload = []
    for fr in raw:
        gray = cv2.cvtColor(fr.rgb, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)
        kpts = np.array([k.pt for k in kp], dtype=np.float64)
        pts3d, _ = _backproject(kpts, fr.depth, K, 8.0)
        payload.append((desc, kpts, pts3d, fr))

    # ground-truth loop poses: a circle returning to the start
    radius = 2.0
    gt = []
    for k in range(n_distinct):
        ang = 2 * math.pi * k / n_distinct
        gt.append(Pose2(radius * math.cos(ang), radius * math.sin(ang),
                        wrap_angle(ang + math.pi / 2)))
    visit_index = list(range(n_distinct)) + list(range(n_revisit))
    gt_seq = [gt[i] for i in visit_index]

    # frontend poses = integration of TRUE relative odometry + small noise,
    # so global drift accumulates and a loop closure can correct it.
    rng = np.random.default_rng(seed)
    keyframes, frames, gt_poses = [], [], []
    scripted_prev = None
    for kid, idx in enumerate(visit_index):
        desc, kpts, pts3d, fr = payload[idx]
        if scripted_prev is None:
            scripted = gt_seq[kid]
        else:
            rel_true = pose_compose(pose_inverse(gt_seq[kid - 1]), gt_seq[kid])
            noise = Pose2(float(rng.normal(0, 0.012)), float(rng.normal(0, 0.012)),
                          float(rng.normal(0, 0.006)))
            scripted = pose_compose(scripted_prev, pose_compose(rel_true, noise))
        scripted_prev = scripted
        keyframes.append(_ScriptedKeyframe(kid, scripted, desc, kpts, pts3d))
        frames.append(_Frame(fr.rgb, fr.depth, float(kid), fr.scan))
        gt_poses.append(gt_seq[kid])
    return keyframes, frames, gt_poses


def _ate(poses_by_id, gt_list, ids):
    errs = [math.hypot(poses_by_id[i].x - gt_list[i].x, poses_by_id[i].y - gt_list[i].y)
            for i in ids]
    return float(np.sqrt(np.mean(np.square(errs))))


def test_mode_c_end_to_end():
    keyframes, frames, gt = _build_run()
    # small STM so early keyframes age into WM and become loop-searchable
    # (RTAB-faithful: STM is protected from loop scoring).
    cfg = FusionConfig(mode=Mode.VLMAIN, dataset_path=str(DATASET), output_dir="out",
                       optimize_every_n_keyframes=3, stm_size=4, rehearsal_similarity=0.9,
                       icp_max_corr=0.5, icp_fitness=0.5, icp_inlier_rmse=0.1)
    detector = BruteForceOrbDetector(min_votes=120, top_k=3)

    result = run_mode_c(cfg, frames, _ScriptedBackend(keyframes),
                        base_T_cam=None, loop_detector=detector,
                        min_index_separation=4)

    ids = result.keyframe_ids
    # 1) at least one ICP-accepted cross-modal loop
    assert result.loop_count >= 1
    assert len(result.accepted_loops) >= 1

    # 2) memory tiers within configured caps
    assert result.memory_stats["stm"] <= cfg.stm_size
    assert result.memory_stats["wm"] <= cfg.wm_cap
    assert result.memory_stats["ltm"] <= cfg.ltm_cap

    # 3) optimized ATE no worse than the front-end (pre-optimization) baseline
    ate_before = _ate(result.frontend_poses, gt, ids)
    ate_after = _ate(result.optimized_poses, gt, ids)
    assert ate_after <= ate_before + 1e-6
    assert ate_before > 0  # there was drift to correct

    # full keyframe stream produced a trajectory
    assert len(result.trajectory) == len(ids)


def test_mode_c_accepted_loop_links_revisit_to_origin():
    keyframes, frames, gt = _build_run()
    cfg = FusionConfig(mode=Mode.VLMAIN, dataset_path=str(DATASET), output_dir="out",
                       optimize_every_n_keyframes=3, stm_size=4, rehearsal_similarity=0.9)
    detector = BruteForceOrbDetector(min_votes=120, top_k=3)
    result = run_mode_c(cfg, frames, _ScriptedBackend(keyframes),
                        loop_detector=detector, min_index_separation=4)
    # every accepted loop must connect a late (revisit) keyframe to an early one
    for source, target in result.accepted_loops:
        assert source > target
        assert (source - target) >= 4
