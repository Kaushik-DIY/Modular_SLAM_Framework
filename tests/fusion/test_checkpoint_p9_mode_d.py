"""
Phase 9 checkpoint — Mode D end-to-end (LiDAR-main + Visual ORB/PnP verifier).

Per CLAUDE.md §4 Phase 9 (symmetric to Phase 8): the LiDAR front-end + memory +
graph + LiDAR (proximity) proposer + visual verifier are wired into the runner.
Checks: >= 1 visual-accepted cross-modal loop; memory caps respected; optimized
ATE no worse than the front-end baseline.

Scripted over REAL fr1_room frames: a looping LiDAR trajectory with injected
odometry noise, where the last keyframes revisit the first (identical RGB ->
identical ORB), so the visual verifier's PnP confirms the proximity proposal.
The LiDAR pipeline's own B&B verification / g2o PGO never run.
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
from slam_core.fusion.runner import run_mode_d

DATASET = Path("datasets/tum/rgbd_dataset_freiburg1_room")
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], dtype=np.float64)

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="TUM fr1_room not present")


class _LidarResult:
    def __init__(self, pose, is_keyframe=True):
        self.pose = pose
        self.is_keyframe = is_keyframe


class _ScriptedLidarBackend:
    def __init__(self, poses):
        self.poses = poses
        self.i = 0

    def process_scan(self, scan, t):
        p = self.poses[min(self.i, len(self.poses) - 1)]
        self.i += 1
        return _LidarResult(p, is_keyframe=True)


class _Frame:
    def __init__(self, rgb, depth, rgb_t, scan):
        self.rgb, self.depth, self.rgb_t, self.scan = rgb, depth, rgb_t, scan


def _build_run(n_distinct=8, n_revisit=3, seed=1):
    ds = FusionDataset(DATASET, num_beams=360, noise_sigma=0.0)
    raw = list(ds.iter_frames(max_frames=n_distinct))

    radius = 2.0
    gt = [Pose2(radius * math.cos(2 * math.pi * k / n_distinct),
               radius * math.sin(2 * math.pi * k / n_distinct),
               wrap_angle(2 * math.pi * k / n_distinct + math.pi / 2))
          for k in range(n_distinct)]
    visit_index = list(range(n_distinct)) + list(range(n_revisit))
    gt_seq = [gt[i] for i in visit_index]

    rng = np.random.default_rng(seed)
    scripted, frames, gt_poses = [], [], []
    prev = None
    for kid, idx in enumerate(visit_index):
        if prev is None:
            pose = gt_seq[kid]
        else:
            rel_true = pose_compose(pose_inverse(gt_seq[kid - 1]), gt_seq[kid])
            noise = Pose2(float(rng.normal(0, 0.012)), float(rng.normal(0, 0.012)),
                          float(rng.normal(0, 0.006)))
            pose = pose_compose(prev, pose_compose(rel_true, noise))
        prev = pose
        scripted.append(pose)
        fr = raw[idx]
        frames.append(_Frame(fr.rgb, fr.depth, float(kid), fr.scan))
        gt_poses.append(gt_seq[kid])
    return scripted, frames, gt_poses


def _ate(poses_by_id, gt_list, ids):
    errs = [math.hypot(poses_by_id[i].x - gt_list[i].x, poses_by_id[i].y - gt_list[i].y)
            for i in ids]
    return float(np.sqrt(np.mean(np.square(errs))))


def test_mode_d_end_to_end():
    scripted, frames, gt = _build_run()
    cfg = FusionConfig(mode=Mode.LVMAIN, dataset_path=str(DATASET), output_dir="out",
                       optimize_every_n_keyframes=3, visual_min_inliers=15, visual_nndr=0.75)

    result = run_mode_d(cfg, frames, _ScriptedLidarBackend(scripted),
                        camera_K=K, min_index_separation=4, proximity_radius=1.0)

    ids = result.keyframe_ids
    assert result.loop_count >= 1
    assert len(result.accepted_loops) >= 1

    assert result.memory_stats["stm"] <= cfg.stm_size
    assert result.memory_stats["wm"] <= cfg.wm_cap
    assert result.memory_stats["ltm"] <= cfg.ltm_cap

    ate_before = _ate(result.frontend_poses, gt, ids)
    ate_after = _ate(result.optimized_poses, gt, ids)
    assert ate_before > 0
    assert ate_after <= ate_before + 1e-6
    assert len(result.trajectory) == len(ids)


def test_mode_d_loops_connect_revisit_to_origin():
    scripted, frames, gt = _build_run()
    cfg = FusionConfig(mode=Mode.LVMAIN, dataset_path=str(DATASET), output_dir="out",
                       optimize_every_n_keyframes=3, visual_min_inliers=15)
    result = run_mode_d(cfg, frames, _ScriptedLidarBackend(scripted),
                        camera_K=K, min_index_separation=4, proximity_radius=1.0)
    for source, target in result.accepted_loops:
        assert source > target
        assert (source - target) >= 4
