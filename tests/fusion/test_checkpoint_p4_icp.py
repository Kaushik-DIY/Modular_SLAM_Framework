"""
Phase 4 checkpoint — ICP verifier (small_gicp).

Per CLAUDE.md §4 Phase 4: a self-loop test (scan vs itself + noise) returns
identity; two overlapping TUM-derived scans align within tolerance; unrelated
scans are rejected with the documented status strings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slam_core.common.types import Pose2
from slam_core.common.se2 import transform_points, pose_inverse
from slam_core.fusion.icp_verifier import ICPLoopVerifier
from slam_core.loop_closure import ClosureTarget, LoopNode

DATASET = Path("datasets/tum/rgbd_dataset_freiburg1_room")


def _node(scan, pose):
    return LoopNode(node_id=1, scan_points=scan, pose_guess_global=pose, timestamp=0.0)


def _target(scan, pose):
    return ClosureTarget(target_id="kf:0", target_type="keyframe", pose_global=pose,
                         is_finished=True, is_fixed=False, map_view=scan)


def _ring_scan(rng, n=400):
    """A structured 2D scan (noisy ring + interior) so alignment is well-posed."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 3.0 + 0.4 * np.sin(5 * ang)
    pts = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    return pts + rng.normal(0, 0.005, pts.shape)


def test_self_loop_returns_identity():
    rng = np.random.default_rng(0)
    scan = _ring_scan(rng)
    noisy = scan + rng.normal(0, 0.01, scan.shape)
    v = ICPLoopVerifier()
    res = v.verify(_node(noisy, Pose2(0, 0, 0)), _target(scan, Pose2(0, 0, 0)))
    assert res.success
    p = res.matched_node_pose_global
    assert abs(p.x) < 0.05 and abs(p.y) < 0.05 and abs(p.theta) < 0.05


def test_known_transform_recovered():
    rng = np.random.default_rng(1)
    target_scan = _ring_scan(rng)            # in the target/world frame
    g = Pose2(0.30, -0.20, 0.15)             # true query global pose
    # query observed the same place from pose g -> its scan in the query frame
    query_scan = transform_points(pose_inverse(g), target_scan)

    v = ICPLoopVerifier()
    # feed a noisy initial guess to prove ICP refines it
    guess = Pose2(g.x + 0.05, g.y - 0.04, g.theta + 0.03)
    res = v.verify(_node(query_scan, guess), _target(target_scan, Pose2(0, 0, 0)))

    assert res.success
    p = res.matched_node_pose_global
    assert abs(p.x - g.x) < 0.05
    assert abs(p.y - g.y) < 0.05
    assert abs(p.theta - g.theta) < 0.05


def test_unrelated_scans_rejected():
    rng = np.random.default_rng(2)
    target_scan = _ring_scan(rng)
    far_scan = rng.normal(0, 1.0, (400, 2)) + 50.0  # no overlap
    v = ICPLoopVerifier()
    res = v.verify(_node(far_scan, Pose2(0, 0, 0)), _target(target_scan, Pose2(0, 0, 0)))
    assert not res.success
    assert res.status in ("matcher_failed", "score_failed")


def test_empty_scan_matcher_failed():
    v = ICPLoopVerifier()
    res = v.verify(_node(np.zeros((0, 2)), Pose2(0, 0, 0)),
                   _target(_ring_scan(np.random.default_rng(3)), Pose2(0, 0, 0)))
    assert not res.success
    assert res.status == "matcher_failed"


@pytest.mark.skipif(not DATASET.exists(), reason="TUM fr1_room not present")
def test_tum_derived_scans_align():
    from slam_core.fusion.dataset import FusionDataset

    ds = FusionDataset(DATASET, num_beams=360, noise_sigma=0.0)
    frame = next(iter(ds.iter_frames(max_frames=1)))
    target_scan = frame.scan
    assert target_scan is not None and len(target_scan) > 50

    g = Pose2(0.10, 0.05, 0.05)
    query_scan = transform_points(pose_inverse(g), target_scan)

    v = ICPLoopVerifier()
    res = v.verify(_node(query_scan, g), _target(target_scan, Pose2(0, 0, 0)))
    assert res.success
    p = res.matched_node_pose_global
    assert abs(p.x - g.x) < 0.08 and abs(p.y - g.y) < 0.08 and abs(p.theta - g.theta) < 0.08
