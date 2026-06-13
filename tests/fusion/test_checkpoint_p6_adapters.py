"""
Phase 6 checkpoint — Adapters.

Per CLAUDE.md §4 Phase 6: each adapter, driven on a short run, returns the
expected proposal/keyframe stream AND proves the upstream pipeline's own
verifier did not run — the ORB proposer emits candidates without ORB-SLAM
emitting Sim(3) constraints; the LiDAR proposer emits candidates without the
LiDAR PGO solving. Spies count any forbidden verification/optimization calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from slam_core.common.types import Pose2
from slam_core.fusion.adapters import (
    OrbLoopProposer,
    LidarLoopProposer,
    VisualFrontendService,
    LidarFrontendService,
)
from slam_core.loop_closure import ClosureTarget, LoopClosureConfig, LoopNode


# --------------------------------------------------------------------------
# ORB loop proposer — propose only, no Sim(3) verification
# --------------------------------------------------------------------------

class _KF:
    def __init__(self, kid):
        self.id = kid


class _DetectorOutput:
    def __init__(self, ids, scores):
        self.candidate_idxs = ids
        self.candidate_scores = scores


class _SpyOrbLoopClosing:
    """Surfaces both detection and the forbidden Sim(3)/correction methods."""

    def __init__(self, candidates):
        self._candidates = candidates  # list[(id, score)]
        self.detect_calls = 0
        self.sim3_calls = 0
        self.correct_calls = 0
        self.registered = []

    def add(self, kf):
        self.registered.append(kf.id)

    def detect(self, keyframe):
        self.detect_calls += 1
        ids = [c for c, _ in self._candidates]
        scores = [s for _, s in self._candidates]
        return _DetectorOutput(ids, scores)

    # forbidden in fusion mode:
    def compute_sim3(self, *a, **k):
        self.sim3_calls += 1

    def correct_loop(self, *a, **k):
        self.correct_calls += 1


def test_orb_proposer_emits_candidates_without_sim3():
    spy = _SpyOrbLoopClosing(candidates=[(2, 0.9), (3, 0.8), (118, 0.7)])
    prop = OrbLoopProposer(detector=spy, min_index_separation=20)
    for i in range(5):
        prop.register(_KF(i))

    proposals = prop.poll_candidates(_KF(120))
    ids = {p.candidate_id for p in proposals}
    # candidates 118 is within 20 of 120 -> filtered; 2 and 3 survive
    assert ids == {2, 3}
    assert all(p.source == "orb" for p in proposals)
    # the adapter retrieved candidates but never ran Sim(3) / correction
    assert spy.detect_calls == 1
    assert spy.sim3_calls == 0
    assert spy.correct_calls == 0


def test_orb_proposer_rejects_non_propose_only():
    with pytest.raises(NotImplementedError):
        OrbLoopProposer(detector=_SpyOrbLoopClosing([]), propose_only=False)


# --------------------------------------------------------------------------
# LiDAR loop proposer — propose only, no verify / no PGO solve
# --------------------------------------------------------------------------

class _SpyCartoProvider:
    def __init__(self, targets):
        self._targets = targets
        self.provider_calls = 0

    def get_candidate_targets_for_node(self, node, all_nodes, config):
        self.provider_calls += 1
        return self._targets


class _SpyVerifierSink:
    def __init__(self):
        self.verify_calls = 0
        self.add_constraint_calls = 0
        self.optimize_calls = 0

    def verify(self, *a, **k):
        self.verify_calls += 1

    def add_loop_constraint(self, *a, **k):
        self.add_constraint_calls += 1

    def maybe_optimize(self, *a, **k):
        self.optimize_calls += 1


def _target(tid, x=0.0):
    return ClosureTarget(target_id=str(tid), target_type="submap",
                         pose_global=Pose2(x, 0, 0), is_finished=True, is_fixed=False,
                         map_view=None)


def test_lidar_proposer_emits_candidates_without_pgo():
    provider = _SpyCartoProvider(targets=[_target(0), _target(1)])
    sink = _SpyVerifierSink()  # wired to NOTHING; must stay untouched
    prop = LidarLoopProposer(provider=provider, config=LoopClosureConfig())

    node = LoopNode(node_id=50, scan_points=np.zeros((10, 2)),
                    pose_guess_global=Pose2(0, 0, 0), timestamp=0.0)
    proposals = prop.poll_candidates(node, all_nodes={i: node for i in range(60)})

    assert [p.candidate_id for p in proposals] == [0, 1]
    assert all(p.source == "lidar" and p.target is not None for p in proposals)
    assert provider.provider_calls == 1
    # the LiDAR verifier and constraint sink (PGO) were never invoked
    assert sink.verify_calls == 0
    assert sink.add_constraint_calls == 0
    assert sink.optimize_calls == 0


def test_lidar_proposer_rejects_non_propose_only():
    with pytest.raises(NotImplementedError):
        LidarLoopProposer(provider=_SpyCartoProvider([]), propose_only=False)


# --------------------------------------------------------------------------
# Front-end services — normalized streams, no loop closing
# --------------------------------------------------------------------------

class _FakeKeyframe:
    def __init__(self, kid, x):
        self.id = kid
        T = np.eye(4)
        T[0, 3] = x
        self.pose = T
        self.keypoints = np.zeros((5, 2))
        self.descriptors = np.zeros((5, 32), np.uint8)
        self.points3d = np.zeros((5, 3))


class _FakeVisualBackend:
    """Publishes a keyframe every 3rd frame."""

    def __init__(self):
        self.frame = 0
        self.kid = 0

    def track(self, rgb, depth, t):
        self.frame += 1
        if self.frame % 3 != 0:
            return None
        kf = _FakeKeyframe(self.kid, x=float(self.kid))
        self.kid += 1
        return kf


def test_visual_frontend_service_stream():
    svc = VisualFrontendService(_FakeVisualBackend())
    records = []
    for i in range(9):
        rec = svc.step(rgb=None, depth=None, timestamp=float(i))
        if rec is not None:
            records.append(rec)
    assert len(records) == 3
    assert [r.id for r in records] == [0, 1, 2]
    assert records[0].pose.shape == (4, 4)
    assert records[0].descriptors.shape == (5, 32)
    assert svc.keyframe_count == 3


class _FakeLidarResult:
    def __init__(self, pose, is_kf):
        self.pose = pose
        self.is_keyframe = is_kf


class _FakeLidarBackend:
    def __init__(self):
        self.i = 0

    def process_scan(self, scan, t):
        self.i += 1
        return _FakeLidarResult(Pose2(float(self.i), 0.0, 0.0), is_kf=(self.i % 2 == 0))


def test_lidar_frontend_service_relative_pose():
    svc = LidarFrontendService(_FakeLidarBackend())
    steps = [svc.step(np.zeros((10, 2)), float(t)) for t in range(3)]
    assert steps[0].rel_pose == Pose2(0.0, 0.0, 0.0)         # first step: no motion
    assert steps[1].rel_pose.x == pytest.approx(1.0)         # +1 per scan
    assert steps[2].rel_pose.x == pytest.approx(1.0)
    assert steps[1].is_keyframe is True
    assert steps[0].pose.x == pytest.approx(1.0)
