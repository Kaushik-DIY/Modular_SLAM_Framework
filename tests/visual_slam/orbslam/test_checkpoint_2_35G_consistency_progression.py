from __future__ import annotations

from visual_slam.orbslam.run_rgbd_slam import LOOP_CONSISTENCY_PROGRESSION_COLUMNS
from visual_slam.orbslam.slam.loop_closing import LoopClosing, LoopGroupConsistencyChecker


class DummyKeyFrame:
    def __init__(self, kid: int):
        self.id = int(kid)
        self.kid = int(kid)
        self._connected: list[DummyKeyFrame] = []
        self._bad = False

    def is_bad(self) -> bool:
        return self._bad

    def get_connected_keyframes(self):
        return list(self._connected)

    def set_connected(self, *keyframes: DummyKeyFrame) -> None:
        self._connected = list(keyframes)


def test_consistency_group_created_for_new_candidate():
    current = DummyKeyFrame(100)
    candidate = DummyKeyFrame(1)
    checker = LoopGroupConsistencyChecker(consistency_threshold=3)

    got_consistent = checker.check_candidates(current, [candidate])

    assert got_consistent is False
    assert len(checker.consistent_groups) == 1
    assert checker.consistent_groups[0].consistency == 0


def test_consistency_increments_on_group_overlap():
    current = DummyKeyFrame(100)
    shared = DummyKeyFrame(50)
    first = DummyKeyFrame(1)
    second = DummyKeyFrame(2)
    first.set_connected(shared)
    second.set_connected(shared)
    checker = LoopGroupConsistencyChecker(consistency_threshold=3)

    checker.check_candidates(current, [first])
    checker.check_candidates(current, [second])

    debug = checker.last_candidate_debug[2]
    assert debug["consistency_score_before"] == 0
    assert debug["consistency_score_after"] == 1
    assert debug["consistency_overlap_count"] == 1


def test_consistency_passes_at_threshold():
    current = DummyKeyFrame(100)
    shared = DummyKeyFrame(50)
    first = DummyKeyFrame(1)
    second = DummyKeyFrame(2)
    first.set_connected(shared)
    second.set_connected(shared)
    checker = LoopGroupConsistencyChecker(consistency_threshold=1)

    checker.check_candidates(current, [first])
    got_consistent = checker.check_candidates(current, [second])

    assert got_consistent is True
    assert checker.enough_consistent_candidates == [second]


def test_consistency_groups_replaced_after_query():
    current = DummyKeyFrame(100)
    shared_a = DummyKeyFrame(50)
    shared_b = DummyKeyFrame(60)
    first = DummyKeyFrame(1)
    second = DummyKeyFrame(2)
    first.set_connected(shared_a)
    second.set_connected(shared_b)
    checker = LoopGroupConsistencyChecker(consistency_threshold=3)

    checker.check_candidates(current, [first])
    checker.check_candidates(current, [second])

    current_group_ids = sorted(int(kf.kid) for kf in checker.consistent_groups[0].keyframes)
    assert current_group_ids == [2, 60]


def test_consistency_trace_records_overlap_and_score():
    current = DummyKeyFrame(100)
    shared = DummyKeyFrame(50)
    first = DummyKeyFrame(1)
    second = DummyKeyFrame(2)
    first.set_connected(shared)
    second.set_connected(shared)
    checker = LoopGroupConsistencyChecker(consistency_threshold=2)

    checker.check_candidates(current, [first])
    checker.check_candidates(current, [second])
    debug = checker.last_candidate_debug[2]

    assert debug["candidate_group_kf_ids"] == [2, 50]
    assert debug["previous_consistency_group_ids"] == [[1, 50]]
    assert debug["consistency_required"] == 2
    assert debug["passed_consistency"] is False


def test_gt_loop_like_consistency_progression_is_logged():
    closing = LoopClosing.__new__(LoopClosing)
    rows = closing._build_loop_consistency_progression_rows(
        [
            {
                "current_kf_id": 100,
                "candidate_kf_id": 2,
                "candidate_group_kf_ids": [2, 50],
                "previous_consistency_group_ids": [[1, 50]],
                "consistency_overlap_count": 1,
                "consistency_score_before": 0,
                "consistency_score_after": 1,
                "consistency_required": 2,
                "passed_consistency": False,
                "gt_loop_like": True,
                "gt_translation_distance": 0.4,
                "gt_rotation_angle_deg": 6.0,
            }
        ]
    )

    assert set(LOOP_CONSISTENCY_PROGRESSION_COLUMNS).issubset(set(rows[0].keys()))
    assert rows[0]["gt_loop_like"] is True
    assert rows[0]["pair_key"] == "2-100"
