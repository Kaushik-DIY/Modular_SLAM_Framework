from __future__ import annotations

from tools.analyze_gt_loop_raw_retrieval_trace import (
    build_group_level_false_negative_analysis,
    build_group_level_recall_summary,
)
from visual_slam.orbslam.slam.keyframe_database import KeyFrameDatabase


class DummyKeyFrame:
    def __init__(self, kid: int, bow: dict[int, float], *, timestamp: float | None = None):
        self.id = int(kid)
        self.kid = int(kid)
        self.timestamp = float(kid if timestamp is None else timestamp)
        self.g_des = dict(bow)
        self.f_des = None
        self.loop_query_id = None
        self.num_loop_words = 0
        self.loop_score = 0.0
        self.reloc_query_id = None
        self.num_reloc_words = 0
        self.reloc_score = 0.0
        self._connected: list[DummyKeyFrame] = []
        self._neighbors: list[DummyKeyFrame] = []
        self._bad = False

    def is_bad(self) -> bool:
        return self._bad

    def get_connected_keyframes(self):
        return list(self._connected)

    def get_best_covisible_keyframes(self, n: int):
        return list(self._neighbors[: int(n)])

    def set_connected(self, *keyframes: DummyKeyFrame) -> None:
        self._connected = list(keyframes)

    def set_neighbors(self, *keyframes: DummyKeyFrame) -> None:
        self._neighbors = list(keyframes)


class FakeVocabulary:
    available = True
    error = ""
    pydbow3 = None
    voc = None

    @staticmethod
    def to_native_bow(bow):
        return bow

    @staticmethod
    def score(bow_a, bow_b) -> float:
        shared = set(bow_a).intersection(bow_b)
        return float(sum(min(float(bow_a[word_id]), float(bow_b[word_id])) for word_id in shared))


def _make_database_with_candidates(*candidates: DummyKeyFrame) -> KeyFrameDatabase:
    database = KeyFrameDatabase(FakeVocabulary())
    database.dbow_database = None
    for candidate in candidates:
        database.add(candidate)
    return database


def test_group_representative_gt_equivalent_counts_as_group_recalled():
    rows = [
        {
            "pair_key": "10-100",
            "current_kf_id": 100,
            "candidate_kf_id": 10,
            "gt_loop_like": True,
            "passed_connected_filter": True,
            "passed_temporal_filter": True,
            "passed_accumulated_score_filter": True,
            "retained_candidate": False,
            "gt_translation_distance": 0.30,
            "gt_rotation_angle_deg": 5.0,
        },
        {
            "pair_key": "20-100",
            "current_kf_id": 100,
            "candidate_kf_id": 20,
            "gt_loop_like": True,
            "passed_connected_filter": True,
            "passed_temporal_filter": True,
            "passed_accumulated_score_filter": True,
            "retained_candidate": True,
            "gt_translation_distance": 0.25,
            "gt_rotation_angle_deg": 4.0,
        },
    ]
    accumulation = {"10-100": {"best_candidate_id_in_group": 20}}

    output = build_group_level_false_negative_analysis(rows, accumulation)
    row = next(item for item in output if item["pair_key"] == "10-100")

    assert row["classification"] == "NOT_RETAINED_BUT_GT_EQUIVALENT_REPRESENTATIVE"
    assert row["group_recalled"] is True


def test_group_representative_false_positive_counts_as_lost():
    rows = [
        {
            "pair_key": "10-100",
            "current_kf_id": 100,
            "candidate_kf_id": 10,
            "gt_loop_like": True,
            "passed_connected_filter": True,
            "passed_temporal_filter": True,
            "passed_accumulated_score_filter": True,
            "retained_candidate": False,
            "gt_translation_distance": 0.30,
            "gt_rotation_angle_deg": 5.0,
        }
    ]
    accumulation = {"10-100": {"best_candidate_id_in_group": 30}}

    output = build_group_level_false_negative_analysis(rows, accumulation)
    row = output[0]

    assert row["classification"] == "NOT_RETAINED_AND_LOST"
    assert row["group_recalled"] is False


def test_connected_gt_pairs_are_reported_separately():
    rows = [
        {
            "pair_key": "10-100",
            "current_kf_id": 100,
            "candidate_kf_id": 10,
            "gt_loop_like": True,
            "passed_connected_filter": False,
            "passed_temporal_filter": True,
            "passed_accumulated_score_filter": False,
            "retained_candidate": False,
        }
    ]

    group_rows = build_group_level_false_negative_analysis(rows, {})
    summary = build_group_level_recall_summary(group_rows)
    counts = {row["stage"]: row["count"] for row in summary}

    assert counts["GT_LOOP_LIKE_CONNECTED_LOCAL"] == 1
    assert counts["GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP"] == 0


def test_loop_recall_denominator_uses_eligible_pairs():
    rows = [
        {
            "pair_key": "10-100",
            "current_kf_id": 100,
            "candidate_kf_id": 10,
            "gt_loop_like": True,
            "passed_connected_filter": False,
            "passed_temporal_filter": True,
            "passed_accumulated_score_filter": False,
            "retained_candidate": False,
        },
        {
            "pair_key": "20-100",
            "current_kf_id": 100,
            "candidate_kf_id": 20,
            "gt_loop_like": True,
            "passed_connected_filter": True,
            "passed_temporal_filter": True,
            "passed_accumulated_score_filter": True,
            "retained_candidate": True,
        },
    ]

    group_rows = build_group_level_false_negative_analysis(rows, {})
    summary = build_group_level_recall_summary(group_rows)
    by_stage = {row["stage"]: row for row in summary}

    assert by_stage["GT_LOOP_LIKE_TOTAL"]["count"] == 2
    assert by_stage["GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP"]["count"] == 1
    assert by_stage["GROUP_RECALLED_EXACT"]["percent_of_eligible_loop_pairs"] == 100.0


def test_accumulation_uses_top10_covisible_neighbors():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate = DummyKeyFrame(1, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3})
    neighbors = [
        DummyKeyFrame(10 + idx, {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1})
        for idx in range(11)
    ]
    candidate.set_neighbors(*neighbors)
    database = _make_database_with_candidates(candidate, *neighbors)
    database.configure_loop_retrieval_trace(enabled=True, raw_k=0)

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="classic_inverted", return_diagnostics=True)
    row = next(item for item in result.trace_rows["accumulation"] if int(item["candidate_kf_id"]) == 1)

    assert int(row["candidate_group_size"]) == 11


def test_accumulation_selects_best_scoring_representative():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25})
    candidate_b = DummyKeyFrame(2, {1: 0.30, 2: 0.30, 3: 0.30, 4: 0.30})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b)
    database.configure_loop_retrieval_trace(enabled=True, raw_k=0)

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="classic_inverted", return_diagnostics=True)
    row = next(item for item in result.trace_rows["accumulation"] if int(item["candidate_kf_id"]) == 1)

    assert int(row["best_candidate_id_in_group"]) == 2


def test_accumulation_deduplicates_representatives():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25})
    candidate_b = DummyKeyFrame(2, {1: 0.30, 2: 0.30, 3: 0.30, 4: 0.30})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b)

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="classic_inverted", return_diagnostics=True)

    assert [int(candidate.kid) for candidate in result.candidates] == [2]
