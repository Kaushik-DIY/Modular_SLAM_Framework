from __future__ import annotations

from dataclasses import dataclass

from tools.analyze_gt_loop_raw_retrieval_trace import (
    _pair_key,
    build_funnel,
    build_summary,
    dominant_failure_stage,
)
from visual_slam.orbslam.run_rgbd_slam import (
    LOOP_ACCUMULATION_TRACE_COLUMNS,
    LOOP_GT_POSITIVE_TRACE_COLUMNS,
    LOOP_INVERTED_WORD_TRACE_COLUMNS,
    LOOP_RAW_DBOW_TRACE_COLUMNS,
    LOOP_RETAINED_CANDIDATE_TRACE_COLUMNS,
    LOOP_SCORE_FILTER_TRACE_COLUMNS,
)
from visual_slam.orbslam.slam.keyframe_database import KeyFrameDatabase
from visual_slam.orbslam.slam.loop_closing import finalize_gt_positive_trace_row


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


@dataclass
class FakeResult:
    id: int
    score: float


class FakeDbowDatabase:
    def __init__(self):
        self.entries = []
        self.query_results: list[FakeResult] = []

    def addBowVector(self, bow):
        entry_id = len(self.entries)
        self.entries.append(dict(bow))
        return entry_id

    def size(self):
        return len(self.entries)

    def query(self, bow, max_results):
        return list(self.query_results[: int(max_results)])


def _make_database_with_candidates(*candidates: DummyKeyFrame) -> KeyFrameDatabase:
    database = KeyFrameDatabase(FakeVocabulary())
    database.dbow_database = FakeDbowDatabase()
    for candidate in candidates:
        database.add(candidate)
    return database


def _set_raw_query_results(database: KeyFrameDatabase, scored_candidates: list[tuple[DummyKeyFrame, float]]) -> None:
    database.dbow_database.query_results = [
        FakeResult(id=int(database._keyframe_to_entry[candidate]), score=float(score))
        for candidate, score in scored_candidates
    ]


def _candidate_ids(result) -> list[int]:
    return [int(getattr(candidate, "kid", getattr(candidate, "id", -1))) for candidate in result.candidates]


def _base_gt_row() -> dict:
    return {
        "raw_dbow_present": True,
        "passed_connected_filter": True,
        "passed_temporal_filter": True,
        "inverted_word_present": True,
        "passed_common_word_filter": True,
        "passed_min_score_filter": True,
        "passed_accumulated_score_filter": True,
        "retained_candidate": True,
        "passed_consistency": True,
        "accepted": False,
        "rejection_reason": "",
        "first_failed_stage": "UNKNOWN",
        "diagnostic_confidence": "limited",
    }


def test_raw_dbow_trace_has_required_columns():
    required = {"current_kf_id", "candidate_kf_id", "raw_rank", "raw_score", "raw_query_k", "raw_result_count"}
    assert required.issubset(set(LOOP_RAW_DBOW_TRACE_COLUMNS))


def test_inverted_word_trace_has_required_columns():
    required = {"current_kf_id", "candidate_kf_id", "shared_words", "passed_common_word_filter"}
    assert required.issubset(set(LOOP_INVERTED_WORD_TRACE_COLUMNS))


def test_score_filter_trace_has_required_columns():
    required = {"current_kf_id", "candidate_kf_id", "bow_score", "min_score", "passed_min_score_filter"}
    assert required.issubset(set(LOOP_SCORE_FILTER_TRACE_COLUMNS))


def test_accumulation_trace_has_required_columns():
    required = {"current_kf_id", "candidate_kf_id", "accumulated_score", "retained_candidate", "retained_rank"}
    assert required.issubset(set(LOOP_ACCUMULATION_TRACE_COLUMNS))


def test_retained_candidate_trace_has_required_columns():
    required = {"current_kf_id", "candidate_kf_id", "passed_consistency", "accepted"}
    assert required.issubset(set(LOOP_RETAINED_CANDIDATE_TRACE_COLUMNS))


def test_gt_positive_trace_has_required_columns():
    required = {"pair_key", "raw_dbow_present", "inverted_word_present", "first_failed_stage", "diagnostic_confidence"}
    assert required.issubset(set(LOOP_GT_POSITIVE_TRACE_COLUMNS))


def test_gt_positive_trace_classifies_missing_raw_dbow():
    row = _base_gt_row()
    row["raw_dbow_present"] = False

    finalize_gt_positive_trace_row(row)

    assert row["first_failed_stage"] == "MISSING_FROM_RAW_DBOW"


def test_gt_positive_trace_classifies_common_word_failure():
    row = _base_gt_row()
    row["passed_common_word_filter"] = False

    finalize_gt_positive_trace_row(row)

    assert row["first_failed_stage"] == "FAILED_COMMON_WORD_FILTER"


def test_gt_positive_trace_classifies_min_score_failure():
    row = _base_gt_row()
    row["passed_min_score_filter"] = False

    finalize_gt_positive_trace_row(row)

    assert row["first_failed_stage"] == "FAILED_MIN_SCORE_FILTER"


def test_gt_positive_trace_classifies_accumulation_failure():
    row = _base_gt_row()
    row["passed_accumulated_score_filter"] = False

    finalize_gt_positive_trace_row(row)

    assert row["first_failed_stage"] == "FAILED_ACCUMULATION_FILTER"


def test_gt_positive_trace_classifies_retained_candidate():
    row = _base_gt_row()
    row["passed_consistency"] = False

    finalize_gt_positive_trace_row(row)

    assert row["first_failed_stage"] == "FAILED_CONSISTENCY"


def test_gt_positive_trace_does_not_change_actual_candidate_list():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3})
    candidate_b = DummyKeyFrame(2, {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b)
    _set_raw_query_results(database, [(candidate_a, 0.91), (candidate_b, 0.90)])

    database.configure_loop_retrieval_trace(enabled=False, raw_k=0)
    without_trace = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow3", return_diagnostics=True)

    database.configure_loop_retrieval_trace(enabled=True, raw_k=100)
    with_trace = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow3", return_diagnostics=True)

    assert _candidate_ids(without_trace) == _candidate_ids(with_trace)


def test_analyzer_builds_recall_funnel():
    rows = [
        {
            "raw_dbow_present": True,
            "inverted_word_present": True,
            "passed_connected_filter": True,
            "passed_temporal_filter": True,
            "passed_common_word_filter": True,
            "passed_min_score_filter": True,
            "passed_accumulated_score_filter": True,
            "retained_candidate": True,
            "passed_consistency": True,
            "passed_geometry_if_available": True,
            "accepted": True,
            "first_failed_stage": "ACCEPTED",
        },
        {
            "raw_dbow_present": False,
            "inverted_word_present": False,
            "passed_connected_filter": False,
            "passed_temporal_filter": False,
            "passed_common_word_filter": False,
            "passed_min_score_filter": False,
            "passed_accumulated_score_filter": False,
            "retained_candidate": False,
            "passed_consistency": False,
            "passed_geometry_if_available": False,
            "accepted": False,
            "first_failed_stage": "MISSING_FROM_RAW_DBOW",
        },
    ]

    funnel = build_funnel(rows)
    counts = {row["stage"]: row["count"] for row in funnel}

    assert counts["GT_LOOP_LIKE_TOTAL"] == 2
    assert counts["RAW_DBOW_PRESENT"] == 1
    assert counts["ACCEPTED"] == 1


def test_analyzer_identifies_dominant_failure_stage():
    rows = [
        {"first_failed_stage": "FAILED_MIN_SCORE_FILTER", "accepted": False},
        {"first_failed_stage": "FAILED_MIN_SCORE_FILTER", "accepted": False},
        {"first_failed_stage": "FAILED_CONSISTENCY", "accepted": False},
    ]

    summary = build_summary(
        rows,
        build_funnel(rows),
        historical_gt_loop_like_count=len(rows),
        min_kf_gap=10,
    )

    assert dominant_failure_stage(rows) == "FAILED_MIN_SCORE_FILTER"
    assert summary["dominant_first_failure_stage"] == "FAILED_MIN_SCORE_FILTER"


def test_pair_key_order_independent():
    assert _pair_key(4, 39) == "4-39"
    assert _pair_key(39, 4) == "4-39"
