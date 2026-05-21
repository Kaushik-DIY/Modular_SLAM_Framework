from __future__ import annotations

from dataclasses import dataclass

from visual_slam.orbslam.slam.config_parameters import Parameters
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


@dataclass
class FakeResult:
    id: int
    score: float


class FakeDbowDatabase:
    def __init__(self):
        self.entries = []
        self.query_results: list[FakeResult] = []
        self.last_query_k = 0

    def addBowVector(self, bow):
        entry_id = len(self.entries)
        self.entries.append(dict(bow))
        return entry_id

    def size(self):
        return len(self.entries)

    def query(self, bow, max_results):
        self.last_query_k = int(max_results)
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


def _with_runtime_k(value: int):
    class _Restore:
        def __enter__(self_inner):
            self_inner.old = Parameters.kLoopDbowDetectorTopK
            Parameters.kLoopDbowDetectorTopK = int(value)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            Parameters.kLoopDbowDetectorTopK = self_inner.old

    return _Restore()


def test_classic_inverted_uses_inverted_file_not_dbow_raw_pool():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate = DummyKeyFrame(1, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3})
    database = _make_database_with_candidates(candidate)
    _set_raw_query_results(database, [])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="classic_inverted", return_diagnostics=True)

    assert result.retrieval_profile["candidate_source"] == "classic_inverted"
    assert _candidate_ids(result) == [1]


def test_classic_inverted_applies_common_words_before_score():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    sparse_high_score = DummyKeyFrame(1, {1: 10.0, 10: 1.0, 11: 1.0})
    structural = DummyKeyFrame(2, {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
    database = _make_database_with_candidates(sparse_high_score, structural)
    _set_raw_query_results(database, [(sparse_high_score, 0.99), (structural, 0.5)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="classic_inverted", return_diagnostics=True)

    assert _candidate_ids(result) == [2]


def test_classic_inverted_accumulates_covisibility_scores():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
    candidate_b = DummyKeyFrame(2, {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
    candidate_c = DummyKeyFrame(3, {1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b, candidate_c)
    _set_raw_query_results(database, [(candidate_c, 0.99), (candidate_a, 0.60), (candidate_b, 0.59)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="classic_inverted", return_diagnostics=True)

    assert 1 in _candidate_ids(result)
    assert 3 not in _candidate_ids(result)


def test_classic_inverted_retains_best_acc_score_candidates():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25})
    candidate_b = DummyKeyFrame(2, {1: 0.30, 2: 0.30, 3: 0.30, 4: 0.30})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b)
    _set_raw_query_results(database, [(candidate_a, 0.91), (candidate_b, 0.90)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="classic_inverted", return_diagnostics=True)

    assert _candidate_ids(result) == [2]


def test_dbow_detector_uses_bounded_top_k():
    query = DummyKeyFrame(100, {1: 1.0})
    first = DummyKeyFrame(1, {1: 1.0})
    second = DummyKeyFrame(2, {1: 1.0})
    database = _make_database_with_candidates(first, second)
    _set_raw_query_results(database, [(first, 0.95), (second, 0.94)])

    with _with_runtime_k(1):
        result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow_detector", return_diagnostics=True)

    assert _candidate_ids(result) == [1]
    assert result.trace_metadata["runtime_dbow_query_k"] == 1


def test_dbow_detector_does_not_apply_common_word_filter():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    sparse_high_score = DummyKeyFrame(1, {1: 10.0, 10: 1.0, 11: 1.0})
    structural = DummyKeyFrame(2, {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
    database = _make_database_with_candidates(sparse_high_score, structural)
    _set_raw_query_results(database, [(sparse_high_score, 0.99), (structural, 0.5)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow_detector", return_diagnostics=True)

    assert _candidate_ids(result) == [1, 2]


def test_dbow_detector_does_not_apply_accumulation_filter():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
    candidate_b = DummyKeyFrame(2, {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
    candidate_c = DummyKeyFrame(3, {1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b, candidate_c)
    _set_raw_query_results(database, [(candidate_c, 0.99), (candidate_a, 0.60), (candidate_b, 0.59)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow_detector", return_diagnostics=True)

    assert _candidate_ids(result) == [3, 1, 2]


def test_dbow_detector_filters_connected_temporal_and_min_score():
    query = DummyKeyFrame(100, {1: 1.0})
    connected = DummyKeyFrame(1, {1: 1.0})
    temporal_near = DummyKeyFrame(95, {1: 1.0})
    low_score = DummyKeyFrame(2, {1: 1.0})
    valid = DummyKeyFrame(3, {1: 1.0})
    query.set_connected(connected)
    database = _make_database_with_candidates(connected, temporal_near, low_score, valid)
    _set_raw_query_results(database, [(connected, 0.99), (temporal_near, 0.98), (low_score, 0.10), (valid, 0.80)])

    result = database.detect_loop_candidates(query, min_score=0.5, candidate_source="dbow_detector", return_diagnostics=True)

    assert _candidate_ids(result) == [3]


def test_dbow_detector_returns_direct_candidates_to_consistency():
    query = DummyKeyFrame(100, {1: 1.0})
    first = DummyKeyFrame(1, {1: 1.0})
    second = DummyKeyFrame(2, {1: 1.0})
    database = _make_database_with_candidates(first, second)
    _set_raw_query_results(database, [(first, 0.95), (second, 0.90)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow_detector", return_diagnostics=True)

    assert result.retrieval_profile["candidate_source"] == "dbow_detector"
    assert result.retrieval_profile["num_candidates_after_common_words"] == 2
    assert result.retrieval_profile["num_candidates_after_accumulation"] == 2
    assert result.candidate_scores == [0.95, 0.9]


def test_hybrid_dbow_scored_available_only_explicitly():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    raw_top = DummyKeyFrame(1, {1: 10.0, 10: 1.0})
    structural = DummyKeyFrame(2, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3})
    database = _make_database_with_candidates(raw_top, structural)
    _set_raw_query_results(database, [(raw_top, 0.99), (structural, 0.40)])

    auto_result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="auto", return_diagnostics=True)
    hybrid_result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="hybrid_dbow_scored", return_diagnostics=True)

    assert auto_result.retrieval_profile["candidate_source"] == "classic_inverted"
    assert hybrid_result.retrieval_profile["candidate_source"] == "hybrid_dbow_scored"


def test_auto_source_resolves_to_documented_primary_mode():
    query = DummyKeyFrame(100, {1: 1.0})
    candidate = DummyKeyFrame(1, {1: 1.0})
    database = _make_database_with_candidates(candidate)
    _set_raw_query_results(database, [(candidate, 0.95)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="auto", return_diagnostics=True)

    assert result.retrieval_profile["candidate_source"] == "classic_inverted"


def test_auto_does_not_select_hybrid_by_default():
    query = DummyKeyFrame(100, {1: 1.0})
    candidate = DummyKeyFrame(1, {1: 1.0})
    database = _make_database_with_candidates(candidate)
    _set_raw_query_results(database, [(candidate, 0.95)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="auto", return_diagnostics=True)

    assert result.retrieval_profile["candidate_source"] != "hybrid_dbow_scored"


def test_trace_raw_k_does_not_change_dbow_detector_decisions():
    query = DummyKeyFrame(100, {1: 1.0})
    first = DummyKeyFrame(1, {1: 1.0})
    second = DummyKeyFrame(2, {1: 1.0})
    third = DummyKeyFrame(3, {1: 1.0})
    database = _make_database_with_candidates(first, second, third)
    _set_raw_query_results(database, [(first, 0.95), (second, 0.94), (third, 0.93)])

    with _with_runtime_k(2):
        database.configure_loop_retrieval_trace(enabled=False, raw_k=0)
        without_trace = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow_detector", return_diagnostics=True)

        database.configure_loop_retrieval_trace(enabled=True, raw_k=100)
        with_trace = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow_detector", return_diagnostics=True)

    assert _candidate_ids(without_trace) == _candidate_ids(with_trace)
    assert with_trace.trace_metadata["trace_dbow_query_k"] == 100


def test_dbow_detector_runtime_k_honors_config():
    query = DummyKeyFrame(100, {1: 1.0})
    first = DummyKeyFrame(1, {1: 1.0})
    second = DummyKeyFrame(2, {1: 1.0})
    database = _make_database_with_candidates(first, second)
    _set_raw_query_results(database, [(first, 0.95), (second, 0.94)])

    with _with_runtime_k(1):
        result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow_detector", return_diagnostics=True)

    assert result.trace_metadata["runtime_dbow_query_k"] == 1
    assert database.dbow_database.last_query_k == 1
