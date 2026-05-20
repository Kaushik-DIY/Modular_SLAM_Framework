from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from visual_slam.orbslam.run_rgbd_slam import (
    LOOP_CANDIDATE_ORACLE_COLUMNS,
    LOOP_KEYFRAME_DENSITY_COLUMNS,
    LOOP_RETRIEVAL_PROFILE_COLUMNS,
)
from visual_slam.orbslam.slam.keyframe_database import KeyFrameDatabase
from visual_slam.orbslam.slam.loop_closing import LoopClosing
from visual_slam.orbslam.slam.loop_detector import LoopDetectorOutput
from visual_slam.orbslam.slam.loop_oracle import TumLoopOracle


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


class FakeKeyFrameDatabase:
    available = True

    def __init__(self, existing_count: int):
        self._size = int(existing_count)
        self.added = []

    def size(self) -> int:
        return int(self._size)

    def add(self, keyframe) -> None:
        self.added.append(keyframe)
        self._size += 1

    @staticmethod
    def unavailable_reason() -> str:
        return ""


def _empty_output(keyframe, source="dbow3_scored") -> LoopDetectorOutput:
    return LoopDetectorOutput(
        keyframe=keyframe,
        candidate_keyframes=[],
        candidate_scores=[],
        min_score=0.0,
        candidate_source=source,
        candidate_details={},
        retrieval_profile={
            "num_db_keyframes_before_query": 0,
            "candidate_source": source,
            "num_raw_dbow_candidates": 0,
            "num_raw_inverted_candidates": 0,
            "num_candidates_after_temporal_filter": 0,
            "num_candidates_after_connected_filter": 0,
            "num_candidates_after_common_words": 0,
            "num_candidates_after_min_score": 0,
            "num_candidates_after_accumulation": 0,
            "num_candidates_after_consistency": 0,
            "top_candidate_id": -1,
            "top_candidate_score": 0.0,
            "top_candidate_acc_score": 0.0,
            "top_candidate_consistency": -1,
            "accepted_candidate_id": -1,
        },
        source_comparison={},
    )


def _make_fake_loop_closing(existing_count: int, detect_fn):
    database = FakeKeyFrameDatabase(existing_count)
    slam = SimpleNamespace(runtime_profiler=None, map=SimpleNamespace(keyframes_map={}))
    closing = LoopClosing(slam, keyframe_database=database)
    closing.loop_detector = SimpleNamespace(detect=lambda keyframe: detect_fn(database, keyframe))
    return closing, database


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


def test_loop_detector_queries_before_adding_current_keyframe():
    events = []
    current = DummyKeyFrame(99, {1: 1.0})

    def detect(database, keyframe):
        events.append(("detect", database.size()))
        return _empty_output(keyframe)

    closing, database = _make_fake_loop_closing(3, detect)
    original_add = database.add

    def recording_add(keyframe):
        events.append(("add", database.size()))
        original_add(keyframe)

    database.add = recording_add
    closing.process_keyframe(current)

    assert events[0] == ("detect", 3)
    assert events[1] == ("add", 3)


def test_current_keyframe_not_consuming_top_dbow_slot():
    current = DummyKeyFrame(42, {1: 1.0})

    def detect(database, keyframe):
        if database.size() > 2:
            return LoopDetectorOutput(
                keyframe=keyframe,
                candidate_keyframes=[keyframe],
                candidate_scores=[1.0],
                min_score=0.0,
                candidate_source="dbow3_scored",
                candidate_details={int(keyframe.kid): {"candidate_source": "dbow3_scored"}},
                retrieval_profile={},
                source_comparison={},
            )
        return _empty_output(keyframe)

    closing, _ = _make_fake_loop_closing(2, detect)
    closing.process_keyframe(current)

    assert closing.last_diagnostics.candidates == 0
    assert closing.last_diagnostics.loop_debug_records == []


def test_database_size_before_query_excludes_current_keyframe():
    sizes = []
    current = DummyKeyFrame(55, {1: 1.0})

    def detect(database, keyframe):
        sizes.append(database.size())
        output = _empty_output(keyframe)
        output.retrieval_profile["num_db_keyframes_before_query"] = database.size()
        return output

    closing, _ = _make_fake_loop_closing(5, detect)
    closing.process_keyframe(current)

    assert sizes == [5]
    assert closing.last_diagnostics.loop_retrieval_profile_rows[0]["num_db_keyframes_before_query"] == 5


def test_dbow3_scored_path_uses_common_word_filter():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    sparse = DummyKeyFrame(1, {1: 1.0, 10: 1.0, 11: 1.0, 12: 1.0})
    dense = DummyKeyFrame(2, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    database = _make_database_with_candidates(sparse, dense)
    _set_raw_query_results(database, [(sparse, 0.99), (dense, 0.5)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow3_scored", return_diagnostics=True)

    assert _candidate_ids(result) == [2]


def test_dbow3_scored_path_uses_min_score_filter():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    low = DummyKeyFrame(1, {1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10})
    high = DummyKeyFrame(2, {1: 0.30, 2: 0.30, 3: 0.30, 4: 0.30, 5: 0.30})
    database = _make_database_with_candidates(low, high)
    _set_raw_query_results(database, [(low, 0.99), (high, 0.98)])

    result = database.detect_loop_candidates(query, min_score=1.0, candidate_source="dbow3_scored", return_diagnostics=True)

    assert _candidate_ids(result) == [2]


def test_dbow3_scored_path_uses_covisibility_score_accumulation():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
    candidate_b = DummyKeyFrame(2, {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15})
    candidate_c = DummyKeyFrame(3, {1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b, candidate_c)
    _set_raw_query_results(database, [(candidate_c, 0.99), (candidate_a, 0.60), (candidate_b, 0.59)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow3_scored", return_diagnostics=True)

    assert 1 in _candidate_ids(result)
    assert 3 not in _candidate_ids(result)


def test_dbow3_scored_path_does_not_return_raw_topk_directly():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    raw_top = DummyKeyFrame(1, {1: 1.0, 10: 1.0})
    structural = DummyKeyFrame(2, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    database = _make_database_with_candidates(raw_top, structural)
    _set_raw_query_results(database, [(raw_top, 0.99), (structural, 0.40)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="dbow3_scored", return_diagnostics=True)

    assert result.retrieval_profile["candidate_source"] == "hybrid_dbow_scored"
    assert _candidate_ids(result) == [2]


def test_inverted_file_and_dbow3_scored_paths_share_candidate_scoring_logic():
    # Use distinct query KFs because the pyslam-aligned classic_inverted walk
    # mutates candidate state (loop_query_id/num_loop_words); re-querying the
    # same KF is intentionally non-idempotent.
    candidate_a = DummyKeyFrame(1, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3})
    candidate_b = DummyKeyFrame(2, {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
    candidate_a.set_neighbors(candidate_b)
    database = _make_database_with_candidates(candidate_a, candidate_b)
    _set_raw_query_results(database, [(candidate_a, 0.91), (candidate_b, 0.90)])

    query_a = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    query_b = DummyKeyFrame(101, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})

    dbow3_result = database.detect_loop_candidates(query_a, min_score=0.0, candidate_source="dbow3_scored", return_diagnostics=True)
    inverted_result = database.detect_loop_candidates(query_b, min_score=0.0, candidate_source="inverted_file", return_diagnostics=True)

    assert _candidate_ids(dbow3_result) == _candidate_ids(inverted_result)


def test_loop_candidate_source_compare_writes_both_sources():
    query = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    candidate_a = DummyKeyFrame(1, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3})
    candidate_b = DummyKeyFrame(2, {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
    database = _make_database_with_candidates(candidate_a, candidate_b)
    _set_raw_query_results(database, [(candidate_a, 0.91), (candidate_b, 0.90)])

    result = database.detect_loop_candidates(query, min_score=0.0, candidate_source="compare", return_diagnostics=True)

    assert "dbow3_candidates" in result.source_comparison
    assert "inverted_file_candidates" in result.source_comparison
    assert result.source_comparison["chosen_candidates"] == _candidate_ids(result)


def test_compare_mode_does_not_change_primary_loop_decision():
    # Use distinct query KFs because the pyslam-aligned classic_inverted walk
    # mutates candidate state (loop_query_id/num_loop_words); re-querying the
    # same KF is intentionally non-idempotent.
    candidate_a = DummyKeyFrame(1, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3})
    candidate_b = DummyKeyFrame(2, {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
    database = _make_database_with_candidates(candidate_a, candidate_b)
    _set_raw_query_results(database, [(candidate_a, 0.91), (candidate_b, 0.90)])

    query_a = DummyKeyFrame(100, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    query_b = DummyKeyFrame(101, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})

    auto_result = database.detect_loop_candidates(query_a, min_score=0.0, candidate_source="auto", return_diagnostics=True)
    compare_result = database.detect_loop_candidates(query_b, min_score=0.0, candidate_source="compare", return_diagnostics=True)

    assert _candidate_ids(auto_result) == _candidate_ids(compare_result)


def test_loop_oracle_loads_tum_groundtruth(tmp_path: Path):
    gt = tmp_path / "groundtruth.txt"
    gt.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0 1\n"
    )

    oracle = TumLoopOracle.from_tum_groundtruth(gt, max_time_diff=0.05)

    assert oracle.has_data()
    assert len(oracle.poses) == 2


def test_loop_oracle_associates_keyframe_timestamps_to_gt(tmp_path: Path):
    gt = tmp_path / "groundtruth.txt"
    gt.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "10.0 0 0 0 0 0 0 1\n"
        "10.1 1 0 0 0 0 0 1\n"
    )
    oracle = TumLoopOracle.from_tum_groundtruth(gt, max_time_diff=0.05)

    pose = oracle.find_pose(10.02)

    assert pose is not None
    assert pose.timestamp == 10.0


def test_loop_oracle_marks_gt_loop_like_pair(tmp_path: Path):
    gt = tmp_path / "groundtruth.txt"
    gt.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "0.0 0 0 0 0 0 0 1\n"
        "5.0 0.5 0 0 0 0 0 1\n"
    )
    oracle = TumLoopOracle.from_tum_groundtruth(gt, max_time_diff=0.05)

    diagnostics = oracle.describe_pair(0.0, 5.0)

    assert diagnostics.gt_available
    assert diagnostics.gt_loop_like
    assert diagnostics.gt_near_loop


def test_loop_candidate_oracle_csv_has_required_columns():
    required = {
        "event_id",
        "current_kf_id",
        "candidate_kf_id",
        "bow_score",
        "gt_loop_like",
        "accepted",
    }
    assert required.issubset(set(LOOP_CANDIDATE_ORACLE_COLUMNS))


def test_loop_retrieval_profile_csv_has_required_columns():
    required = {
        "kf_id",
        "num_db_keyframes_before_query",
        "candidate_source",
        "num_candidates_after_accumulation",
        "accepted_candidate_id",
    }
    assert required.issubset(set(LOOP_RETRIEVAL_PROFILE_COLUMNS))


def test_loop_keyframe_density_profile_csv_has_required_columns():
    required = {
        "current_kf_id",
        "candidate_kf_id",
        "shared_bow_words",
        "final_matched_map_points",
        "rejection_reason",
    }
    assert required.issubset(set(LOOP_KEYFRAME_DENSITY_COLUMNS))
