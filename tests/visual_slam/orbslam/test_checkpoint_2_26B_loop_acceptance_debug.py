from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.build_tum_reference_cloud import write_ascii_ply
from tools.plot_fr1_room_evaluation import apply_alignment, generate_plots
from tools.probe_loop_candidates_fr1_room import OraclePair, find_oracle_pairs, probe_pairs
from tools.run_fr1_room_full_evaluation import LOOP_DEBUG_COLUMNS, write_csv
from tests.visual_slam.orbslam.test_checkpoint_2_21_loop_closing import (
    build_loop_scene,
    make_base_points,
)
from tests.visual_slam.orbslam.test_checkpoint_2_26A_fr1_room_evaluation_tools import (
    _write_minimal_run,
    _write_tum,
)
from visual_slam.orbslam.slam import KeyFrameDatabase, load_default_vocabulary
from visual_slam.orbslam.slam.bow_matcher import BoWGuidedMatcher
from visual_slam.orbslam.slam.loop_closing import LoopGeometryChecker, LoopGroupConsistencyChecker


class FakeCovisibleKeyFrame:
    def __init__(self, kid, connected):
        self.kid = kid
        self.id = kid
        self._connected = list(connected)

    def is_bad(self):
        return False

    def get_connected_keyframes(self):
        return list(self._connected)


def test_loop_debug_csv_has_required_columns(tmp_path):
    path = tmp_path / "loop_debug_candidates.csv"
    write_csv(path, [], LOOP_DEBUG_COLUMNS)

    assert path.read_text().splitlines()[0].split(",") == LOOP_DEBUG_COLUMNS


def test_candidate_pair_report_contains_required_fields():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf = build_loop_scene(n=80)
    database.add(loop_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert checker.check_candidates(current_kf, [loop_kf])
    report = checker.last_candidate_reports[loop_kf.kid]

    required = {
        "current_kf_id",
        "candidate_kf_id",
        "bow_match_pairs",
        "descriptor_distances",
        "bow_matches_with_valid_mappoints",
        "geometry_ransac_inliers",
        "guided_projection_matches",
        "final_inliers",
        "rejection_reason",
    }
    assert required.issubset(report)


def test_consistency_group_overlap_accepts_covisible_candidate():
    _, _, loop_kf, current_kf = build_loop_scene(n=30)
    neighbor = FakeCovisibleKeyFrame(99, [loop_kf])
    loop_kf.add_connection(neighbor, 20)
    checker = LoopGroupConsistencyChecker(consistency_threshold=1)

    assert not checker.check_candidates(current_kf, [loop_kf])
    assert checker.check_candidates(current_kf, [neighbor])


def test_consistency_exact_candidate_id_not_required():
    _, _, loop_kf, current_kf = build_loop_scene(n=30)
    neighbor = FakeCovisibleKeyFrame(98, [loop_kf])
    loop_kf.add_connection(neighbor, 20)
    checker = LoopGroupConsistencyChecker(consistency_threshold=1)

    checker.check_candidates(current_kf, [loop_kf])
    checker.check_candidates(current_kf, [neighbor])

    assert checker.enough_consistent_candidates == [neighbor]


def test_bow_loop_matcher_reports_filter_counts():
    descriptors = np.zeros((6, 32), dtype=np.uint8)
    frame1 = SimpleNamespace(des=descriptors, feature_vector={1: list(range(6))}, f_des={1: list(range(6))}, angles=np.zeros(6))
    frame2 = SimpleNamespace(des=descriptors.copy(), feature_vector={1: list(range(6))}, f_des={1: list(range(6))}, angles=np.zeros(6))

    result = BoWGuidedMatcher().match(frame1, frame2, max_descriptor_distance=0)

    assert result.available
    assert result.diagnostics.shared_words == 1
    assert result.diagnostics.matches_after_ratio >= len(result.idxs1)
    assert result.diagnostics.matches_after_orientation == len(result.idxs1)


def test_rgbd_se3_loop_verifier_accepts_synthetic_true_loop():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf = build_loop_scene(n=80)
    database.add(loop_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert checker.check_candidates(current_kf, [loop_kf])
    assert checker.num_last_inliers >= checker.min_matches


def test_rgbd_se3_loop_verifier_rejects_false_loop():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf = build_loop_scene(n=80)
    current_kf.des = np.random.default_rng(42).integers(0, 256, size=current_kf.des.shape, dtype=np.uint8)
    current_kf.g_des = None
    current_kf.f_des = None
    current_kf.feature_vector = None
    database.add(loop_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert not checker.check_candidates(current_kf, [loop_kf])


def test_guided_projection_refinement_increases_or_preserves_matches():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf = build_loop_scene(n=80)
    database.add(loop_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert checker.check_candidates(current_kf, [loop_kf])
    report = checker.last_candidate_reports[loop_kf.kid]
    assert report["final_inliers"] >= report["geometry_ransac_inliers"]


def test_oracle_loop_probe_outputs_pair_reports(tmp_path, monkeypatch):
    import tools.probe_loop_candidates_fr1_room as probe

    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf = build_loop_scene(n=60)
    database.add(loop_kf)
    slam = SimpleNamespace(keyframe_database=database)
    Twc = np.eye(4, dtype=np.float64)

    monkeypatch.setattr(probe, "build_slam_keyframes", lambda dataset, backend, max_frames=0: (slam, [loop_kf, current_kf]))
    monkeypatch.setattr(
        probe,
        "read_tum_groundtruth",
        lambda path: [
            SimpleNamespace(timestamp=0.0, Twc=Twc.copy()),
            SimpleNamespace(timestamp=100.0, Twc=Twc.copy()),
        ],
    )

    summary = probe_pairs(tmp_path, tmp_path / "oracle", "pyslam_orb2", max_pairs=1)

    assert summary["oracle_pairs"] == 1
    assert (tmp_path / "oracle" / "oracle_pairs.csv").exists()
    assert list((tmp_path / "oracle" / "pair_reports").glob("*.json"))


def test_map_alignment_transform_is_applied_to_map_points():
    points = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = np.array([2.0, 3.0, 4.0], dtype=np.float64)

    aligned = apply_alignment(points, (R, t))

    np.testing.assert_allclose(aligned, [[2.0, 4.0, 4.0]])


def test_aligned_map_side_by_side_output_exists(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_tum(dataset / "groundtruth.txt", [1.0, 2.0, 3.0], [(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    root = tmp_path / "eval"
    for run_name in ["run_A_no_loop", "run_B_loop_only", "run_C_loop_plus_gba"]:
        _write_minimal_run(root, run_name)
    (root / "reference_map").mkdir(parents=True)
    write_ascii_ply(root / "reference_map" / "reference_cloud_gt.ply", np.array([[0, 0, 1], [1, 1, 1]], dtype=float))

    generate_plots(root, dataset=dataset)

    assert (root / "comparison" / "map_side_by_side_xy_aligned.png").exists()
    assert (root / "comparison" / "estimated_sparse_map_xy_aligned.png").exists()


def test_find_oracle_pairs_uses_gt_distance_and_time_separation():
    Twc0 = np.eye(4, dtype=np.float64)
    Twc1 = np.eye(4, dtype=np.float64)
    keyframes = [
        SimpleNamespace(kid=0, id=0, timestamp=0.0),
        SimpleNamespace(kid=1, id=1, timestamp=25.0),
    ]
    groundtruth = [
        SimpleNamespace(timestamp=0.0, Twc=Twc0),
        SimpleNamespace(timestamp=25.0, Twc=Twc1),
    ]

    pairs = find_oracle_pairs(keyframes, groundtruth, max_pairs=1)

    assert pairs == [OraclePair(1, 0, 25.0, 0.0, 0.0, 0.0)]
