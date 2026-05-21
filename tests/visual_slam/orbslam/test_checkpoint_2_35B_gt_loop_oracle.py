from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tools.analyze_gt_loop_recall import (
    KeyframeRecord,
    PoseSample,
    _pair_key,
    associate_keyframes_to_gt,
    build_summary_json,
    classify_gt_pairs,
    classify_pipeline_stage,
    generate_gt_pairs,
    load_loop_candidate_oracle,
    load_tum_groundtruth,
    rotation_angle_degrees,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _pose(timestamp: float, xyz: tuple[float, float, float], rotation: np.ndarray | None = None) -> PoseSample:
    return PoseSample(
        timestamp=timestamp,
        translation=np.asarray(xyz, dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64) if rotation is None else np.asarray(rotation, dtype=np.float64),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def test_load_tum_groundtruth_skips_comments(tmp_path: Path):
    path = tmp_path / "groundtruth.txt"
    path.write_text(
        "# comment\n"
        "# timestamp tx ty tz qx qy qz qw\n"
        "1.0 0 0 0 0 0 0 1\n"
        "2.0 1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    poses = load_tum_groundtruth(path)

    assert len(poses) == 2
    assert poses[0].timestamp == 1.0
    assert poses[1].translation.tolist() == [1.0, 0.0, 0.0]


def test_associate_keyframes_to_nearest_gt():
    keyframes = [
        KeyframeRecord(kf_id=0, timestamp=10.02, num_map_points=100),
        KeyframeRecord(kf_id=1, timestamp=10.11, num_map_points=90),
    ]
    gt = [_pose(10.0, (0.0, 0.0, 0.0)), _pose(10.1, (1.0, 0.0, 0.0))]

    rows = associate_keyframes_to_gt(keyframes, gt, max_dt=0.05)

    assert rows[0]["gt_available"] is True
    assert rows[0]["gt_timestamp"] == 10.0
    assert rows[1]["gt_timestamp"] == 10.1


def test_gt_pair_distance_and_rotation_computation():
    rot_b = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    angle = rotation_angle_degrees(np.eye(3, dtype=np.float64), rot_b)

    assert angle == 90.0


def test_gt_loop_like_classification():
    keyframes = [
        KeyframeRecord(kf_id=0, timestamp=0.0, num_map_points=100, estimated_pose=_pose(0.0, (0.0, 0.0, 0.0))),
        KeyframeRecord(kf_id=12, timestamp=12.0, num_map_points=90, estimated_pose=_pose(12.0, (0.3, 0.0, 0.0))),
    ]
    associations = [
        {
            "kf_id": 0,
            "kf_timestamp": 0.0,
            "gt_timestamp": 0.0,
            "dt_sec": 0.0,
            "gt_available": True,
            "gt_tx": 0.0,
            "gt_ty": 0.0,
            "gt_tz": 0.0,
            "gt_qx": 0.0,
            "gt_qy": 0.0,
            "gt_qz": 0.0,
            "gt_qw": 1.0,
        },
        {
            "kf_id": 12,
            "kf_timestamp": 12.0,
            "gt_timestamp": 12.0,
            "dt_sec": 0.0,
            "gt_available": True,
            "gt_tx": 0.5,
            "gt_ty": 0.0,
            "gt_tz": 0.0,
            "gt_qx": 0.0,
            "gt_qy": 0.0,
            "gt_qz": 0.0,
            "gt_qw": 1.0,
        },
    ]

    pairs = generate_gt_pairs(
        keyframes,
        associations,
        min_time_gap_sec=10.0,
        min_kf_gap=10,
        loop_trans_threshold_m=0.75,
        loop_rot_threshold_deg=45.0,
        near_loop_trans_threshold_m=1.5,
    )

    assert len(pairs) == 1
    assert pairs[0]["gt_loop_like"] is True
    assert pairs[0]["gt_near_loop"] is True
    assert pairs[0]["gt_translation_distance"] == 0.5


def test_pair_key_is_order_independent():
    assert _pair_key(7, 2) == "2-7"
    assert _pair_key(2, 7) == "2-7"


def test_actual_loop_candidate_join_by_unordered_pair(tmp_path: Path):
    oracle_path = tmp_path / "loop_candidate_oracle.csv"
    _write_csv(
        oracle_path,
        ["current_kf_id", "candidate_kf_id", "accepted", "candidate_rank", "final_matched_map_points"],
        [
            {"current_kf_id": 9, "candidate_kf_id": 3, "accepted": "False", "candidate_rank": 2, "final_matched_map_points": 5},
            {"current_kf_id": 3, "candidate_kf_id": 9, "accepted": "True", "candidate_rank": 1, "final_matched_map_points": 25},
        ],
    )

    loaded = load_loop_candidate_oracle(tmp_path)

    assert list(loaded) == ["3-9"]
    assert loaded["3-9"]["accepted"] == "True"


def test_pipeline_stage_classification_accepted():
    stage = classify_pipeline_stage({"accepted": "True", "rejection_reason": "", "rejection_stage": ""}, None, {"kf_i": 1, "kf_j": 9})

    assert stage == "ACCEPTED"


def test_pipeline_stage_classification_consistency_failure():
    stage = classify_pipeline_stage(
        {"accepted": "False", "rejection_reason": "rejected_by_consistency", "rejection_stage": "consistency"},
        None,
        {"kf_i": 1, "kf_j": 9},
    )

    assert stage == "FAILED_CONSISTENCY"


def test_pipeline_stage_classification_not_retrieved():
    stage = classify_pipeline_stage(None, {"kf_id": 9, "dbow3_candidates": [], "inverted_file_candidates": [], "chosen_candidates": []}, {"kf_i": 1, "kf_j": 9})

    assert stage == "NOT_RETRIEVED"


def test_gt_loop_recall_summary_counts():
    keyframes = [
        KeyframeRecord(kf_id=0, timestamp=0.0, num_map_points=100),
        KeyframeRecord(kf_id=15, timestamp=15.0, num_map_points=90),
    ]
    associations = [
        {"kf_id": 0, "gt_available": True},
        {"kf_id": 15, "gt_available": True},
    ]
    gt_pairs = [
        {"pair_key": "0-15", "gt_loop_like": True, "gt_near_loop": True},
    ]
    classified_rows = [
        {"pair_key": "0-15", "gt_loop_like": True, "gt_near_loop": True, "pipeline_stage": "FAILED_CONSISTENCY", "density_concern": False},
    ]

    summary = build_summary_json(
        keyframes=keyframes,
        gt_associations=associations,
        gt_pairs=gt_pairs,
        classified_rows=classified_rows,
    )

    assert summary["num_keyframes"] == 2
    assert summary["num_gt_associated_keyframes"] == 2
    assert summary["num_gt_loop_like_pairs"] == 1
    assert summary["gt_loop_like_stage_counts"] == {"FAILED_CONSISTENCY": 1}


def test_classify_gt_pairs_joins_oracle_row():
    gt_pairs = [
        {
            "pair_key": "2-8",
            "kf_i": 2,
            "kf_j": 8,
            "timestamp_i": 1.0,
            "timestamp_j": 12.0,
            "time_gap_sec": 11.0,
            "kf_id_gap": 6,
            "gt_available_i": True,
            "gt_available_j": True,
            "gt_translation_distance": 0.5,
            "gt_rotation_angle_deg": 10.0,
            "estimated_translation_distance": "",
            "estimated_rotation_angle_deg": "",
            "gt_loop_like": True,
            "gt_near_loop": True,
            "num_map_points_i": 80,
            "num_map_points_j": 70,
        }
    ]
    oracle_by_pair = {
        "2-8": {
            "current_kf_id": "8",
            "candidate_kf_id": "2",
            "accepted": "False",
            "rejection_stage": "geometry",
            "rejection_reason": "not enough SE3 RANSAC seed inliers",
            "candidate_source": "dbow3_scored",
        }
    }

    rows = classify_gt_pairs(gt_pairs, oracle_by_pair, {}, {})

    assert rows[0]["pipeline_stage"] == "FAILED_SEED_GEOMETRY"
    assert rows[0]["actual_candidate_seen"] is True
