from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.evaluate_tum_trajectory import (
    associate_poses,
    evaluate_trajectories,
    read_tum_poses,
)


def write_tum(path: Path, stamps, positions) -> Path:
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for stamp, pos in zip(stamps, positions):
            f.write(
                f"{stamp:.6f} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                "0.000000000 0.000000000 0.000000000 1.000000000\n"
            )
    return path


def base_positions():
    return [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    ]


def test_identical_trajectories_have_near_zero_ate_and_rpe(tmp_path):
    stamps = [1.0, 2.0, 3.0, 4.0]
    gt = write_tum(tmp_path / "groundtruth.txt", stamps, base_positions())
    est = write_tum(tmp_path / "trajectory.txt", stamps, base_positions())

    metrics = evaluate_trajectories(gt, est, tmp_path / "eval")

    assert metrics["num_associations"] == 4
    assert metrics["ate_rmse_se3_m"] < 1e-9
    assert metrics["ate_rmse_sim3_m"] < 1e-9
    assert metrics["rpe_trans_rmse_m"] < 1e-9
    assert metrics["rpe_rot_rmse_deg"] < 1e-9


def test_translated_trajectory_is_handled_by_alignment(tmp_path):
    stamps = [1.0, 2.0, 3.0, 4.0]
    gt_positions = base_positions()
    est_positions = [(x + 10.0, y - 3.0, z + 2.0) for x, y, z in gt_positions]
    gt = write_tum(tmp_path / "groundtruth.txt", stamps, gt_positions)
    est = write_tum(tmp_path / "trajectory.txt", stamps, est_positions)

    metrics = evaluate_trajectories(gt, est, tmp_path / "eval")

    assert metrics["ate_rmse_se3_m"] < 1e-9
    assert metrics["ate_rmse_sim3_m"] < 1e-9


def test_timestamp_association_uses_nearest_pose_with_threshold(tmp_path):
    gt = write_tum(tmp_path / "groundtruth.txt", [1.0, 2.0, 3.0], base_positions()[:3])
    est = write_tum(tmp_path / "trajectory.txt", [1.01, 2.018, 3.05], base_positions()[:3])

    associations = associate_poses(read_tum_poses(gt), read_tum_poses(est), max_time_diff=0.02)

    assert len(associations) == 2
    assert associations[0].time_diff == pytest.approx(0.01)
    assert associations[1].time_diff == pytest.approx(0.018)


def test_missing_and_empty_files_give_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_tum_poses(tmp_path / "missing.txt")

    empty = tmp_path / "empty.txt"
    empty.write_text("# no poses\n")
    with pytest.raises(ValueError, match="No valid TUM poses"):
        read_tum_poses(empty)


def test_output_json_markdown_and_association_csv_are_generated(tmp_path):
    stamps = [1.0, 2.0, 3.0, 4.0]
    gt = write_tum(tmp_path / "groundtruth.txt", stamps, base_positions())
    est = write_tum(tmp_path / "trajectory.txt", stamps, base_positions())
    output = tmp_path / "eval"

    metrics = evaluate_trajectories(gt, est, output)

    json_path = output / "trajectory_metrics.json"
    md_path = output / "trajectory_metrics.md"
    csv_path = output / "associated_poses.csv"

    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()
    assert json.loads(json_path.read_text())["num_associations"] == metrics["num_associations"]
    assert "ATE RMSE SE(3)" in md_path.read_text()
