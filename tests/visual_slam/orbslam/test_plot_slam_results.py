from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tools.plot_slam_results import generate_all, load_run_data


def _write_tum_pose_file(path: Path, stamps_and_xyz: list[tuple[float, tuple[float, float, float]]]) -> Path:
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for stamp, (x, y, z) in stamps_and_xyz:
            f.write(f"{stamp:.6f} {x:.9f} {y:.9f} {z:.9f} 0.0 0.0 0.0 1.0\n")
    return path


def _write_map_ply(path: Path) -> Path:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex 3\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        f.write("0.0 0.0 1.0 255 0 0\n")
        f.write("0.5 0.0 1.2 0 255 0\n")
        f.write("1.0 0.0 1.4 0 0 255\n")
    return path


def _write_frame_log(path: Path) -> Path:
    columns = [
        "i",
        "timestamp",
        "ok",
        "state",
        "keyframes",
        "points",
        "frames",
        "poses",
        "history",
        "last_tracked",
        "last_ba_mse",
        "lm_last_fused",
        "lm_last_triangulated",
        "loop_global_ba_started",
        "loop_global_ba_success",
        "loop_global_ba_reason",
        "loop_global_ba_edges",
        "loop_global_ba_inliers",
        "loop_global_ba_mse_after",
    ]
    rows = [
        {
            "i": 0,
            "timestamp": 1.0,
            "ok": 1,
            "state": "OK",
            "keyframes": 1,
            "points": 3,
            "frames": 1,
            "poses": 1,
            "history": 1,
            "last_tracked": 80,
            "last_ba_mse": 0.1,
            "lm_last_fused": 0,
            "lm_last_triangulated": 0,
            "loop_global_ba_started": 0,
            "loop_global_ba_success": 0,
            "loop_global_ba_reason": "",
            "loop_global_ba_edges": 0,
            "loop_global_ba_inliers": 0,
            "loop_global_ba_mse_after": "",
        },
        {
            "i": 1,
            "timestamp": 2.0,
            "ok": 1,
            "state": "OK",
            "keyframes": 2,
            "points": 3,
            "frames": 2,
            "poses": 2,
            "history": 2,
            "last_tracked": 75,
            "last_ba_mse": 0.2,
            "lm_last_fused": 1,
            "lm_last_triangulated": 1,
            "loop_global_ba_started": 1,
            "loop_global_ba_success": 1,
            "loop_global_ba_reason": "accepted",
            "loop_global_ba_edges": 4,
            "loop_global_ba_inliers": 10,
            "loop_global_ba_mse_after": 0.05,
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_associated_poses_csv(path: Path) -> Path:
    columns = [
        "timestamp_est",
        "timestamp_gt",
        "time_diff",
        "gt_tx",
        "gt_ty",
        "gt_tz",
        "est_tx",
        "est_ty",
        "est_tz",
        "est_se3_tx",
        "est_se3_ty",
        "est_se3_tz",
        "est_sim3_tx",
        "est_sim3_ty",
        "est_sim3_tz",
    ]
    rows = [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.02, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def _make_dataset(tmp_path: Path, name: str, *, with_gt: bool) -> Path:
    dataset = tmp_path / name
    (dataset / "rgb").mkdir(parents=True)
    (dataset / "depth").mkdir()
    rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
    depth = np.full((8, 8), 1000, dtype=np.uint16)
    cv2.imwrite(str(dataset / "rgb" / "1.0.png"), rgb)
    cv2.imwrite(str(dataset / "depth" / "1.0.png"), depth)
    cv2.imwrite(str(dataset / "rgb" / "2.0.png"), rgb)
    cv2.imwrite(str(dataset / "depth" / "2.0.png"), depth)
    (dataset / "associations.txt").write_text(
        "# rgb_timestamp rgb_file depth_timestamp depth_file\n"
        "1.0 rgb/1.0.png 1.0 depth/1.0.png\n"
        "2.0 rgb/2.0.png 2.0 depth/2.0.png\n"
    )
    (dataset / "camera.yaml").write_text(
        "\n".join(
            [
                "Camera.width: 640",
                "Camera.height: 480",
                "Camera.fx: 517.3",
                "Camera.fy: 516.5",
                "Camera.cx: 318.6",
                "Camera.cy: 255.3",
                "Camera.k1: 0.0",
                "Camera.k2: 0.0",
                "Camera.p1: 0.0",
                "Camera.p2: 0.0",
                "Camera.k3: 0.0",
                "Camera.fps: 30.0",
                "DepthMapFactor: 5000.0",
            ]
        )
        + "\n"
    )
    if with_gt:
        _write_tum_pose_file(dataset / "groundtruth.txt", [(1.0, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
    return dataset


def _make_run(tmp_path: Path, run_name: str, dataset: Path, dataset_type: str) -> Path:
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True)
    _write_tum_pose_file(run_dir / f"trajectory_{dataset.name}.txt", [(1.0, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))])
    _write_frame_log(run_dir / f"frame_log_{dataset.name}.csv")
    _write_map_ply(run_dir / "map_points.ply")
    (run_dir / "keyframes.json").write_text(
        json.dumps(
            [
                {"kid": 0, "position": [0.0, 0.0, 0.0], "timestamp": 1.0, "Twc": np.eye(4).tolist()},
                {"kid": 1, "position": [1.0, 0.0, 0.0], "timestamp": 2.0, "Twc": np.eye(4).tolist()},
            ]
        )
    )
    (run_dir / "keyframe_graph.json").write_text(
        json.dumps(
            {
                "nodes": [0, 1],
                "spanning_tree_edges": [{"source": 0, "target": 1}],
                "covisibility_edges": [],
                "loop_edges": [{"source": 0, "target": 1}],
            }
        )
    )
    (run_dir / "trajectory_eval").mkdir()
    _write_associated_poses_csv(run_dir / "trajectory_eval" / "associated_poses.csv")
    (run_dir / "trajectory_eval" / "trajectory_metrics.json").write_text(
        json.dumps(
            {
                "ate_rmse_se3_m": 0.01,
                "ate_mean_se3_m": 0.01,
                "ate_median_se3_m": 0.01,
                "ate_max_se3_m": 0.01,
                "rpe_trans_rmse_m": 0.02,
                "rpe_rot_rmse_deg": 0.0,
                "num_associations": 2,
            }
        )
    )
    (run_dir / "effective_run_config.json").write_text(
        json.dumps(
            {
                "dataset_path": str(dataset),
                "dataset_type": dataset_type,
                "camera": {
                    "fx": 517.3,
                    "fy": 516.5,
                    "cx": 318.6,
                    "cy": 255.3,
                    "depth_factor": 1.0 / 5000.0,
                    "depth_map_factor": 5000.0,
                },
            }
        )
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset.name,
                "dataset_type": dataset_type,
                "frames_attempted": 2,
                "tracking_ok_count": 2,
                "tracking_lost_count": 0,
                "final_state": "OK",
                "keyframes": 2,
                "map_points": 3,
                "trajectory_poses": 2,
                "avg_fps": 1.0,
            }
        )
    )
    return run_dir


def test_load_run_data_auto_resolves_dataset_and_tum_gt(tmp_path):
    dataset = _make_dataset(tmp_path, "rgbd_dataset_freiburg1_test", with_gt=True)
    run_dir = _make_run(tmp_path, "run_a", dataset, "tum_rgbd")
    data = load_run_data(run_dir, gt_path=None, dataset_path=None, label="run_a")
    assert data["dataset_path"] == dataset.resolve()
    assert data["gt_path"] == (dataset / "groundtruth.txt").resolve()
    assert data["has_gt"] is True


def test_generate_all_single_run_outputs_expected_files(tmp_path):
    dataset = _make_dataset(tmp_path, "rgbd_dataset_freiburg1_test", with_gt=True)
    run_dir = _make_run(tmp_path, "run_single", dataset, "tum_rgbd")
    out_dir = tmp_path / "plots_single"
    generate_all(run_dir, None, None, "single", None, None, None, "", out_dir)
    for name in [
        "trajectory_xy.png",
        "ate_rpe_over_time.png",
        "tracking_quality.png",
        "map_xy.png",
        "metrics_panel.png",
        "metrics_table.md",
        "sparse_vs_gt_map.png",
    ]:
        assert (out_dir / name).exists()


def test_generate_all_compare_outputs_expected_files(tmp_path):
    dataset_a = _make_dataset(tmp_path, "rgbd_dataset_freiburg1_test", with_gt=True)
    dataset_b = _make_dataset(tmp_path, "lab_rgbd_test", with_gt=False)
    run_a = _make_run(tmp_path, "run_a", dataset_a, "tum_rgbd")
    run_b = _make_run(tmp_path, "run_b", dataset_b, "lab_rgbd")
    out_dir = tmp_path / "plots_compare"
    generate_all(run_a, None, None, "tum", run_b, None, None, "lab", out_dir)
    for name in [
        "compare_trajectory.png",
        "compare_map.png",
        "compare_trajectory_map.png",
    ]:
        assert (out_dir / name).exists()
