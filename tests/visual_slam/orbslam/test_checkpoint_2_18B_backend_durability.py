from __future__ import annotations

import csv
from pathlib import Path

from tools.run_orb_backend_durability import (
    DurabilityResult,
    create_arg_parser,
    evaluate_result_trajectory,
    parse_smoke_stdout,
    write_metrics_csv,
    write_summary_markdown,
)


def make_result(backend="opencv_orb", frame_count="100") -> DurabilityResult:
    return DurabilityResult(
        backend=backend,
        frame_count=frame_count,
        output_dir=Path("/tmp/run"),
        returncode=0,
        frames_attempted=100,
        tracking_ok_count=100,
        tracking_lost_count=0,
        errors=0,
        final_state="OK",
        final_keyframes=8,
        final_map_points=4000,
        avg_fps=0.12,
        trajectory_file="/tmp/traj.txt",
        frame_log_file="/tmp/frame_log.csv",
        command_log="/tmp/command.log",
        trajectory_eval_status="ok",
        ate_rmse_se3_m=0.1,
        ate_rmse_sim3_m=0.09,
        rpe_trans_rmse_m=0.02,
        rpe_rot_rmse_deg=1.5,
        num_associations=95,
    )


def test_tool_imports_and_parser_accepts_requested_arguments():
    parser = create_arg_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/data/tum",
            "--output",
            "/tmp/out",
            "--frame-counts",
            "100",
            "300",
            "full",
            "--backends",
            "opencv_orb",
            "pyslam_orb2",
        ]
    )

    assert args.frame_counts == ["100", "300", "full"]
    assert args.backends == ["opencv_orb", "pyslam_orb2"]


def test_smoke_stdout_summary_parser_extracts_metrics():
    summary = parse_smoke_stdout(
        """
frames_attempted:     100
tracking_ok_count:    99
tracking_lost_count:  1
errors:               0
final_state:          OK
final_keyframes:      8
final_map_points:     4010
avg_fps:              0.12
trajectory_file:      /tmp/traj.txt
frame_log_file:       /tmp/log.csv
"""
    )

    assert summary["frames_attempted"] == "100"
    assert summary["tracking_ok_count"] == "99"
    assert summary["trajectory_file"] == "/tmp/traj.txt"


def test_summary_generation_works_with_mocked_results(tmp_path):
    path = tmp_path / "summary.md"
    results = [make_result("opencv_orb"), make_result("pyslam_orb2")]

    write_summary_markdown(path, results, dataset=Path("/data/tum"), groundtruth=Path("/data/tum/groundtruth.txt"))

    text = path.read_text()
    assert "opencv_orb" in text
    assert "pyslam_orb2" in text
    assert "Backend Recommendation" in text


def test_csv_output_creation_works_with_mocked_results(tmp_path):
    path = tmp_path / "metrics.csv"
    write_metrics_csv(path, [make_result()])

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["backend"] == "opencv_orb"
    assert rows[0]["ok_ratio"] == "1.000000"
    assert rows[0]["ate_rmse_se3_m"] == "0.100000000"


def test_missing_groundtruth_is_handled_gracefully(tmp_path):
    result = make_result()
    result.trajectory_eval_status = "not_run"
    result.ate_rmse_se3_m = None

    updated = evaluate_result_trajectory(result, tmp_path / "missing_groundtruth.txt")

    assert updated.trajectory_eval_status == "groundtruth_missing"
    assert updated.ate_rmse_se3_m is None
