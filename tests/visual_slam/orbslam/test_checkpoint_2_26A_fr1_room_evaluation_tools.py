from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tools.build_tum_reference_cloud import (
    build_reference_cloud,
    quaternion_to_rotation,
    read_tum_associations,
    read_tum_groundtruth,
    write_ascii_ply,
)
from tools.export_orbslam_map import collect_exportable_points, export_orbslam_map
from tools.plot_fr1_room_evaluation import EXPECTED_FIGURES, generate_plots
from tools.run_fr1_room_full_evaluation import (
    GLOBAL_BA_EVENT_COLUMNS,
    LOOP_EVENT_COLUMNS,
    create_empty_event_logs,
    create_output_structure,
    has_real_loop_triggered_gba,
)


def _write_tum(path: Path, stamps, positions) -> Path:
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for stamp, pos in zip(stamps, positions):
            f.write(
                f"{stamp:.6f} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                "0.000000000 0.000000000 0.000000000 1.000000000\n"
            )
    return path


def test_reference_cloud_builder_imports():
    assert callable(build_reference_cloud)


def test_quaternion_to_rotation_conversion_is_valid():
    rotation = quaternion_to_rotation(np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_tum_groundtruth_parser_reads_valid_poses(tmp_path):
    gt = _write_tum(tmp_path / "groundtruth.txt", [1.0], [(1.0, 2.0, 3.0)])
    poses = read_tum_groundtruth(gt)
    assert len(poses) == 1
    assert poses[0].timestamp == 1.0
    assert np.allclose(poses[0].Twc[:3, 3], [1.0, 2.0, 3.0])


def test_tum_association_parser_reads_rgb_depth_pairs(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    (tmp_path / "associations.txt").write_text("1.0 rgb/1.png 1.0 depth/1.png\n")
    pairs = read_tum_associations(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].rgb_path.name == "1.png"
    assert pairs[0].depth_path.parent.name == "depth"


def test_ply_writer_creates_valid_header(tmp_path):
    ply = tmp_path / "cloud.ply"
    write_ascii_ply(ply, np.array([[1.0, 2.0, 3.0]]), np.array([[10, 20, 30]], dtype=np.uint8))
    text = ply.read_text()
    assert text.startswith("ply\nformat ascii 1.0\n")
    assert "element vertex 1" in text
    assert "property uchar red" in text
    assert "end_header" in text


class DummyPoint:
    def __init__(self, point_id, position, *, bad=False, replacement=None):
        self.id = point_id
        self._position = np.asarray(position, dtype=np.float64)
        self._bad = bad
        self.replacement = replacement
        self.color = np.array([1, 2, 3], dtype=np.uint8)

    def get_position(self):
        return self._position

    def is_bad(self):
        return self._bad


class DummyKeyFrame:
    def __init__(self, kid, x):
        self.kid = kid
        self.id = kid + 10
        self.img_id = kid + 100
        self.timestamp = float(kid)
        self.points = []
        self.connected_keyframes_weights = {}
        self.loop_edges = set()
        self.parent = None
        self._Tcw = np.eye(4)
        self._Tcw[0, 3] = -float(x)

    def is_bad(self):
        return False

    def Tcw(self):
        return self._Tcw

    def Twc(self):
        return np.linalg.inv(self._Tcw)

    def get_parent(self):
        return self.parent

    def get_loop_edges(self):
        return self.loop_edges


class DummyMap:
    def __init__(self, points, keyframes):
        self._points = points
        self._keyframes = keyframes

    def get_points(self):
        return list(self._points)

    def get_keyframes(self):
        return list(self._keyframes)


def test_map_export_skips_bad_replaced_and_nonfinite_points():
    replacement = DummyPoint(99, [0.0, 0.0, 1.0])
    points = [
        DummyPoint(1, [1.0, 2.0, 3.0]),
        DummyPoint(2, [np.nan, 0.0, 0.0]),
        DummyPoint(3, [4.0, 5.0, 6.0], bad=True),
        DummyPoint(4, [7.0, 8.0, 9.0], replacement=replacement),
    ]
    exported, colors, ids = collect_exportable_points(DummyMap(points, []))
    assert ids == [1]
    assert exported.shape == (1, 3)
    assert colors.shape == (1, 3)


def test_keyframe_json_export_contains_required_fields(tmp_path):
    kf0 = DummyKeyFrame(0, 0.0)
    kf1 = DummyKeyFrame(1, 1.0)
    kf1.parent = kf0
    kf0.connected_keyframes_weights[kf1] = 12
    kf0.loop_edges.add(kf1)
    summary = export_orbslam_map(DummyMap([DummyPoint(1, [0.0, 0.0, 1.0])], [kf0, kf1]), tmp_path)
    assert Path(summary["map_points_ply"]).exists()
    keyframes = json.loads((tmp_path / "keyframes.json").read_text())
    assert {"kid", "frame_id", "timestamp", "position", "Tcw", "Twc"}.issubset(keyframes[0])
    graph = json.loads((tmp_path / "keyframe_graph.json").read_text())
    assert graph["nodes"] == [0, 1]
    assert graph["loop_edges"]


def test_loop_events_and_global_ba_events_contain_required_columns(tmp_path):
    create_empty_event_logs(tmp_path)
    with open(tmp_path / "loop_events.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames == LOOP_EVENT_COLUMNS
    with open(tmp_path / "global_ba_events.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames == GLOBAL_BA_EVENT_COLUMNS


def _write_minimal_run(root: Path, run_name: str):
    run = root / run_name
    run.mkdir(parents=True)
    _write_tum(run / "trajectory_rgbd_dataset_freiburg1_room_smoke.txt", [1.0, 2.0, 3.0], [(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "frames_attempted": 3,
                "tracking_ok_count": 3,
                "tracking_lost_count": 0,
                "accepted_loops": 0,
                "global_ba_success": 0,
            }
        )
    )
    (run / "trajectory_eval").mkdir()
    (run / "trajectory_eval" / "trajectory_metrics.json").write_text(
        json.dumps({"ate_rmse_se3_m": 0.0, "rpe_trans_rmse_m": 0.0, "rpe_rot_rmse_deg": 0.0})
    )
    with open(run / "trajectory_eval" / "associated_poses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )
        for i, pos in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0)], start=1):
            writer.writerow([i, i, 0, *pos, *pos, *pos, *pos])
    write_ascii_ply(run / "map_points.ply", np.array([[0, 0, 1], [1, 1, 1]], dtype=float))
    (run / "keyframes.json").write_text(
        json.dumps(
            [
                {"kid": 0, "position": [0, 0, 0]},
                {"kid": 1, "position": [1, 0, 0]},
            ]
        )
    )
    (run / "keyframe_graph.json").write_text(
        json.dumps(
            {
                "nodes": [0, 1],
                "spanning_tree_edges": [{"source": 0, "target": 1}],
                "covisibility_edges": [],
                "loop_edges": [],
            }
        )
    )
    create_empty_event_logs(run)


def test_plot_script_generates_expected_output_filenames(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_tum(dataset / "groundtruth.txt", [1.0, 2.0, 3.0], [(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    root = tmp_path / "eval"
    for run_name in ["run_A_no_loop", "run_B_loop_only", "run_C_loop_plus_gba"]:
        _write_minimal_run(root, run_name)
    (root / "reference_map").mkdir(parents=True)
    write_ascii_ply(root / "reference_map" / "reference_cloud_gt.ply", np.array([[0, 0, 1], [1, 1, 1]], dtype=float))
    paths = generate_plots(root, dataset=dataset)
    assert [path.name for path in paths] == EXPECTED_FIGURES
    assert all(path.exists() for path in paths)


def test_evaluation_runner_creates_expected_folder_structure(tmp_path):
    paths = create_output_structure(tmp_path / "out")
    for key in ["run_A_no_loop", "run_B_loop_only", "run_C_loop_plus_gba", "reference_map", "comparison"]:
        assert paths[key].exists()


def test_metrics_summary_detects_missing_real_loop_trigger():
    assert not has_real_loop_triggered_gba({"accepted_loops": 0, "global_ba_started": 1})
    assert not has_real_loop_triggered_gba({"accepted_loops": 1, "global_ba_started": 0})
    assert has_real_loop_triggered_gba({"accepted_loops": 1, "global_ba_started": 1})
