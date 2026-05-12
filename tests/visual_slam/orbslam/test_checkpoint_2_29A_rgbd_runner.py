from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_slam.orbslam.io import make_tum_rgbd_camera
from visual_slam.orbslam.io.rgbd_dataset import (
    DATASET_TYPE_LAB,
    DATASET_TYPE_TUM,
    detect_dataset_type,
    load_lab_camera_config,
    make_rgbd_camera,
)
from visual_slam.orbslam.run_rgbd_slam import (
    build_effective_run_config,
    build_run_summary,
    build_standardized_artifact_paths,
    build_standardized_output_stem,
    create_arg_parser,
    write_effective_run_config,
    write_run_summary,
)
from visual_slam.orbslam.run_tum_rgbd_smoke import run_tum_rgbd_smoke


LAB_CAMERA_YAML = """
dataset_name: lab_run_01
sensor_type: RGBD

image:
  width: 640
  height: 480
  fps: 30.0

camera:
  fx: 615.23
  fy: 614.87
  cx: 320.12
  cy: 241.09
  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]

depth:
  depth_map_factor: 1000.0
  depth_threshold: 40.0
  baseline_m: 0.075
""".strip()

LAB_CAMERA_YAML_FLAT = """
# Camera parameters exported from ROS camera_info
Camera.width: 640
Camera.height: 480
Camera.fx: 609.883300781
Camera.fy: 609.177246094
Camera.cx: 324.920776367
Camera.cy: 229.748153687
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.k3: 0.0
Camera.fps: 30.0
Camera.RGB: 1
DepthMapFactor: 1000.0
""".strip()


def _make_lab_dataset(root: Path, *, with_camera: bool = True) -> Path:
    dataset = root / "lab_run_01"
    (dataset / "rgb").mkdir(parents=True)
    (dataset / "depth").mkdir()
    (dataset / "rgb.txt").write_text("1.0 rgb/1.png\n")
    (dataset / "depth.txt").write_text("1.0 depth/1.png\n")
    (dataset / "associations.txt").write_text("1.0 rgb/1.png 1.0 depth/1.png\n")
    if with_camera:
        (dataset / "camera.yaml").write_text(LAB_CAMERA_YAML + "\n")
    return dataset


def test_tum_dataset_type_auto_detection_works(tmp_path):
    dataset = tmp_path / "rgbd_dataset_freiburg1_desk"
    dataset.mkdir()
    assert detect_dataset_type(dataset) == DATASET_TYPE_TUM


def test_lab_rgbd_auto_detection_works(tmp_path):
    dataset = _make_lab_dataset(tmp_path)
    assert detect_dataset_type(dataset) == DATASET_TYPE_LAB


def test_lab_rgbd_without_camera_yaml_raises_clear_error(tmp_path):
    dataset = _make_lab_dataset(tmp_path, with_camera=False)
    with pytest.raises(FileNotFoundError, match=r"camera\.yaml|--camera-config"):
        make_rgbd_camera(dataset, dataset_type=DATASET_TYPE_LAB)


def test_camera_yaml_depth_map_factor_1000_produces_depth_factor_0001(tmp_path):
    dataset = _make_lab_dataset(tmp_path)
    config = load_lab_camera_config(dataset / "camera.yaml")
    camera = make_rgbd_camera(dataset, dataset_type=DATASET_TYPE_LAB, camera_config=dataset / "camera.yaml")
    assert config["depth_map_factor"] == pytest.approx(1000.0)
    assert config["depth_factor"] == pytest.approx(0.001)
    assert camera.depth_factor == pytest.approx(0.001)


def test_flat_orbslam2_style_camera_yaml_is_supported(tmp_path):
    dataset = _make_lab_dataset(tmp_path)
    (dataset / "camera.yaml").write_text(LAB_CAMERA_YAML_FLAT + "\n")
    config = load_lab_camera_config(dataset / "camera.yaml")
    camera = make_rgbd_camera(dataset, dataset_type=DATASET_TYPE_LAB, camera_config=dataset / "camera.yaml")
    assert config["width"] == 640
    assert config["height"] == 480
    assert config["fx"] == pytest.approx(609.883300781)
    assert config["fy"] == pytest.approx(609.177246094)
    assert config["cx"] == pytest.approx(324.920776367)
    assert config["cy"] == pytest.approx(229.748153687)
    assert config["depth_factor"] == pytest.approx(0.001)
    assert config["baseline_source"] == "default_rgbd_virtual_baseline_0p08m"
    assert config["depth_threshold_source"] == "default_th_depth_40"
    assert camera.depth_factor == pytest.approx(0.001)


def test_tum_camera_creation_still_uses_existing_tum_logic():
    camera = make_rgbd_camera(Path("rgbd_dataset_freiburg2_desk"), dataset_type=DATASET_TYPE_TUM, camera_profile="auto")
    tum_camera = make_tum_rgbd_camera("rgbd_dataset_freiburg2_desk")
    assert camera.fx == pytest.approx(tum_camera.fx)
    assert camera.fy == pytest.approx(tum_camera.fy)
    assert camera.cx == pytest.approx(tum_camera.cx)
    assert camera.cy == pytest.approx(tum_camera.cy)
    assert camera.depth_factor == pytest.approx(tum_camera.depth_factor)


def test_effective_run_config_contains_camera_parameters_and_dataset_type(tmp_path):
    output_dir = tmp_path / "out"
    config = build_effective_run_config(
        dataset=tmp_path / "rgbd_dataset_freiburg1_desk",
        dataset_name="rgbd_dataset_freiburg1_desk",
        dataset_type=DATASET_TYPE_TUM,
        output_dir=output_dir,
        camera_profile="auto",
        camera_config=None,
        associations=None,
        camera_metadata={
            "camera_source": "tum_fr1_auto",
            "sensor_type": "RGBD",
            "width": 640,
            "height": 480,
            "fps": 30.0,
            "fx": 517.3,
            "fy": 516.5,
            "cx": 318.6,
            "cy": 255.3,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "depth_map_factor": 5000.0,
            "depth_factor": 0.0002,
            "depth_threshold": 3.2,
            "baseline_m": 0.08,
            "bf": 41.384,
        },
        feature_backend="pyslam_orb2",
        enable_loop_closing=True,
        enable_global_ba=True,
        global_ba_after_loop=True,
        global_ba_iterations=10,
        max_frames=30,
        start_index=0,
        print_every=1,
        loop_debug=False,
        stop_after_loop_events=0,
        stop_after_accepted_loops=0,
        dump_loop_candidate_reports=False,
        start_local_mapping_thread=False,
        lm_wait_timeout=0.5,
    )
    path = write_effective_run_config(output_dir, config)
    payload = json.loads(path.read_text())
    assert payload["dataset_type"] == DATASET_TYPE_TUM
    assert payload["camera"]["fx"] == pytest.approx(517.3)
    assert payload["camera"]["depth_map_factor"] == pytest.approx(5000.0)


def test_standardized_output_stem_contains_dataset_type_and_timestamp():
    stem = build_standardized_output_stem("lab_rgbd", "lab_rgbd_run_2", "20260510_221500")
    assert stem == "lab_rgbd__lab_rgbd_run_2__completed_20260510_221500"


def test_standardized_artifact_paths_use_completed_timestamp(tmp_path):
    paths = build_standardized_artifact_paths(tmp_path, "lab_rgbd", "lab_rgbd_run_2", "20260510_221500")
    assert paths["trajectory_file"].name == "trajectory__lab_rgbd__lab_rgbd_run_2__completed_20260510_221500.txt"
    assert paths["run_summary_json"].name == "run_summary__lab_rgbd__lab_rgbd_run_2__completed_20260510_221500.json"


def test_run_summary_contains_required_fields(tmp_path):
    output_dir = tmp_path / "out"
    summary = build_run_summary(
        dataset_name="rgbd_dataset_freiburg1_desk",
        dataset_type=DATASET_TYPE_TUM,
        frames_attempted=30,
        tracking_ok_count=30,
        tracking_lost_count=0,
        errors=0,
        final_state="OK",
        keyframes=4,
        map_points=250,
        trajectory_poses=30,
        elapsed_sec=2.5,
        avg_fps=12.0,
        feature_backend="pyslam_orb2",
        enable_loop_closing=False,
        enable_global_ba=False,
        global_ba_after_loop=False,
        loop_debug_events=0,
        accepted_loops=0,
        output_files={
            "trajectory_file": "trajectory.txt",
            "frame_log_file": "frame_log.csv",
            "map_points_ply": "map_points.ply",
            "keyframes_json": "keyframes.json",
            "keyframe_graph_json": "keyframe_graph.json",
            "effective_run_config_json": "effective_run_config.json",
            "loop_debug_file": None,
            "standardized_trajectory_file": "trajectory__tum_rgbd__rgbd_dataset_freiburg1_desk__completed_20260510_221500.txt",
            "standardized_frame_log_file": "frame_log__tum_rgbd__rgbd_dataset_freiburg1_desk__completed_20260510_221500.csv",
            "standardized_map_points_ply": "map_points__tum_rgbd__rgbd_dataset_freiburg1_desk__completed_20260510_221500.ply",
            "standardized_keyframes_json": "keyframes__tum_rgbd__rgbd_dataset_freiburg1_desk__completed_20260510_221500.json",
            "standardized_keyframe_graph_json": "keyframe_graph__tum_rgbd__rgbd_dataset_freiburg1_desk__completed_20260510_221500.json",
            "standardized_effective_run_config_json": "effective_run_config__tum_rgbd__rgbd_dataset_freiburg1_desk__completed_20260510_221500.json",
            "standardized_loop_debug_file": None,
        },
        completed_timestamp="20260510_221500",
        standardized_output_stem="tum_rgbd__rgbd_dataset_freiburg1_desk__completed_20260510_221500",
    )
    path = write_run_summary(output_dir, summary)
    payload = json.loads(path.read_text())
    for key in [
        "completed_timestamp",
        "standardized_output_stem",
        "frames_attempted",
        "tracking_ok_count",
        "tracking_lost_count",
        "final_state",
        "keyframes",
        "map_points",
        "trajectory_poses",
        "elapsed_sec",
        "avg_fps",
    ]:
        assert key in payload


def test_cli_parser_accepts_tum_command():
    parser = create_arg_parser()
    args = parser.parse_args(
        [
            "datasets/tum/rgbd_dataset_freiburg1_desk",
            "--dataset-type",
            "tum_rgbd",
            "--camera-profile",
            "auto",
            "--output",
            "visual_slam_outputs/checkpoint_2_29A/fr1_desk_30",
            "--feature-backend",
            "pyslam_orb2",
            "--enable-loop-closing",
            "--enable-global-ba",
            "--global-ba-after-loop",
        ]
    )
    assert args.dataset_type == DATASET_TYPE_TUM
    assert args.camera_profile == "auto"
    assert args.feature_backend == "pyslam_orb2"
    assert args.enable_loop_closing is True
    assert args.enable_global_ba is True
    assert args.global_ba_after_loop is True


def test_cli_parser_accepts_lab_rgbd_command(tmp_path):
    parser = create_arg_parser()
    dataset = _make_lab_dataset(tmp_path)
    args = parser.parse_args(
        [
            str(dataset),
            "--dataset-type",
            "lab_rgbd",
            "--camera-config",
            str(dataset / "camera.yaml"),
            "--output",
            str(tmp_path / "out"),
            "--feature-backend",
            "pyslam_orb2",
            "--enable-loop-closing",
            "--enable-global-ba",
            "--global-ba-after-loop",
        ]
    )
    assert args.dataset_type == DATASET_TYPE_LAB
    assert args.camera_config == dataset / "camera.yaml"
    assert args.feature_backend == "pyslam_orb2"


def test_existing_run_tum_rgbd_smoke_import_still_works():
    assert callable(run_tum_rgbd_smoke)
