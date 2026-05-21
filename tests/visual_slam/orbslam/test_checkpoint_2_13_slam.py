from types import SimpleNamespace
import inspect

import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    LocalMapping,
    Map,
    Parameters,
    PinholeCamera,
    SensorType,
    Slam,
    SlamMode,
    SlamState,
    Tracking,
)


def make_camera():
    return PinholeCamera.from_params(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
        th_depth=40.0,
    )


def test_slam_initialization_matches_pyslam_module_ownership():
    cam = make_camera()
    slam = Slam(camera=cam, sensor_type=SensorType.RGBD, headless=True)

    assert slam.camera is cam
    assert slam.sensor_type == SensorType.RGBD
    assert isinstance(slam.map, Map)
    assert isinstance(slam.local_mapping, LocalMapping)
    assert isinstance(slam.tracking, Tracking)
    assert slam.feature_tracker is FeatureTrackerShared.feature_tracker
    assert FeatureTrackerShared.feature_manager.max_descriptor_distance == 100
    assert Parameters.kMaxDescriptorDistance == 100


def test_slam_track_delegates_with_pyslam_argument_order():
    cam = make_camera()
    slam = Slam(camera=cam, sensor_type=SensorType.RGBD, headless=True)

    received = {}

    def fake_track(img, img_right=None, depth=None, img_id=None, timestamp=None, mask=None, mask_right=None):
        received["img"] = img
        received["img_right"] = img_right
        received["depth"] = depth
        received["img_id"] = img_id
        received["timestamp"] = timestamp
        received["mask"] = mask
        received["mask_right"] = mask_right
        return "ok"

    slam.tracking.track = fake_track

    result = slam.track(
        img="left",
        img_right="right",
        depth="depth",
        img_id=7,
        timestamp=1.25,
        mask="mask_l",
        mask_right="mask_r",
    )

    assert result == "ok"
    assert received == {
        "img": "left",
        "img_right": "right",
        "depth": "depth",
        "img_id": 7,
        "timestamp": 1.25,
        "mask": "mask_l",
        "mask_right": "mask_r",
    }


def test_tracking_track_signature_is_pyslam_compatible():
    sig = inspect.signature(Tracking.track)
    params = list(sig.parameters.keys())

    assert params[:8] == [
        "self",
        "img",
        "img_right",
        "depth",
        "img_id",
        "timestamp",
        "mask",
        "mask_right",
    ]


def test_slam_config_distribution_to_tracking_and_local_mapping():
    cam = make_camera()
    config = SimpleNamespace(
        far_points_threshold=12.5,
        use_fov_centers_based_kf_generation=True,
        max_fov_centers_distance=0.3,
    )

    slam = Slam(camera=cam, sensor_type=SensorType.RGBD, config=config, headless=True)

    assert slam.tracking.far_points_threshold == 12.5
    assert slam.tracking.use_fov_centers_based_kf_generation is True
    assert slam.tracking.max_fov_centers_distance == 0.3

    assert slam.local_mapping.far_points_threshold == 12.5
    assert slam.local_mapping.use_fov_centers_based_kf_generation is True
    assert slam.local_mapping.max_fov_centers_distance == 0.3


def test_slam_state_and_trajectory_accessors():
    cam = make_camera()
    slam = Slam(camera=cam, sensor_type=SensorType.RGBD, headless=True)

    slam.set_tracking_state(SlamState.OK)

    assert slam.get_tracking_state() == SlamState.OK
    assert slam.is_ok()

    traj = slam.get_final_trajectory()

    assert "poses" in traj
    assert "timestamps" in traj
    assert "slam_states" in traj
    assert "ids" in traj
    assert isinstance(traj["poses"], list)
    assert isinstance(traj["timestamps"], list)
    assert isinstance(traj["slam_states"], list)
    assert len(traj["poses"]) == len(traj["timestamps"]) == len(traj["slam_states"])


def test_slam_reset_preserves_module_ownership_and_clears_request():
    cam = make_camera()
    slam = Slam(camera=cam, sensor_type=SensorType.RGBD, headless=True)

    slam.request_reset()
    assert slam.reset_requested is True

    slam.reset()

    assert slam.reset_requested is False
    assert isinstance(slam.map, Map)
    assert isinstance(slam.local_mapping, LocalMapping)
    assert isinstance(slam.tracking, Tracking)
