"""
pySLAM-aligned ORB-SLAM core package.

This package contains the self-contained ORB/RGB-D SLAM implementation.
The legacy flat visual_slam modules remain outside this package.
"""

from visual_slam.orbslam.slam.sensor_types import (
    DatasetEnvironmentType,
    DatasetType,
    SensorType,
    get_sensor_type,
    is_depth_available,
    is_monocular,
    is_rgbd,
    is_stereo,
)
from visual_slam.orbslam.slam.slam_commons import SlamState
from visual_slam.orbslam.slam.config_parameters import Parameters, OrbSlamSettings
from visual_slam.orbslam.slam.camera_pose import CameraPose
from visual_slam.orbslam.slam.camera import CameraType, CameraUtils, Camera, PinholeCamera, fov2focal, focal2fov
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared, SlamFeatureManagerInfo
from visual_slam.orbslam.slam.frame import FrameBase, Frame, detect_and_compute, match_frames, are_map_points_visible_in_frame

__all__ = [
    "DatasetEnvironmentType",
    "DatasetType",
    "SensorType",
    "SlamState",
    "Parameters",
    "OrbSlamSettings",
    "CameraPose",
    "CameraType",
    "CameraUtils",
    "Camera",
    "PinholeCamera",
    "fov2focal",
    "focal2fov",
    "FeatureTrackerShared",
    "SlamFeatureManagerInfo",
    "FrameBase",
    "Frame",
    "detect_and_compute",
    "match_frames",
    "are_map_points_visible_in_frame",
    "get_sensor_type",
    "is_depth_available",
    "is_monocular",
    "is_rgbd",
    "is_stereo",
]
