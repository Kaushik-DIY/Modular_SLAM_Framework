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

__all__ = [
    "DatasetEnvironmentType",
    "DatasetType",
    "SensorType",
    "SlamState",
    "Parameters",
    "OrbSlamSettings",
    "get_sensor_type",
    "is_depth_available",
    "is_monocular",
    "is_rgbd",
    "is_stereo",
]
