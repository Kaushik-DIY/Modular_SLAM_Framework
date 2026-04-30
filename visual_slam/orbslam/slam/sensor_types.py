"""
=============================================================================
visual_slam/orbslam/slam/sensor_types.py

pySLAM-aligned minimal dataset/sensor type definitions.

Reference:
- pySLAM: pyslam/io/dataset_types.py

Only the ORB/RGB-D SLAM-relevant subset is kept, but enum values are preserved
to stay compatible with pySLAM-style logic.
=============================================================================
"""

from __future__ import annotations

from enum import Enum


class DatasetType(Enum):
    """pySLAM-compatible dataset type values."""

    NONE = 1
    KITTI = 2
    TUM = 3
    EUROC = 4
    REPLICA = 5
    TARTANAIR = 6
    VIDEO = 7
    FOLDER = 8
    ROS1BAG = 9
    ROS2BAG = 10
    LIVE = 11
    SCANNET = 12
    ICL_NUIM = 13
    MCAP = 14
    SEVEN_SCENES = 15
    NEURAL_RGBD = 16
    ROVER = 17


class DatasetEnvironmentType(Enum):
    """pySLAM-compatible environment type values."""

    INDOOR = 1
    OUTDOOR = 2


class SensorType(Enum):
    """pySLAM-compatible sensor type values."""

    MONOCULAR = 0
    STEREO = 1
    RGBD = 2


def get_sensor_type(sensor_str: str | SensorType) -> SensorType:
    """pySLAM-compatible sensor string parser."""
    if isinstance(sensor_str, SensorType):
        return sensor_str

    sensor_str = str(sensor_str).lower()

    if sensor_str in ("mono", "monocular"):
        return SensorType.MONOCULAR
    if sensor_str == "stereo":
        return SensorType.STEREO
    if sensor_str == "rgbd":
        return SensorType.RGBD

    return SensorType.MONOCULAR


def is_monocular(sensor_type: SensorType) -> bool:
    return sensor_type == SensorType.MONOCULAR


def is_stereo(sensor_type: SensorType) -> bool:
    return sensor_type == SensorType.STEREO


def is_rgbd(sensor_type: SensorType) -> bool:
    return sensor_type == SensorType.RGBD


def is_depth_available(sensor_type: SensorType) -> bool:
    return sensor_type in (SensorType.STEREO, SensorType.RGBD)
