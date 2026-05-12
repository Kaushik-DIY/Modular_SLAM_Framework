"""
Dataset and sensor type definitions.
This module stores the enums and helpers used to configure sensor mode and dataset metadata.
"""

from __future__ import annotations

from enum import Enum


# Enumerate the dataset families recognized by the project.
class DatasetType(Enum):
    """Dataset family identifiers."""
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


# Enumerate the operating environments associated with a dataset.
class DatasetEnvironmentType(Enum):
    """Environment labels attached to a dataset."""
    INDOOR = 1
    OUTDOOR = 2


# Enumerate the camera sensing modes supported by the pipeline.
class SensorType(Enum):
    """Sensor mode identifiers used by the pipeline."""
    MONOCULAR = 0
    STEREO = 1
    RGBD = 2


def get_sensor_type(sensor_str: str | SensorType) -> SensorType:
    """Normalize a sensor specification into a SensorType enum."""
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
