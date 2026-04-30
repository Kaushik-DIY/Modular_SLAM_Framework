"""
=============================================================================
visual_slam/orbslam/local_features/feature_types.py

pySLAM-aligned minimal feature type definitions for ORB/RGB-D SLAM.

Reference:
- pySLAM: pyslam/local_features/feature_types.py

Only the ORB/ORB2 subset is implemented now. Enum names are kept pySLAM-like
so later modules can be ported with minimal changes.
=============================================================================
"""

from __future__ import annotations

from enum import Enum


class FeatureDetectorTypes(Enum):
    NONE = 0
    SHI_TOMASI = 1
    FAST = 2
    ORB = 3
    ORB2 = 4


class FeatureDescriptorTypes(Enum):
    NONE = 0
    ORB = 1
    ORB2 = 2


class FeatureInfo:
    """
    Minimal pySLAM-like feature metadata helper.

    The ORB/ORB2 path uses binary descriptors and Hamming distance.
    """

    @staticmethod
    def is_binary_descriptor(descriptor_type: FeatureDescriptorTypes) -> bool:
        return descriptor_type in (
            FeatureDescriptorTypes.ORB,
            FeatureDescriptorTypes.ORB2,
        )

    @staticmethod
    def is_oriented_features(detector_type: FeatureDetectorTypes) -> bool:
        return detector_type in (
            FeatureDetectorTypes.ORB,
            FeatureDetectorTypes.ORB2,
        )
