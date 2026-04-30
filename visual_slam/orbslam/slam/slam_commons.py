"""
=============================================================================
visual_slam/orbslam/slam/slam_commons.py

pySLAM-aligned SLAM state definitions.

Reference:
- pySLAM: pyslam/slam/slam_commons.py
=============================================================================
"""

from __future__ import annotations

from enum import Enum


class SlamState(Enum):
    """pySLAM-compatible SLAM state values."""

    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2
    LOST = 3
    RELOCALIZE = 4
    INIT_RELOCALIZE = 5
