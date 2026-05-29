"""
Fused-keyframe Signature and the SE(3) -> SE(2) pose projection.

A Signature is the unit stored in the memory tiers and represented as a node in
the fusion graph. Its payload (CLAUDE.md §2.8): an SE(2) pose, ORB keypoints +
descriptors + 3D points, the most-recent 2D LiDAR scan (or None), a small local
occupancy grid, an RTAB-Map weight, and link bookkeeping.

Phase 1 builds it as a plain data container; the memory/graph behaviour that
consumes it arrives in Phases 2 and 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle


def _as_matrix(pose) -> np.ndarray:
    """Coerce an SE(3) pose into a 4x4 numpy matrix.

    Accepts either a 4x4 array-like or an object exposing ``.matrix()``
    (e.g. ``g2o.Isometry3d``, the ``Pose3D`` alias used by ORB-SLAM).
    """
    if hasattr(pose, "matrix"):
        m = np.asarray(pose.matrix(), dtype=np.float64)
    else:
        m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 SE(3) matrix, got shape {m.shape}")
    return m


def project_pose3d_to_pose2(pose3d, base_T_cam: Optional[np.ndarray] = None) -> Pose2:
    """Project an SE(3) camera pose to a planar SE(2) base pose.

    ORB-SLAM keyframe poses live in SE(3). At the fusion-layer boundary
    (CLAUDE.md §2.6) they are projected to ``Pose2`` under the planar-motion
    assumption using a known camera->base extrinsic.

    Parameters
    ----------
    pose3d : 4x4 array-like or object with ``.matrix()``
        The camera pose in world frame, ``T_world_cam``.
    base_T_cam : 4x4 array-like, optional
        The fixed camera->base extrinsic (``base_from_camera``). If omitted,
        the SE(3) pose is taken to already be a base-frame world pose and its
        (x, y, yaw) are extracted directly.

    Returns
    -------
    Pose2
        Planar pose ``(x, y, theta)`` with ``theta`` wrapped to [-pi, pi].
    """
    T_world_cam = _as_matrix(pose3d)

    if base_T_cam is not None:
        # T_world_base = T_world_cam @ inv(base_T_cam)
        cam_T_base = np.linalg.inv(_as_matrix(base_T_cam))
        T_world_base = T_world_cam @ cam_T_base
    else:
        T_world_base = T_world_cam

    x = float(T_world_base[0, 3])
    y = float(T_world_base[1, 3])
    theta = float(np.arctan2(T_world_base[1, 0], T_world_base[0, 0]))
    return Pose2(x, y, wrap_angle(theta))


@dataclass
class Signature:
    """A fused keyframe: pose + sensor payload + memory bookkeeping.

    Attributes
    ----------
    id : int
        Unique, monotonically increasing keyframe id.
    timestamp : float
        Source RGB-D timestamp (seconds).
    pose : Pose2
        Planar keyframe pose in the fusion world frame.
    keypoints : (N, 2) float array, optional
        ORB keypoint pixel coordinates.
    descriptors : (N, 32) uint8 array, optional
        ORB binary descriptors.
    points3d : (N, 3) float array, optional
        Back-projected 3D points (camera frame).
    scan : (M, 2) float array, optional
        Most-recent 2D LiDAR scan within sync tolerance, in sensor frame.
        ``None`` when no scan fell inside the sync window (CLAUDE.md §2.12).
    local_grid : 2D array, optional
        Small local occupancy grid centred on the keyframe.
    weight : int
        RTAB-Map signature weight (rehearsal inheritance, Phase 2).
    neighbors : list of int
        Ids linked by spine (consecutive-keyframe) edges.
    loops : list of int
        Ids linked by accepted loop-closure edges.
    """

    id: int
    timestamp: float
    pose: Pose2
    keypoints: Optional[np.ndarray] = None
    descriptors: Optional[np.ndarray] = None
    points3d: Optional[np.ndarray] = None
    scan: Optional[np.ndarray] = None
    local_grid: Optional[np.ndarray] = None
    weight: int = 0
    neighbors: List[int] = field(default_factory=list)
    loops: List[int] = field(default_factory=list)

    @property
    def has_scan(self) -> bool:
        return self.scan is not None and len(self.scan) > 0
