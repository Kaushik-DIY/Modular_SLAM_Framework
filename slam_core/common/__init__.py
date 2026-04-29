"""
slam_core/common/__init__.py

Exports for common types and SE(2)/SE(3) transformations.
"""

from .types import Pose2, RangeData2D
from .types3d import Pose3D, CameraIntrinsics, PoseEstimate  # Pose3D is g2o.Isometry3d
from .se2 import (
    wrap_angle,
    pose_compose,
    pose_inverse,
    inverse_pose,
    transform_points,
    transform_points_pose,
)
from .se3 import (
    pose3d_compose,
    pose3d_inverse,
    transform_points_3d,
    matrix_to_pose3d,
    pose3d_to_matrix,
)

__all__ = [
    # 2D types (existing)
    "Pose2",
    "RangeData2D",
    # 3D types (new - following pyslam)
    "Pose3D",  # Type alias for g2o.Isometry3d
    "CameraIntrinsics",
    "PoseEstimate",
    # SE(2) functions (existing)
    "wrap_angle",
    "pose_compose",
    "pose_inverse",
    "inverse_pose",
    "transform_points",
    "transform_points_pose",
    # SE(3) functions (new - operate on g2o.Isometry3d)
    "pose3d_compose",
    "pose3d_inverse",
    "transform_points_3d",
    "matrix_to_pose3d",
    "pose3d_to_matrix",
]