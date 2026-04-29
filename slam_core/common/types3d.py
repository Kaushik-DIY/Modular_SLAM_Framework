"""
=============================================================================
slam_core/common/types3d.py

3D (SE3) pose types and camera intrinsics for Visual SLAM.

IMPORTANT: This module follows pyslam's exact implementation approach.
-----------------------------------------------------------------------
pyslam uses g2o.Isometry3d directly for SE(3) poses rather than creating
wrapper classes. This allows seamless integration with g2o optimization
without conversion overhead.

Design:
- Internal Visual SLAM operations: use g2o.Isometry3d directly (pyslam style)
- Cross-SLAM interface: use PoseEstimate with 4x4 matrix (framework style)  
- Conversion happens at module boundaries (adapters), not in hot paths

Classes
-------
CameraIntrinsics
    Pinhole camera model (ported from pyslam's camera class).
    
PoseEstimate  
    Unified pose for cross-SLAM interface. Wraps 4x4 matrix + metadata.

Type Aliases
------------
Pose3D : g2o.Isometry3d (the type pyslam uses internally)

=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

# Import g2o for Isometry3d type
try:
    import g2o
    Pose3D = g2o.Isometry3d  # Type alias matching pyslam's usage
except ImportError:
    Pose3D = None
    print("WARNING: g2o not found. SE(3) functionality unavailable.")

if TYPE_CHECKING:
    from slam_core.common.types import Pose2


# ===========================================================================
# Camera Intrinsics (ported from pyslam)
# ===========================================================================

@dataclass
class CameraIntrinsics:
    """
    Pinhole camera model intrinsics.
    
    Ported from pyslam's camera.py to match their parameter structure.
    Used by Visual SLAM for projection/back-projection.
    
    Attributes
    ----------
    fx, fy : float
        Focal length in pixels.
    cx, cy : float
        Principal point in pixels (optical center).
    width, height : int
        Image dimensions in pixels.
    depth_scale : float
        Depth scale factor. TUM: 5000, RealSense: 1000.
    k1, k2, k3, p1, p2 : float
        Distortion coefficients (Brown-Conrady model).
    
    Notes
    -----
    Projection: u = fx*(X/Z) + cx, v = fy*(Y/Z) + cy
    Back-projection: X = (u-cx)*Z/fx, Y = (v-cy)*Z/fy, Z = depth_raw/depth_scale
    
    Examples
    --------
    >>> # TUM fr1/desk camera
    >>> cam = CameraIntrinsics(
    ...     fx=517.3, fy=516.5, cx=318.6, cy=255.3,
    ...     width=640, height=480, depth_scale=5000.0
    ... )
    """
    
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 5000.0  # TUM default
    
    # Distortion coefficients
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0


# ===========================================================================
# PoseEstimate — Cross-SLAM Interface
# ===========================================================================

@dataclass
class PoseEstimate:
    """
    Unified pose estimate for cross-SLAM interface.
    
    Wraps a 4x4 SE(3) matrix with metadata. Both Lidar and Visual SLAM
    produce PoseEstimate objects for downstream consumption.
    
    Attributes
    ----------
    timestamp : float
        Pose timestamp in seconds.
    matrix : np.ndarray
        4x4 homogeneous transformation (SE3).
        Visual SLAM: from g2o.Isometry3d.matrix()
        Lidar SLAM: Pose2 lifted to 3D (z=0)
    source : str
        SLAM source: "hector" | "carto" | "visual_orbslam"
    confidence : float
        Quality in [0,1]. Lidar: match score. Visual: tracking quality.
    is_keyframe : bool
        True if this is a keyframe pose.
    
    Properties
    ----------
    x, y, z : float
        Translation extracted from matrix (computed on-the-fly).
    theta : float  
        Yaw angle in radians (computed on-the-fly).
    
    Methods
    -------
    to_pose2() -> Pose2
        Project to 2D ground plane.
    
    Examples
    --------
    >>> # From Visual SLAM (g2o.Isometry3d)
    >>> import g2o
    >>> pose_g2o = g2o.Isometry3d(np.eye(4))
    >>> pe = PoseEstimate(
    ...     timestamp=0.1,
    ...     matrix=pose_g2o.matrix(),
    ...     source="visual_orbslam",
    ...     confidence=0.88
    ... )
    >>> print(f"x={pe.x}, y={pe.y}, z={pe.z}")
    x=0.0, y=0.0, z=0.0
    """
    
    timestamp: float
    matrix: np.ndarray  # 4x4 SE(3)
    source: str
    confidence: float = 1.0
    is_keyframe: bool = False
    
    @property
    def x(self) -> float:
        """Translation x from matrix."""
        return float(self.matrix[0, 3])
    
    @property
    def y(self) -> float:
        """Translation y from matrix."""
        return float(self.matrix[1, 3])
    
    @property
    def z(self) -> float:
        """Translation z from matrix."""
        return float(self.matrix[2, 3])
    
    @property
    def theta(self) -> float:
        """
        Yaw angle (rotation around z-axis) from matrix.
        
        Returns
        -------
        float
            Yaw in radians, range [-π, π].
        
        Notes
        -----
        Formula: theta = atan2(R[1,0], R[0,0])
        """
        R = self.matrix[:3, :3]
        return float(np.arctan2(R[1, 0], R[0, 0]))
    
    def to_pose2(self) -> Pose2:
        """
        Project to 2D ground plane.
        
        Extracts (x, y, theta) from the SE(3) matrix for comparison
        with Lidar SLAM or 2D navigation.
        
        Returns
        -------
        Pose2
            Ground plane projection.
        """
        from slam_core.common.types import Pose2
        return Pose2(x=self.x, y=self.y, theta=self.theta)