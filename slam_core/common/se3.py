"""
=============================================================================
slam_core/common/se3.py

SE(3) transformation utilities for g2o.Isometry3d objects.

IMPORTANT: This module follows pyslam's implementation approach.
----------------------------------------------------------------
All functions operate on g2o.Isometry3d objects directly (the type pyslam uses).
No custom wrapper classes. This matches pyslam's design and ensures zero-cost
integration when we port their Visual SLAM pipeline in Phase 2.

Functions
---------
pose3d_compose(a, b) -> g2o.Isometry3d
    Compose two SE(3) poses: result = a ⊕ b.
    
pose3d_inverse(a) -> g2o.Isometry3d
    Invert an SE(3) pose: result = a^(-1).
    
transform_points_3d(pose, pts) -> np.ndarray
    Transform 3D points from local to world frame.
    
matrix_to_pose3d(T) -> g2o.Isometry3d
    Convert 4x4 matrix to g2o.Isometry3d.
    
pose3d_to_matrix(pose) -> np.ndarray
    Convert g2o.Isometry3d to 4x4 matrix.

Notes
-----
These are thin wrappers around g2o's SE(3) operations. They exist to:
1. Provide a consistent API matching our se2.py functions
2. Add documentation for the framework
3. Handle any edge cases or validations

For internal Visual SLAM code (Phase 2+), you can call g2o methods directly.
These utilities are provided for framework-level code that might not want
to import g2o everywhere.

Examples
--------
>>> import g2o
>>> import numpy as np
>>> from slam_core.common.se3 import pose3d_compose, transform_points_3d
>>> 
>>> # Create two poses
>>> T1 = np.eye(4)
>>> T1[:3, 3] = [1.0, 0.0, 0.0]  # 1m forward
>>> pose1 = g2o.Isometry3d(T1)
>>> 
>>> T2 = np.eye(4)
>>> T2[:3, 3] = [0.0, 2.0, 0.0]  # 2m left
>>> pose2 = g2o.Isometry3d(T2)
>>> 
>>> # Compose
>>> pose3 = pose3d_compose(pose1, pose2)
>>> print(pose3.matrix()[:3, 3])
[1. 2. 0.]
=============================================================================
"""

from __future__ import annotations

import numpy as np

try:
    import g2o
except ImportError:
    g2o = None
    print("WARNING: g2o not found. SE(3) functions unavailable.")


def pose3d_compose(a, b):
    """
    Compose two SE(3) poses: result = a ⊕ b.
    
    Computes the compound transformation: first apply b, then a.
    Equivalent to matrix multiplication: T_result = T_a @ T_b.
    
    Parameters
    ----------
    a, b : g2o.Isometry3d
        Input poses to compose.
    
    Returns
    -------
    g2o.Isometry3d
        Composed pose.
    
    Notes
    -----
    SE(3) composition is NOT commutative: a ⊕ b ≠ b ⊕ a.
    This function uses g2o's native matrix multiplication.
    
    Examples
    --------
    >>> import g2o
    >>> import numpy as np
    >>> T1 = np.eye(4)
    >>> T1[0, 3] = 1.0
    >>> a = g2o.Isometry3d(T1)
    >>> T2 = np.eye(4)
    >>> T2[1, 3] = 2.0
    >>> b = g2o.Isometry3d(T2)
    >>> c = pose3d_compose(a, b)
    >>> assert abs(c.matrix()[0, 3] - 1.0) < 1e-9
    >>> assert abs(c.matrix()[1, 3] - 2.0) < 1e-9
    """
    if g2o is None:
        raise RuntimeError("g2o not available")
    
    # Use g2o's native matrix multiplication
    Ta = a.matrix()
    Tb = b.matrix()
    Tc = Ta @ Tb
    return g2o.Isometry3d(Tc)


def pose3d_inverse(a):
    """
    Invert an SE(3) pose: result = a^(-1).
    
    The inverse transforms points from world frame to local frame.
    Satisfies: a ⊕ a^(-1) = identity.
    
    Parameters
    ----------
    a : g2o.Isometry3d
        Input pose to invert.
    
    Returns
    -------
    g2o.Isometry3d
        Inverted pose.
    
    Notes
    -----
    For SE(3) matrix T = [R t; 0 1]:
        T^(-1) = [R^T  -R^T*t; 0  1]
    
    Examples
    --------
    >>> import g2o
    >>> import numpy as np
    >>> T = np.eye(4)
    >>> T[:3, 3] = [2.0, 1.0, 0.5]
    >>> a = g2o.Isometry3d(T)
    >>> a_inv = pose3d_inverse(a)
    >>> identity = pose3d_compose(a, a_inv)
    >>> assert np.allclose(identity.matrix(), np.eye(4), atol=1e-9)
    """
    if g2o is None:
        raise RuntimeError("g2o not available")
    
    Ta = a.matrix()
    Ta_inv = np.linalg.inv(Ta)
    return g2o.Isometry3d(Ta_inv)


def transform_points_3d(pose, pts: np.ndarray) -> np.ndarray:
    """
    Transform 3D points from local frame to world frame.
    
    Applies: p_world = R @ p_local + t
    
    Parameters
    ----------
    pose : g2o.Isometry3d
        Transformation from local to world frame.
    pts : np.ndarray
        Points in local frame, shape (N, 3).
    
    Returns
    -------
    np.ndarray
        Points in world frame, shape (N, 3).
    
    Examples
    --------
    >>> import g2o
    >>> import numpy as np
    >>> T = np.eye(4)
    >>> T[:3, 3] = [1.0, 2.0, 3.0]
    >>> pose = g2o.Isometry3d(T)
    >>> pts_local = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> pts_world = transform_points_3d(pose, pts_local)
    >>> assert np.allclose(pts_world[0], [1.0, 2.0, 3.0])
    >>> assert np.allclose(pts_world[1], [2.0, 2.0, 3.0])
    """
    if g2o is None:
        raise RuntimeError("g2o not available")
    
    T = pose.matrix()
    R = T[:3, :3]
    t = T[:3, 3]
    
    # pts_world = (R @ pts_local.T).T + t
    return (R @ pts.T).T + t


def matrix_to_pose3d(T: np.ndarray):
    """
    Convert 4x4 matrix to g2o.Isometry3d.
    
    Parameters
    ----------
    T : np.ndarray
        4x4 homogeneous transformation matrix.
    
    Returns
    -------
    g2o.Isometry3d
        Pose object.
    
    Examples
    --------
    >>> import numpy as np
    >>> T = np.eye(4)
    >>> T[:3, 3] = [1.0, 2.0, 3.0]
    >>> pose = matrix_to_pose3d(T)
    >>> assert np.allclose(pose.matrix(), T)
    """
    if g2o is None:
        raise RuntimeError("g2o not available")
    
    return g2o.Isometry3d(T)


def pose3d_to_matrix(pose) -> np.ndarray:
    """
    Convert g2o.Isometry3d to 4x4 matrix.
    
    Parameters
    ----------
    pose : g2o.Isometry3d
        Pose object.
    
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix.
    
    Examples
    --------
    >>> import g2o
    >>> import numpy as np
    >>> T_orig = np.eye(4)
    >>> T_orig[:3, 3] = [1.0, 2.0, 3.0]
    >>> pose = g2o.Isometry3d(T_orig)
    >>> T_back = pose3d_to_matrix(pose)
    >>> assert np.allclose(T_back, T_orig)
    """
    if g2o is None:
        raise RuntimeError("g2o not available")
    
    return pose.matrix()