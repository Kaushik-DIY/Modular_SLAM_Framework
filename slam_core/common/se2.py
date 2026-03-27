from __future__ import annotations
import numpy as np

from .types import Pose2


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def pose_compose(a: Pose2, b: Pose2) -> Pose2:
    ca, sa = np.cos(a.theta), np.sin(a.theta)
    x = a.x + ca * b.x - sa * b.y
    y = a.y + sa * b.x + ca * b.y
    th = wrap_angle(a.theta + b.theta)
    return Pose2(float(x), float(y), float(th))


def pose_inverse(a: Pose2) -> Pose2:
    ca, sa = np.cos(a.theta), np.sin(a.theta)
    x = -(ca * a.x + sa * a.y)
    y = -(-sa * a.x + ca * a.y)
    return Pose2(float(x), float(y), float(wrap_angle(-a.theta)))


def inverse_pose(p: Pose2) -> Pose2:
    ca, sa = np.cos(p.theta), np.sin(p.theta)
    x = -(ca * p.x + sa * p.y)
    y = -(-sa * p.x + ca * p.y)
    return Pose2(float(x), float(y), float(wrap_angle(-p.theta)))


def transform_points(pose: Pose2, pts_xy: np.ndarray) -> np.ndarray:
    """
    pts_xy: (N,2) in local frame -> (N,2) in world frame
    """
    ca, sa = np.cos(pose.theta), np.sin(pose.theta)
    R = np.array([[ca, -sa], [sa, ca]], dtype=float)
    return pts_xy @ R.T + np.array([pose.x, pose.y], dtype=float)


def transform_points_pose(pose: Pose2, pts_xy: np.ndarray) -> np.ndarray:
    """
    Same as transform_points(): pose ⊕ points
    """
    ca, sa = np.cos(pose.theta), np.sin(pose.theta)
    R = np.array([[ca, -sa], [sa, ca]], dtype=float)
    return pts_xy @ R.T + np.array([pose.x, pose.y], dtype=float)