"""Motion-based filtering for deciding whether a scan is sufficiently different to keep."""

import numpy as np

from carto.common.se2 import wrap_angle
from carto.common.types import Pose2


class MotionFilter:
    """Suppress scans whose pose change is smaller than configured translation and rotation thresholds."""

    def __init__(self, max_dist: float, max_angle: float):
        self.max_dist = float(max_dist)
        self.max_angle = float(max_angle)
        self._last_pose = None

    def is_similar(self, pose: Pose2) -> bool:
        """Return True when the incoming pose is too close to the last accepted pose."""
        if self._last_pose is None:
            self._last_pose = pose
            return False

        dx = pose.x - self._last_pose.x
        dy = pose.y - self._last_pose.y
        dtrans = np.hypot(dx, dy)
        dtheta = abs(wrap_angle(pose.theta - self._last_pose.theta))

        # Only update the anchor pose when motion exceeds the thresholds; this preserves the intended debounce effect.
        if dtrans < self.max_dist and dtheta < self.max_angle:
            return True

        self._last_pose = pose
        return False
