import numpy as np
from carto.common.types import Pose2
from carto.common.se2 import wrap_angle

class MotionFilter:
    def __init__(self, max_dist: float, max_angle: float):
        self.max_dist = float(max_dist)
        self.max_angle = float(max_angle)
        self._last_pose = None

    def is_similar(self, pose: Pose2) -> bool:
        if self._last_pose is None:
            self._last_pose = pose
            return False
        dx = pose.x - self._last_pose.x
        dy = pose.y - self._last_pose.y
        dtrans = np.hypot(dx, dy)
        dtheta = abs(wrap_angle(pose.theta - self._last_pose.theta))
        if dtrans < self.max_dist and dtheta < self.max_angle:
            return True
        self._last_pose = pose
        return False