"""Pose extrapolation utilities used to predict the next local SLAM pose."""

import numpy as np

from carto.common.se2 import wrap_angle
from carto.common.types import Pose2


class PoseExtrapolatorCV:
    """Constant-velocity SE(2) extrapolator with smoothed velocity updates."""

    def __init__(self, max_dt: float = 0.5, init_vxy: float = 0.0, init_wz: float = 0.0):
        self.max_dt = float(max_dt)
        self.vx = float(init_vxy)
        self.vy = float(init_vxy)
        self.wz = float(init_wz)
        self._last_t = None
        self._last_pose = None

    def has_state(self) -> bool:
        """Return True once the extrapolator has seen at least one timestamped pose."""
        return self._last_t is not None and self._last_pose is not None

    def predict(self, t: float) -> Pose2:
        """Predict the pose at time t using the current constant-velocity estimate."""
        if not self.has_state():
            return Pose2(0.0, 0.0, 0.0)

        # Clamp the horizon so stale timestamps do not create unrealistic dead-reckoning jumps.
        dt = float(t - self._last_t)
        dt = max(0.0, min(dt, self.max_dt))

        x = self._last_pose.x + self.vx * dt
        y = self._last_pose.y + self.vy * dt
        th = wrap_angle(self._last_pose.theta + self.wz * dt)
        return Pose2(x, y, th)

    def update(self, t: float, pose: Pose2):
        """Update the internal velocity estimate from the newest pose observation."""
        if not self.has_state():
            self._last_t = float(t)
            self._last_pose = pose
            return

        dt = float(t - self._last_t)
        if dt <= 1e-6:
            self._last_t = float(t)
            self._last_pose = pose
            return

        dx = float(pose.x - self._last_pose.x)
        dy = float(pose.y - self._last_pose.y)
        dth = float(wrap_angle(pose.theta - self._last_pose.theta))

        vx_new = dx / dt
        vy_new = dy / dt
        wz_new = dth / dt

        MAX_V = 2.0
        MAX_W = 2.0

        # Saturation limits keep single-frame outliers from dominating the velocity estimate.
        vx_new = float(np.clip(vx_new, -MAX_V, MAX_V))
        vy_new = float(np.clip(vy_new, -MAX_V, MAX_V))
        wz_new = float(np.clip(wz_new, -MAX_W, MAX_W))

        # Exponential smoothing keeps prediction stable while still adapting to recent motion.
        alpha = 0.7
        self.vx = alpha * self.vx + (1.0 - alpha) * vx_new
        self.vy = alpha * self.vy + (1.0 - alpha) * vy_new
        self.wz = alpha * self.wz + (1.0 - alpha) * wz_new

        self._last_t = float(t)
        self._last_pose = pose
