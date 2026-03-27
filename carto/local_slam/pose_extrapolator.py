import numpy as np
from carto.common.types import Pose2
from carto.common.se2 import wrap_angle

class PoseExtrapolatorCV:
    """
    Constant-velocity extrapolator in SE2.
    Maintains last pose and estimated body velocity (vx, vy, wz) in world frame.
    """
    def __init__(self, max_dt: float = 0.5, init_vxy: float = 0.0, init_wz: float = 0.0):
        self.max_dt = float(max_dt)
        self.vx = float(init_vxy)
        self.vy = float(init_vxy)
        self.wz = float(init_wz)
        self._last_t = None
        self._last_pose = None

    def has_state(self) -> bool:
        return self._last_t is not None and self._last_pose is not None

    def predict(self, t: float) -> Pose2:
        if not self.has_state():
            return Pose2(0.0, 0.0, 0.0)

        dt = float(t - self._last_t)
        dt = max(0.0, min(dt, self.max_dt))

        x = self._last_pose.x + self.vx * dt
        y = self._last_pose.y + self.vy * dt
        th = wrap_angle(self._last_pose.theta + self.wz * dt)
        return Pose2(x, y, th)

    # def update(self, t: float, pose: Pose2):
    #     """
    #     Update velocity estimate from finite differences.
    #     """
    #     if not self.has_state():
    #         self._last_t = float(t)
    #         self._last_pose = pose
    #         return

    #     dt = float(t - self._last_t)
    #     if dt <= 1e-6:
    #         self._last_t = float(t)
    #         self._last_pose = pose
    #         return

    #     # estimate world-frame velocity
    #     MAX_V = 2.0   # m/s (fr079 is slow; adjust later)
    #     MAX_W = 2.0   # rad/s

    #     self.vx = np.clip(self.vx, -MAX_V, MAX_V)
    #     self.vy = np.clip(self.vy, -MAX_V, MAX_V)
    #     self.wz = np.clip(self.wz, -MAX_W, MAX_W)

    #     self._last_t = float(t)
    #     self._last_pose = pose

    def update(self, t: float, pose: Pose2):
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

        vx_new = float(np.clip(vx_new, -MAX_V, MAX_V))
        vy_new = float(np.clip(vy_new, -MAX_V, MAX_V))
        wz_new = float(np.clip(wz_new, -MAX_W, MAX_W))

        alpha = 0.7
        self.vx = alpha * self.vx + (1.0 - alpha) * vx_new
        self.vy = alpha * self.vy + (1.0 - alpha) * vy_new
        self.wz = alpha * self.wz + (1.0 - alpha) * wz_new

        self._last_t = float(t)
        self._last_pose = pose
