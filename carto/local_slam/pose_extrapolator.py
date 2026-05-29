from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from carto.common.se2 import wrap_angle
from carto.common.types import Pose2


class PoseExtrapolatorCV:
    """
    Cartographer-like 2D constant-velocity pose extrapolator.

    Design goals
    ------------
    - The extrapolator owns the prior used by local scan matching.
    - Recent matched poses are kept in a short queue to estimate motion.
    - Odometry, when available, is also queued and converted into a velocity
      estimate inside the extrapolator.
    - Prediction blends pose-derived and odometry-derived velocities.

    This is still a simplified 2D extrapolator, but the ownership model now
    matches Cartographer much more closely than adapter-side odometry blending.
    """

    def __init__(
        self,
        max_dt: float = 0.5,
        init_vxy: float = 0.0,
        init_wz: float = 0.0,
        *,
        pose_queue_duration_s: float = 1.5,
        odom_queue_duration_s: float = 1.5,
        odom_trust: float = 0.35,
        max_linear_speed_mps: float = 2.0,
        max_angular_speed_rps: float = 2.0,
        use_imu: bool = False,
        imu_yaw_correction_alpha: float = 0.02,
    ) -> None:
        self.max_dt = float(max_dt)

        self.pose_queue_duration_s = float(pose_queue_duration_s)
        self.odom_queue_duration_s = float(odom_queue_duration_s)
        self.odom_trust = float(np.clip(odom_trust, 0.0, 1.0))

        self.max_linear_speed_mps = float(max_linear_speed_mps)
        self.max_angular_speed_rps = float(max_angular_speed_rps)

        # Optional IMU aiding (Cartographer ImuTracker style, 2D): the IMU angular
        # velocity drives the rotation term and the absolute quaternion-yaw provides a
        # drift-bounding heading reference. Off by default -> behaviour is pure CV.
        self.use_imu = bool(use_imu)
        self.imu_yaw_correction_alpha = float(np.clip(imu_yaw_correction_alpha, 0.0, 1.0))
        self._imu_wz: Optional[float] = None     # latest IMU yaw-rate [rad/s]
        self._imu_yaw: Optional[float] = None     # latest IMU absolute yaw [rad]
        self._imu_yaw_offset: Optional[float] = None  # slam_yaw - imu_yaw, set once
        self._imu_t: Optional[float] = None

        # Exposed for debugging / plotting convenience.
        self.vx = float(init_vxy)
        self.vy = float(init_vxy)
        self.wz = float(init_wz)

        self._pose_queue: Deque[Tuple[float, Pose2]] = deque()
        self._odom_queue: Deque[Tuple[float, Pose2]] = deque()

        self._last_extrapolated_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def has_state(self) -> bool:
        """Return True once at least one matched pose has been added."""
        return len(self._pose_queue) > 0

    def get_last_pose_time(self) -> Optional[float]:
        """Return the timestamp of the latest matched pose, if available."""
        if not self._pose_queue:
            return None
        return float(self._pose_queue[-1][0])

    def get_last_extrapolated_time(self) -> Optional[float]:
        """Return the timestamp used by the most recent prediction call."""
        return self._last_extrapolated_time

    def add_pose(self, t: float, pose: Pose2) -> None:
        """
        Add a matched local SLAM pose to the extrapolator history.
        """
        self._append_state(self._pose_queue, float(t), pose, self.pose_queue_duration_s)

    def update(self, t: float, pose: Pose2) -> None:
        """
        Backward-compatible alias for add_pose().
        """
        self.add_pose(t, pose)

    def add_odometry(self, t: float, odom_pose: Pose2) -> None:
        """
        Add an odometry pose sample to the extrapolator history.
        """
        self._append_state(self._odom_queue, float(t), odom_pose, self.odom_queue_duration_s)

    def add_imu(self, t: float, wz: float, yaw: Optional[float] = None) -> None:
        """Feed one IMU sample: angular velocity wz [rad/s] and optional absolute
        yaw [rad] (from the orientation quaternion). No-op for prediction unless
        `use_imu` was enabled at construction."""
        self._imu_t = float(t)
        self._imu_wz = float(wz)
        if yaw is not None:
            self._imu_yaw = float(wrap_angle(float(yaw)))

    def correct_pose(self, t: float, pose: Pose2) -> None:
        """
        Replace or append the latest matched pose after a pose-graph correction.
        """
        self._append_state(self._pose_queue, float(t), pose, self.pose_queue_duration_s)

    def predict(self, t: float) -> Pose2:
        """
        Predict the pose at time t using the internal motion estimate.

        The extrapolator first estimates motion from the recent matched-pose
        queue and, when odometry is available, blends that with motion derived
        from the odometry queue.
        """
        if not self.has_state():
            return Pose2(0.0, 0.0, 0.0)

        last_t, last_pose = self._pose_queue[-1]

        dt = float(t - last_t)
        dt = max(0.0, min(dt, self.max_dt))

        vx, vy, wz = self._estimate_blended_velocity()

        # IMU aiding: the gyro yaw-rate is a far better short-horizon rotation estimate
        # than differencing matched poses (especially during fast turns, where the
        # pose-derived wz lags). Translation stays CV (no reliable 2D accel integration).
        if self.use_imu and self._imu_wz is not None:
            wz = float(np.clip(self._imu_wz, -self.max_angular_speed_rps, self.max_angular_speed_rps))

        self.vx = float(vx)
        self.vy = float(vy)
        self.wz = float(wz)

        pred_theta = wrap_angle(float(last_pose.theta) + float(wz) * dt)

        # Absolute-heading reference from the IMU orientation quaternion: bound long-term
        # yaw drift by gently nudging the prediction toward the IMU-implied heading.
        # The offset (slam_yaw - imu_yaw) is captured once so the IMU yaw frame aligns
        # with the SLAM world frame; thereafter imu_yaw + offset is the absolute heading.
        if self.use_imu and self._imu_yaw is not None and self.imu_yaw_correction_alpha > 0.0:
            if self._imu_yaw_offset is None:
                self._imu_yaw_offset = float(wrap_angle(float(last_pose.theta) - float(self._imu_yaw)))
            imu_heading = wrap_angle(float(self._imu_yaw) + float(self._imu_yaw_offset))
            a = float(self.imu_yaw_correction_alpha)
            pred_theta = wrap_angle(pred_theta + a * wrap_angle(imu_heading - pred_theta))

        pred = Pose2(
            x=float(last_pose.x) + float(vx) * dt,
            y=float(last_pose.y) + float(vy) * dt,
            theta=pred_theta,
        )
        self._last_extrapolated_time = float(t)
        return pred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_state(
        self,
        queue: Deque[Tuple[float, Pose2]],
        t: float,
        pose: Pose2,
        duration_s: float,
    ) -> None:
        """
        Append one timestamped SE(2) state to a bounded duration queue.

        If the latest entry already has the same timestamp, replace it.
        """
        if queue and abs(float(queue[-1][0]) - float(t)) <= 1e-9:
            queue[-1] = (float(t), pose)
        else:
            queue.append((float(t), pose))

        self._trim_queue(queue, duration_s)

    def _trim_queue(
        self,
        queue: Deque[Tuple[float, Pose2]],
        duration_s: float,
    ) -> None:
        """
        Keep only the recent portion of a timestamped queue.
        """
        if not queue:
            return

        newest_t = float(queue[-1][0])
        while len(queue) > 1 and (newest_t - float(queue[0][0])) > float(duration_s):
            queue.popleft()

    def _estimate_velocity_from_queue(
        self,
        queue: Deque[Tuple[float, Pose2]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Estimate SE(2) velocity from the oldest and newest elements of a queue.
        """
        if len(queue) < 2:
            return None

        t0, p0 = queue[0]
        t1, p1 = queue[-1]
        dt = float(t1 - t0)
        if dt <= 1e-6:
            return None

        vx = float(p1.x - p0.x) / dt
        vy = float(p1.y - p0.y) / dt
        wz = float(wrap_angle(p1.theta - p0.theta)) / dt

        vx = float(np.clip(vx, -self.max_linear_speed_mps, self.max_linear_speed_mps))
        vy = float(np.clip(vy, -self.max_linear_speed_mps, self.max_linear_speed_mps))
        wz = float(np.clip(wz, -self.max_angular_speed_rps, self.max_angular_speed_rps))

        return vx, vy, wz

    def _estimate_blended_velocity(self) -> Tuple[float, float, float]:
        """
        Blend pose-derived and odometry-derived motion estimates.

        Fallback order:
        1. blend pose and odometry velocities when both exist
        2. use pose-derived velocity if only that exists
        3. use odometry-derived velocity if only that exists
        4. use the currently stored velocity state otherwise
        """
        pose_vel = self._estimate_velocity_from_queue(self._pose_queue)
        odom_vel = self._estimate_velocity_from_queue(self._odom_queue)

        if pose_vel is not None and odom_vel is not None:
            a = float(self.odom_trust)
            vx = (1.0 - a) * float(pose_vel[0]) + a * float(odom_vel[0])
            vy = (1.0 - a) * float(pose_vel[1]) + a * float(odom_vel[1])
            wz = (1.0 - a) * float(pose_vel[2]) + a * float(odom_vel[2])
            return float(vx), float(vy), float(wz)

        if pose_vel is not None:
            return pose_vel

        if odom_vel is not None:
            return odom_vel

        return float(self.vx), float(self.vy), float(self.wz)