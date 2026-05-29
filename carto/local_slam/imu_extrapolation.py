"""IMU helpers for the 2D pose extrapolator.

Converts the lab IMU CSV rows (read by `slam_core.dataio.imu_csv.read_imu_csv`,
columns: timestamp, frame_id, qx, qy, qz, qw, wx, wy, wz, ax, ay, az) into the
(timestamp, yaw_rate, absolute_yaw) samples that `PoseExtrapolatorCV.add_imu`
consumes. Only the yaw channel matters for planar SLAM.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rotation about +Z) from an orientation quaternion, in radians."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def imu_rows_to_samples(rows: List[Dict[str, Any]]) -> List[Tuple[float, float, float]]:
    """Map IMU CSV rows -> sorted list of (timestamp, wz, yaw).

    wz is the gyro z angular velocity [rad/s]; yaw is the absolute heading [rad]
    from the orientation quaternion.
    """
    samples: List[Tuple[float, float, float]] = []
    for r in rows:
        try:
            t = float(r["timestamp"])
            wz = float(r["wz"])
            yaw = quaternion_to_yaw(
                float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])
            )
        except (KeyError, ValueError, TypeError):
            continue
        samples.append((t, wz, yaw))
    samples.sort(key=lambda s: s[0])
    return samples
