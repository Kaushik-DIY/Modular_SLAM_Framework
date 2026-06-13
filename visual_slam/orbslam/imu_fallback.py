"""IMU-assisted pose fallback for RGB-D ORB-SLAM runs.

The ORB map and keyframes stay visual-only. This helper fills a separate
trajectory during visual tracking loss using IMU yaw delta plus recent visual
translation velocity, and can update the tracking recovery anchor so depth
reinitialization starts near the extrapolated pose instead of an old pose.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carto.local_slam.imu_extrapolation import imu_rows_to_samples
from slam_core.dataio.imu_csv import read_imu_csv


def _as_matrix(pose) -> np.ndarray:
    if hasattr(pose, "matrix"):
        pose = pose.matrix()
    T = np.asarray(pose, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), got {T.shape}")
    return T


def _yaw_rot(delta_yaw: float) -> np.ndarray:
    c = float(np.cos(delta_yaw))
    s = float(np.sin(delta_yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass
class ImuFallbackResult:
    timestamp: float
    Tcw: np.ndarray
    source: str
    state: str
    imu_yaw: float | None
    dt_since_visual: float
    speed_mps: float

    def row(self) -> dict:
        Twc = np.linalg.inv(self.Tcw)
        p = Twc[:3, 3]
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "state": self.state,
            "tx": float(p[0]),
            "ty": float(p[1]),
            "tz": float(p[2]),
            "imu_yaw": self.imu_yaw,
            "dt_since_visual": self.dt_since_visual,
            "speed_mps": self.speed_mps,
        }


class ImuFallbackExtrapolator:
    """Maintain a TUM-compatible fallback trajectory for lost visual frames."""

    def __init__(self, imu_path: str | Path):
        self.imu_path = Path(imu_path).expanduser().resolve()
        rows = read_imu_csv(str(self.imu_path))
        samples = imu_rows_to_samples(rows)
        if not samples:
            raise RuntimeError(f"No usable IMU samples in {self.imu_path}")

        self.times = np.asarray([s[0] for s in samples], dtype=np.float64)
        self.yaws = np.unwrap(np.asarray([s[2] for s in samples], dtype=np.float64))

        self.last_visual_t = None
        self.last_visual_Tcw = None
        self.last_visual_yaw = None
        self.prev_visual_t = None
        self.prev_visual_Tcw = None
        self.velocity_w = np.zeros(3, dtype=np.float64)

    @property
    def num_samples(self) -> int:
        return int(len(self.times))

    def yaw_at(self, timestamp: float) -> float | None:
        t = float(timestamp)
        if t < float(self.times[0]) or t > float(self.times[-1]):
            return None
        idx = bisect_right(self.times, t)
        if idx <= 0:
            return float(self.yaws[0])
        if idx >= len(self.times):
            return float(self.yaws[-1])
        t0 = float(self.times[idx - 1])
        t1 = float(self.times[idx])
        if abs(t1 - t0) < 1e-12:
            return float(self.yaws[idx])
        a = (t - t0) / (t1 - t0)
        return float((1.0 - a) * self.yaws[idx - 1] + a * self.yaws[idx])

    def observe(self, timestamp: float, visual_Tcw, state: str) -> ImuFallbackResult | None:
        t = float(timestamp)
        imu_yaw = self.yaw_at(t)

        if visual_Tcw is not None:
            Tcw = _as_matrix(visual_Tcw).copy()
            if self.last_visual_t is not None and self.last_visual_Tcw is not None:
                dt = max(t - float(self.last_visual_t), 1e-9)
                p_prev = np.linalg.inv(self.last_visual_Tcw)[:3, 3]
                p_cur = np.linalg.inv(Tcw)[:3, 3]
                self.velocity_w = (p_cur - p_prev) / dt
                self.prev_visual_t = self.last_visual_t
                self.prev_visual_Tcw = self.last_visual_Tcw

            self.last_visual_t = t
            self.last_visual_Tcw = Tcw.copy()
            self.last_visual_yaw = imu_yaw
            return ImuFallbackResult(
                timestamp=t,
                Tcw=Tcw,
                source="visual",
                state=state,
                imu_yaw=imu_yaw,
                dt_since_visual=0.0,
                speed_mps=float(np.linalg.norm(self.velocity_w)),
            )

        extrapolated = self._extrapolate_Tcw(t, imu_yaw)
        if extrapolated is None:
            return None
        Tcw_pred, dt = extrapolated

        return ImuFallbackResult(
            timestamp=t,
            Tcw=Tcw_pred,
            source="imu_fallback",
            state=state,
            imu_yaw=imu_yaw,
            dt_since_visual=dt,
            speed_mps=float(np.linalg.norm(self.velocity_w)),
        )

    def _extrapolate_Tcw(self, t: float, imu_yaw: float | None):
        """Dead-reckon a camera pose (Tcw) from the last visual anchor.

        Constant-velocity translation (from recent visual motion) plus IMU
        yaw-delta rotation. Returns ``(Tcw, dt_since_visual)`` or ``None`` when
        there is no anchor yet. Pure function of current state — does NOT mutate.
        """
        if self.last_visual_t is None or self.last_visual_Tcw is None:
            return None

        dt = max(t - float(self.last_visual_t), 0.0)
        Twc_anchor = np.linalg.inv(self.last_visual_Tcw)
        Twc_pred = Twc_anchor.copy()
        Twc_pred[:3, 3] = Twc_anchor[:3, 3] + self.velocity_w * dt

        if imu_yaw is not None and self.last_visual_yaw is not None:
            Twc_pred[:3, :3] = _yaw_rot(imu_yaw - self.last_visual_yaw) @ Twc_anchor[:3, :3]

        return np.linalg.inv(Twc_pred), dt

    def predict(self, timestamp: float):
        """Read-only IMU-extrapolated camera pose (Tcw 4x4) for ``timestamp``.

        Used by the tracker as a motion prior / carry-through pose during visual
        loss. Returns ``None`` if no visual anchor has been observed yet. Does
        not mutate extrapolator state (unlike :meth:`observe`).
        """
        t = float(timestamp)
        extrapolated = self._extrapolate_Tcw(t, self.yaw_at(t))
        if extrapolated is None:
            return None
        return extrapolated[0]
