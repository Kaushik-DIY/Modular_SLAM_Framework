"""Native windowed C++ VO front-end wrapper (fusion v3).

Drives fusion_core.VoFrontend per RGB-D frame. IMU usage (v3.6): the IMU is
used ONLY as dead-reckoning during tracking dropouts — per-frame IMU priors
measurably hurt projection matching on this robot (camera-mount tilt, V3.3
A/B), but during a dropout there is nothing to match, and integrating the
real measured rotation beats freezing the last visual velocity (which keeps
"turning" at a stale rate through 2-4 s failure bursts). Two parameters the
extrinsic file would normally give are learned online from data instead:

- yaw SIGN: correlate VO heading deltas with IMU yaw deltas during good
  tracking; commit the sign once enough correlated rotation has been seen.
- up AXIS: PCA over keyframe positions (the robot drives on a plane); the
  smallest-variance direction is gravity in the camera world frame, which
  the mount tilt moves away from the naive (0,-1,0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

import fusion_core as fc


@dataclass
class NativeKeyframe:
    id: int
    stamp: float
    Twc: np.ndarray            # 4x4 camera pose (world from camera)
    kpts: np.ndarray           # (N,2) f32
    des: np.ndarray            # (N,32) u8
    pts3d_cam: np.ndarray      # (N,3) f32, NaN where no depth
    state: "fc.VoState"
    prev_Twc: Optional[np.ndarray] = None  # BA-refined pose of prev emitted KF


def _rot_about_axis(u: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about unit axis u."""
    c, s = math.cos(angle), math.sin(angle)
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(u, u)


class NativeOrbFrontend:
    def __init__(self, dataset: Path, imu_path: Optional[str] = None,
                 n_features: int = 2000, depth_max: float = 4.0,
                 imu_dropout: bool = True, vo_overrides: Optional[dict] = None,
                 imu_buffer=None):
        sc = yaml.safe_load(open(Path(dataset) / "sensor_config.yaml"))["camera"]
        cfg = fc.VoConfig()
        cfg.fx, cfg.fy = float(sc["fx"]), float(sc["fy"])
        cfg.cx, cfg.cy = float(sc["cx"]), float(sc["cy"])
        cfg.depth_factor = float(sc.get("depth_map_factor", 1000.0))
        # camera.depth_threshold equivalent: bf * th_depth / fx with the virtual
        # 8cm baseline and th=40 => ~3.2 m; allow a little more for the lab.
        cfg.depth_max = float(depth_max)
        cfg.n_features = int(n_features)
        # optional VoConfig overrides (tracking/keyframe/BA knobs) for tuning
        for k, v in (vo_overrides or {}).items():
            if not hasattr(cfg, k):
                raise AttributeError(f"VoConfig has no field {k!r}")
            setattr(cfg, k, v)
        self._imu_dropout_enabled = bool(imu_dropout)
        self.cfg = cfg
        self.vo = fc.VoFrontend(cfg)
        self.K = np.array([[cfg.fx, 0, cfg.cx], [0, cfg.fy, cfg.cy], [0, 0, 1]])

        # IMU for the dropout prior: a live buffer (online/ROS) takes precedence
        # over the imu.csv file; both expose yaw_at(t).
        self._imu = imu_buffer
        if self._imu is None and imu_path and Path(imu_path).exists():
            from visual_slam.orbslam.imu_fallback import ImuFallbackExtrapolator
            self._imu = ImuFallbackExtrapolator(imu_path)
        self._last_result = None
        self._anchor = None          # (t, Twc, imu_yaw) at last well-tracked frame
        self._prev_Twc = None
        self._vel_t = np.zeros(3)    # per-frame world translation velocity

        # online IMU calibration state
        self._up = np.array([0.0, -1.0, 0.0])   # gravity-up in camera world
        self._e1 = None                           # in-plane heading basis
        self._e2 = None
        self._yaw_sign = 0.0                      # 0 until learned
        self._sign_corr = 0.0                     # sum(vo_dyaw * imu_dyaw)
        self._sign_energy = 0.0                   # sum(|vo_dyaw * imu_dyaw|)
        self._prev_ok = None                      # (heading, imu_yaw) last OK frame
        self._kf_positions: list = []             # OK keyframe positions for PCA

    # ---- online calibration helpers -------------------------------------

    def _heading(self, R: np.ndarray) -> Optional[float]:
        """Signed bearing of the camera forward axis in the ground plane."""
        f = R[:, 2] - self._up * float(self._up @ R[:, 2])
        n = np.linalg.norm(f)
        if n < 1e-6:
            return None
        f /= n
        if self._e1 is None:
            self._e1 = f.copy()
            self._e2 = np.cross(self._up, self._e1)
        return math.atan2(float(f @ self._e2), float(f @ self._e1))

    def _update_calibration(self, t: float, Twc: np.ndarray) -> None:
        imu_yaw = self._imu.yaw_at(t) if self._imu is not None else None
        h = self._heading(Twc[:3, :3])
        if imu_yaw is not None and h is not None and self._prev_ok is not None:
            h0, y0 = self._prev_ok
            dvo = math.atan2(math.sin(h - h0), math.cos(h - h0))
            dimu = math.atan2(math.sin(imu_yaw - y0), math.cos(imu_yaw - y0))
            if abs(dvo) < 0.3 and abs(dimu) < 0.3:   # same-frame-pair deltas only
                self._sign_corr += dvo * dimu
                self._sign_energy += abs(dvo * dimu)
        if imu_yaw is not None and h is not None:
            self._prev_ok = (h, imu_yaw)
        # commit the sign after enough correlated rotation (rad^2)
        if self._yaw_sign == 0.0 and self._sign_energy > 0.05 \
                and abs(self._sign_corr) > 0.6 * self._sign_energy:
            self._yaw_sign = math.copysign(1.0, self._sign_corr)

    def _update_up_axis(self, pos: np.ndarray) -> None:
        self._kf_positions.append(pos.copy())
        n = len(self._kf_positions)
        if n < 60 or n % 30 != 0:
            return
        P = np.asarray(self._kf_positions)
        span = P.max(axis=0) - P.min(axis=0)
        if np.sort(span)[-2] < 1.5:               # need 2D spread in the plane
            return
        C = np.cov((P - P.mean(axis=0)).T)
        w, V = np.linalg.eigh(C)
        u = V[:, 0]                                # smallest-variance direction
        if u @ np.array([0.0, -1.0, 0.0]) < 0:
            u = -u
        self._up = u / np.linalg.norm(u)
        self._e1 = None                            # rebasis heading on new up

    def _dropout_prior(self, t: float) -> Optional[np.ndarray]:
        """IMU dead-reckoned pose during a tracking dropout: real measured
        rotation about the calibrated up axis + constant-velocity translation."""
        if not self._imu_dropout_enabled:
            return None
        if self._imu is None or self._anchor is None or self._yaw_sign == 0.0:
            return None
        t_a, Twc_a, yaw_a = self._anchor
        yaw_now = self._imu.yaw_at(t)
        if yaw_now is None or yaw_a is None:
            return None
        dpsi = self._yaw_sign * math.atan2(math.sin(yaw_now - yaw_a),
                                           math.cos(yaw_now - yaw_a))
        prior = np.eye(4)
        prior[:3, :3] = _rot_about_axis(self._up, dpsi) @ Twc_a[:3, :3]
        n_frames = max(1.0, round((t - t_a) * 15.0))
        prior[:3, 3] = Twc_a[:3, 3] + self._vel_t * n_frames
        return prior

    # ---- main entry -------------------------------------------------------

    def track(self, rgb: np.ndarray, depth: np.ndarray, t: float
              ) -> tuple[np.ndarray, "fc.VoState", Optional[NativeKeyframe]]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
        # IMU prior ONLY while tracking is down (last frame under the reinit
        # threshold). Never during healthy tracking — that A/B-tested worse.
        prior = None
        last = self._last_result
        if last is not None and last.n_inliers < self.cfg.min_inliers_reinit:
            prior = self._dropout_prior(t)
        res = self.vo.track(gray, np.ascontiguousarray(depth, np.uint16), t,
                            prior_Twc=prior)
        self._last_result = res
        Twc = np.asarray(res.Twc)

        well_tracked = (res.state == fc.VoState.INIT
                        or (res.state == fc.VoState.OK
                            and res.n_inliers >= self.cfg.min_inliers_reinit))
        if well_tracked:
            self._anchor = (t, Twc.copy(),
                            self._imu.yaw_at(t) if self._imu is not None else None)
            self._update_calibration(t, Twc)
            if self._prev_Twc is not None:
                self._vel_t = Twc[:3, 3] - self._prev_Twc[:3, 3]
        self._prev_Twc = Twc.copy()

        kf = None
        if res.new_keyframe:
            payload = self.vo.keyframe_payload(res.kf_id)
            if payload is not None:
                kpts, des, pts3d = payload
                prev = np.asarray(res.prev_Twc) if res.has_prev else None
                kf = NativeKeyframe(id=res.kf_id, stamp=t, Twc=Twc,
                                    kpts=kpts, des=des, pts3d_cam=pts3d,
                                    state=res.state, prev_Twc=prev)
                if res.state == fc.VoState.OK:
                    self._update_up_axis(Twc[:3, 3])
        return Twc, res.state, kf

    @property
    def last_track_ms(self) -> float:
        return self._last_result.track_ms if self._last_result else 0.0

    @property
    def imu_calibration(self) -> dict:
        return dict(yaw_sign=self._yaw_sign,
                    up_axis=self._up.round(4).tolist())
