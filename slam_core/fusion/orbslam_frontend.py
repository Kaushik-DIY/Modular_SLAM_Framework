"""
Real ORB-SLAM front-end backend for Mode C (replaces the OrbRgbdVoBackend stand-in).

This is the production Mode C visual front-end the plan always intended
(CLAUDE.md §2.1, §3): the existing ORB-SLAM RGB-D pipeline driven in-process as
an odometry/keyframe service, with **its own loop closing disabled**
(``enable_loop_closing=False``) so it runs *propose-only* — exactly how
RTAB-Map consumes an external odometry source (ORB-SLAM2 is one of its seven
integrated odometry libraries, JFR'19 §3.1) and lets the fusion layer own loop
closure.

What it provides per frame:
- real ORB-SLAM tracking + local mapping + **local bundle adjustment** (good,
  drift-bounded keyframe poses — the thing the crude VO stand-in lacked),
- new ``KeyFrame`` objects (pose ``Twc``, ORB descriptors, keypoints),
- the live DBoW3 ``KeyFrameDatabase`` for the real appearance loop detector.

No ORB-SLAM source is modified — this is a pure consumer of the public Slam API.

NOTE on depth: ORB-SLAM expects RAW uint16 depth (it applies the camera's
``depth_map_factor`` internally), so callers must feed raw depth here, not metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from visual_slam.orbslam.slam import Slam, SensorType
from visual_slam.orbslam.slam.loop_detector import LoopDetector


@dataclass
class _OrbKeyframe:
    """Lightweight record the VisualFrontendService normalizes; carries the
    live KeyFrame in ``source`` so the real LoopDetector can score it."""
    id: int
    pose: np.ndarray            # 4x4 T_world_cam (Twc)
    keypoints: np.ndarray
    descriptors: np.ndarray
    source: object              # the ORB-SLAM KeyFrame


class OrbSlamFrontendBackend:
    def __init__(self, camera, sensor_type: SensorType = SensorType.RGBD,
                 feature_backend: str = "pyslam_orb2"):
        # pyslam_orb2 = the real ORB-SLAM2 quadtree-distributed extractor; far
        # more robust than the opencv_orb fallback (see Mode-A lab diagnosis).
        ft_cfg = None if feature_backend in (None, "auto") else {"extractor_backend": feature_backend}
        self.slam = Slam(camera=camera, sensor_type=sensor_type,
                         feature_tracker_config=ft_cfg,
                         enable_loop_closing=False, headless=True)
        self.frame_idx = -1
        self._seen = set()
        self._buffer: List[_OrbKeyframe] = []

    @property
    def keyframe_database(self):
        return self.slam.keyframe_database

    def make_loop_detector(self) -> LoopDetector:
        return LoopDetector(self.slam.keyframe_database)

    def track(self, rgb, depth, t) -> Optional[_OrbKeyframe]:
        """Feed one RGB-D frame (raw uint16 depth); return a new keyframe or None."""
        self.frame_idx += 1
        self.slam.track(img=rgb, img_right=None, depth=depth,
                        img_id=self.frame_idx, timestamp=t)
        # Non-threaded local mapping: drain the queue so keyframes + local BA run.
        while self.slam.local_mapping.queue_size() > 0:
            self.slam.local_mapping.step()

        for kid, kf in self.slam.map.keyframes_map.items():
            if kid in self._seen or kf.is_bad():
                continue
            self._seen.add(kid)
            self._buffer.append(self._record(kf))

        return self._buffer.pop(0) if self._buffer else None

    def flush(self) -> List[_OrbKeyframe]:
        out, self._buffer = self._buffer, []
        return out

    @staticmethod
    def _record(kf) -> _OrbKeyframe:
        Tw = kf.Twc()
        Tw = Tw.matrix() if hasattr(Tw, "matrix") else np.asarray(Tw, dtype=float)
        kpts = np.array([k.pt for k in kf.kps], dtype=np.float64) if len(kf.kps) else np.zeros((0, 2))
        return _OrbKeyframe(id=int(kf.id), pose=np.asarray(Tw, dtype=float).reshape(4, 4),
                            keypoints=kpts, descriptors=kf.des, source=kf)
