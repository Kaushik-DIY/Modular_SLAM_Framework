"""
VisualFrontendService — front-end-only wrapper over ORB-SLAM tracking.

RTAB_inspired_implementation_plan.md §6.2. Exposes a clean "give me the next
keyframe pose + ORB descriptors + tracking status" service. It runs ORB-SLAM's
tracking/local-mapping only; loop closing is NOT part of this service (the
fusion layer owns loop closure via the proposer + verifier).

The tracking backend is injected so the heavy ORB-SLAM stack plugs in during
Mode C (Phase 8) and a lightweight fake drives the Phase 6 test. The backend
must expose ``track(rgb, depth, timestamp) -> keyframe_or_None``; a returned
keyframe exposes ``id``/``kid``, a 4x4 pose (``pose``/``Tcw``/``get_pose``),
and optionally ``keypoints``/``descriptors``/``points3d``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from slam_core.fusion.adapters.types import KeyframeRecord


def _kid(kf) -> int:
    for attr in ("kid", "id"):
        v = getattr(kf, attr, None)
        if v is not None:
            return int(v)
    raise AttributeError("keyframe has neither .kid nor .id")


def _pose_matrix(kf) -> np.ndarray:
    pose = getattr(kf, "pose", None)
    if pose is None:
        getter = getattr(kf, "get_pose", None)
        pose = getter() if getter is not None else getattr(kf, "Tcw", None)
    if pose is None:
        raise AttributeError("keyframe exposes no pose (.pose/.get_pose/.Tcw)")
    if hasattr(pose, "matrix"):
        pose = pose.matrix()
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


class VisualFrontendService:
    def __init__(self, backend):
        self.backend = backend
        self.keyframe_count = 0

    def step(self, rgb, depth, timestamp: float) -> Optional[KeyframeRecord]:
        kf = self.backend.track(rgb, depth, timestamp)
        if kf is None:
            return None
        self.keyframe_count += 1
        return KeyframeRecord(
            id=_kid(kf),
            timestamp=float(timestamp),
            pose=_pose_matrix(kf),
            keypoints=_opt_array(getattr(kf, "keypoints", None)),
            descriptors=_opt_array(getattr(kf, "descriptors", None)),
            points3d=_opt_array(getattr(kf, "points3d", None)),
            is_keyframe=True,
            source=getattr(kf, "source", None),
        )

    def finalize(self) -> None:
        fin = getattr(self.backend, "finalize", None)
        if fin is not None:
            fin()


def _opt_array(v):
    return None if v is None else np.asarray(v)
