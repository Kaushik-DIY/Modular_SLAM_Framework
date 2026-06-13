"""
LidarFrontendService — wrapper exposing the LiDAR front-end as a service.

RTAB_inspired_implementation_plan.md §6.2. Exposes a clean "give me the next
per-scan relative pose + raw scan" interface over the existing LiDAR front-end
(``scan_to_submap`` matcher via the Hector/Carto adapter). Underlying behaviour
is unchanged; in fusion modes the front-end's own g2o PGO is simply not driven
(the fusion graph owns optimization).

The backend is injected. It must expose
``process_scan(scan, timestamp) -> result`` (or ``step``), where the result
exposes a world ``pose`` (Pose2) and optionally ``is_keyframe``. The relative
pose between consecutive steps is computed here in SE(2).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import pose_compose, pose_inverse
from slam_core.fusion.adapters.types import FrontendStep


def _as_pose2(p) -> Pose2:
    if isinstance(p, Pose2):
        return p
    return Pose2(float(p.x), float(p.y), float(p.theta))


class LidarFrontendService:
    def __init__(self, backend):
        self.backend = backend
        self._prev_pose: Optional[Pose2] = None
        self.step_count = 0

    def step(self, scan, timestamp: float) -> FrontendStep:
        proc = getattr(self.backend, "process_scan", None) or getattr(self.backend, "step")
        result = proc(scan, timestamp)

        pose = _as_pose2(getattr(result, "pose", result))
        is_kf = bool(getattr(result, "is_keyframe", False))

        if self._prev_pose is None:
            rel = Pose2(0.0, 0.0, 0.0)
        else:
            rel = pose_compose(pose_inverse(self._prev_pose), pose)
        self._prev_pose = pose
        self.step_count += 1

        return FrontendStep(
            timestamp=float(timestamp),
            pose=pose,
            rel_pose=rel,
            scan=None if scan is None else np.asarray(scan),
            is_keyframe=is_kf,
        )

    def finalize(self) -> None:
        fin = getattr(self.backend, "finalize", None)
        if fin is not None:
            fin()
