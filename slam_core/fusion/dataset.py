"""
FusionDataset — TUM RGB-D frames paired with synthetic 2D LiDAR.

RTAB_inspired_implementation_plan.md §11.3. Wraps the existing TUM loader
(``visual_slam/orbslam/io/tum_rgbd.py``) and the synthetic-LiDAR utility, and
yields ``FusionFrame`` records that a ``SoftSync`` instance consumes.

The RGB and scan timestamps differ: the scan is synthesized from the depth
image, so its timestamp is the depth file's own timestamp (parsed from the TUM
filename), which the associator matched to the RGB frame within ~20 ms. This
gives SoftSync a realistic, non-trivial sub-tolerance offset to resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from visual_slam.orbslam.io.tum_rgbd import (
    load_tum_rgbd_associations,
    make_tum_rgbd_camera,
)
from slam_core.fusion.lidar_synth import synthesize_2d_scan


@dataclass
class FusionFrame:
    """One raw dataset sample, before synchronization."""

    rgb_t: float
    rgb: np.ndarray              # (H, W, 3) BGR uint8
    depth: np.ndarray            # (H, W) float metres
    scan_t: float
    scan: Optional[np.ndarray]   # (M, 2) synthetic LiDAR, sensor frame


class FusionDataset:
    def __init__(
        self,
        dataset_path: str | Path,
        num_beams: int = 360,
        noise_sigma: float = 0.02,
        range_min: float = 0.3,
        range_max: float = 10.0,
        seed: Optional[int] = 0,
    ):
        self.dataset_path = Path(dataset_path)
        self.frames = load_tum_rgbd_associations(self.dataset_path)
        if not self.frames:
            raise RuntimeError(f"No RGB-D associations found under {self.dataset_path}")

        self.camera = make_tum_rgbd_camera(self.dataset_path.name)
        self.K = np.array(
            [
                [self.camera.fx, 0.0, self.camera.cx],
                [0.0, self.camera.fy, self.camera.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # camera.depth_factor converts raw uint16 -> metres (TUM: 1/5000).
        self.depth_factor = float(self.camera.depth_factor)

        self.num_beams = num_beams
        self.noise_sigma = noise_sigma
        self.range_min = range_min
        self.range_max = range_max
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frames)

    def _depth_timestamp(self, depth_path: Path, fallback: float) -> float:
        # TUM depth filenames are the timestamp, e.g. "1305031910.771502.png".
        try:
            return float(depth_path.stem)
        except ValueError:
            return fallback

    def _load_depth_metres(self, depth_path: Path) -> np.ndarray:
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read depth image: {depth_path}")
        return raw.astype(np.float64) * self.depth_factor

    def __iter__(self) -> Iterator[FusionFrame]:
        yield from self.iter_frames()

    def iter_frames(self, max_frames: Optional[int] = None) -> Iterator[FusionFrame]:
        n = len(self.frames) if max_frames is None else min(max_frames, len(self.frames))
        for i in range(n):
            frame = self.frames[i]
            rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Could not read RGB image: {frame.rgb_path}")
            depth = self._load_depth_metres(frame.depth_path)
            scan = synthesize_2d_scan(
                depth,
                self.K,
                num_beams=self.num_beams,
                range_min=self.range_min,
                range_max=self.range_max,
                noise_sigma=self.noise_sigma,
                rng=self._rng,
            )
            scan_t = self._depth_timestamp(frame.depth_path, frame.timestamp)
            yield FusionFrame(
                rgb_t=float(frame.timestamp),
                rgb=rgb,
                depth=depth,
                scan_t=scan_t,
                scan=scan,
            )
