"""lab_hybrid dataset streams for fusion v2.

Reuses the existing loaders (`read_lab_hybrid_lidar_csv`, association files,
`ranges_to_points`) and exposes two iterators:
  * lidar_stream():  (t, scan_xy float32 (N,2))           — LiDAR-led modes
  * rgbd_stream():   (t, rgb_path, depth_path, scan_or_None) — visual-led modes
Scan points are produced with the lab geometry (full 360°, 909 beams, 16 m).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from carto.local_slam.range_to_points import ranges_to_points
from slam_core.dataio.lab_carmen import read_lab_hybrid_lidar_csv

# Fallback geometry (the original lab scanner), used when the dataset does not
# declare a `lidar:` section in sensor_config.yaml.
LAB_ANGLE_MIN = -3.1415927410125732
LAB_ANGLE_INC = 0.006919807754456997
LAB_RANGE_MIN = 0.10
LAB_RANGE_MAX = 16.0


class LabHybridStream:
    def __init__(self, dataset: Path, sync_tolerance_s: float = 0.05):
        self.root = Path(dataset)
        self.sync_tol = float(sync_tolerance_s)
        self._scans = read_lab_hybrid_lidar_csv(str(self.root / "lidar" / "scans.csv"))
        self._scan_ts = np.array([s["t"] for s in self._scans])
        # LiDAR geometry from the dataset itself (V4.3 portability); falls back
        # to the lab constants for dataset copies without a lidar section.
        self.angle_min, self.angle_inc = LAB_ANGLE_MIN, LAB_ANGLE_INC
        self.range_min, self.range_max = LAB_RANGE_MIN, LAB_RANGE_MAX
        sc_path = self.root / "sensor_config.yaml"
        if sc_path.exists():
            import yaml
            lid = (yaml.safe_load(open(sc_path)) or {}).get("lidar") or {}
            self.angle_min = float(lid.get("angle_min", self.angle_min))
            self.angle_inc = float(lid.get("angle_increment", self.angle_inc))
            self.range_min = float(lid.get("range_min", self.range_min))
            self.range_max = float(lid.get("range_max", self.range_max))

    # ------------------------------------------------------------------
    def num_scans(self) -> int:
        return len(self._scans)

    def scan_points(self, idx: int) -> np.ndarray:
        pts = ranges_to_points(self._scans[idx]["ranges"], self.angle_min,
                               self.angle_inc, self.range_min, self.range_max)
        return np.ascontiguousarray(pts, dtype=np.float32)

    def lidar_stream(self, max_scans: int = 0) -> Iterator[Tuple[float, np.ndarray]]:
        n = len(self._scans) if max_scans <= 0 else min(max_scans, len(self._scans))
        for i in range(n):
            yield float(self._scans[i]["t"]), self.scan_points(i)

    # ------------------------------------------------------------------
    def rgbd_entries(self):
        """(t, rgb_path, depth_path) from associations_rgbd.txt."""
        out = []
        for line in open(self.root / "associations_rgbd.txt"):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            out.append((float(p[0]), self.root / p[1], self.root / p[3]))
        return out

    def nearest_scan(self, t: float) -> Optional[np.ndarray]:
        """Scan within the sync window of t, or None (CLAUDE.md §2.12 soft sync)."""
        i = int(np.searchsorted(self._scan_ts, t))
        best, best_dt = -1, self.sync_tol
        for j in (i - 1, i):
            if 0 <= j < len(self._scan_ts):
                dt = abs(float(self._scan_ts[j]) - t)
                if dt <= best_dt:
                    best, best_dt = j, dt
        return self.scan_points(best) if best >= 0 else None

    def rgbd_stream(self, max_frames: int = 0):
        entries = self.rgbd_entries()
        if max_frames > 0:
            entries = entries[:max_frames]
        for t, rgb_p, depth_p in entries:
            yield t, rgb_p, depth_p, self.nearest_scan(t)

    def nearest_rgbd(self, t: float):
        """(rgb_path, depth_path) within the sync window of t, or None."""
        if not hasattr(self, "_rgbd_cache"):
            self._rgbd_cache = self.rgbd_entries()
            self._rgbd_ts = np.array([e[0] for e in self._rgbd_cache])
        i = int(np.searchsorted(self._rgbd_ts, t))
        best, best_dt = -1, self.sync_tol
        for j in (i - 1, i):
            if 0 <= j < len(self._rgbd_ts):
                dt = abs(float(self._rgbd_ts[j]) - t)
                if dt <= best_dt:
                    best, best_dt = j, dt
        if best < 0:
            return None
        return self._rgbd_cache[best][1], self._rgbd_cache[best][2]
