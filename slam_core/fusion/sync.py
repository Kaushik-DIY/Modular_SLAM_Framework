"""
SoftSync — timestamp matching of RGB-D frames against 2D LiDAR scans.

RTAB_inspired_implementation_plan.md §11.1. RGB-D frames and scans are pushed
with their own timestamps. ``try_pop`` returns the oldest RGB-D frame paired
with the nearest scan inside a ±tolerance window; if no scan falls in the
window the sample carries ``scan=None`` (Mode C still proceeds, LiDAR-side
verification is skipped — CLAUDE.md §2.12).

Streaming safety: a frame at time ``t0`` is only finalized once a scan with
timestamp ≥ ``t0 + tolerance`` has arrived (so no closer future scan can
exist), unless ``flush=True`` drains the remaining buffered frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class FusedSample:
    """One synchronized output of SoftSync."""

    t: float
    rgb: np.ndarray
    depth: np.ndarray
    scan: Optional[np.ndarray]  # (M, 2) in sensor frame, or None


class SoftSync:
    def __init__(self, tolerance_s: float = 0.050):
        if tolerance_s <= 0:
            raise ValueError("tolerance_s must be positive")
        self.tol = float(tolerance_s)
        self._rgbd: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque()
        self._scans: List[Tuple[float, np.ndarray]] = []  # kept sorted by time

    def push_rgbd(self, t: float, rgb: np.ndarray, depth: np.ndarray) -> None:
        self._rgbd.append((float(t), rgb, depth))

    def push_scan(self, t: float, scan: np.ndarray) -> None:
        self._scans.append((float(t), scan))
        # Buffers are small (sync window); insertion sort keeps order cheaply.
        self._scans.sort(key=lambda s: s[0])

    def _latest_scan_time(self) -> Optional[float]:
        return self._scans[-1][0] if self._scans else None

    def try_pop(self, flush: bool = False) -> Optional[FusedSample]:
        """Return the next finalizable FusedSample, or None if not ready.

        With ``flush=False`` a frame is held until a scan past its window has
        been seen. With ``flush=True`` the oldest buffered frame is finalized
        immediately (used to drain at end-of-stream).
        """
        if not self._rgbd:
            return None

        t0, rgb, depth = self._rgbd[0]

        if not flush:
            latest = self._latest_scan_time()
            if latest is None or latest < t0 + self.tol:
                return None  # a closer scan might still arrive

        self._rgbd.popleft()
        scan = self._match_scan(t0)
        self._prune_scans(t0)
        return FusedSample(t0, rgb, depth, scan)

    def _match_scan(self, t0: float) -> Optional[np.ndarray]:
        best: Optional[np.ndarray] = None
        best_dt = float("inf")
        for ts, scan in self._scans:
            dt = abs(ts - t0)
            if dt <= self.tol and dt < best_dt:
                best_dt = dt
                best = scan
        return best

    def _prune_scans(self, t0: float) -> None:
        # RGB-D frames are finalized in increasing time order, so scans older
        # than (t0 - tol) can never match a future frame.
        cutoff = t0 - self.tol
        self._scans = [(ts, s) for ts, s in self._scans if ts >= cutoff]

    def __len__(self) -> int:
        """Number of RGB-D frames still buffered (not yet popped)."""
        return len(self._rgbd)
