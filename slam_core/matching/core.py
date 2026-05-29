from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, List, Optional, Sequence

import numpy as np

from slam_core.common.types import Pose2


@dataclass
class MatchResult:
    """
    Unified output of any scan matcher.
    All swappable matchers must return this structure.
    """
    pose_world: Pose2
    score: float
    success: bool
    method: str

    refine_delta: Optional[np.ndarray] = None
    inliers: Optional[int] = None
    debug: Optional[dict[str, Any]] = None


@dataclass
class BufferedScan:
    """
    One matched scan stored in the rolling buffer.
    Important: stored AFTER matching, not before.
    """
    t: float
    scan_points_local: np.ndarray
    pose_world: Pose2
    score: float


class RollingScanBuffer:
    """
    Rolling buffer of recently matched scans.
    This buffer survives matcher switching and is used to warm-start
    a newly activated matcher.
    """

    def __init__(self, max_size: int = 30):
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self.max_size = int(max_size)
        self._buf: Deque[BufferedScan] = deque(maxlen=self.max_size)

    def push(self, item: BufferedScan) -> None:
        self._buf.append(item)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_size: int) -> bool:
        return len(self._buf) >= int(min_size)

    def latest(self) -> Optional[BufferedScan]:
        if not self._buf:
            return None
        return self._buf[-1]

    def oldest(self) -> Optional[BufferedScan]:
        if not self._buf:
            return None
        return self._buf[0]

    def to_list(self) -> List[BufferedScan]:
        return list(self._buf)

    def iter_items(self) -> Iterable[BufferedScan]:
        return iter(self._buf)


class ScanMatcherBase(ABC):
    """
    Base interface for any scan matching strategy.
    Every matcher must obey the same input/output contract.
    """

    def __init__(self, name: str):
        self.name = str(name)
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @abstractmethod
    def initialize_from_buffer(self, scans: Sequence[BufferedScan]) -> None:
        """
        Warm-start this matcher from a matched-scan buffer.
        Called during activation or switching.
        """
        raise NotImplementedError

    @abstractmethod
    def match(
        self,
        t: float,
        scan_points_local: np.ndarray,
        predicted_pose_world: Pose2,
        odom_pose_world: Optional[Pose2] = None,
    ) -> MatchResult:
        """
        Estimate pose of the current scan.
        """
        raise NotImplementedError

    @abstractmethod
    def update_target(
        self,
        pose_world: Pose2,
        scan_points_local: np.ndarray,
        t: Optional[float] = None,
    ) -> bool:
        """
        Update internal target representation after a scan has been accepted.
        Return:
            bool = whether internal state/map was actually updated
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """
        Clean up memory/state when matcher is deactivated.
        """
        raise NotImplementedError


class MatcherManager:
    """
    Manages:
      - active matcher
      - pending matcher switch
      - rolling matched-scan buffer

    Safe design:
      - current matcher continues running until the new matcher is ready
      - switching happens BETWEEN scans, never during scan processing
    """

    def __init__(
        self,
        active_matcher: ScanMatcherBase,
        rolling_buffer_size: int = 30,
        min_buffer_for_switch: int = 20,
    ):
        self.active_matcher = active_matcher
        self.pending_matcher: Optional[ScanMatcherBase] = None

        self.buffer = RollingScanBuffer(max_size=rolling_buffer_size)
        self.min_buffer_for_switch = int(min_buffer_for_switch)

        self._switch_requested = False
        self._target_matcher_name: Optional[str] = None
        # Scans the ORIGINAL matcher keeps running after a switch request before the
        # switch takes effect (the grace/buffer window). Decremented once per scan in
        # maybe_activate_pending (called exactly once per scan by the adapter).
        self._grace_remaining: int = 0

    @property
    def switch_requested(self) -> bool:
        return self._switch_requested

    @property
    def grace_remaining(self) -> int:
        return self._grace_remaining

    def request_switch(self, new_matcher: ScanMatcherBase, grace_scans: int = 0) -> None:
        """
        Request a switch to ``new_matcher``, effective after ``grace_scans`` more scans
        (during which the current matcher keeps running). The actual switch additionally
        waits until the rolling matched-scan buffer is ready, then warm-starts the new
        matcher from it.

        Edge handling:
          - switching to the already-active matcher cancels any pending switch (no-op);
          - re-requesting the SAME pending target is ignored (the countdown is not reset);
          - requesting a DIFFERENT target while one is pending replaces it and resets the
            countdown (last request wins).
        """
        if new_matcher is self.active_matcher:
            # Cancel any in-flight switch back toward where we already are.
            self.pending_matcher = None
            self._target_matcher_name = None
            self._switch_requested = False
            self._grace_remaining = 0
            return
        if self._switch_requested and new_matcher is self.pending_matcher:
            return
        self.pending_matcher = new_matcher
        self._target_matcher_name = new_matcher.name
        self._switch_requested = True
        self._grace_remaining = max(0, int(grace_scans))

    def maybe_activate_pending(self) -> bool:
        """
        Try to activate pending matcher from rolling buffer.
        Returns True if switch was completed.
        """
        if not self._switch_requested or self.pending_matcher is None:
            return False

        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            return False

        if not self.buffer.is_ready(self.min_buffer_for_switch):
            return False

        scans = self.buffer.to_list()
        self.pending_matcher.initialize_from_buffer(scans)

        old_matcher = self.active_matcher
        self.active_matcher = self.pending_matcher
        self.pending_matcher = None
        self._switch_requested = False
        self._target_matcher_name = None

        old_matcher.shutdown()
        return True

    def match(
        self,
        t: float,
        scan_points_local: np.ndarray,
        predicted_pose_world: Pose2,
        odom_pose_world: Optional[Pose2] = None,
    ) -> MatchResult:
        return self.active_matcher.match(
            t=t,
            scan_points_local=scan_points_local,
            predicted_pose_world=predicted_pose_world,
            odom_pose_world=odom_pose_world,
        )

    def update_active_target(
        self,
        pose_world: Pose2,
        scan_points_local: np.ndarray,
        t: Optional[float] = None,
    ) -> bool:
        return self.active_matcher.update_target(
            pose_world=pose_world,
            scan_points_local=scan_points_local,
            t=t,
        )

    def push_buffered_scan(
        self,
        t: float,
        scan_points_local: np.ndarray,
        pose_world: Pose2,
        score: float,
    ) -> None:
        self.buffer.push(
            BufferedScan(
                t=float(t),
                scan_points_local=scan_points_local,
                pose_world=pose_world,
                score=float(score),
            )
        )