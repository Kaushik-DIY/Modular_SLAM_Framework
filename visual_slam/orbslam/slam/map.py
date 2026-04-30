"""
=============================================================================
visual_slam/orbslam/slam/map.py

pySLAM-aligned Map subset for ORB/RGB-D SLAM.

Reference:
- pySLAM: pyslam/slam/map.py

Scope retained:
- recent frame buffer
- ordered keyframes
- map points
- keyframe origins
- frame-id -> keyframe lookup
- max frame/keyframe/point ID counters
- add/remove/get methods
- local covisibility map

Excluded for now:
- serialization/reload
- viewer drawing arrays
- semantic/dense map hooks
- add_points triangulation path
=============================================================================
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Iterable, Optional

import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.frame import Frame
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map_point import MapPoint

kMaxLenFrameDeque = 20


class OrderedSetLite:
    """
    Minimal ordered-set replacement.

    pySLAM uses ordered_set.OrderedSet. To keep this package self-contained, this
    small container preserves insertion order and supports the subset used by the
    ORB-SLAM path.
    """

    def __init__(self, values: Optional[Iterable] = None):
        self._items = []
        if values is not None:
            for value in values:
                self.add(value)

    def add(self, value) -> None:
        if value not in self._items:
            self._items.append(value)

    def discard(self, value) -> None:
        try:
            self._items.remove(value)
        except ValueError:
            pass

    def remove(self, value) -> None:
        self._items.remove(value)

    def clear(self) -> None:
        self._items.clear()

    def copy(self) -> "OrderedSetLite":
        return OrderedSetLite(self._items)

    def to_list(self) -> list:
        return list(self._items)

    def __contains__(self, value) -> bool:
        return value in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return OrderedSetLite(self._items[item])
        return self._items[item]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"OrderedSetLite({self._items!r})"


@dataclass
class ReloadedSessionMapInfo:
    num_keyframes: int
    num_points: int
    max_point_id: int
    max_frame_id: int
    max_keyframe_id: int


class MapStateData:
    """Small pySLAM-compatible map state container for later visualization."""

    def __init__(self):
        self.poses = []
        self.pose_timestamps = []
        self.fov_centers = []
        self.fov_centers_colors = []
        self.points = []
        self.colors = []
        self.semantic_colors = []
        self.covisibility_graph = []
        self.spanning_tree = []
        self.loops = []


class LocalCovisibilityMap:
    """
    pySLAM-style local map based on covisibility.

    The local map consists of:
    - the reference keyframe
    - its best covisible keyframes
    - map points observed by those keyframes
    """

    def __init__(self, map: "Map"):
        self.map = map
        self.reference_keyframe = None
        self.local_keyframes = OrderedSetLite()
        self.local_points = OrderedSetLite()

    def reset(self) -> None:
        self.reference_keyframe = None
        self.local_keyframes.clear()
        self.local_points.clear()

    def reset_session(self, keyframes_to_remove=None, points_to_remove=None) -> None:
        if keyframes_to_remove:
            for kf in keyframes_to_remove:
                self.local_keyframes.discard(kf)
        if points_to_remove:
            for p in points_to_remove:
                self.local_points.discard(p)

    def update(self, reference_keyframe: KeyFrame, num_best: int = Parameters.kNumBestCovisibilityKeyFrames):
        self.reset()
        self.reference_keyframe = reference_keyframe

        if reference_keyframe is None:
            return

        self.local_keyframes.add(reference_keyframe)

        for kf in reference_keyframe.get_best_covisible_keyframes(num_best):
            if kf is not None and not kf.is_bad():
                self.local_keyframes.add(kf)

        for kf in self.local_keyframes:
            for p in kf.get_matched_good_points():
                if p is not None and not p.is_bad():
                    self.local_points.add(p)

    def get_keyframes(self) -> OrderedSetLite:
        return self.local_keyframes.copy()

    def get_points(self) -> OrderedSetLite:
        return self.local_points.copy()

    def num_keyframes(self) -> int:
        return len(self.local_keyframes)

    def num_points(self) -> int:
        return len(self.local_points)


class Map:
    """pySLAM-like sparse SLAM map."""

    def __init__(self):
        self._lock = RLock()
        self._update_lock = RLock()

        self.frames: deque[Frame] = deque(maxlen=kMaxLenFrameDeque)
        self.keyframes: OrderedSetLite = OrderedSetLite()
        self.points: OrderedSetLite = OrderedSetLite()
        self.keyframe_origins: OrderedSetLite = OrderedSetLite()

        # pySLAM map: frame id -> keyframe
        self.keyframes_map: dict[int, KeyFrame] = {}

        self.max_point_id = 0
        self.max_frame_id = 0
        self.max_keyframe_id = 0

        self.reloaded_session_map_info: ReloadedSessionMapInfo | None = None
        self.local_map = LocalCovisibilityMap(map=self)
        self.viewer_scale = -1

    @property
    def lock(self):
        return self._lock

    @property
    def update_lock(self):
        return self._update_lock

    def is_reloaded(self) -> bool:
        return self.reloaded_session_map_info is not None

    def reset(self) -> None:
        with self._lock:
            with self._update_lock:
                self.frames.clear()
                self.keyframes.clear()
                self.points.clear()
                self.keyframe_origins.clear()
                self.keyframes_map.clear()
                self.local_map.reset()
                self.max_point_id = 0
                self.max_frame_id = 0
                self.max_keyframe_id = 0

    def reset_session(self) -> None:
        # Full reload-session behavior will be expanded when map persistence is ported.
        self.reset()

    def delete(self) -> None:
        with self._lock:
            for frame in self.frames:
                frame.reset_points()
            for keyframe in self.keyframes:
                keyframe.reset_points()

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------

    def get_points(self) -> OrderedSetLite:
        with self._lock:
            return self.points.copy()

    def num_points(self) -> int:
        with self._lock:
            return len(self.points)

    def add_point(self, point: MapPoint) -> int:
        with self._lock:
            ret = self.max_point_id
            point.id = ret
            point.map = self
            self.max_point_id += 1
            self.points.add(point)
            return ret

    def remove_point(self, point: MapPoint) -> None:
        with self._lock:
            self.points.discard(point)
            if getattr(point, "map", None) is self:
                point.map = None

    def remove_point_no_lock(self, point: MapPoint) -> None:
        self.points.discard(point)
        if getattr(point, "map", None) is self:
            point.map = None

    # Compatibility alias.
    def add_map_point(self, point: MapPoint) -> int:
        return self.add_point(point)

    def remove_map_point(self, point: MapPoint) -> None:
        self.remove_point(point)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def get_frame(self, idx: int):
        with self._lock:
            try:
                return self.frames[idx]
            except Exception:
                return None

    def get_frames(self):
        with self._lock:
            return self.frames.copy()

    def num_frames(self) -> int:
        with self._lock:
            return len(self.frames)

    def add_frame(self, frame: Frame, override_id: bool = False) -> int:
        with self._lock:
            ret = frame.id
            if override_id:
                ret = self.max_frame_id
                frame.id = ret
                self.max_frame_id += 1
            else:
                self.max_frame_id = max(self.max_frame_id, frame.id + 1)

            self.frames.append(frame)
            return ret

    def remove_frame(self, frame: Frame) -> None:
        with self._lock:
            try:
                self.frames.remove(frame)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Keyframes
    # ------------------------------------------------------------------

    def get_keyframes(self) -> OrderedSetLite:
        with self._lock:
            return self.keyframes.copy()

    def get_first_keyframe(self):
        with self._lock:
            if len(self.keyframes) == 0:
                return None
            return self.keyframes[0]

    def get_last_keyframe(self):
        with self._lock:
            if len(self.keyframes) == 0:
                return None
            return self.keyframes[-1]

    def get_last_keyframes(self, local_window_size: int = Parameters.kLocalBAWindowSize) -> OrderedSetLite:
        with self._lock:
            return self.keyframes[-int(local_window_size):]

    def num_keyframes(self) -> int:
        with self._lock:
            return len(self.keyframes)

    def num_keyframes_session(self) -> int:
        with self._lock:
            if self.reloaded_session_map_info is not None:
                return len(self.keyframes) - self.reloaded_session_map_info.num_keyframes
            return len(self.keyframes)

    def add_keyframe(self, keyframe: KeyFrame) -> int:
        with self._lock:
            assert keyframe.is_keyframe

            ret = self.max_keyframe_id
            keyframe.kid = ret
            keyframe.is_keyframe = True
            keyframe.map = self

            self.keyframes.add(keyframe)
            self.keyframes_map[keyframe.id] = keyframe
            self.max_keyframe_id += 1

            if ret == 0:
                self.keyframe_origins.add(keyframe)

            return ret

    def remove_keyframe(self, keyframe: KeyFrame) -> None:
        with self._lock:
            assert keyframe.is_keyframe
            self.keyframes.discard(keyframe)
            self.keyframe_origins.discard(keyframe)
            self.keyframes_map.pop(keyframe.id, None)
            if getattr(keyframe, "map", None) is self:
                keyframe.map = None

    def get_keyframe_by_frame_id(self, frame_id: int):
        with self._lock:
            return self.keyframes_map.get(int(frame_id), None)

    # ------------------------------------------------------------------
    # Local map
    # ------------------------------------------------------------------

    def update_local_map(self, reference_keyframe: KeyFrame) -> None:
        with self._lock:
            self.local_map.update(reference_keyframe)

    def get_local_keyframes(self) -> OrderedSetLite:
        return self.local_map.get_keyframes()

    def get_local_points(self) -> OrderedSetLite:
        return self.local_map.get_points()

    # ------------------------------------------------------------------

    def add_stereo_points(self, pts3d, pts3d_mask, f, kf, idxs, img=None) -> int:
        """
        pySLAM-compatible helper used by TrackingCore.

        Creates RGB-D/stereo map points from valid 3D coordinates and attaches
        them to the new keyframe observations.
        """
        count = 0
        idxs = list(np.asarray(idxs, dtype=np.int32).reshape(-1))

        for p, is_valid, idx in zip(pts3d, pts3d_mask, idxs):
            if not bool(is_valid):
                continue
            if idx < 0 or idx >= len(kf.points):
                continue

            existing = kf.points[idx]
            if existing is not None and existing.num_observations() > 0:
                continue

            mp = MapPoint(np.asarray(p, dtype=np.float64).reshape(3), keyframe=kf, idx=int(idx))
            self.add_point(mp)

            if f is not None and hasattr(f, "points") and idx < len(f.points):
                f.points[idx] = mp

            mp.update_info()
            count += 1

        return count

    def __repr__(self) -> str:
        return (
            f"Map(frames={self.num_frames()}, "
            f"keyframes={self.num_keyframes()}, "
            f"points={self.num_points()})"
        )
