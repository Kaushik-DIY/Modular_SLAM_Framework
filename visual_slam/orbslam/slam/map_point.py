"""
=============================================================================
visual_slam/orbslam/slam/map_point.py

pySLAM-aligned MapPointBase and MapPoint subset for ORB/RGB-D SLAM.

Reference:
- pySLAM: pyslam/slam/map_point.py

Scope retained:
- global map-point ID handling
- keyframe observations
- transient frame views
- found/visible statistics
- bad/replacement state
- descriptor medoid computation
- normal/depth-range update
- scale prediction

Excluded for now:
- semantic descriptors
- JSON serialization
- drawing/color utilities
=============================================================================
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

import cv2
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _get_frame_like(kf_or_frame):
    return getattr(kf_or_frame, "frame", kf_or_frame)


def _get_keypoint_octave(kf_or_frame, idx: int) -> int:
    frame = _get_frame_like(kf_or_frame)
    kps = getattr(frame, "kps", getattr(frame, "keypoints", []))
    if idx < 0 or idx >= len(kps):
        return 0
    return max(0, int(getattr(kps[idx], "octave", 0)))


def _get_kps_ur(kf_or_frame):
    if hasattr(kf_or_frame, "kps_ur"):
        return getattr(kf_or_frame, "kps_ur")
    if hasattr(kf_or_frame, "uRs"):
        return getattr(kf_or_frame, "uRs")
    frame = getattr(kf_or_frame, "frame", None)
    if frame is not None and hasattr(frame, "uRs"):
        return frame.uRs
    return None


def _get_descriptor(kf_or_frame, idx: int):
    frame = _get_frame_like(kf_or_frame)
    des = getattr(frame, "des", getattr(frame, "descriptors", None))
    if des is None or idx < 0 or idx >= len(des):
        return None
    return np.asarray(des[idx], dtype=np.uint8)


def _get_camera_center(kf_or_frame) -> np.ndarray:
    frame = _get_frame_like(kf_or_frame)
    Ow = getattr(frame, "Ow", None)
    if callable(Ow):
        return np.asarray(Ow(), dtype=np.float64).reshape(3)
    if Ow is not None:
        return np.asarray(Ow, dtype=np.float64).reshape(3)
    return np.zeros(3, dtype=np.float64)


def _set_point_match(frame_or_keyframe, point, idx: int) -> None:
    if hasattr(frame_or_keyframe, "set_point_match"):
        frame_or_keyframe.set_point_match(point, idx)
    else:
        frame = _get_frame_like(frame_or_keyframe)
        frame.set_point_match(point, idx)


def _remove_point_match(frame_or_keyframe, idx: int) -> None:
    if hasattr(frame_or_keyframe, "remove_point_match"):
        frame_or_keyframe.remove_point_match(idx)
    else:
        frame = _get_frame_like(frame_or_keyframe)
        frame.remove_point_match(idx)


def _remove_point(frame_or_keyframe, point) -> None:
    if hasattr(frame_or_keyframe, "remove_point"):
        frame_or_keyframe.remove_point(point)
    else:
        frame = _get_frame_like(frame_or_keyframe)
        frame.remove_point(point)


class MapPointBase:
    """pySLAM-like map-point base object."""

    _id = 0
    _id_lock = Lock()

    def __init__(self, id: Optional[int] = None):
        if id is not None:
            self.id = int(id)
        else:
            with MapPointBase._id_lock:
                self.id = MapPointBase._id
                MapPointBase._id += 1

        self._lock_pos = Lock()
        self._lock_features = Lock()

        self.map = None

        self._observations = {}
        self._frame_views = {}
        self._is_bad = False
        self._num_observations = 0

        self.num_times_visible = 1
        self.num_times_found = 1
        self.last_frame_id_seen = -1

        self.replacement = None

        self.corrected_by_kf = 0
        self.corrected_reference = 0
        self.kf_ref = None

    @staticmethod
    def next_id() -> int:
        with MapPointBase._id_lock:
            return MapPointBase._id

    @staticmethod
    def set_id(id_value: int) -> None:
        with MapPointBase._id_lock:
            MapPointBase._id = int(id_value)

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return isinstance(rhs, MapPointBase) and self.id == rhs.id

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __le__(self, rhs):
        return self.id <= rhs.id

    def observations_string(self) -> str:
        obs = sorted(
            [
                (
                    getattr(kf, "id", getattr(kf, "kid", -1)),
                    kidx,
                    getattr(_get_frame_like(kf), "get_point_match", lambda _: None)(kidx) is not None,
                )
                for kf, kidx in self.observations()
            ],
            key=lambda x: x[0],
        )
        return "observations: " + str(obs)

    def frame_views_string(self) -> str:
        obs = sorted(
            [
                (
                    getattr(f, "id", -1),
                    idx,
                    getattr(f, "get_point_match", lambda _: None)(idx) is not None,
                )
                for f, idx in self.frame_views()
            ],
            key=lambda x: x[0],
        )
        return "views: " + str(obs)

    def __str__(self):
        return f"MapPoint {self.id} {{ {self.observations_string()}, {self.frame_views_string()} }}"

    def observations(self):
        with self._lock_features:
            return list(self._observations.items())

    def observations_iter(self):
        return iter(self._observations.items())

    def keyframes(self):
        with self._lock_features:
            return list(self._observations.keys())

    def keyframes_iter(self):
        return iter(self._observations.keys())

    def is_in_keyframe(self, keyframe) -> bool:
        with self._lock_features:
            return keyframe in self._observations

    def get_observation_idx(self, keyframe) -> int:
        with self._lock_features:
            return self._observations.get(keyframe, -1)

    def get_frame_view_idx(self, frame) -> int:
        with self._lock_features:
            return self._frame_views.get(frame, -1)

    def _observation_weight(self, keyframe, idx: int) -> int:
        kps_ur = _get_kps_ur(keyframe)
        if kps_ur is not None and idx < len(kps_ur) and kps_ur[idx] >= 0:
            return 2
        return 1

    def add_observation_no_lock_(self, keyframe, idx: int) -> bool:
        success = False

        if keyframe not in self._observations:
            self._observations[keyframe] = int(idx)
            self._num_observations += self._observation_weight(keyframe, idx)
            success = True

        if success:
            _set_point_match(keyframe, self, idx)

        return success

    def add_observation(self, keyframe, idx: int) -> bool:
        with self._lock_features:
            success = self.add_observation_no_lock_(keyframe, idx)
        return success

    def remove_observation(self, keyframe, idx=None, map_no_lock: bool = False) -> None:
        kf_remove_point_match = False
        kf_remove_point = False
        set_bad = False

        with self._lock_features:
            if keyframe not in self._observations:
                return

            obs_idx = self._observations[keyframe] if idx is None else int(idx)

            if idx is not None:
                kf_remove_point_match = True
            else:
                kf_remove_point = True

            del self._observations[keyframe]
            self._num_observations = max(
                0,
                self._num_observations - self._observation_weight(keyframe, obs_idx),
            )

            set_bad = self._num_observations <= 2

            if self.kf_ref is keyframe:
                self.kf_ref = next(iter(self._observations.keys()), None)

        if kf_remove_point_match:
            _remove_point_match(keyframe, obs_idx)

        if kf_remove_point:
            _remove_point(keyframe, self)

        if set_bad:
            self.set_bad(map_no_lock=map_no_lock)

    def frame_views(self):
        with self._lock_features:
            return list(self._frame_views.items())

    def frame_views_iter(self):
        return iter(self._frame_views.items())

    def frames(self):
        with self._lock_features:
            return list(self._frame_views.keys())

    def frames_iter(self):
        return iter(self._frame_views.keys())

    def is_in_frame(self, frame) -> bool:
        with self._lock_features:
            return frame in self._frame_views

    def add_frame_view(self, frame, idx: int) -> bool:
        if getattr(frame, "is_keyframe", False):
            raise AssertionError("add_frame_view expects a non-keyframe Frame")

        with self._lock_features:
            if frame in self._frame_views:
                return False
            self._frame_views[frame] = int(idx)

        _set_point_match(frame, self, idx)
        return True

    def remove_frame_view(self, frame, idx=None) -> None:
        frame_remove_point_match = False
        frame_remove_point = False

        with self._lock_features:
            if frame not in self._frame_views:
                return

            obs_idx = self._frame_views[frame] if idx is None else int(idx)

            if idx is not None:
                frame_remove_point_match = True
            else:
                frame_remove_point = True

            del self._frame_views[frame]

        if frame_remove_point_match:
            _remove_point_match(frame, obs_idx)

        if frame_remove_point:
            _remove_point(frame, self)

    def is_bad(self) -> bool:
        with self._lock_features:
            return self._is_bad

    def is_bad_or_is_in_keyframe(self, keyframe) -> bool:
        with self._lock_features:
            return self._is_bad or (keyframe in self._observations)

    def num_observations(self) -> int:
        with self._lock_features:
            return int(self._num_observations)

    def is_good_with_min_obs(self, min_obs: int) -> bool:
        with self._lock_features:
            return (not self._is_bad) and (self._num_observations >= int(min_obs))

    def is_bad_and_is_good_with_min_obs(self, min_obs: int):
        with self._lock_features:
            return (
                self._is_bad,
                (not self._is_bad) and (self._num_observations >= int(min_obs)),
            )

    def increase_visible(self, num_times: int = 1) -> None:
        with self._lock_features:
            self.num_times_visible += int(num_times)

    def increase_found(self, num_times: int = 1) -> None:
        with self._lock_features:
            self.num_times_found += int(num_times)

    def get_found_ratio(self) -> float:
        with self._lock_features:
            if self.num_times_visible <= 0:
                return 0.0
            return float(self.num_times_found) / float(self.num_times_visible)


class MapPoint(MapPointBase):
    """pySLAM-like 3D landmark."""

    def __init__(
        self,
        position: np.ndarray,
        color=None,
        keyframe=None,
        idx: Optional[int] = None,
        id: Optional[int] = None,
    ):
        super().__init__(id=id)

        self._position = np.asarray(position, dtype=np.float64).reshape(3)
        self.color = color

        self.des = None
        self.descriptor = None

        self.normal = np.zeros(3, dtype=np.float64)
        self.min_distance = 0.0
        self.max_distance = 0.0

        self.first_kid = -1
        self._last_update_time = 0.0

        if keyframe is not None and idx is not None:
            self.kf_ref = keyframe
            self.first_kid = int(getattr(keyframe, "kid", getattr(keyframe, "id", -1)))
            self.add_observation(keyframe, int(idx))
            self.update_info()

    @property
    def position(self) -> np.ndarray:
        return self.get_position()

    @position.setter
    def position(self, value) -> None:
        self.set_position(value)

    @property
    def position_world(self) -> np.ndarray:
        return self.get_position()

    @position_world.setter
    def position_world(self, value) -> None:
        self.set_position(value)

    def pt(self) -> np.ndarray:
        return self.get_position()

    def get_position(self) -> np.ndarray:
        with self._lock_pos:
            return self._position.copy()

    def set_position(self, position: np.ndarray) -> None:
        with self._lock_pos:
            self._position = np.asarray(position, dtype=np.float64).reshape(3)

    def update_position(self, position: np.ndarray) -> None:
        self.set_position(position)

    def get_descriptor(self):
        with self._lock_features:
            if self.des is None:
                return None
            return self.des.copy()

    def set_descriptor(self, descriptor: np.ndarray) -> None:
        with self._lock_features:
            self.des = np.asarray(descriptor, dtype=np.uint8).copy()
            self.descriptor = self.des

    def get_normal(self) -> np.ndarray:
        with self._lock_pos:
            return self.normal.copy()

    def get_reference_keyframe(self):
        with self._lock_features:
            return self.kf_ref

    def compute_distinctive_descriptor(self) -> None:
        descriptors = []

        with self._lock_features:
            observations = list(self._observations.items())

        for kf, idx in observations:
            des = _get_descriptor(kf, idx)
            if des is not None:
                descriptors.append(des)

        if len(descriptors) == 0:
            self.des = None
            self.descriptor = None
            return

        if len(descriptors) == 1:
            self.set_descriptor(descriptors[0])
            return

        descriptors = np.asarray(descriptors, dtype=np.uint8)

        n = len(descriptors)
        distances = np.zeros((n, n), dtype=np.float32)

        descriptor_distance = FeatureTrackerShared.descriptor_distance
        if descriptor_distance is None:
            descriptor_distance = lambda a, b: cv2.norm(a, b, cv2.NORM_HAMMING)

        for i in range(n):
            for j in range(i + 1, n):
                d = descriptor_distance(descriptors[i], descriptors[j])
                distances[i, j] = d
                distances[j, i] = d

        median_distances = np.median(distances, axis=1)
        best_idx = int(np.argmin(median_distances))
        self.set_descriptor(descriptors[best_idx])

    def compute_descriptor(self) -> None:
        self.compute_distinctive_descriptor()

    def update_normal_and_depth(self) -> None:
        with self._lock_features:
            observations = list(self._observations.items())
            ref_kf = self.kf_ref
            if ref_kf is None and observations:
                ref_kf = observations[0][0]
                self.kf_ref = ref_kf

        if not observations or ref_kf is None:
            return

        position = self.get_position()
        normal = np.zeros(3, dtype=np.float64)

        for kf, _ in observations:
            camera_center = _get_camera_center(kf)
            normal += _normalize_vector(position - camera_center)

        normal = _normalize_vector(normal / max(1, len(observations)))

        ref_center = _get_camera_center(ref_kf)
        dist = float(np.linalg.norm(position - ref_center))

        ref_idx = self.get_observation_idx(ref_kf)
        ref_level = _get_keypoint_octave(ref_kf, ref_idx)

        feature_manager = FeatureTrackerShared.feature_manager
        if feature_manager is not None:
            scale_factors = feature_manager.scale_factors
            ref_level = min(max(ref_level, 0), len(scale_factors) - 1)
            level_scale_factor = float(scale_factors[ref_level])
            max_distance = dist * level_scale_factor
            min_distance = max_distance / float(scale_factors[-1])
        else:
            max_distance = dist
            min_distance = dist

        with self._lock_pos:
            self.normal = normal
            self.max_distance = float(max_distance)
            self.min_distance = float(min_distance)

    def update_info(self) -> None:
        self.compute_distinctive_descriptor()
        self.update_normal_and_depth()

    def predict_scale(self, dist: float, frame_or_keyframe) -> int:
        feature_manager = FeatureTrackerShared.feature_manager
        if feature_manager is None:
            return 0

        with self._lock_pos:
            max_distance = max(self.max_distance, 1e-12)

        ratio = max_distance / max(float(dist), 1e-12)
        n_scale = int(np.ceil(np.log(ratio) / feature_manager.log_scale_factor))

        if n_scale < 0:
            return 0
        if n_scale >= feature_manager.num_levels:
            return feature_manager.num_levels - 1
        return n_scale

    def get_min_distance_invariance(self) -> float:
        with self._lock_pos:
            return 0.8 * self.min_distance

    def get_max_distance_invariance(self) -> float:
        with self._lock_pos:
            return 1.2 * self.max_distance

    def set_bad(self, map_no_lock: bool = False) -> None:
        with self._lock_features:
            if self._is_bad:
                return

            observations = list(self._observations.items())
            frame_views = list(self._frame_views.items())

            self._observations.clear()
            self._frame_views.clear()
            self._num_observations = 0
            self._is_bad = True

        for kf, idx in observations:
            try:
                _remove_point_match(kf, idx)
            except Exception:
                pass

        for frame, idx in frame_views:
            try:
                _remove_point_match(frame, idx)
            except Exception:
                pass

        if self.map is not None:
            remove_fn = getattr(self.map, "remove_point", None) or getattr(self.map, "remove_map_point", None)
            if remove_fn is not None:
                try:
                    remove_fn(self)
                except TypeError:
                    remove_fn(self, map_no_lock=map_no_lock)

    def replace_with(self, replacement: "MapPoint") -> None:
        if replacement is self:
            return

        with self._lock_features:
            observations = list(self._observations.items())
            frame_views = list(self._frame_views.items())
            self.replacement = replacement

        for kf, idx in observations:
            if not replacement.is_in_keyframe(kf):
                replacement.add_observation(kf, idx)
            else:
                _remove_point_match(kf, idx)

        for frame, idx in frame_views:
            if not replacement.is_in_frame(frame):
                replacement.add_frame_view(frame, idx)
            else:
                _remove_point_match(frame, idx)

        replacement.increase_found(self.num_times_found)
        replacement.increase_visible(self.num_times_visible)
        replacement.update_info()
        self.set_bad()

    def get_replaced(self):
        with self._lock_features:
            return self.replacement

    def delete(self) -> None:
        self.set_bad()

    def __repr__(self) -> str:
        p = self.get_position()
        return f"MapPoint(id={self.id}, p=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], obs={self.num_observations()})"
