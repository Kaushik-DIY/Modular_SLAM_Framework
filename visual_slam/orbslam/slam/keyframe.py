"""
Keyframe representation and graph structure.
This module stores covisibility, spanning-tree, loop-edge, and observation relationships.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from threading import RLock, Lock
from typing import Optional

import numpy as np

from visual_slam.orbslam.slam.camera_pose import CameraPose
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.frame import Frame, _as_points_array, kMinDepth

# F1 (12fps plan): optional C++ KeyFrame base (covisibility graph / spanning tree /
# loop edges / points in C++). Mirrors the MapPoint precedent in map_point.py.
try:
    import cpp_slam_core as _cpp_slam_core
    _CppKeyFrameBase = getattr(_cpp_slam_core, "KeyFrame", None)
except Exception:
    _CppKeyFrameBase = None

# Resolved at import (the base class is fixed at class-definition time). Toggle via
# Parameters.USE_CPP_KEYFRAME, which defaults from $SLAM_USE_CPP_KEYFRAME so a run/test
# can opt in before this module is imported.
_USE_CPP_KF = bool(getattr(Parameters, "USE_CPP_KEYFRAME", False)) and (_CppKeyFrameBase is not None)


def _make_keyframe_bases():
    """Base class(es) for KeyFrame: the C++ KeyFrame when enabled, else the proven
    pure-Python (Frame, KeyFrameGraph). One class either way, so isinstance() holds."""
    if _USE_CPP_KF:
        return (_CppKeyFrameBase,)
    return (Frame, KeyFrameGraph)


def build_cpp_keyframe_from_frame(frame, kid):
    """Populate a fresh C++ KeyFrame base from a Python Frame (F1b construction).

    Feature arrays (kpsu/octaves/des/kps_ur) are READONLY C++ properties, so they
    must be set via ``init_feature_arrays`` — never assigned directly. Mirrors the
    proven recipe in tests/.../test_cpp_slam_core_phase3_keyframe.py.
    """
    if _CppKeyFrameBase is None:
        raise RuntimeError("cpp_slam_core.KeyFrame is unavailable")
    kf = _CppKeyFrameBase(kid=int(kid), frame_id=int(frame.id), camera=frame.camera)
    n = len(frame.kps)
    # Canonical stereo right-coords live in uRs (matches the Python KeyFrame:
    # self.kps_ur = frame.uRs). frame.kps_ur may be a stale all-(-1) placeholder,
    # so prefer uRs — using kps_ur would zero the stereo weights and halve
    # num_observations downstream (breaks num_tracked_points / KF insertion).
    kps_ur = getattr(frame, "uRs", None)
    if kps_ur is None:
        kps_ur = getattr(frame, "kps_ur", None)
    octaves = getattr(frame, "octaves", None)
    des = frame.des if frame.des is not None else np.empty((0, 32), dtype=np.uint8)
    kf.init_feature_arrays(list(frame.kps), np.ascontiguousarray(des, dtype=np.uint8),
                           kps_ur, octaves, n)
    kf.kps = list(frame.kps)  # init_feature_arrays keeps only kpsu; retain the kps list
    kf.timestamp = float(frame.timestamp)   # ctor doesn't set it; gates KF culling
    kf.img_id = int(getattr(frame, "img_id", -1) or -1)
    kf.update_pose(np.ascontiguousarray(frame.Tcw(), dtype=np.float64))
    if getattr(frame, "depths", None) is not None:
        kf.depths = frame.depths
    for idx, p in enumerate(frame.points):
        if p is not None:
            kf.set_point_match(p, idx)
    return kf


# Store the graph relationships attached to one keyframe.
class KeyFrameGraph:
    """Graph container storing parent, child, loop, and covisibility edges."""
    def __init__(self):
        self._lock_features = RLock()
        self._lock_connections = Lock()

        # Spanning tree
        self.init_parent = False
        self.parent = None
        self.children = set()
        self.is_first_connection = True

        # Loop edges
        self.loop_edges = set()
        self.not_to_erase = False

        # Covisibility graph
        self.connected_keyframes_weights = Counter()
        self.ordered_keyframes_weights = OrderedDict()
        self.last_tracking_frame_id = -1
        self.tracking_vote_count = 0

    # ------------------------------------------------------------------
    # Spanning tree
    # ------------------------------------------------------------------

    def add_child_no_lock_(self, keyframe) -> None:
        self.children.add(keyframe)

    def add_child(self, keyframe) -> None:
        with self._lock_connections:
            self.add_child_no_lock_(keyframe)

    def erase_child_no_lock_(self, keyframe) -> None:
        try:
            self.children.remove(keyframe)
        except KeyError:
            pass

    def erase_child(self, keyframe) -> None:
        with self._lock_connections:
            self.erase_child_no_lock_(keyframe)

    def set_parent_no_lock_(self, keyframe) -> None:
        if keyframe is None or keyframe is self:
            return
        self.parent = keyframe
        self.init_parent = True
        keyframe.add_child(self)

    def set_parent(self, keyframe) -> None:
        with self._lock_connections:
            self.set_parent_no_lock_(keyframe)

    def get_children(self):
        with self._lock_connections:
            return self.children.copy()

    def get_parent(self):
        with self._lock_connections:
            return self.parent

    def has_child(self, keyframe) -> bool:
        with self._lock_connections:
            return keyframe in self.children

    # ------------------------------------------------------------------
    # Loop edges
    # ------------------------------------------------------------------

    def add_loop_edge(self, keyframe) -> None:
        with self._lock_connections:
            self.not_to_erase = True
            if keyframe is not None and keyframe is not self:
                self.loop_edges.add(keyframe)

    def get_loop_edges(self):
        with self._lock_connections:
            return self.loop_edges.copy()

    # ------------------------------------------------------------------
    # Covisibility graph
    # ------------------------------------------------------------------

    def reset_covisibility(self) -> None:
        self.connected_keyframes_weights = Counter()
        self.ordered_keyframes_weights = OrderedDict()

    def update_best_covisibles_no_lock_(self) -> None:
        # Deterministic covisibility ordering: weight DESC, then keyframe id ASC as
        # a STABLE tie-breaker (M0 reproducibility). Without the id tie-break,
        # equal-weight keyframes inherit the insertion order of
        # connected_keyframes_weights, which derives from MapPoint.observations()
        # iteration (C++ std::map keyed by pointer ADDRESS, PyObjCompare) — stable
        # within a run but run-to-run non-deterministic (heap/ASLR) — so the local
        # map / trajectory diverged ~65 mm between identical runs. Sorting by a
        # stable id makes get_best_covisible_keyframes() reproducible. Matches
        # pySLAM's id-ordered covisibility intent.
        self.ordered_keyframes_weights = OrderedDict(
            sorted(
                self.connected_keyframes_weights.items(),
                key=lambda item: (
                    -int(item[1]),
                    int(getattr(item[0], "kid", getattr(item[0], "id", 0))),
                ),
            )
        )

    def add_connection_no_lock_(self, keyframe, weight: int) -> None:
        if keyframe is None or keyframe is self:
            return
        self.connected_keyframes_weights[keyframe] = int(weight)
        self.update_best_covisibles_no_lock_()

    def add_connection(self, keyframe, weight: int) -> None:
        with self._lock_connections:
            self.add_connection_no_lock_(keyframe, weight)

    def erase_connection_no_lock_(self, keyframe) -> None:
        try:
            del self.connected_keyframes_weights[keyframe]
            self.update_best_covisibles_no_lock_()
        except KeyError:
            pass

    def erase_connection(self, keyframe) -> None:
        with self._lock_connections:
            self.erase_connection_no_lock_(keyframe)

    def get_connected_keyframes_no_lock_(self):
        return list(self.connected_keyframes_weights.keys())

    def get_connected_keyframes(self):
        with self._lock_connections:
            return self.get_connected_keyframes_no_lock_()

    def get_covisible_keyframes_no_lock_(self):
        return list(self.ordered_keyframes_weights.keys())

    def get_covisible_keyframes(self):
        with self._lock_connections:
            return self.get_covisible_keyframes_no_lock_()

    def get_best_covisible_keyframes(self, N: int):
        with self._lock_connections:
            return list(self.ordered_keyframes_weights.keys())[: int(N)]

    def get_covisible_by_weight(self, weight: int):
        with self._lock_connections:
            return [kf for kf, w in self.ordered_keyframes_weights.items() if w > weight]

    def get_weight_no_lock_(self, keyframe) -> int:
        return int(self.connected_keyframes_weights[keyframe])

    def get_weight(self, keyframe) -> int:
        with self._lock_connections:
            return self.get_weight_no_lock_(keyframe)

    def get_connected_keyframes_weights(self):
        with self._lock_connections:
            return {
                getattr(kf, "id", None): w
                for kf, w in self.connected_keyframes_weights.items()
                if kf is not None
            }


# Represent a selected map keyframe with graph and observation state.
class KeyFrame(*_make_keyframe_bases()):

    def __init__(
        self,
        frame: Frame,
        img=None,
        img_right=None,
        depth=None,
        kid: Optional[int] = None,
    ):
        if _USE_CPP_KF:
            self._init_from_frame_cpp(frame, img, img_right, depth, kid)
            return
        KeyFrameGraph.__init__(self)

        # Create a Frame shell without recomputing features.
        Frame.__init__(
            self,
            img=None,
            camera=frame.camera,
            pose=frame.pose(),
            id=frame.id,
            timestamp=frame.timestamp,
            img_id=frame.img_id,
        )

        self.img = frame.img if frame.img is not None else img
        self.img_right = frame.img_right if frame.img_right is not None else img_right
        self.depth_img = frame.depth_img if frame.depth_img is not None else depth

        self.map = None
        self.is_keyframe = True
        self.kid = kid if kid is not None else frame.id

        self._is_bad = False
        self.to_be_erased = False
        self.lba_count = 0

        self.is_blurry = getattr(frame, "is_blurry", False)
        self.laplacian_var = getattr(frame, "laplacian_var", None)

        # Pose relative to parent. Computed when the keyframe is marked bad.
        self._pose_Tcp = CameraPose()

        # Share immutable feature information with the source frame.
        self.kps = frame.kps
        self.kpsu = frame.kpsu
        self.des = frame.des
        self.depths = frame.depths
        self.uRs = frame.uRs
        self.kps_ur = frame.uRs

        self.kpsn = getattr(frame, "kpsn", None)
        self.octaves = np.array([max(0, int(getattr(kp, "octave", 0))) for kp in self.kps], dtype=np.int32)
        self.sizes = np.array([float(getattr(kp, "size", 0.0)) for kp in self.kps], dtype=np.float32)
        self.angles = np.array([float(getattr(kp, "angle", -1.0)) for kp in self.kps], dtype=np.float32)

        self.median_depth = frame.median_depth
        self.fov_center_c = frame.fov_center_c
        self.fov_center_w = frame.fov_center_w

        # Loop closing fields
        self.g_des = None
        self.f_des = None
        self.bow_vector = None
        self.feature_vector = None
        self.loop_query_id = None
        self.num_loop_words = 0
        self.loop_score = 0.0

        # Relocalization fields
        self.reloc_query_id = None
        self.num_reloc_words = 0
        self.reloc_score = 0.0

        # GBA fields
        self.GBA_kf_id = 0
        self.is_Tcw_GBA_valid = False
        self.Tcw_GBA = None
        self.Tcw_before_GBA = None

        # Copy map-point associations from source frame.
        self.points = list(frame.points)
        self.outliers = np.zeros(len(self.kps), dtype=bool)

    def _init_from_frame_cpp(self, frame, img, img_right, depth, kid):
        """C++ KeyFrame path: populate the C++ base from a Python Frame.

        Feature arrays go through init_feature_arrays (kpsu/octaves/des/kps_ur are
        READONLY C++ properties). _is_bad is a readonly C++ property (never assigned).
        loop_query_id/reloc_query_id keep their C++ int(-1) default (consumers compare
        with != / == only). Python-only state lives on the instance via dynamic_attr.
        """
        kid_val = int(kid) if kid is not None else int(frame.id)
        _CppKeyFrameBase.__init__(self, kid=kid_val, frame_id=int(frame.id),
                                  camera=frame.camera)
        n = len(frame.kps)
        # Stereo right-coords: prefer uRs (canonical; matches the Python KeyFrame).
        # frame.kps_ur may be a stale all-(-1) placeholder -> would zero stereo
        # weights and halve num_observations (breaks num_tracked_points/KF insertion).
        kps_ur = getattr(frame, "uRs", None)
        if kps_ur is None:
            kps_ur = getattr(frame, "kps_ur", None)
        des = frame.des if frame.des is not None else np.empty((0, 32), dtype=np.uint8)
        self.init_feature_arrays(list(frame.kps), np.ascontiguousarray(des, dtype=np.uint8),
                                 kps_ur, getattr(frame, "octaves", None), n)
        # init_feature_arrays normalizes kps_in into kpsu but does NOT retain the
        # cv2.KeyPoint list; consumers still read kf.kps, so store it (settable).
        self.kps = list(frame.kps)
        self.update_pose(np.ascontiguousarray(frame.Tcw(), dtype=np.float64))
        if getattr(frame, "depths", None) is not None:
            self.depths = frame.depths
        for idx, p in enumerate(frame.points):
            if p is not None:
                self.set_point_match(p, idx)

        # C++ fields (typed) — keep loop_query_id/reloc_query_id at their -1 default.
        # The C++ ctor takes only (kid, frame_id, camera); timestamp/img_id are NOT
        # set by it and must be copied — timestamp gates keyframe culling
        # (kKeyframeMaxTimeDistanceInSecForCulling), so a 0 timestamp disables culling.
        self.timestamp = float(frame.timestamp)
        self.img_id = int(getattr(frame, "img_id", -1) or -1)
        self.kid = kid_val
        self.map = None
        self.is_keyframe = True
        self.to_be_erased = False
        self.GBA_kf_id = 0
        self.is_Tcw_GBA_valid = False
        self.Tcw_GBA = None
        self.Tcw_before_GBA = None
        self.num_loop_words = 0
        self.loop_score = 0.0
        self.num_reloc_words = 0
        self.reloc_score = 0.0

        # Python-only state (held on the C++ instance via dynamic_attr).
        self.img = frame.img if frame.img is not None else img
        self.img_right = frame.img_right if frame.img_right is not None else img_right
        self.depth_img = frame.depth_img if frame.depth_img is not None else depth
        # NB: uRs / kps_ur are readonly C++ properties (aliases of kps_ur, already
        # populated by init_feature_arrays) — readable, not assignable.
        self.lba_count = 0
        self.is_blurry = getattr(frame, "is_blurry", False)
        self.laplacian_var = getattr(frame, "laplacian_var", None)
        self._pose_Tcp = CameraPose()
        self.kpsn = getattr(frame, "kpsn", None)
        self.sizes = np.array([float(getattr(kp, "size", 0.0)) for kp in frame.kps], dtype=np.float32)
        self.angles = np.array([float(getattr(kp, "angle", -1.0)) for kp in frame.kps], dtype=np.float32)
        self.median_depth = frame.median_depth
        self.fov_center_c = frame.fov_center_c
        self.fov_center_w = frame.fov_center_w
        self.g_des = None
        self.f_des = None
        self.bow_vector = None
        self.feature_vector = None
        self.outliers = np.zeros(n, dtype=bool)

    if _USE_CPP_KF:
        # ---- Frame-API compatibility for the C++ KeyFrame base -------------
        # The fork's C++ Frame is a partial "mirror" (pySLAM's is complete), so
        # these Python-Frame methods absent on the C++ base are provided here via
        # the C++ pose accessors. Defined only on the C++ path (else they would
        # shadow the proven Python Frame versions).
        def pose(self):
            return self.Tcw()

        def Rcw(self):
            return self.Tcw()[:3, :3].copy()

        def Rwc(self):
            return self.Twc()[:3, :3].copy()

        def tcw(self):
            return self.Tcw()[:3, 3].copy()

        def position(self):
            return self.Ow()

        def isometry3d(self):
            import g2o
            return g2o.Isometry3d(np.ascontiguousarray(self.Tcw(), dtype=np.float64))

        # Point-query family (operate on the C++ self.points / Python self.outliers;
        # identical logic to the Python Frame versions).
        def get_matched_points(self):
            return [p for p in self.points if p is not None]

        def get_matched_points_idxs(self):
            return np.array([i for i, p in enumerate(self.points) if p is not None], dtype=np.int32)

        def get_unmatched_points_idxs(self):
            return np.array([i for i, p in enumerate(self.points) if p is None], dtype=np.int32)

        def get_matched_inlier_points(self):
            return self.get_matched_good_points()

        def num_matched_inlier_map_points(self):
            outliers = getattr(self, "outliers", None)
            count = 0
            for idx, p in enumerate(self.points):
                if p is None:
                    continue
                if outliers is not None and idx < len(outliers) and bool(outliers[idx]):
                    continue
                if p.num_observations() > 0:
                    count += 1
            return count

        # Projection family — the fork's C++ "mirror" Frame omits these, but
        # fuse_map_points (search_and_fuse -> keyframe.are_visible) and
        # triangulation need them. Their absence was SILENTLY swallowed by
        # fuse's bare `except`, disabling map-point fusion on C++ keyframes
        # (=> 2x duplicate points, sparser observations, under-culling). Reuse
        # the exact Python Frame logic; transform via the C++ Tcw (the only
        # _pose-private dependency). ~1 Hz path (not the tracking hot loop).
        def transform_points(self, points):
            Tcw = np.ascontiguousarray(self.Tcw(), dtype=np.float64)
            points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
            return (Tcw[:3, :3] @ points.T + Tcw[:3, 3].reshape(3, 1)).T

        def transform_point(self, pw):
            Tcw = np.ascontiguousarray(self.Tcw(), dtype=np.float64)
            pw = np.asarray(pw, dtype=np.float64).reshape(3)
            return (Tcw[:3, :3] @ pw) + Tcw[:3, 3]

        def project_points(self, points, do_stereo_project: bool = False):
            pcs = self.transform_points(points)
            return (self.camera.project_stereo(pcs) if do_stereo_project
                    else self.camera.project(pcs))

        def project_point(self, pw, do_stereo_project: bool = False):
            pc = self.transform_point(pw)
            proj, zs = (self.camera.project_stereo(pc.reshape(1, 3)) if do_stereo_project
                        else self.camera.project(pc.reshape(1, 3)))
            return proj.reshape(-1), float(zs[0])

        def project_map_points(self, map_points, do_stereo_project: bool = False):
            points = _as_points_array(map_points)
            if len(points) == 0:
                w = 3 if do_stereo_project else 2
                return np.empty((0, w), dtype=np.float64), np.empty((0,), dtype=np.float64)
            return self.project_points(points, do_stereo_project=do_stereo_project)

        def are_in_image(self, uvs, zs):
            return self.camera.are_in_image(uvs, zs)

        def are_visible(self, map_points, do_stereo_project: bool = False):
            projs, depths = self.project_map_points(map_points, do_stereo_project=do_stereo_project)
            pts = _as_points_array(map_points)
            if len(pts) == 0:
                return (np.empty((0,), dtype=bool), projs, depths,
                        np.empty((0,), dtype=np.float64))
            dists = np.linalg.norm(pts - self.Ow().reshape(1, 3), axis=1)
            visible = self.are_in_image(projs[:, :2], depths) & (depths > kMinDepth)
            return visible, projs, depths, dists

        def unproject_points_3d(self, idxs, transform_in_world: bool = True):
            idxs = np.asarray(idxs, dtype=np.int32).reshape(-1)
            pts3d = np.zeros((len(idxs), 3), dtype=np.float64)
            valid = np.zeros(len(idxs), dtype=bool)
            if len(idxs) == 0:
                return pts3d, valid
            kpsu, depths = self.kpsu, self.depths
            Rwc, Ow = self.Rwc(), self.Ow()
            for out_i, idx in enumerate(idxs):
                if idx < 0 or idx >= len(kpsu) or idx >= len(depths):
                    continue
                depth = float(depths[idx])
                if not np.isfinite(depth) or depth <= kMinDepth:
                    continue
                _kp = kpsu[idx]
                uv = np.array(_kp.pt if hasattr(_kp, "pt") else _kp, dtype=np.float64)
                pc = self.camera.unproject_3d(uv, depth)
                pts3d[out_i] = (Rwc @ pc.reshape(3) + Ow.reshape(3)) if transform_in_world else pc.reshape(3)
                valid[out_i] = True
            return pts3d, valid

    def init_observations(self) -> None:
        """Associate all currently matched map points as keyframe observations."""
        if not hasattr(self, "_lock_features"):
            self._lock_features = RLock()
        with self._lock_features:
            points = list(self.points)

        for idx, point in enumerate(points):
            if point is None:
                continue
            if hasattr(point, "is_bad") and point.is_bad():
                continue
            if point.add_observation(self, idx):
                point.update_info()

    def update_connections(self) -> None:
        """
        """
        if _USE_CPP_KF:
            return _CppKeyFrameBase.update_connections(self)
        points = self.get_matched_good_points()
        if len(points) == 0:
            return

        viewing_keyframes = Counter()

        for point in points:
            if point is None:
                continue
            for kf in point.keyframes():
                if kf is self:
                    continue
                if getattr(kf, "kid", None) == self.kid:
                    continue
                if hasattr(kf, "is_bad") and kf.is_bad():
                    continue
                viewing_keyframes[kf] += 1

        if not viewing_keyframes:
            return

        covisible_keyframes = viewing_keyframes.most_common()
        kf_max, w_max = covisible_keyframes[0]

        with self._lock_connections:
            self.connected_keyframes_weights = viewing_keyframes

            if w_max >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                self.ordered_keyframes_weights = OrderedDict()
                for kf, w in covisible_keyframes:
                    if w >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                        kf.add_connection_no_lock_(self, w)
                        self.ordered_keyframes_weights[kf] = w
                    else:
                        break
            else:
                self.ordered_keyframes_weights = OrderedDict([(kf_max, w_max)])
                kf_max.add_connection_no_lock_(self, w_max)

            if (
                self.is_first_connection
                and self.kid != 0
                and kf_max is not None
                and kf_max is not self
                and not kf_max.is_bad()
            ):
                self.set_parent_no_lock_(kf_max)
                self.is_first_connection = False

    def Tcp(self):
        if _USE_CPP_KF:
            return self._pose_Tcp.get_matrix()
        with self._lock_connections:
            return self._pose_Tcp.get_matrix()

    def is_bad(self) -> bool:
        if _USE_CPP_KF:
            return _CppKeyFrameBase.is_bad(self)
        with self._lock_connections:
            return self._is_bad

    def compute_bow(self, vocabulary):
        from visual_slam.orbslam.slam.bow import compute_bow_for_frame

        return compute_bow_for_frame(self, vocabulary)

    def set_not_erase(self) -> None:
        if _USE_CPP_KF:
            return _CppKeyFrameBase.set_not_erase(self)
        with self._lock_connections:
            self.not_to_erase = True

    def set_erase(self) -> None:
        if _USE_CPP_KF:
            return _CppKeyFrameBase.set_erase(self)
        should_set_bad = False
        with self._lock_connections:
            if len(self.loop_edges) == 0:
                self.not_to_erase = False
            if self.to_be_erased:
                should_set_bad = True
        if should_set_bad:
            self.set_bad()

    def set_bad(self) -> None:
        """Mark this keyframe bad and detach its graph and point links."""
        if _USE_CPP_KF:
            # The C++ set_bad stubs the Tcp computation; do it here (Python),
            # faithful to pySLAM, BEFORE the C++ side erases the parent link.
            # Guards mirror the C++ early-returns (kid==0 / not_to_erase).
            if self.kid != 0 and not self.not_to_erase:
                parent = self.get_parent()
                if parent is not None:
                    try:
                        self._pose_Tcp.update(self.Tcw() @ parent.Twc())
                    except Exception:
                        pass
            return _CppKeyFrameBase.set_bad(self)
        with self._lock_connections:
            if self.kid == 0:
                return

            if self.not_to_erase:
                self.to_be_erased = True
                return

            connected = self.get_connected_keyframes_no_lock_()

        for kf_connected in connected:
            kf_connected.erase_connection(self)

        # Remove observations from map points.
        for idx, point in enumerate(list(self.points)):
            if point is not None:
                point.remove_observation(self, idx)
                self.points[idx] = None

        with self._lock_connections:
            self.reset_covisibility()

            if self.parent is not None:
                try:
                    self._pose_Tcp.update(self.Tcw() @ self.parent.Twc())
                except Exception:
                    pass
                self.parent.erase_child_no_lock_(self)

            self.children.clear()
            self._is_bad = True

        if self.map is not None:
            remove_fn = getattr(self.map, "remove_keyframe", None)
            if remove_fn is not None:
                remove_fn(self)

    def __repr__(self) -> str:
        return f"KeyFrame(kid={self.kid}, frame_id={self.id}, kps={len(self.kps)}, bad={self._is_bad})"

    def get_matched_good_points_and_idxs(self):
        """

        Returns:
            list[(MapPoint, keypoint_idx)]

        The index must be the original keypoint index in self.points, not the
        compact index of get_points().
        """
        pairs = []
        points = getattr(self, "points", [])
        outliers = getattr(self, "outliers", None)

        for idx, p in enumerate(points):
            if p is None:
                continue
            if hasattr(p, "is_bad") and p.is_bad():
                continue
            if outliers is not None and idx < len(outliers) and bool(outliers[idx]):
                continue
            pairs.append((p, idx))

        return pairs

    def get_matched_good_points(self):
        """Return non-bad matched map points."""
        return [p for p, _ in self.get_matched_good_points_and_idxs()]

    def get_matched_good_points_idxs(self):
        """Return original keypoint indices of non-bad matched map points."""
        return np.asarray(
            [idx for _, idx in self.get_matched_good_points_and_idxs()],
            dtype=np.int32,
        )

    def num_tracked_points(self, min_num_observations=0):
        """Count tracked map points with at least min_num_observations."""
        count = 0

        for p, _ in self.get_matched_good_points_and_idxs():
            if p is None:
                continue
            if hasattr(p, "is_bad") and p.is_bad():
                continue
            if min_num_observations > 0 and p.num_observations() < min_num_observations:
                continue
            count += 1

        return count

    def check_replaced_map_points(self):
        """
        Replace frame associations when a MapPoint has been substituted.

        mapping / fusion can replace map points between frames.
        """
        replaced = 0

        points = getattr(self, "points", [])
        for idx, p in enumerate(list(points)):
            if p is None:
                continue

            replacement = None

            if hasattr(p, "get_replacement"):
                try:
                    replacement = p.get_replacement()
                except Exception:
                    replacement = None
            # Fallback: check .replacement attribute even when get_replacement() returned None
            if replacement is None and hasattr(p, "replacement"):
                try:
                    replacement = p.replacement
                except Exception:
                    replacement = None

            if replacement is not None and replacement is not p:
                points[idx] = replacement
                try:
                    replacement.add_frame_view(self, idx)
                except Exception:
                    pass
                replaced += 1

        return replaced

    def release_depth_image(self) -> None:
        self.depth_img = None

    def release_rgb_image(self) -> None:
        self.img = None
        self.img_right = None

    def release_heavy_data(self, release_rgb=False, release_depth=True, release_kd=False) -> None:
        if release_rgb:
            self.release_rgb_image()
        if release_depth:
            self.release_depth_image()
        if release_kd:
            self.kd = None

    def heavy_memory_bytes(self) -> int:
        if _USE_CPP_KF:
            # The C++ base has no Python heavy-data accounting; sum the
            # Python-held image buffers (diagnostics only).
            total = 0
            for attr in ("img", "img_right", "depth_img"):
                a = getattr(self, attr, None)
                if a is not None and hasattr(a, "nbytes"):
                    total += int(a.nbytes)
            return total
        return super().heavy_memory_bytes()
