"""
=============================================================================
visual_slam/orbslam/slam/tracking.py

pySLAM-aligned Tracking subset for ORB/RGB-D SLAM.

Reference:
- pySLAM: pyslam/slam/tracking.py

Implemented in this checkpoint:
- TrackingHistory
- Tracking.__init__/reset
- pose_optimization
- track_previous_frame
- track_reference_frame
- track_keyframe
- update_local_map
- track_local_map
- clean_vo_points
- need_new_keyframe
- create_new_keyframe
- create_vo_points_on_last_frame
- create_and_add_stereo_map_points_on_new_kf
- update_history
- minimal RGB-D track() entry point

Deferred:
- full relocalizer/database
- initializer for monocular
- local mapping thread synchronization
- dynamic descriptor threshold estimator class
- drawing/debug viewers
=============================================================================
"""

from __future__ import annotations

from itertools import chain

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import Frame, ensure_frame_feature_arrays
from visual_slam.orbslam.slam.geometry_matchers import ProjectionMatcher
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map import Map
from visual_slam.orbslam.slam.map_point import MapPoint
from visual_slam.orbslam.slam.motion_model import MotionModel
from visual_slam.orbslam.slam.optimizer_g2o import pose_optimization as g2o_pose_optimization
from visual_slam.orbslam.slam.rotation_histogram import RotationHistogram
from visual_slam.orbslam.slam.sensor_types import SensorType
from visual_slam.orbslam.slam.slam_commons import SlamState
from visual_slam.orbslam.slam.tracking_core import TrackingCore
from visual_slam.orbslam.utilities.logging import Printer


kVerbose = True
kUseDynamicDesDistanceTh = Parameters.kUseDynamicDesDistanceTh
kUseMotionModel = Parameters.kUseMotionModel or Parameters.kUseSearchFrameByProjection
kUseSearchFrameByProjection = (
    Parameters.kUseSearchFrameByProjection and not Parameters.kUseEssentialMatrixFitting
)
kNumMinInliersPoseOptimizationTrackFrame = 10
kNumMinInliersPoseOptimizationTrackLocalMap = 20
kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle = 50
kNumMinObsForKeyFrameDefault = 3
kMinDepth = Parameters.kMinDepth


class TrackingHistory(object):
    def __init__(self):
        self.relative_frame_poses = []
        self.kf_references = []
        self.timestamps = []
        self.ids = []
        self.slam_states = []

    def reset(self):
        self.relative_frame_poses.clear()
        self.kf_references.clear()
        self.timestamps.clear()
        self.ids.clear()
        self.slam_states.clear()


class Tracking:
    def __init__(self, slam):
        self.slam = slam
        self.motion_model = MotionModel()

        self.descriptor_distance_sigma = FeatureTrackerShared.feature_manager.max_descriptor_distance
        self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMap
        if self.sensor_type == SensorType.RGBD:
            self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMapRgbd

        self.max_frames_between_kfs = int(self.camera.fps) if getattr(self.camera, "fps", None) is not None else 1
        self.max_frames_between_kfs_after_reloc = self.max_frames_between_kfs
        self.min_frames_between_kfs = 0

        self.far_points_threshold = None
        self.use_fov_centers_based_kf_generation = False
        self.max_fov_centers_distance = -1

        self.state = SlamState.NO_IMAGES_YET

        self.num_matched_kps = None
        self.num_inliers = None
        self.num_matched_map_points = None
        self.num_matched_map_points_in_last_pose_opt = None
        self.num_kf_ref_tracked_points = None

        self.last_num_static_stereo_map_points = None
        self.total_num_static_stereo_map_points = 0
        self.last_reloc_frame_id = -float("inf")

        self.pose_is_ok = False
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None
        self.velocity = None

        self.f_cur: Frame | None = None
        self.idxs_cur = None
        self.f_ref: Frame | None = None
        self.idxs_ref = None

        self.kf_ref = None
        self.kf_last = None
        self.kid_last_BA = -1

        self.local_keyframes = []
        self.local_points = []
        self.vo_points: list[MapPoint] = []
        self.tracking_history = TrackingHistory()

        self.init_history = True
        self.poses = []
        self.pose_timestamps = []
        self.traj3d_est = []

    @property
    def feature_tracker(self):
        return self.slam.feature_tracker

    @property
    def map(self) -> Map:
        return self.slam.map

    @property
    def camera(self):
        return self.slam.camera

    @property
    def sensor_type(self):
        return self.slam.sensor_type

    @property
    def local_mapping(self):
        return getattr(self.slam, "local_mapping", None)

    def reset(self):
        Printer.orange("Tracking: reset...")
        self.motion_model.reset()
        self.state = SlamState.NO_IMAGES_YET

        self.num_matched_kps = None
        self.num_inliers = None
        self.num_matched_map_points = None
        self.num_matched_map_points_in_last_pose_opt = None
        self.num_kf_ref_tracked_points = None

        self.last_num_static_stereo_map_points = None
        self.total_num_static_stereo_map_points = 0
        self.pose_is_ok = False
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None
        self.velocity = None

        self.f_cur = None
        self.idxs_cur = None
        self.f_ref = None
        self.idxs_ref = None

        self.kf_ref = None
        self.kf_last = None
        self.kid_last_BA = -1

        self.local_keyframes.clear()
        self.local_points.clear()
        self.vo_points.clear()
        self.tracking_history.reset()

        self.init_history = True
        self.poses.clear()
        self.pose_timestamps.clear()
        self.traj3d_est.clear()

    def pose_optimization(self, f_cur: Frame, name=""):
        pose_before = f_cur.pose()

        num_inliers, mean_chi2 = g2o_pose_optimization(f_cur, verbose=False)

        self.num_matched_map_points_in_last_pose_opt = int(num_inliers)
        self.mean_pose_opt_chi2_error = float(mean_chi2)
        self.pose_is_ok = (
            np.isfinite(mean_chi2)
            and num_inliers >= Parameters.kRelocalizationPoseOpt1MinMatches
        )

        if not self.pose_is_ok:
            f_cur.update_pose(g2o.Isometry3d(pose_before))

        return self.pose_is_ok, self.mean_pose_opt_chi2_error

    def track_previous_frame(self, f_ref: Frame, f_cur: Frame):
        ensure_frame_feature_arrays(f_ref)
        ensure_frame_feature_arrays(f_cur)

        is_search_frame_by_projection_failure = False

        use_search_frame_by_projection = (
            self.motion_model.is_ok and kUseSearchFrameByProjection and kUseMotionModel
        )

        if use_search_frame_by_projection:
            search_radius = Parameters.kMaxReprojectionDistanceFrame
            if self.sensor_type != SensorType.STEREO:
                search_radius = Parameters.kMaxReprojectionDistanceFrameNonStereo

            f_cur.reset_points()

            idxs_ref, idxs_cur, num_found_map_pts = ProjectionMatcher.search_frame_by_projection(
                f_ref,
                f_cur,
                max_reproj_distance=search_radius,
                max_descriptor_distance=self.descriptor_distance_sigma,
                ratio_test=Parameters.kMatchRatioTestFrameByProjection,
                is_monocular=(self.sensor_type == SensorType.MONOCULAR),
            )

            self.num_matched_kps = len(idxs_cur)

            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()

                idxs_ref, idxs_cur, num_found_map_pts = ProjectionMatcher.search_frame_by_projection(
                    f_ref,
                    f_cur,
                    max_reproj_distance=2 * search_radius,
                    max_descriptor_distance=self.descriptor_distance_sigma,
                    ratio_test=Parameters.kMatchRatioTestFrameByProjection,
                    is_monocular=(self.sensor_type == SensorType.MONOCULAR),
                )
                self.num_matched_kps = len(idxs_cur)

            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()
                is_search_frame_by_projection_failure = True
                Printer.red("Not enough matches in search frame by projection:", self.num_matched_kps)
            else:
                self.idxs_ref = idxs_ref
                self.idxs_cur = idxs_cur

                pose_before_pos_opt = f_cur.pose()

                self.pose_optimization(f_cur, "proj-frame-frame")
                self.num_matched_map_points = f_cur.clean_outlier_map_points()

                if (
                    not self.pose_is_ok
                    or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame
                ):
                    Printer.red(
                        "failure in tracking previous frame, # matched map points:",
                        self.num_matched_map_points,
                    )
                    self.pose_is_ok = False
                    f_cur.update_pose(g2o.Isometry3d(pose_before_pos_opt))
                    is_search_frame_by_projection_failure = True

        if not use_search_frame_by_projection or is_search_frame_by_projection_failure:
            self.track_reference_frame(f_ref, f_cur, "match-frame-frame")

        return self.pose_is_ok

    def track_reference_frame(self, f_ref: Frame, f_cur: Frame, name=""):
        if f_ref is not None:
            ensure_frame_feature_arrays(f_ref)
        ensure_frame_feature_arrays(f_cur)

        if f_ref is None:
            self.pose_is_ok = False
            Printer.red("[track_reference_frame]: f_ref is None")
            return False

        idxs_ref_map_points = np.asarray(f_ref.get_matched_good_points_idxs(), dtype=int)

        if len(idxs_ref_map_points) == 0:
            self.pose_is_ok = False
            Printer.orange("[track_reference_frame]: reference frame has no valid map points")
            return False

        des_ref = f_ref.des[idxs_ref_map_points]
        kps_ref = [f_ref.kps[i] for i in idxs_ref_map_points]

        matching_result = FeatureTrackerShared.feature_matcher.match(
            f_cur.img,
            f_ref.img,
            f_cur.des,
            des_ref,
            kps1=f_cur.kps,
            kps2=kps_ref,
        )

        idxs_cur = (
            np.asarray(matching_result.idxs1, dtype=int)
            if matching_result.idxs1 is not None
            else np.array([], dtype=int)
        )
        idxs_ref_local = (
            np.asarray(matching_result.idxs2, dtype=int)
            if matching_result.idxs2 is not None
            else np.array([], dtype=int)
        )

        idxs_ref = idxs_ref_map_points[idxs_ref_local] if len(idxs_ref_local) > 0 else np.array([], dtype=int)
        self.num_matched_kps = len(idxs_cur)

        if FeatureTrackerShared.oriented_features and len(idxs_cur) > 0 and len(idxs_ref) > 0:
            valid_match_idxs = RotationHistogram.filter_matches_with_histogram_orientation(
                idxs_cur,
                idxs_ref,
                f_cur.angles,
                f_ref.angles,
            )

            des_distances = FeatureTrackerShared.descriptor_distances(
                f_cur.des[idxs_cur],
                f_ref.des[idxs_ref],
            )

            valid_match_idxs = np.intersect1d(
                valid_match_idxs,
                np.where(des_distances <= 0.5 * self.descriptor_distance_sigma)[0],
            )

            if len(valid_match_idxs) > 0:
                idxs_cur = idxs_cur[valid_match_idxs]
                idxs_ref = idxs_ref[valid_match_idxs]
            else:
                idxs_cur = np.array([], dtype=int)
                idxs_ref = np.array([], dtype=int)

        self.num_matched_kps = len(idxs_cur)

        if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchReferenceFrame:
            self.pose_is_ok = False
            Printer.orange("Not enough matches in reference-frame matching:", self.num_matched_kps)
            return False

        if Parameters.kUseEssentialMatrixFitting:
            idxs_ref, idxs_cur, self.num_inliers = TrackingCore.estimate_pose_by_fitting_ess_mat(
                f_ref,
                f_cur,
                idxs_ref,
                idxs_cur,
            )
            self.num_matched_kps = len(idxs_cur)

        max_descriptor_distance = (
            self.descriptor_distance_sigma
            if not getattr(f_ref, "is_keyframe", False)
            else 0.5 * self.descriptor_distance_sigma
        )

        num_found_map_pts_inter_frame, idx_ref_prop, idx_cur_prop = (
            TrackingCore.propagate_map_point_matches(
                f_ref,
                f_cur,
                idxs_ref,
                idxs_cur,
                max_descriptor_distance=max_descriptor_distance,
            )
        )

        self.idxs_ref = idxs_ref
        self.idxs_cur = idxs_cur

        pose_before_pos_opt = f_cur.pose()

        self.pose_optimization(f_cur, name)
        self.num_matched_map_points = f_cur.clean_outlier_map_points()

        if (
            not self.pose_is_ok
            or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame
        ):
            f_cur.remove_frame_views(idxs_cur)
            f_cur.reset_points()
            Printer.red(
                f"failure in tracking reference {f_ref.id}, # matched map points:",
                self.num_matched_map_points,
            )
            self.pose_is_ok = False
            f_cur.update_pose(g2o.Isometry3d(pose_before_pos_opt))
            return False

        return True

    def track_keyframe(self, keyframe: KeyFrame, f_cur: Frame, name="match-frame-keyframe"):
        f_cur.update_pose(keyframe.pose())
        return self.track_reference_frame(keyframe, f_cur, name)

    def update_local_map(self):
        self.f_cur.clean_bad_map_points()

        reference = self.kf_ref
        if reference is None:
            reference = self.kf_last

        if reference is not None:
            self.map.update_local_map(reference)
            self.local_keyframes = self.map.get_local_keyframes().to_list()
            self.local_points = self.map.get_local_points().to_list()
            self.kf_ref = reference
            self.f_cur.kf_ref = reference
        else:
            self.local_keyframes = []
            self.local_points = []

    def track_local_map(self):
        if self.f_cur is not None:
            ensure_frame_feature_arrays(self.f_cur)

        if self.f_cur is None:
            self.pose_is_ok = False
            return False

        self.update_local_map()

        if len(self.local_points) > 0:
            found_pts_count, found_pts_fidxs = ProjectionMatcher.search_map_by_projection(
                self.local_points,
                self.f_cur,
                max_reproj_distance=self.reproj_err_frame_map_sigma,
                max_descriptor_distance=self.descriptor_distance_sigma,
                ratio_test=Parameters.kMatchRatioTestMap,
                far_points_threshold=self.far_points_threshold,
            )
        else:
            found_pts_count = 0

        pose_before_pos_opt = self.f_cur.pose()
        self.pose_optimization(self.f_cur, "local-map")

        self.num_matched_map_points = self.f_cur.clean_outlier_map_points()

        if (
            not self.pose_is_ok
            or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackLocalMap
        ):
            self.f_cur.update_pose(g2o.Isometry3d(pose_before_pos_opt))
            self.pose_is_ok = False
            return False

        return True

    def clean_vo_points(self):
        self.vo_points = [p for p in self.vo_points if p is not None and not p.is_bad()]

    def need_new_keyframe(self):
        if self.f_cur is None:
            return False

        if self.kf_last is None:
            return True

        if self.sensor_type == SensorType.MONOCULAR:
            return False

        if self.local_mapping is not None:
            is_idle = getattr(self.local_mapping, "is_idle", lambda: True)()
        else:
            is_idle = True

        num_kfs = self.map.num_keyframes()

        if num_kfs == 0:
            return True

        frames_since_last_kf = self.f_cur.id - self.kf_last.id

        num_tracked_close, num_non_tracked_close, _ = (
            TrackingCore.count_tracked_and_non_tracked_close_points(self.f_cur, self.sensor_type)
        )

        need_to_insert_close = (
            num_tracked_close < Parameters.kNumMinTrackedClosePointsForNewKfNonMonocular
            and num_non_tracked_close > Parameters.kNumMaxNonTrackedClosePointsForNewKfNonMonocular
        )

        if self.num_kf_ref_tracked_points is None:
            self.num_kf_ref_tracked_points = max(1, len(self.kf_ref.get_matched_good_points()) if self.kf_ref else 1)

        ref_ratio = Parameters.kThNewKfRefRatioStereo

        c1a = frames_since_last_kf >= self.max_frames_between_kfs
        c1b = frames_since_last_kf >= self.min_frames_between_kfs and is_idle
        c2 = (
            self.num_matched_map_points is not None
            and self.num_matched_map_points < ref_ratio * self.num_kf_ref_tracked_points
        ) or need_to_insert_close

        return bool((c1a or c1b) and c2)

    def create_new_keyframe(self, img=None):
        if self.f_cur is None:
            return None

        kf_new = KeyFrame(self.f_cur, img=img)
        self.map.add_keyframe(kf_new)

        kf_new.init_observations()

        if self.sensor_type != SensorType.MONOCULAR:
            count = self.create_and_add_stereo_map_points_on_new_kf(self.f_cur, kf_new, img=img)
            self.last_num_static_stereo_map_points = count
            self.total_num_static_stereo_map_points += count

        kf_new.update_connections()

        self.kf_last = kf_new
        self.kf_ref = kf_new
        self.f_cur.kf_ref = kf_new
        self.clean_vo_points()

        if self.local_mapping is not None and hasattr(self.local_mapping, "insert_keyframe"):
            self.local_mapping.insert_keyframe(kf_new)

        return kf_new

    def relocalize(self, frame: Frame):
        Printer.orange("Relocalization is deferred until keyframe database / loop-closing port.")
        return False

    def create_vo_points_on_last_frame(self):
        if self.f_ref is None:
            return []
        self.vo_points = TrackingCore.create_vo_points(self.f_ref)
        return self.vo_points

    def create_and_add_stereo_map_points_on_new_kf(self, f, kf, img=None):
        return TrackingCore.create_and_add_stereo_map_points_on_new_kf(f, kf, self.map, img)

    def wait_for_local_mapping(self):
        # Full pySLAM threading synchronization will be ported with local_mapping.py.
        return True

    def update_history(self):
        if self.f_cur is None:
            return

        self.poses.append(self.f_cur.pose())
        self.pose_timestamps.append(self.f_cur.timestamp)

        if self.kf_ref is not None:
            Tcr = self.f_cur.pose() @ np.linalg.inv(self.kf_ref.pose())
            self.tracking_history.relative_frame_poses.append(g2o.Isometry3d(Tcr))
            self.tracking_history.kf_references.append(self.kf_ref)
            self.tracking_history.timestamps.append(self.f_cur.timestamp)
            self.tracking_history.ids.append(self.f_cur.id)
            self.tracking_history.slam_states.append(self.state)

    def _create_initial_rgbd_map(self, f_cur: Frame, img=None):
        f_cur.update_pose(g2o.Isometry3d(np.eye(4, dtype=np.float64)))

        kf0 = KeyFrame(f_cur, img=img)
        self.map.add_keyframe(kf0)

        num_created = TrackingCore.create_and_add_stereo_map_points_on_new_kf(
            f_cur,
            kf0,
            self.map,
            img=img,
        )

        kf0.update_connections()

        self.kf_ref = kf0
        self.kf_last = kf0
        self.f_ref = f_cur
        self.f_cur = f_cur

        self.motion_model.update_pose_from_matrix(f_cur.timestamp, f_cur.pose())

        self.state = SlamState.OK
        self.pose_is_ok = True
        self.num_matched_map_points = num_created

        return num_created >= Parameters.kInitializerNumMinTriangulatedPointsStereo

    def track(
        self,
        img,
        img_right=None,
        depth=None,
        img_id=None,
        timestamp=None,
        mask=None,
        mask_right=None,
    ):
        """
        pySLAM-style tracking entry point.

        Signature follows pySLAM:
            track(img, img_right, depth, img_id, timestamp, mask, mask_right)

        The mask arguments are accepted for API compatibility and are reserved
        for the later masked feature-extraction path.
        """
        f_cur = Frame(
            camera=self.camera,
            img=img,
            depth_img=depth,
            pose=g2o.Isometry3d(np.eye(4, dtype=np.float64)),
            timestamp=timestamp,
            img_id=img_id,
            img_right=img_right,
        )

        self.map.add_frame(f_cur)
        self.f_cur = f_cur

        if self.state in (SlamState.NO_IMAGES_YET, SlamState.NOT_INITIALIZED):
            ok = self._create_initial_rgbd_map(f_cur, img=img)
            self.update_history()
            return ok

        if self.motion_model.is_ok:
            predicted_pose, _ = self.motion_model.predict_pose(timestamp)
            f_cur.update_pose(predicted_pose)
        elif self.f_ref is not None:
            f_cur.update_pose(self.f_ref.pose())

        ok_prev = self.track_previous_frame(self.f_ref, f_cur) if self.f_ref is not None else False

        if ok_prev:
            self.track_local_map()

        if self.pose_is_ok:
            self.motion_model.update_pose_from_matrix(timestamp, f_cur.pose())
            self.state = SlamState.OK

            if self.need_new_keyframe():
                self.create_new_keyframe(img=img)
        else:
            self.state = SlamState.LOST

        self.f_ref = f_cur
        self.update_history()

        return self.pose_is_ok
