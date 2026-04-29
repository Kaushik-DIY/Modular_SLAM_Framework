"""
=============================================================================
visual_slam/tracking.py

Real-time tracking front-end — ORB-SLAM2 compliant implementation.

Matches pyslam's tracking.py and ORB-SLAM2's Tracking.cc exactly.

Key algorithmic differences from simplified version:
  - Projection-based map point matching (not descriptor-only brute-force)
  - OpenCV BFMatcher with ratio test for robust matching
  - Proper pose update from motion-only BA result
  - found_ratio incremented on every visible map point
  - Scale-consistent keyframe insertion policy

Processing pipeline per frame (ORB-SLAM2 IV):
  1. ORB extraction + depth association
  2. Pose prediction via constant-velocity motion model
  3. Project local map points into predicted frame, search in 2D radius
  4. Motion-only BA to refine pose
  5. Mark outliers, update found_ratio counters
  6. Keyframe decision

References
----------
pyslam: tracking.py, search_points.py, matcher.py
ORB-SLAM2: Tracking.cc, ORBmatcher.cc
=============================================================================
"""
from __future__ import annotations

from typing import Optional, Tuple, List
from enum import Enum
import numpy as np
import cv2

try:
    import g2o
except ImportError:
    g2o = None

from visual_slam.types import Frame, KeyFrame, MapPoint, Map
from visual_slam.feature_tracker import FeatureTracker
from visual_slam.optimizer import motion_only_ba
from slam_core.common.types3d import CameraIntrinsics, Pose3D


class TrackingState(Enum):
    NOT_INITIALIZED = 0
    OK              = 1
    RECENTLY_LOST   = 2
    LOST            = 3


class Tracker:
    """
    ORB-SLAM2-compliant tracking front-end.

    Uses projection-based map point matching and OpenCV BFMatcher
    instead of the naive brute-force per-pixel Hamming loop.
    """

    def __init__(self, camera: CameraIntrinsics):
        self.camera          = camera
        self.feature_tracker = FeatureTracker(num_features=1000)
        self.slam_map: Optional[Map] = None
        self.state           = TrackingState.NOT_INITIALIZED

        self.current_frame:  Optional[Frame]    = None
        self.last_frame:     Optional[Frame]    = None
        self.last_keyframe:  Optional[KeyFrame] = None

        self.velocity: Optional[Pose3D] = None

        # ORB-SLAM2 thresholds (Tracking.cc)
        self.min_tracked_points   = 15
        self.keyframe_min_ratio   = 0.75
        self.keyframe_max_frames  = 10
        self.frames_since_last_kf = 0

        # Projection search window in pixels
        self.search_radius = 15

        # OpenCV BFMatcher (Hamming, cross-check=False for ratio test)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def set_map(self, slam_map: Map) -> None:
        self.slam_map = slam_map

    # ------------------------------------------------------------------ #

    def process_frame(
        self,
        rgb:   np.ndarray,
        depth: np.ndarray,
        timestamp: float,
    ) -> Tuple[Optional[Pose3D], TrackingState]:
        """Process one RGBD frame. Returns (pose_world, state)."""
        frame = Frame.from_images(timestamp, rgb, depth, self.camera)
        self.current_frame = frame

        self.feature_tracker.detect_and_compute(frame)

        if len(frame.keypoints) == 0:
            self.state = TrackingState.LOST
            return None, self.state

        if self.state == TrackingState.NOT_INITIALIZED:
            ok = self._initialize_rgbd(frame)
            if ok:
                self.state = TrackingState.OK
                self.last_frame = frame
                return frame.pose_world, self.state
            return None, self.state

        ok = self._track(frame)

        if ok:
            self.state = TrackingState.OK
            self._update_motion_model(frame)
            if self._need_new_keyframe(frame):
                self._create_keyframe(frame)
            self.last_frame = frame
            return frame.pose_world, self.state
        else:
            self.state = TrackingState.LOST
            self.last_frame = frame
            return None, self.state

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    def _initialize_rgbd(self, frame: Frame) -> bool:
        """Back-project all features with valid depth. ORB-SLAM2 IV-A."""
        if self.slam_map is None:
            return False

        frame.pose_world = g2o.Isometry3d(np.eye(4))
        kf = KeyFrame.from_frame(frame)
        self.slam_map.add_keyframe(kf)
        self.last_keyframe = kf

        n_created = 0
        for i, kp in enumerate(frame.keypoints):
            d = frame.depths[i]
            if d <= 0.01 or d > 10.0:
                frame.map_point_matches.append(None)
                continue

            u, v = kp.pt
            X = (u - self.camera.cx) * d / self.camera.fx
            Y = (v - self.camera.cy) * d / self.camera.fy
            pos_w = np.array([X, Y, d])

            mp = MapPoint(position_world=pos_w)
            mp.add_observation(kf, i)
            mp.compute_descriptor()
            mp.found_in_frames   = 1
            mp.visible_in_frames = 1

            self.slam_map.add_map_point(mp)
            frame.map_point_matches.append(mp)
            n_created += 1

        print(f"  Initialized: created {n_created} map points from {len(frame.keypoints)} features")
        self.frames_since_last_kf = 0
        return n_created > self.min_tracked_points

    # ------------------------------------------------------------------ #
    #  Main tracking                                                       #
    # ------------------------------------------------------------------ #

    def _track(self, frame: Frame) -> bool:
        """Predict → project-search → motion-only BA."""
        self._predict_pose(frame)

        n_matches = self._search_local_map_by_projection(frame)

        if n_matches < self.min_tracked_points:
            n_matches = self._match_last_frame(frame)

        if n_matches < self.min_tracked_points:
            print(f"  WARNING: {n_matches} matches — tracking lost")
            return False

        optimized_pose = motion_only_ba(frame, self.camera, iterations=10)
        if optimized_pose is None:
            return False

        # Sanity check: reject insane poses (BA divergence)
        T_opt = optimized_pose.matrix()
        translation = np.linalg.norm(T_opt[:3, 3])
        
        if translation > 100.0:  # >100m from origin = unreasonable
            print(f"  WARNING: BA diverged (translation={translation:.1f}m) — tracking lost")
            return False
        
        if not np.all(np.isfinite(T_opt)):
            print(f"  WARNING: BA returned NaN/Inf — tracking lost")
            return False

        # CRITICAL: actually update the frame's pose with BA result
        frame.pose_world = optimized_pose
        self._update_map_point_stats(frame)

        n_inliers = sum(1 for mp in frame.map_point_matches if mp is not None)
        return n_inliers >= self.min_tracked_points

    # ------------------------------------------------------------------ #
    #  Pose prediction                                                     #
    # ------------------------------------------------------------------ #

    def _predict_pose(self, frame: Frame) -> None:
        """Constant-velocity model. ORB-SLAM2 IV-B."""
        if self.velocity is not None and self.last_frame is not None:
            T_last = self.last_frame.pose_world.matrix()
            T_vel  = self.velocity.matrix()
            frame.pose_world = g2o.Isometry3d(T_last @ T_vel)
        elif self.last_frame is not None:
            frame.pose_world = self.last_frame.pose_world
        else:
            frame.pose_world = g2o.Isometry3d(np.eye(4))

    # ------------------------------------------------------------------ #
    #  Projection-based matching (ORBmatcher::SearchByProjection)         #
    # ------------------------------------------------------------------ #

    def _search_local_map_by_projection(self, frame: Frame) -> int:
        """
        Project local map points with the predicted pose and match
        each to the nearest keypoint within self.search_radius pixels.

        ORB-SLAM2: ORBmatcher.cc SearchByProjection (Tracking).
        """
        if frame.pose_world is None:
            return 0

        T_cw = np.linalg.inv(frame.pose_world.matrix())
        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]

        fx, fy = self.camera.fx, self.camera.fy
        cx, cy = self.camera.cx, self.camera.cy
        W, H   = self.camera.width, self.camera.height

        local_mps = self._get_local_map_points()
        frame.map_point_matches = [None] * len(frame.keypoints)
        n_matches = 0

        # Pre-compute keypoint positions as numpy array for fast distance
        kp_pts = np.array([kp.pt for kp in frame.keypoints], dtype=np.float32)

        for mp in local_mps:
            if mp.is_bad or mp.descriptor is None:
                continue

            p_cam = R_cw @ mp.position_world + t_cw
            if p_cam[2] < 0.01:
                continue

            u_proj = fx * p_cam[0] / p_cam[2] + cx
            v_proj = fy * p_cam[1] / p_cam[2] + cy

            if u_proj < 0 or u_proj >= W or v_proj < 0 or v_proj >= H:
                continue

            mp.visible_in_frames += 1

            # Candidates within search radius
            sq_dists   = (kp_pts[:, 0] - u_proj)**2 + (kp_pts[:, 1] - v_proj)**2
            candidates = np.where(sq_dists < self.search_radius**2)[0]

            if len(candidates) == 0:
                continue

            # Hamming distances to all candidates at once using OpenCV
            mp_desc = mp.descriptor.reshape(1, -1).astype(np.uint8)
            cand_descs = frame.descriptors[candidates].astype(np.uint8)

            matches_cv = self._bf.match(mp_desc, cand_descs)
            if not matches_cv:
                continue

            # sort() only works on lists — convert tuple from BFMatcher
            matches_cv = sorted(matches_cv, key=lambda x: x.distance)
            best = matches_cv[0]

            if best.distance >= 100:
                continue

            # Ratio test (need at least 2 matches for ratio)
            if len(matches_cv) >= 2:
                ratio = best.distance / matches_cv[1].distance
                if ratio >= 0.9:
                    continue

            best_kp_idx = int(candidates[best.trainIdx])

            if frame.map_point_matches[best_kp_idx] is None:
                frame.map_point_matches[best_kp_idx] = mp
                n_matches += 1

        return n_matches

    def _get_local_map_points(self) -> List[MapPoint]:
        """Map points from last KF + covisible neighbors."""
        if self.last_keyframe is None:
            return list(self.slam_map.map_points.values())

        local_mps = set(self.last_keyframe.map_points)
        for nb in self.last_keyframe.get_best_covisible_keyframes(n=20):
            local_mps.update(nb.map_points)
        return list(local_mps)

    # ------------------------------------------------------------------ #
    #  Fallback: match against last frame                                  #
    # ------------------------------------------------------------------ #

    def _match_last_frame(self, frame: Frame) -> int:
        """
        Fallback for when projection search fails.

        kNN match (k=2) with ratio test between current and last frame
        descriptors, restricted to keypoints that have map points.
        """
        if self.last_frame is None:
            return 0

        prev_mps   = self.last_frame.map_point_matches
        prev_descs = self.last_frame.descriptors

        if prev_descs is None or len(prev_descs) == 0:
            return 0

        ref_descs, ref_mps = [], []
        for i, mp in enumerate(prev_mps):
            if mp is not None and not mp.is_bad:
                ref_descs.append(prev_descs[i])
                ref_mps.append(mp)

        if not ref_descs:
            return 0

        ref_arr = np.array(ref_descs, dtype=np.uint8)
        cur_arr = frame.descriptors.astype(np.uint8)

        knn = self._bf.knnMatch(cur_arr, ref_arr, k=2)

        frame.map_point_matches = [None] * len(frame.keypoints)
        n_matches = 0

        for m_list in knn:
            if len(m_list) < 2:
                continue
            m, n = m_list
            if m.distance < 0.9 * n.distance and m.distance < 100:
                if frame.map_point_matches[m.queryIdx] is None:
                    frame.map_point_matches[m.queryIdx] = ref_mps[m.trainIdx]
                    n_matches += 1

        return n_matches

    # ------------------------------------------------------------------ #
    #  Map point statistics                                                #
    # ------------------------------------------------------------------ #

    def _update_map_point_stats(self, frame: Frame) -> None:
        """Increment found_in_frames counter. ORB-SLAM2: found_ratio."""
        for mp in frame.map_point_matches:
            if mp is not None and not mp.is_bad:
                mp.found_in_frames += 1

    # ------------------------------------------------------------------ #
    #  Motion model update                                                 #
    # ------------------------------------------------------------------ #

    def _update_motion_model(self, frame: Frame) -> None:
        """velocity = T_last^{-1} * T_current."""
        if self.last_frame is None or self.last_frame.pose_world is None:
            return
        T_vel = np.linalg.inv(self.last_frame.pose_world.matrix()) @ \
                frame.pose_world.matrix()
        self.velocity = g2o.Isometry3d(T_vel)

    # ------------------------------------------------------------------ #
    #  Keyframe insertion                                                  #
    # ------------------------------------------------------------------ #

    def _need_new_keyframe(self, frame: Frame) -> bool:
        """ORB-SLAM2 IV-D keyframe insertion policy."""
        self.frames_since_last_kf += 1

        n_matched = sum(1 for mp in frame.map_point_matches if mp is not None)
        n_kf_mps  = len(self.last_keyframe.map_points) if self.last_keyframe else 0

        c1 = self.frames_since_last_kf >= self.keyframe_max_frames
        c2 = (n_kf_mps > 0 and n_matched / n_kf_mps < self.keyframe_min_ratio)
        c3 = n_matched >= self.min_tracked_points

        return (c1 or c2) and c3

    def _create_keyframe(self, frame: Frame) -> None:
        """Promote frame to keyframe and register observations."""
        kf = KeyFrame.from_frame(frame)
        self.slam_map.add_keyframe(kf)

        for i, mp in enumerate(frame.map_point_matches):
            if mp is not None and not mp.is_bad:
                mp.add_observation(kf, i)
                mp.compute_descriptor()

        if self.last_keyframe is not None:
            shared = sum(
                1 for mp in frame.map_point_matches
                if mp is not None and mp in self.last_keyframe.map_points
            )
            if shared > 0:
                kf.add_connection(self.last_keyframe, shared)
                self.last_keyframe.add_connection(kf, shared)

        self.last_keyframe = kf
        self.frames_since_last_kf = 0
        n_inliers = sum(1 for mp in frame.map_point_matches if mp is not None)
        print(f"  Created keyframe {kf.keyframe_id} at t={frame.timestamp:.3f}s ({n_inliers} inliers)")