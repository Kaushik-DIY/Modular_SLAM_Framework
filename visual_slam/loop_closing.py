"""
=============================================================================
visual_slam/loop_closing.py

Loop closing for Visual SLAM.

IMPORTANT: This module follows pyslam's loop closing approach (simplified).
---------------------------------------------------------------------------
Ported from pyslam's loop_closing.py.

The LoopCloser detects when the camera returns to a previously visited
location and corrects accumulated drift through:
- Loop detection (simplified: spatial proximity + feature matching)
- Geometric validation (feature matching + pose estimation)
- Pose graph optimization (PGO)
- Global bundle adjustment (GBA)

Classes
-------
LoopCloser
    Detects and corrects loop closures.

Processing Pipeline Per New Keyframe
--------------------------------------
1. Check if enough time has passed since last loop
2. Detect loop candidates (keyframes spatially close)
3. Validate geometrically (feature matching)
4. Compute relative pose constraint
5. Run PGO to distribute correction
6. Run GBA to refine full map

References
----------
pyslam: loop_closing.py
ORB-SLAM2: LoopClosing.cc

=============================================================================
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

try:
    import g2o
except ImportError:
    g2o = None

from visual_slam.types import KeyFrame, Map
from visual_slam.feature_tracker import FeatureTracker
from visual_slam.optimizer import pose_graph_optimization, global_ba
from slam_core.common.types3d import Pose3D


class LoopCloser:
    """
    Loop closing module.
    
    Detects and corrects loop closures to reduce accumulated drift.
    In a full system, this would run on a separate thread.
    
    Attributes
    ----------
    slam_map : Map
        The global map.
    feature_tracker : FeatureTracker
        For matching between keyframes.
    
    Methods
    -------
    detect_and_correct(kf)
        Check for loop closure and correct if found.
    
    Examples
    --------
    >>> loop_closer = LoopCloser(slam_map, feature_tracker)
    >>> # When tracking creates a new keyframe:
    >>> loop_closer.detect_and_correct(new_kf)
    """
    
    def __init__(
        self,
        slam_map: Map,
        feature_tracker: FeatureTracker,
    ):
        """
        Initialize loop closer.
        
        Parameters
        ----------
        slam_map : Map
            The global map.
        feature_tracker : FeatureTracker
            Feature matcher.
        """
        self.slam_map = slam_map
        self.feature_tracker = feature_tracker
        
        # Loop detection parameters
        self.min_keyframes_between_loops = 10  # Min KFs between loop attempts
        self.loop_spatial_threshold = 3.0  # Max distance (m) for loop candidate
        self.min_loop_matches = 50  # Min feature matches to validate loop
        
        self.last_loop_kf_id = -1  # Last KF where loop was closed
    
    def detect_and_correct(self, kf: KeyFrame) -> bool:
        """
        Detect and correct loop closure for new keyframe.
        
        Parameters
        ----------
        kf : KeyFrame
            New keyframe from tracking.
        
        Returns
        -------
        bool
            True if loop was detected and corrected.
        """
        # Check if enough time has passed since last loop
        if kf.keyframe_id - self.last_loop_kf_id < self.min_keyframes_between_loops:
            return False
        
        # Detect loop candidates
        candidates = self._detect_loop_candidates(kf)
        
        if len(candidates) == 0:
            return False
        
        print(f"  LoopClosing: Found {len(candidates)} candidates for KF {kf.keyframe_id}")
        
        # Validate candidates geometrically
        for candidate in candidates:
            loop_constraint = self._validate_loop_candidate(kf, candidate)
            
            if loop_constraint is not None:
                print(f"    Loop validated: KF {kf.keyframe_id} <-> KF {candidate.keyframe_id}")
                self._correct_loop(kf, candidate, loop_constraint)
                self.last_loop_kf_id = kf.keyframe_id
                return True
        
        return False
    
    def _detect_loop_candidates(self, kf: KeyFrame) -> List[KeyFrame]:
        """
        Detect potential loop closure candidates.
        
        Simplified version: uses spatial proximity.
        Full ORB-SLAM2 would use DBoW3 place recognition.
        
        Parameters
        ----------
        kf : KeyFrame
            Query keyframe.
        
        Returns
        -------
        List[KeyFrame]
            Candidate keyframes for loop closure.
        """
        if kf.frame.pose_world is None:
            return []
        
        candidates = []
        kf_pos = kf.frame.pose_world.matrix()[:3, 3]
        
        for candidate in self.slam_map.keyframes.values():
            if candidate.is_bad or candidate.keyframe_id >= kf.keyframe_id:
                continue
            
            # Skip recent keyframes (need temporal separation)
            if kf.keyframe_id - candidate.keyframe_id < self.min_keyframes_between_loops:
                continue
            
            if candidate.frame.pose_world is None:
                continue
            
            # Check spatial proximity
            candidate_pos = candidate.frame.pose_world.matrix()[:3, 3]
            dist = np.linalg.norm(kf_pos - candidate_pos)
            
            if dist < self.loop_spatial_threshold:
                candidates.append(candidate)
        
        return candidates
    
    def _validate_loop_candidate(
        self,
        kf: KeyFrame,
        candidate: KeyFrame,
    ) -> Optional[Pose3D]:
        """
        Validate loop candidate geometrically.
        
        Performs feature matching and checks if enough matches exist.
        Computes relative pose constraint if validated.
        
        Parameters
        ----------
        kf : KeyFrame
            Query keyframe.
        candidate : KeyFrame
            Candidate keyframe.
        
        Returns
        -------
        g2o.Isometry3d or None
            Relative pose constraint (T_candidate_from_kf), or None if invalid.
        """
        # Match features
        matches = self.feature_tracker.match_frames(kf.frame, candidate.frame)
        
        if len(matches) < self.min_loop_matches:
            return None
        
        print(f"      Matched {len(matches)} features")
        
        # Compute relative pose from current poses
        # In full ORB-SLAM2, would use PnP with matched features
        # For simplicity, use current pose estimates
        
        T_world_kf = kf.frame.pose_world.matrix()
        T_world_candidate = candidate.frame.pose_world.matrix()
        
        # T_candidate_from_kf = T_world_candidate^(-1) * T_world_kf
        T_candidate_inv = np.linalg.inv(T_world_candidate)
        T_relative = T_candidate_inv @ T_world_kf
        
        return g2o.Isometry3d(T_relative)
    
    def _correct_loop(
        self,
        kf: KeyFrame,
        loop_kf: KeyFrame,
        relative_pose: Pose3D,
    ) -> None:
        """
        Correct loop closure.
        
        1. Run pose graph optimization (PGO) with loop constraint
        2. Run global bundle adjustment (GBA)
        
        Parameters
        ----------
        kf : KeyFrame
            Current keyframe.
        loop_kf : KeyFrame
            Loop keyframe (old).
        relative_pose : g2o.Isometry3d
            Relative pose constraint.
        """
        print(f"    Correcting loop: KF {kf.keyframe_id} -> KF {loop_kf.keyframe_id}")
        
        # Collect all keyframes
        keyframes = list(self.slam_map.keyframes.values())
        
        # Build loop closure edges
        # Edge: from current KF to loop KF with relative pose
        loop_edges = [
            (kf.keyframe_id, loop_kf.keyframe_id, relative_pose)
        ]
        
        # Add odometry edges (sequential keyframes)
        # In full system, would use covisibility graph
        for i in range(len(keyframes) - 1):
            kf1 = keyframes[i]
            kf2 = keyframes[i + 1]
            
            if kf1.is_bad or kf2.is_bad:
                continue
            
            if kf1.frame.pose_world is None or kf2.frame.pose_world is None:
                continue
            
            # Relative pose between consecutive keyframes
            T1 = kf1.frame.pose_world.matrix()
            T2 = kf2.frame.pose_world.matrix()
            T1_inv = np.linalg.inv(T1)
            T_rel = T1_inv @ T2
            
            loop_edges.append(
                (kf1.keyframe_id, kf2.keyframe_id, g2o.Isometry3d(T_rel))
            )
        
        # Run pose graph optimization
        print(f"    Running PGO with {len(keyframes)} keyframes, {len(loop_edges)} edges")
        pose_graph_optimization(keyframes, loop_edges, iterations=20)
        
        # Run global bundle adjustment
        print(f"    Running global BA")
        global_ba(self.slam_map, iterations=10)
        
        print(f"    Loop closure corrected!")