"""
=============================================================================
visual_slam/local_mapping.py

Local mapping back-end for Visual SLAM.

IMPORTANT: This module follows pyslam's local mapping approach.
---------------------------------------------------------------------------
Ported from pyslam's local_mapping.py.

The LocalMapper processes new keyframes to refine and expand the map:
- Triangulate new map points from unmatched features
- Run local bundle adjustment
- Cull bad map points
- Cull redundant keyframes

This runs in parallel with tracking (in a real system, would be on separate thread).

Classes
-------
LocalMapper
    Processes new keyframes to maintain and expand the map.

Processing Pipeline Per Keyframe
----------------------------------
1. Process new keyframe (compute BoW for loop closure)
2. Cull recent map points with low tracking quality
3. Create new map points via triangulation
4. Run local bundle adjustment
5. Cull redundant local keyframes

References
----------
pyslam: local_mapping.py
ORB-SLAM2: LocalMapping.cc

=============================================================================
"""

from __future__ import annotations

from typing import List, Set, Tuple, Optional
import numpy as np

try:
    import g2o
except ImportError:
    g2o = None

from visual_slam.types import Frame, KeyFrame, MapPoint, Map
from visual_slam.feature_tracker import FeatureTracker
from visual_slam.optimizer import local_ba
from slam_core.common.types3d import CameraIntrinsics


class LocalMapper:
    """
    Local mapping back-end.
    
    Processes new keyframes to refine and expand the map.
    In a full system, this would run on a separate thread.
    
    Attributes
    ----------
    slam_map : Map
        The global map (shared with tracking).
    feature_tracker : FeatureTracker
        For matching between keyframes.
    
    Methods
    -------
    process_new_keyframe(kf)
        Process a new keyframe from tracking.
    
    Examples
    --------
    >>> mapper = LocalMapper(slam_map, feature_tracker)
    >>> # When tracking creates a new keyframe:
    >>> mapper.process_new_keyframe(new_kf)
    """
    
    def __init__(
        self,
        slam_map: Map,
        feature_tracker: FeatureTracker,
    ):
        """
        Initialize local mapper.
        
        Parameters
        ----------
        slam_map : Map
            The global map.
        feature_tracker : FeatureTracker
            Feature matcher.
        """
        self.slam_map = slam_map
        self.feature_tracker = feature_tracker
        
        # Quality thresholds
        self.min_found_ratio = 0.25  # Cull MPs with found_ratio < 0.25
        self.recent_map_points_age = 2  # MPs created in last N KFs are "recent"
    
    def process_new_keyframe(self, kf: KeyFrame) -> None:
        """
        Process a new keyframe.
        
        Main local mapping pipeline.
        
        Parameters
        ----------
        kf : KeyFrame
            New keyframe from tracking.
        """
        print(f"  LocalMapping: Processing KF {kf.keyframe_id}")
        
        # 1. Compute BoW vector (for future loop closure)
        # Skipped for now - would need DBoW3 vocabulary
        
        # 2. Cull recent map points with low quality
        self._cull_map_points()
        
        # 3. Create new map points via triangulation
        num_created = self._create_new_map_points(kf)
        print(f"    Created {num_created} new map points")
        
        # 4. Run local bundle adjustment
        self._local_bundle_adjustment(kf)
        
        # 5. Cull redundant keyframes
        # Skipped for simplicity - would check if KF observations are redundant
    
    def _cull_map_points(self) -> None:
        """
        Remove low-quality map points.
        
        A map point is culled if:
        - found_ratio < threshold (not reliably tracked)
        - Too few observations (but only if not from first KF)
        """
        to_remove = []
        
        # Don't cull if we have very few keyframes (map is just starting)
        if len(self.slam_map.keyframes) <= 1:
            return  # Don't cull anything on first keyframe
        
        for mp_id, mp in self.slam_map.map_points.items():
            if mp.is_bad:
                continue
            
            # Check found ratio (if tracked)
            if hasattr(mp, 'found_ratio') and mp.found_ratio < self.min_found_ratio:
                to_remove.append(mp)
                continue
            
            # Check minimum observations
            if len(mp.observations) < 2:
                to_remove.append(mp)
                continue
        
        for mp in to_remove:
            self.slam_map.remove_map_point(mp)
    
    def _create_new_map_points(self, kf: KeyFrame) -> int:
        """
        Create new map points via triangulation.
        
        Triangulates unmatched features between the new keyframe and
        its connected keyframes.
        
        Parameters
        ----------
        kf : KeyFrame
            New keyframe.
        
        Returns
        -------
        int
            Number of new map points created.
        """
        # Get connected keyframes (covisible neighbors)
        connected_kfs = kf.get_best_covisible_keyframes(n=10)
        
        if len(connected_kfs) == 0:
            # No neighbors yet, use all keyframes
            connected_kfs = [k for k in self.slam_map.keyframes.values() 
                            if k.keyframe_id != kf.keyframe_id and not k.is_bad]
        
        num_created = 0
        
        for ref_kf in connected_kfs[:5]:  # Limit to 5 neighbors
            new_points = self._triangulate_between_keyframes(kf, ref_kf)
            num_created += new_points
        
        return num_created
    
    def _triangulate_between_keyframes(
        self,
        kf1: KeyFrame,
        kf2: KeyFrame,
    ) -> int:
        """
        Triangulate new map points between two keyframes.
        
        Matches unmatched features and triangulates their 3D positions.
        
        Parameters
        ----------
        kf1, kf2 : KeyFrame
            Keyframes to triangulate between.
        
        Returns
        -------
        int
            Number of new map points created.
        """
        # Match features between keyframes
        matches = self.feature_tracker.match_frames(kf1.frame, kf2.frame)
        
        if len(matches) == 0:
            return 0
        
        num_created = 0
        
        for m in matches:
            idx1 = m.queryIdx
            idx2 = m.trainIdx
            
            # Skip if already have map points
            if (idx1 < len(kf1.frame.map_point_matches) and 
                kf1.frame.map_point_matches[idx1] is not None):
                continue
            if (idx2 < len(kf2.frame.map_point_matches) and 
                kf2.frame.map_point_matches[idx2] is not None):
                continue
            
            # Get depths (for RGBD, use depth directly)
            depth1 = kf1.frame.depths[idx1] if idx1 < len(kf1.frame.depths) else -1
            depth2 = kf2.frame.depths[idx2] if idx2 < len(kf2.frame.depths) else -1
            
            # For RGBD, we can back-project directly from depth
            # Use depth from kf1 (could average both)
            if depth1 > 0 and depth1 < 10.0:
                kp = kf1.frame.keypoints[idx1]
                cam = kf1.frame.camera
                
                # Back-project to camera frame
                u, v = kp.pt
                X_cam = (u - cam.cx) * depth1 / cam.fx
                Y_cam = (v - cam.cy) * depth1 / cam.fy
                Z_cam = depth1
                
                # Transform to world frame
                T = kf1.frame.pose_world.matrix()
                p_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
                p_world = (T @ p_cam)[:3]
                
                # Sanity check: reject map points too far from origin
                if np.linalg.norm(p_world) > 50.0:
                    continue  # Skip this point, likely from bad pose
                
                if not np.all(np.isfinite(p_world)):
                    continue  # Skip NaN/Inf
                
                # Create map point
                mp = MapPoint(position_world=p_world)
                mp.add_observation(kf1, idx1)
                mp.add_observation(kf2, idx2)
                mp.compute_descriptor()
                
                self.slam_map.add_map_point(mp)
                
                # Update keyframe matches
                while len(kf1.frame.map_point_matches) <= idx1:
                    kf1.frame.map_point_matches.append(None)
                while len(kf2.frame.map_point_matches) <= idx2:
                    kf2.frame.map_point_matches.append(None)
                
                kf1.frame.map_point_matches[idx1] = mp
                kf2.frame.map_point_matches[idx2] = mp
                
                num_created += 1
        
        return num_created
    
    def _local_bundle_adjustment(self, kf: KeyFrame) -> None:
        """
        Run local bundle adjustment.
        
        Optimizes the new keyframe and its neighbors along with their
        observed map points.
        
        Parameters
        ----------
        kf : KeyFrame
            New keyframe (center of local window).
        """
        # Get local keyframes (new KF + neighbors)
        local_kfs = [kf] + kf.get_best_covisible_keyframes(n=5)
        
        # Get local map points (observed by local KFs)
        local_mps: Set[MapPoint] = set()
        for k in local_kfs:
            local_mps.update(k.map_points)
        
        # Get fixed keyframes (observe local MPs but not in local window)
        fixed_kfs: List[KeyFrame] = []
        for mp in local_mps:
            for obs_kf in mp.observations.keys():
                if obs_kf not in local_kfs and not obs_kf.is_bad:
                    fixed_kfs.append(obs_kf)
        
        # Remove duplicates
        fixed_kfs = list(set(fixed_kfs))
        
        # Run local BA
        if len(local_kfs) > 0 and len(local_mps) > 0:
            local_ba(
                local_keyframes=local_kfs,
                local_map_points=list(local_mps),
                fixed_keyframes=fixed_kfs,
                iterations=5,
            )
            print(f"    Local BA: {len(local_kfs)} KFs, {len(local_mps)} MPs, {len(fixed_kfs)} fixed KFs")