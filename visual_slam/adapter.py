"""
=============================================================================
visual_slam/adapter.py

Visual SLAM adapter/orchestrator.

IMPORTANT: This module orchestrates all Visual SLAM components.
---------------------------------------------------------------------------
Integrates tracking, local mapping, and loop closing into a unified
interface for processing RGBD image sequences.

Classes
-------
VisualSlamAdapter
    Main adapter that orchestrates the full SLAM pipeline.

Processing Pipeline
-------------------
For each RGBD frame:
1. Tracking processes the frame (pose estimation)
2. If new keyframe created → Local mapping refines map
3. Every N keyframes → Loop closing checks for loops

The adapter manages the three parallel threads/processes:
- Tracking (real-time, runs every frame)
- Local Mapping (processes new keyframes)
- Loop Closing (runs periodically)

In this simplified version, we run sequentially (no threading).

References
----------
pyslam: slam.py, visual_odometry.py
ORB-SLAM2: System.cc

=============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from visual_slam.types import Map
from visual_slam.tracking import Tracker, TrackingState
from visual_slam.local_mapping import LocalMapper
from visual_slam.loop_closing import LoopCloser
from visual_slam.feature_tracker import FeatureTracker
from slam_core.common.types3d import CameraIntrinsics, Pose3D, PoseEstimate


class VisualSlamAdapter:
    """
    Visual SLAM adapter/orchestrator.
    
    Integrates tracking, local mapping, and loop closing into a
    unified pipeline for processing RGBD sequences.
    
    Attributes
    ----------
    camera : CameraIntrinsics
        Camera parameters.
    slam_map : Map
        The global SLAM map.
    tracker : Tracker
        Real-time tracking front-end.
    mapper : LocalMapper
        Local mapping back-end.
    loop_closer : LoopCloser
        Loop closing module.
    
    Methods
    -------
    process_frame(rgb, depth, timestamp) -> Tuple[PoseEstimate, TrackingState]
        Process one RGBD frame through the full pipeline.
    get_trajectory() -> List[PoseEstimate]
        Get the estimated camera trajectory.
    
    Examples
    --------
    >>> # TUM RGBD camera
    >>> cam = CameraIntrinsics(fx=517.3, fy=516.5, cx=318.6, cy=255.3,
    ...                        width=640, height=480, depth_scale=5000.0)
    >>> 
    >>> # Create SLAM system
    >>> slam = VisualSlamAdapter(cam)
    >>> 
    >>> # Process frames
    >>> for rgb, depth, timestamp in dataset:
    ...     pose, state = slam.process_frame(rgb, depth, timestamp)
    ...     if state == TrackingState.OK:
    ...         print(f"Pose: {pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}")
    """
    
    def __init__(self, camera: CameraIntrinsics):
        """
        Initialize Visual SLAM system.
        
        Parameters
        ----------
        camera : CameraIntrinsics
            Camera parameters.
        """
        self.camera = camera
        
        # Create global map
        self.slam_map = Map()
        
        # Create modules
        feature_tracker = FeatureTracker(num_features=1000)
        
        self.tracker = Tracker(camera)
        self.tracker.set_map(self.slam_map)
        
        self.mapper = LocalMapper(self.slam_map, feature_tracker)
        self.loop_closer = LoopCloser(self.slam_map, feature_tracker)
        
        # Trajectory storage
        self.trajectory: list[PoseEstimate] = []
        
        # Loop closing parameters
        self.frames_since_last_loop_check = 0
        self.loop_check_interval = 20  # Check for loops every 20 frames
        
        # Statistics
        self.frame_count = 0
        self.keyframe_count = 0
        self.loop_count = 0
    
    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
    ) -> Tuple[Optional[PoseEstimate], TrackingState]:
        """
        Process one RGBD frame through the full SLAM pipeline.
        
        This is the main entry point for the SLAM system.
        
        Parameters
        ----------
        rgb : np.ndarray
            RGB image, shape (H, W, 3).
        depth : np.ndarray
            Depth image, shape (H, W).
        timestamp : float
            Frame timestamp.
        
        Returns
        -------
        PoseEstimate or None
            Estimated camera pose, or None if tracking lost.
        TrackingState
            Current tracking state.
        """
        self.frame_count += 1
        
        # 1. Tracking: estimate pose for current frame
        num_kfs_before = len(self.slam_map.keyframes)
        pose_g2o, state = self.tracker.process_frame(rgb, depth, timestamp)
        num_kfs_after = len(self.slam_map.keyframes)
        
        # Check if new keyframe was created
        new_keyframe_created = (num_kfs_after > num_kfs_before)
        
        if new_keyframe_created:
            self.keyframe_count += 1
            new_kf = self.slam_map.keyframes[num_kfs_after - 1]
            
            # 2. Local Mapping: process new keyframe
            self.mapper.process_new_keyframe(new_kf)
            
            # 3. Loop Closing: check periodically
            self.frames_since_last_loop_check += 1
            if self.frames_since_last_loop_check >= self.loop_check_interval:
                loop_detected = self.loop_closer.detect_and_correct(new_kf)
                if loop_detected:
                    self.loop_count += 1
                self.frames_since_last_loop_check = 0
        
        # Convert pose to PoseEstimate for cross-SLAM compatibility
        if pose_g2o is not None:
            pose_estimate = self._to_pose_estimate(pose_g2o, timestamp, state)
            self.trajectory.append(pose_estimate)
            return pose_estimate, state
        else:
            return None, state
    
    def _to_pose_estimate(
        self,
        pose_g2o: Pose3D,
        timestamp: float,
        state: TrackingState,
    ) -> PoseEstimate:
        """
        Convert g2o.Isometry3d to PoseEstimate.
        
        Parameters
        ----------
        pose_g2o : g2o.Isometry3d
            Pose in g2o format.
        timestamp : float
            Frame timestamp.
        state : TrackingState
            Current tracking state.
        
        Returns
        -------
        PoseEstimate
            Pose in framework-standard format.
        """
        T = pose_g2o.matrix()
        
        # Compute confidence from tracking state
        if state == TrackingState.OK:
            confidence = 1.0
        elif state == TrackingState.RECENTLY_LOST:
            confidence = 0.5
        else:
            confidence = 0.1
        
        return PoseEstimate(
            timestamp=timestamp,
            matrix=T,
            source="visual_orbslam",
            confidence=confidence,
            is_keyframe=(len(self.slam_map.keyframes) > len(self.trajectory)),
        )
    
    def get_trajectory(self) -> list[PoseEstimate]:
        """
        Get the estimated camera trajectory.
        
        Returns
        -------
        List[PoseEstimate]
            List of poses (one per processed frame).
        """
        return self.trajectory
    
    def get_stats(self) -> dict:
        """
        Get SLAM statistics.
        
        Returns
        -------
        dict
            Statistics about the SLAM session.
        """
        return {
            'frames_processed': self.frame_count,
            'keyframes_created': self.keyframe_count,
            'loops_closed': self.loop_count,
            'map_points': len(self.slam_map.map_points),
            'tracking_state': self.tracker.state.name,
        }
    
    def reset(self) -> None:
        """
        Reset the SLAM system.
        
        Clears the map and reinitializes tracking.
        """
        self.slam_map = Map()
        self.tracker = Tracker(self.camera)
        self.tracker.set_map(self.slam_map)
        
        feature_tracker = FeatureTracker(num_features=1000)
        self.mapper = LocalMapper(self.slam_map, feature_tracker)
        self.loop_closer = LoopCloser(self.slam_map, feature_tracker)
        
        self.trajectory = []
        self.frame_count = 0
        self.keyframe_count = 0
        self.loop_count = 0
        self.frames_since_last_loop_check = 0