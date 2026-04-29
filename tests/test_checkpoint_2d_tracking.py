#!/usr/bin/env python3
"""Checkpoint 2D: Test tracking on 10 consecutive TUM frames"""

import sys
from pathlib import Path
import numpy as np

# Load TUM frames helper (reuse from 2B)
sys.path.insert(0, str(Path.cwd()))

from visual_slam.tracking import Tracker, TrackingState
from visual_slam.types import Map
from slam_core.common.types3d import CameraIntrinsics

# Import TUM loader from checkpoint 2B
from tests.test_checkpoint_2b_feature_tracker import _load_tum_frame

def test_tracking():
    # TUM fr1/desk camera
    cam = CameraIntrinsics(
        fx=517.3, fy=516.5, cx=318.6, cy=255.3,
        width=640, height=480, depth_scale=5000.0
    )
    
    # Create tracker and map
    tracker = Tracker(cam)
    slam_map = Map()
    tracker.set_map(slam_map)
    
    # Load 10 frames
    tum_path = Path("datasets/tum/rgbd_dataset_freiburg1_desk")
    poses = []
    
    for i in range(10):
        result = _load_tum_frame(tum_path, i)
        if result is None:
            return False
        
        rgb, depth, timestamp = result
        pose, state = tracker.process_frame(rgb, depth, timestamp)
        poses.append(pose)
        
        print(f"Frame {i}: state={state.name}, pose={'OK' if pose else 'NONE'}")
    
    # Checks
    assert all(p is not None for p in poses), "Some poses are None"
    assert len(slam_map.keyframes) >= 1, f"No keyframes: {len(slam_map.keyframes)}"
    assert len(slam_map.map_points) >= 50, f"Too few map points: {len(slam_map.map_points)}"
    
    # Check plausible movement
    total_dist = sum(
        np.linalg.norm(poses[i+1].matrix()[:3,3] - poses[i].matrix()[:3,3])
        for i in range(len(poses)-1)
    )
    
    assert 0.01 < total_dist < 2.0, f"Distance {total_dist:.3f}m implausible"
    
    print(f"\nCHECKPOINT 2D PASSED:")
    print(f"  Tracked 10 frames")
    print(f"  {len(slam_map.keyframes)} keyframes")
    print(f"  {len(slam_map.map_points)} map points")
    print(f"  {total_dist:.3f}m traveled")
    return True

if __name__ == "__main__":
    try:
        passed = test_tracking()
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)