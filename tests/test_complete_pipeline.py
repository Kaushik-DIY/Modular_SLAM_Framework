#!/usr/bin/env python3
"""
=============================================================================
VALIDATION CHECKPOINT 2G+2H — Complete Visual SLAM Pipeline
=============================================================================

Purpose:
    Test the complete Visual SLAM pipeline end-to-end:
    - Adapter orchestrates all modules
    - Run on 100 TUM frames
    - Output valid trajectory

Pass criteria:
    - Process 100 frames without crashes
    - >= 10 keyframes created
    - >= 500 map points
    - Valid trajectory output
    - At least 90% of frames tracked

Usage (from slam_ws root, with venv activated):
    python3 tests/test_complete_pipeline.py
=============================================================================
"""

import sys
from pathlib import Path
import traceback
import numpy as np


def _run_test() -> bool:
    """Execute complete pipeline test."""
    
    print("[Test] Complete Visual SLAM Pipeline (100 frames)...")
    
    try:
        from visual_slam.adapter import VisualSlamAdapter
        from visual_slam.tracking import TrackingState
        from slam_core.common.types3d import CameraIntrinsics
        from tests.test_checkpoint_2b_feature_tracker import _load_tum_frame
        
        # TUM fr1/desk camera
        camera = CameraIntrinsics(
            fx=517.3, fy=516.5, cx=318.6, cy=255.3,
            width=640, height=480, depth_scale=5000.0
        )
        
        # Create SLAM system
        slam = VisualSlamAdapter(camera)
        
        # Process 100 frames
        tum_path = Path("datasets/tum/rgbd_dataset_freiburg1_desk")
        
        if not tum_path.exists():
            print(f"  SKIP: TUM dataset not found at {tum_path}")
            return False
        
        num_ok = 0
        num_lost = 0
        
        for i in range(100):
            result = _load_tum_frame(tum_path, i)
            if result is None:
                return False
            
            rgb, depth, timestamp = result
            
            # Process frame
            pose, state = slam.process_frame(rgb, depth, timestamp)
            
            if state == TrackingState.OK:
                num_ok += 1
            elif state == TrackingState.LOST:
                num_lost += 1
            
            if (i + 1) % 25 == 0:
                stats = slam.get_stats()
                print(f"  Frame {i+1}/100: {state.name} | "
                      f"KFs: {stats['keyframes_created']} | "
                      f"MPs: {stats['map_points']}")
        
        # Get final stats
        stats = slam.get_stats()
        trajectory = slam.get_trajectory()
        
        print(f"\n  Final Statistics:")
        print(f"    Frames processed:  {stats['frames_processed']}")
        print(f"    Frames tracked:    {num_ok}")
        print(f"    Frames lost:       {num_lost}")
        print(f"    Keyframes created: {stats['keyframes_created']}")
        print(f"    Map points:        {stats['map_points']}")
        print(f"    Loops closed:      {stats['loops_closed']}")
        print(f"    Trajectory length: {len(trajectory)}")
        
        # Checks
        assert stats['frames_processed'] == 100, \
            f"Expected 100 frames, got {stats['frames_processed']}"
        
        assert stats['keyframes_created'] >= 10, \
            f"Too few keyframes: {stats['keyframes_created']} (expected >= 10)"
        
        assert stats['map_points'] >= 500, \
            f"Too few map points: {stats['map_points']} (expected >= 500)"
        
        assert len(trajectory) >= 90, \
            f"Too few trajectory poses: {len(trajectory)} (expected >= 90)"
        
        tracking_rate = num_ok / 100
        assert tracking_rate >= 0.9, \
            f"Tracking rate too low: {tracking_rate:.1%} (expected >= 90%)"
        
        # Check trajectory validity
        for i, pose in enumerate(trajectory):
            assert pose is not None, f"Pose {i} is None"
            assert pose.matrix is not None, f"Pose {i} has no matrix"
            assert pose.matrix.shape == (4, 4), f"Pose {i} has wrong shape"
            assert not np.any(np.isnan(pose.matrix)), f"Pose {i} contains NaN"
        
        print(f"\n  PASSED: Complete Visual SLAM pipeline working!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("VALIDATION: Complete Visual SLAM Pipeline")
    print("=" * 70)
    print()
    
    if not Path("datasets/tum/rgbd_dataset_freiburg1_desk").exists():
        print("ERROR: TUM dataset not found")
        sys.exit(1)
    
    try:
        passed = _run_test()
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        traceback.print_exc()
        passed = False
    
    print()
    print("=" * 70)
    if passed:
        print("COMPLETE PIPELINE: PASSED")
        print()
        print("✓ Adapter orchestrates all modules correctly")
        print("✓ 100 frames processed successfully")
        print("✓ Keyframes and map points created")
        print("✓ Valid trajectory output")
        print()
        print("Visual SLAM implementation complete!")
        print("You can now run on full datasets with:")
        print("  python3 visual_slam/run_slam.py <dataset_path>")
    else:
        print("COMPLETE PIPELINE: FAILED")
        print()
        print("Fix the issues above before proceeding.")
    print("=" * 70)
    
    sys.exit(0 if passed else 1)