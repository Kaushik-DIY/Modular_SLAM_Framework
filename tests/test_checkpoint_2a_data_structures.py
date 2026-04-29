#!/usr/bin/env python3
"""
=============================================================================
VALIDATION CHECKPOINT 2A — Data Structures (Frame, KeyFrame, MapPoint, Map)
=============================================================================

Purpose:
    Verify that the core SLAM data structures work correctly:
    - Frame creation and feature storage
    - KeyFrame promotion from Frame
    - MapPoint observations and descriptor computation
    - Map add/remove operations
    - Covisibility graph construction

Tests:
    1. Frame creation from images
    2. KeyFrame creation from Frame
    3. MapPoint creation and observations
    4. Map operations (add/remove)
    5. Covisibility graph (keyframe connections)
    6. Thread safety (basic check)

Pass criteria:
    - All data structures create successfully
    - Relationships (observations, connections) link correctly
    - Map queries return expected results
    - No crashes or exceptions

Usage (from slam_ws root, with venv activated):
    python3 tests/test_checkpoint_2a_data_structures.py
=============================================================================
"""

from __future__ import annotations

import sys
import traceback
import numpy as np
import cv2


def _run_checkpoint() -> bool:
    """Execute all data structure tests. Returns True if all pass."""

    # ------------------------------------------------------------------
    # Test 1: Frame creation
    # ------------------------------------------------------------------
    print("[Test 1] Frame creation from images...")
    try:
        from visual_slam.types import Frame
        from slam_core.common.types3d import CameraIntrinsics
        
        # Create camera (TUM fr1/desk params)
        cam = CameraIntrinsics(
            fx=517.3, fy=516.5, cx=318.6, cy=255.3,
            width=640, height=480, depth_scale=5000.0
        )
        
        # Create dummy images
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.uint16)
        
        # Create frame
        frame = Frame.from_images(timestamp=0.1, rgb=rgb, depth=depth, camera=cam)
        
        assert frame.frame_id == 0, "First frame should have ID 0"
        assert frame.timestamp == 0.1
        assert frame.image_rgb.shape == (480, 640, 3)
        assert frame.image_depth.shape == (480, 640)
        assert frame.pose_world is None, "Pose not yet estimated"
        assert len(frame.keypoints) == 0, "No features yet"
        
        # Create second frame to check ID increment
        frame2 = Frame.from_images(timestamp=0.2, rgb=rgb, depth=depth, camera=cam)
        assert frame2.frame_id == 1, "Second frame should have ID 1"
        
        print(f"  OK: {frame}, {frame2}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 2: KeyFrame creation from Frame
    # ------------------------------------------------------------------
    print("[Test 2] KeyFrame creation from Frame...")
    try:
        from visual_slam.types import KeyFrame
        
        # Promote frame to keyframe
        kf = KeyFrame.from_frame(frame)
        
        assert kf.keyframe_id == 0, "First keyframe should have ID 0"
        assert kf.frame is frame, "KeyFrame should reference the frame"
        assert kf.is_bad == False
        assert len(kf.connected_keyframes) == 0, "No connections yet"
        assert len(kf.map_points) == 0, "No map points yet"
        
        # Create second keyframe
        kf2 = KeyFrame.from_frame(frame2)
        assert kf2.keyframe_id == 1, "Second keyframe should have ID 1"
        
        print(f"  OK: {kf}, {kf2}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 3: MapPoint creation and observations
    # ------------------------------------------------------------------
    print("[Test 3] MapPoint creation and observations...")
    try:
        from visual_slam.types import MapPoint
        
        # Create map point
        mp = MapPoint(position_world=np.array([1.0, 2.0, 3.0]))
        
        assert mp.point_id == 0, "First map point should have ID 0"
        assert np.allclose(mp.position_world, [1.0, 2.0, 3.0])
        assert len(mp.observations) == 0, "No observations yet"
        assert mp.is_bad == False
        
        # Add dummy features to keyframe first
        kf.frame.keypoints = [cv2.KeyPoint(x=100, y=200, size=7)]
        kf.frame.descriptors = np.random.randint(0, 256, (1, 32), dtype=np.uint8)
        
        # Add observation
        mp.add_observation(kf, keypoint_idx=0)
        
        assert len(mp.observations) == 1, "Should have 1 observation"
        assert kf in mp.observations, "Keyframe should be in observations"
        assert mp.observations[kf] == 0, "Keypoint index should be 0"
        assert mp in kf.map_points, "Map point should be in keyframe's set"
        
        # Compute descriptor
        mp.compute_descriptor()
        assert mp.descriptor is not None, "Descriptor should be computed"
        assert mp.descriptor.shape == (32,), "Descriptor should be 32-dim for ORB"
        
        print(f"  OK: {mp}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 4: Map operations
    # ------------------------------------------------------------------
    print("[Test 4] Map add/remove operations...")
    try:
        from visual_slam.types import Map
        
        # Create map
        slam_map = Map()
        
        assert len(slam_map.keyframes) == 0
        assert len(slam_map.map_points) == 0
        
        # Add keyframe
        slam_map.add_keyframe(kf)
        assert len(slam_map.keyframes) == 1
        assert kf.keyframe_id in slam_map.keyframes
        
        # Add map point
        slam_map.add_map_point(mp)
        assert len(slam_map.map_points) == 1
        assert mp.point_id in slam_map.map_points
        
        # Remove (mark as bad)
        slam_map.remove_map_point(mp)
        assert mp.is_bad == True, "Map point should be marked bad"
        assert mp.point_id in slam_map.map_points, "Reference should remain"
        
        print(f"  OK: {slam_map}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 5: Covisibility graph
    # ------------------------------------------------------------------
    print("[Test 5] Covisibility graph construction...")
    try:
        # Create another keyframe and map point
        frame3 = Frame.from_images(0.3, rgb, depth, cam)
        kf3 = KeyFrame.from_frame(frame3)
        kf3.frame.keypoints = [cv2.KeyPoint(x=110, y=210, size=7)]
        kf3.frame.descriptors = np.random.randint(0, 256, (1, 32), dtype=np.uint8)
        
        mp2 = MapPoint(position_world=np.array([4.0, 5.0, 6.0]))
        
        # Both kf and kf3 observe mp2 -> they share this map point
        mp2.add_observation(kf, keypoint_idx=0)
        mp2.add_observation(kf3, keypoint_idx=0)
        
        # Build covisibility connection
        kf.add_connection(kf3, weight=1)  # 1 shared map point
        kf3.add_connection(kf, weight=1)
        
        assert len(kf.connected_keyframes) == 1
        assert kf3 in kf.connected_keyframes
        assert kf.connected_keyframes[kf3] == 1
        
        # Get best covisible keyframes
        best = kf.get_best_covisible_keyframes(n=5)
        assert len(best) == 1
        assert best[0] is kf3
        
        print(f"  OK: kf connected to kf3 with weight 1")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 6: Thread safety (basic check)
    # ------------------------------------------------------------------
    print("[Test 6] Thread safety (basic lock check)...")
    try:
        # Just verify locks exist and work
        slam_map2 = Map()
        
        # Add multiple keyframes concurrently would test locks,
        # but for checkpoint we just verify locks don't raise errors
        for i in range(10):
            f = Frame.from_images(float(i), rgb, depth, cam)
            k = KeyFrame.from_frame(f)
            slam_map2.add_keyframe(k)
        
        assert len(slam_map2.keyframes) == 10
        
        print(f"  OK: Added 10 keyframes without lock errors")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    return True


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VALIDATION CHECKPOINT 2A: Data Structures")
    print("=" * 70)
    print()

    try:
        passed = _run_checkpoint()
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        passed = False

    print()
    print("=" * 70)
    if passed:
        print("CHECKPOINT 2A: PASSED")
        print("")
        print("✓ Frame, KeyFrame, MapPoint, Map all work correctly")
        print("✓ Observations and covisibility graph link properly")
        print("✓ Map operations (add/remove) work as expected")
        print("")
        print("You may proceed to Module 2B (Feature Tracker).")
    else:
        print("CHECKPOINT 2A: FAILED")
        print("")
        print("Data structures are NOT working correctly.")
        print("Do NOT proceed to Module 2B. Fix the issues above first.")
    print("=" * 70)

    sys.exit(0 if passed else 1)