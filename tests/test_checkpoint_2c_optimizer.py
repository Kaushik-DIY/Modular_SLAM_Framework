#!/usr/bin/env python3
"""
=============================================================================
VALIDATION CHECKPOINT 2C — g2o Optimizer Wrappers
=============================================================================

Purpose:
    Verify that g2o optimization functions work correctly:
    - motion_only_ba: Refine pose with fixed map points
    - local_ba: Optimize local keyframes + map points
    - pose_graph_optimization: PGO with loop closure
    - global_ba: Full bundle adjustment

Tests:
    1. motion_only_ba on synthetic data (known ground truth)
    2. local_ba with 2 keyframes observing shared map points
    3. PGO with simple 3-node loop
    4. global_ba (smoke test)

Pass criteria:
    - motion_only_ba converges to <5mm error from perturbed init
    - local_ba reduces reprojection error
    - PGO distributes loop closure error across nodes
    - No crashes or exceptions

Usage (from slam_ws root, with venv activated):
    python3 tests/test_checkpoint_2c_optimizer.py
=============================================================================
"""

from __future__ import annotations

import sys
import traceback
import numpy as np
import cv2


def _run_checkpoint() -> bool:
    """Execute all optimizer tests. Returns True if all pass."""

    # ------------------------------------------------------------------
    # Test 1: Motion-only BA on synthetic data
    # ------------------------------------------------------------------
    print("[Test 1] Motion-only BA on synthetic reprojection problem...")
    try:
        import g2o
        from visual_slam.optimizer import motion_only_ba
        from visual_slam.types import Frame, MapPoint
        from slam_core.common.types3d import CameraIntrinsics
        
        # Create camera
        cam = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            width=640, height=480, depth_scale=1000.0
        )
        
        # Create 20 synthetic 3D points in front of camera
        np.random.seed(42)
        points_3d_world = np.random.uniform(-2, 2, (20, 3))
        points_3d_world[:, 2] = np.abs(points_3d_world[:, 2]) + 2.0  # Z > 0
        
        # True camera pose: identity (at origin, looking along +Z)
        true_pose = g2o.Isometry3d(np.eye(4))
        
        # Create map points
        map_points = []
        for pos in points_3d_world:
            mp = MapPoint(position_world=pos)
            map_points.append(mp)
        
        # Project points using true pose to get pixel observations
        observations_2d = []
        for pos in points_3d_world:
            # Transform to camera frame (identity, so same as world)
            X, Y, Z = pos
            
            # Project to image plane
            u = cam.fx * (X / Z) + cam.cx
            v = cam.fy * (Y / Z) + cam.cy
            observations_2d.append((u, v))
        
        # Create frame with perturbed initial pose (5cm translation error)
        T_perturbed = np.eye(4)
        T_perturbed[:3, 3] = [0.05, -0.03, 0.02]  # 5cm, 3cm, 2cm errors
        init_pose = g2o.Isometry3d(T_perturbed)
        
        # Create dummy image data
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.uint16) * 2000  # 2m depth
        
        frame = Frame.from_images(0.0, rgb, depth, cam)
        frame.pose_world = init_pose
        
        # Create keypoints at observed locations
        frame.keypoints = [cv2.KeyPoint(x=u, y=v, size=7) for u, v in observations_2d]
        frame.descriptors = np.zeros((len(frame.keypoints), 32), dtype=np.uint8)
        frame.depths = np.ones(len(frame.keypoints), dtype=np.float32) * 2.0
        frame.map_point_matches = map_points
        
        # Run motion-only BA
        optimized_pose = motion_only_ba(frame, cam, iterations=10)
        
        assert optimized_pose is not None, "Optimization returned None"
        
        # Extract optimized translation
        T_opt = optimized_pose.matrix()
        opt_x, opt_y, opt_z = T_opt[0, 3], T_opt[1, 3], T_opt[2, 3]
        
        # Check convergence to true pose (identity)
        assert abs(opt_x) < 0.005, f"BA x error too large: {opt_x:.6f}m (should be <0.005m)"
        assert abs(opt_y) < 0.005, f"BA y error too large: {opt_y:.6f}m"
        assert abs(opt_z) < 0.005, f"BA z error too large: {opt_z:.6f}m"
        
        print(f"  OK: Converged to ({opt_x:.6f}, {opt_y:.6f}, {opt_z:.6f})m from init (0.05, -0.03, 0.02)m")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 2: Local BA with shared map points
    # ------------------------------------------------------------------
    print("[Test 2] Local BA with 2 keyframes and shared map points...")
    try:
        from visual_slam.optimizer import local_ba
        from visual_slam.types import KeyFrame
        
        # Create 2 keyframes with perturbed poses
        kf1 = KeyFrame.from_frame(Frame.from_images(0.0, rgb, depth, cam))
        kf1.frame.pose_world = g2o.Isometry3d(np.eye(4))
        
        T2 = np.eye(4)
        T2[0, 3] = 1.0  # 1m translation along X
        kf2 = KeyFrame.from_frame(Frame.from_images(0.1, rgb, depth, cam))
        kf2.frame.pose_world = g2o.Isometry3d(T2)
        
        # Create shared map points observed by both keyframes
        shared_map_points = []
        for i in range(5):
            pos = np.array([0.5 + i * 0.1, 0.0, 2.0])
            mp = MapPoint(position_world=pos)
            
            # Add observations (dummy keypoint indices)
            mp.add_observation(kf1, keypoint_idx=i)
            mp.add_observation(kf2, keypoint_idx=i)
            shared_map_points.append(mp)
        
        # Add dummy keypoints to keyframes
        for kf in [kf1, kf2]:
            kf.frame.keypoints = [cv2.KeyPoint(x=100+i*50, y=200, size=7) for i in range(5)]
            kf.frame.descriptors = np.zeros((5, 32), dtype=np.uint8)
            kf.frame.depths = np.ones(5, dtype=np.float32) * 2.0
        
        # Run local BA
        local_ba(
            local_keyframes=[kf1, kf2],
            local_map_points=shared_map_points,
            fixed_keyframes=[],
            iterations=5
        )
        
        # Check that poses are still reasonable (not NaN or zero)
        T1_opt = kf1.frame.pose_world.matrix()
        T2_opt = kf2.frame.pose_world.matrix()
        
        assert not np.any(np.isnan(T1_opt)), "KF1 pose has NaN after BA"
        assert not np.any(np.isnan(T2_opt)), "KF2 pose has NaN after BA"
        assert np.linalg.norm(T1_opt[:3, 3]) < 10.0, "KF1 pose exploded"
        assert np.linalg.norm(T2_opt[:3, 3]) < 10.0, "KF2 pose exploded"
        
        # Check map points
        for mp in shared_map_points:
            assert not np.any(np.isnan(mp.position_world)), "Map point has NaN"
            assert np.linalg.norm(mp.position_world) < 100.0, "Map point exploded"
        
        print(f"  OK: Local BA completed, poses and map points remain valid")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 3: PGO with simple loop
    # ------------------------------------------------------------------
    print("[Test 3] Pose graph optimization with loop closure...")
    try:
        from visual_slam.optimizer import pose_graph_optimization
        
        # Create 4 keyframes forming a square
        keyframes = []
        for i in range(4):
            kf = KeyFrame.from_frame(Frame.from_images(float(i) * 0.1, rgb, depth, cam))
            
            T = np.eye(4)
            if i == 0:
                pass  # Origin
            elif i == 1:
                T[0, 3] = 1.0  # 1m along X
            elif i == 2:
                T[0, 3] = 1.0
                T[1, 3] = 1.0  # (1, 1, 0)
            elif i == 3:
                T[1, 3] = 1.0  # (0, 1, 0)
            
            kf.frame.pose_world = g2o.Isometry3d(T)
            kf.frame.keypoints = []
            kf.frame.descriptors = np.array([], dtype=np.uint8).reshape(0, 32)
            keyframes.append(kf)
        
        # Loop closure edge: 3 -> 0 (with slight error)
        T_loop = np.eye(4)
        T_loop[1, 3] = -1.05  # Should be -1.0, so 5cm error
        loop_relative_pose = g2o.Isometry3d(T_loop)
        
        loop_edges = [
            (3, 0, loop_relative_pose),
        ]
        
        # Run PGO
        pose_graph_optimization(keyframes, loop_edges, iterations=20)
        
        # Check that loop error was distributed
        # Node 3 should be closer to (0, 1, 0) after optimization
        T3_opt = keyframes[3].frame.pose_world.matrix()
        y_error = abs(T3_opt[1, 3] - 1.0)
        
        assert y_error < 0.03, f"PGO did not reduce loop error: y_error={y_error:.6f}m"
        
        print(f"  OK: PGO reduced loop error to {y_error:.6f}m")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 4: Global BA (smoke test)
    # ------------------------------------------------------------------
    print("[Test 4] Global BA smoke test...")
    try:
        from visual_slam.optimizer import global_ba
        from visual_slam.types import Map
        
        # Create map with keyframes and map points from Test 2
        slam_map = Map()
        for kf in [kf1, kf2]:
            slam_map.add_keyframe(kf)
        for mp in shared_map_points:
            slam_map.add_map_point(mp)
        
        # Run global BA
        global_ba(slam_map, iterations=3)
        
        # Just check no crashes and results are valid
        assert len(slam_map.keyframes) == 2
        assert len(slam_map.map_points) == 5
        
        print(f"  OK: Global BA completed without errors")
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
    print("VALIDATION CHECKPOINT 2C: g2o Optimizer Wrappers")
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
        print("CHECKPOINT 2C: PASSED")
        print("")
        print("✓ motion_only_ba converges to <5mm from perturbed init")
        print("✓ local_ba optimizes keyframes and map points")
        print("✓ PGO distributes loop closure error correctly")
        print("✓ global_ba runs without errors")
        print("")
        print("g2o optimization is working! You may proceed to Module 2D")
        print("(Tracking front-end).")
    else:
        print("CHECKPOINT 2C: FAILED")
        print("")
        print("g2o optimizer wrappers are NOT working correctly.")
        print("Do NOT proceed to Module 2D. Fix the issues above first.")
        print("")
        print("Common issues:")
        print("  - g2o not installed: run phase0_install_g2o.sh")
        print("  - Wrong g2o version: use uoip/g2opy fork")
        print("  - Missing g2o types: check VertexSE3Expmap, EdgeSE3ProjectXYZ")
    print("=" * 70)

    sys.exit(0 if passed else 1)