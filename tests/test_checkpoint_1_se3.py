#!/usr/bin/env python3
"""
=============================================================================
VALIDATION CHECKPOINT 1 — SE(3) Types and Math (pyslam-compatible)
=============================================================================

Purpose:
    Verify that our SE(3) implementation follows pyslam's approach using
    g2o.Isometry3d directly. This checkpoint confirms:
    1. g2o.Isometry3d works as our Pose3D type
    2. SE(3) utilities operate correctly on g2o objects
    3. PoseEstimate cross-SLAM interface works
    4. Existing Pose2 code remains unbroken (regression)

Tests:
    1. Pose2 regression check (existing code must work)
    2. g2o.Isometry3d creation and matrix conversion
    3. SE(3) composition (pose3d_compose)
    4. SE(3) inversion (pose3d_inverse)
    5. Transform 3D points
    6. PoseEstimate wraps both 2D and 3D sources
    7. CameraIntrinsics creation

Pass criteria:
    - All g2o.Isometry3d operations work correctly
    - PoseEstimate provides unified interface
    - Pose2 regression passes
    - All numerical errors < 1e-6

Usage (from slam_ws root, with venv activated):
    python3 tests/test_checkpoint_1_se3.py
=============================================================================
"""

from __future__ import annotations

import sys
import traceback
import math
import numpy as np


def _run_checkpoint() -> bool:
    """Execute all tests. Returns True if all pass."""

    # ------------------------------------------------------------------
    # Test 1: Pose2 regression (existing code must work)
    # ------------------------------------------------------------------
    print("[Test 1] Regression: Pose2 unchanged...")
    try:
        from slam_core.common.types import Pose2
        
        p2 = Pose2(1.0, 2.0, 0.5)
        assert abs(p2.x - 1.0) < 1e-9
        assert abs(p2.y - 2.0) < 1e-9
        assert abs(p2.theta - 0.5) < 1e-9
        
        arr = p2.as_array()
        assert arr.shape == (3,)
        
        print("  OK: Pose2 API unchanged")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # ------------------------------------------------------------------
    # Test 2: g2o.Isometry3d creation (pyslam's Pose3D type)
    # ------------------------------------------------------------------
    print("[Test 2] g2o.Isometry3d creation and matrix conversion...")
    try:
        import g2o
        from slam_core.common.types3d import Pose3D
        
        # Verify Pose3D is an alias for g2o.Isometry3d
        assert Pose3D is g2o.Isometry3d, "Pose3D should be g2o.Isometry3d"
        
        # Create identity pose
        pose = g2o.Isometry3d()
        T = pose.matrix()
        assert T.shape == (4, 4)
        assert np.allclose(T, np.eye(4), atol=1e-9)
        
        # Create pose with translation
        T2 = np.eye(4)
        T2[:3, 3] = [1.0, 2.0, 3.0]
        pose2 = g2o.Isometry3d(T2)
        T2_back = pose2.matrix()
        assert np.allclose(T2_back, T2, atol=1e-9)
        
        print(f"  OK: g2o.Isometry3d creates and converts matrices")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 3: SE(3) composition
    # ------------------------------------------------------------------
    print("[Test 3] SE(3) composition (pose3d_compose)...")
    try:
        from slam_core.common.se3 import pose3d_compose
        
        # Two pure translations
        T1 = np.eye(4)
        T1[:3, 3] = [1.0, 0.0, 0.0]
        pose1 = g2o.Isometry3d(T1)
        
        T2 = np.eye(4)
        T2[:3, 3] = [0.0, 2.0, 0.0]
        pose2 = g2o.Isometry3d(T2)
        
        pose3 = pose3d_compose(pose1, pose2)
        T3 = pose3.matrix()
        
        assert abs(T3[0, 3] - 1.0) < 1e-6, f"x wrong: {T3[0,3]}"
        assert abs(T3[1, 3] - 2.0) < 1e-6, f"y wrong: {T3[1,3]}"
        assert abs(T3[2, 3] - 0.0) < 1e-6, f"z wrong: {T3[2,3]}"
        
        print(f"  OK: (1,0,0) ⊕ (0,2,0) = ({T3[0,3]:.3f}, {T3[1,3]:.3f}, {T3[2,3]:.3f})")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 4: SE(3) inversion
    # ------------------------------------------------------------------
    print("[Test 4] SE(3) inversion (pose3d_inverse)...")
    try:
        from slam_core.common.se3 import pose3d_inverse
        
        T = np.eye(4)
        T[:3, 3] = [3.0, 4.0, 5.0]
        pose = g2o.Isometry3d(T)
        
        pose_inv = pose3d_inverse(pose)
        identity = pose3d_compose(pose, pose_inv)
        T_id = identity.matrix()
        
        assert np.allclose(T_id, np.eye(4), atol=1e-6), \
            f"Inverse roundtrip failed: {T_id}"
        
        print(f"  OK: p ⊕ p^(-1) ≈ identity")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 5: Transform 3D points
    # ------------------------------------------------------------------
    print("[Test 5] Transform 3D points...")
    try:
        from slam_core.common.se3 import transform_points_3d
        
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        pose = g2o.Isometry3d(T)
        
        pts_local = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        pts_world = transform_points_3d(pose, pts_local)
        
        assert np.allclose(pts_world[0], [1.0, 2.0, 3.0], atol=1e-6)
        assert np.allclose(pts_world[1], [2.0, 2.0, 3.0], atol=1e-6)
        assert np.allclose(pts_world[2], [1.0, 3.0, 3.0], atol=1e-6)
        
        print(f"  OK: Transformed 3 points, e.g. (0,0,0) → ({pts_world[0][0]:.1f}, "
              f"{pts_world[0][1]:.1f}, {pts_world[0][2]:.1f})")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 6: PoseEstimate cross-SLAM interface
    # ------------------------------------------------------------------
    print("[Test 6] PoseEstimate cross-SLAM interface...")
    try:
        from slam_core.common.types3d import PoseEstimate
        
        # Test 6a: From Visual SLAM (g2o.Isometry3d source)
        T_visual = np.eye(4)
        T_visual[:3, 3] = [1.5, 2.5, 0.3]
        pose_g2o = g2o.Isometry3d(T_visual)
        
        pe_visual = PoseEstimate(
            timestamp=0.123,
            matrix=pose_g2o.matrix(),
            source="visual_orbslam",
            confidence=0.88
        )
        
        assert pe_visual.source == "visual_orbslam"
        assert abs(pe_visual.x - 1.5) < 1e-9
        assert abs(pe_visual.y - 2.5) < 1e-9
        assert abs(pe_visual.z - 0.3) < 1e-9
        
        # Test 6b: From Lidar SLAM (Pose2 lifted to 3D)
        p2 = Pose2(x=1.0, y=2.0, theta=math.pi/4)
        T_lidar = np.eye(4)
        T_lidar[0, 3] = p2.x
        T_lidar[1, 3] = p2.y
        c, s = math.cos(p2.theta), math.sin(p2.theta)
        T_lidar[:2, :2] = [[c, -s], [s, c]]
        
        pe_lidar = PoseEstimate(
            timestamp=0.2,
            matrix=T_lidar,
            source="hector",
            confidence=0.95
        )
        
        assert pe_lidar.source == "hector"
        assert abs(pe_lidar.z) < 1e-9  # Lidar is 2D
        assert abs(pe_lidar.theta - math.pi/4) < 1e-6
        
        # Test 6c: to_pose2() projection
        p2_proj = pe_visual.to_pose2()
        assert abs(p2_proj.x - 1.5) < 1e-9
        assert abs(p2_proj.y - 2.5) < 1e-9
        
        print(f"  OK: PoseEstimate handles Visual (3D) and Lidar (2D) sources")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 7: CameraIntrinsics
    # ------------------------------------------------------------------
    print("[Test 7] CameraIntrinsics...")
    try:
        from slam_core.common.types3d import CameraIntrinsics
        
        # TUM fr1/desk camera
        cam = CameraIntrinsics(
            fx=517.3, fy=516.5, cx=318.6, cy=255.3,
            width=640, height=480, depth_scale=5000.0
        )
        
        assert cam.fx == 517.3
        assert cam.depth_scale == 5000.0
        assert cam.width == 640
        
        print(f"  OK: CameraIntrinsics created (fx={cam.fx}, depth_scale={cam.depth_scale})")
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
    print("VALIDATION CHECKPOINT 1: SE(3) Types (pyslam-compatible)")
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
        print("CHECKPOINT 1: PASSED")
        print("")
        print("✓ g2o.Isometry3d works as Pose3D (pyslam approach)")
        print("✓ SE(3) utilities operate correctly")
        print("✓ PoseEstimate cross-SLAM interface works")
        print("✓ Pose2 regression passed")
        print("")
        print("slam_core has been successfully extended with pyslam-compatible")
        print("SE(3) support. You may proceed to Phase 2 (Visual SLAM porting).")
    else:
        print("CHECKPOINT 1: FAILED")
        print("")
        print("SE(3) implementation is NOT working correctly.")
        print("Do NOT proceed to Phase 2. Fix the issues above first.")
        print("")
        print("Common issues:")
        print("  - g2o not installed: run phase0_install_g2o.sh")
        print("  - numpy not in venv: pip install numpy")
        print("  - slam_core/common/types.py missing")
    print("=" * 70)

    sys.exit(0 if passed else 1)