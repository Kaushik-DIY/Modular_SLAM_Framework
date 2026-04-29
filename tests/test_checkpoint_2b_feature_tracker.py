#!/usr/bin/env python3
"""
=============================================================================
VALIDATION CHECKPOINT 2B — Feature Tracker (ORB2 + Matching + Depth)
=============================================================================

Purpose:
    Verify that the FeatureTracker works correctly:
    - ORB2 keypoint detection
    - ORB descriptor computation
    - Brute-force matching with ratio test
    - Depth association from RGBD images

Tests:
    1. ORB detection on synthetic image (verify parameters work)
    2. ORB detection on real TUM image (verify real-world performance)
    3. Descriptor computation (verify shape and dtype)
    4. Feature matching between two frames (verify ratio test)
    5. Depth association (verify depth lookup)
    6. End-to-end: detect + match on consecutive TUM frames

Pass criteria:
    - ORB detector creates features (>100 on real images)
    - Descriptors are binary np.uint8, shape (N, 32)
    - Matching finds good correspondences (>50 on similar frames)
    - Depth values are plausible (0.3m - 10m range for TUM)
    - No crashes or exceptions

Usage (from slam_ws root, with venv activated):
    python3 tests/test_checkpoint_2b_feature_tracker.py

Note: This test requires TUM dataset downloaded (phase0_download_tum.sh).
=============================================================================
"""

from __future__ import annotations

import sys
import os
import traceback
import numpy as np
import cv2
from pathlib import Path


def _load_tum_frame(dataset_path: Path, index: int = 0):
    """
    Load a single RGB+Depth frame from TUM dataset.
    
    Parameters
    ----------
    dataset_path : Path
        Path to TUM dataset directory (e.g., rgbd_dataset_freiburg1_desk).
    index : int
        Frame index to load.
    
    Returns
    -------
    tuple
        (rgb_image, depth_image, timestamp) or None if failed.
    """
    # Read associations file
    assoc_file = dataset_path / "associations.txt"
    if not assoc_file.exists():
        print(f"  ERROR: {assoc_file} not found")
        print(f"  Run: python3 datasets/tum/associate.py ...")
        return None
    
    with open(assoc_file, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
    
    if index >= len(lines):
        print(f"  ERROR: Index {index} out of range (only {len(lines)} frames)")
        return None
    
    # Parse line: timestamp1 rgb/file1.png timestamp2 depth/file2.png
    parts = lines[index].split()
    if len(parts) < 4:
        print(f"  ERROR: Malformed association line: {lines[index]}")
        return None
    
    timestamp = float(parts[0])
    rgb_path = dataset_path / parts[1]
    depth_path = dataset_path / parts[3]
    
    # Load images
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        print(f"  ERROR: Failed to load RGB image: {rgb_path}")
        return None
    
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR
    
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"  ERROR: Failed to load depth image: {depth_path}")
        return None
    
    return rgb, depth, timestamp


def _run_checkpoint() -> bool:
    """Execute all feature tracker tests. Returns True if all pass."""

    # ------------------------------------------------------------------
    # Test 1: ORB detection on synthetic image
    # ------------------------------------------------------------------
    print("[Test 1] ORB detection on synthetic image...")
    try:
        from visual_slam.feature_tracker import FeatureTracker
        from visual_slam.types import Frame
        from slam_core.common.types3d import CameraIntrinsics
        
        # Create feature tracker
        tracker = FeatureTracker(num_features=1000, num_levels=8)
        
        # Create synthetic image with corners
        img_synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some checkerboard pattern (corners for ORB to detect)
        for i in range(0, 480, 60):
            for j in range(0, 640, 60):
                if (i // 60 + j // 60) % 2 == 0:
                    img_synthetic[i:i+60, j:j+60] = 255
        
        depth_synthetic = np.ones((480, 640), dtype=np.uint16) * 2500  # 0.5m
        
        cam = CameraIntrinsics(
            fx=517.3, fy=516.5, cx=318.6, cy=255.3,
            width=640, height=480, depth_scale=5000.0
        )
        
        frame = Frame.from_images(0.0, img_synthetic, depth_synthetic, cam)
        tracker.detect_and_compute(frame)
        
        assert len(frame.keypoints) > 0, "Should detect some features on checkerboard"
        assert frame.descriptors.shape[0] == len(frame.keypoints)
        assert frame.descriptors.shape[1] == 32, "ORB descriptors are 32-dim"
        assert frame.descriptors.dtype == np.uint8, "ORB descriptors are binary (uint8)"
        
        print(f"  OK: Detected {len(frame.keypoints)} features on synthetic image")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 2: ORB detection on real TUM image
    # ------------------------------------------------------------------
    print("[Test 2] ORB detection on real TUM image...")
    try:
        # Find TUM dataset
        slam_ws = Path.cwd()
        tum_path = slam_ws / "datasets" / "tum" / "rgbd_dataset_freiburg1_desk"
        
        if not tum_path.exists():
            print(f"  SKIP: TUM dataset not found at {tum_path}")
            print(f"  Run phase0_download_tum.sh first")
            return False
        
        # Load first frame
        result = _load_tum_frame(tum_path, index=0)
        if result is None:
            return False
        
        rgb, depth, timestamp = result
        
        # Detect features
        frame_tum = Frame.from_images(timestamp, rgb, depth, cam)
        tracker.detect_and_compute(frame_tum)
        
        assert len(frame_tum.keypoints) > 100, \
            f"Should detect >100 features on real image (got {len(frame_tum.keypoints)})"
        assert len(frame_tum.depths) == len(frame_tum.keypoints), \
            "Depth array should match keypoint count"
        
        # Check depth values are plausible
        valid_depths = frame_tum.depths[frame_tum.depths > 0]
        if len(valid_depths) > 0:
            assert np.all(valid_depths < 10.0), "TUM depths should be < 10m"
            assert np.all(valid_depths > 0.1), "TUM depths should be > 0.1m"
        
        print(f"  OK: Detected {len(frame_tum.keypoints)} features, "
              f"{len(valid_depths)} with valid depth")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 3: Descriptor properties
    # ------------------------------------------------------------------
    print("[Test 3] Descriptor computation...")
    try:
        # Already computed in Test 2
        assert frame_tum.descriptors.shape == (len(frame_tum.keypoints), 32)
        assert frame_tum.descriptors.dtype == np.uint8
        
        # Check descriptors are not all zeros (would indicate bug)
        assert np.any(frame_tum.descriptors > 0), "Descriptors should not be all zeros"
        
        print(f"  OK: Descriptors shape {frame_tum.descriptors.shape}, dtype {frame_tum.descriptors.dtype}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 4: Feature matching between two frames
    # ------------------------------------------------------------------
    print("[Test 4] Feature matching with ratio test...")
    try:
        # Load second frame (should be similar to first)
        result2 = _load_tum_frame(tum_path, index=1)
        if result2 is None:
            return False
        
        rgb2, depth2, timestamp2 = result2
        frame2 = Frame.from_images(timestamp2, rgb2, depth2, cam)
        tracker.detect_and_compute(frame2)
        
        # Match
        matches = tracker.match_frames(frame_tum, frame2)
        
        assert len(matches) > 50, \
            f"Should find >50 matches between consecutive frames (got {len(matches)})"
        
        # Check match properties
        for m in matches[:5]:  # Check first 5 matches
            assert 0 <= m.queryIdx < len(frame_tum.keypoints)
            assert 0 <= m.trainIdx < len(frame2.keypoints)
            assert m.distance >= 0, "Hamming distance should be non-negative"
        
        print(f"  OK: Matched {len(matches)} features between consecutive frames")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 5: Depth association
    # ------------------------------------------------------------------
    print("[Test 5] Depth association from RGBD...")
    try:
        # Check depth values were associated correctly
        valid_depth_count = np.sum(frame_tum.depths > 0)
        total_kpts = len(frame_tum.keypoints)
        
        assert valid_depth_count > 0, "Should have some valid depths"
        
        # Most keypoints should have valid depth (TUM is RGBD)
        # Allow some to fail (edges, reflective surfaces, etc.)
        valid_ratio = valid_depth_count / total_kpts
        assert valid_ratio > 0.5, \
            f"At least 50% of keypoints should have valid depth (got {valid_ratio:.1%})"
        
        # Check depth statistics
        valid_depths = frame_tum.depths[frame_tum.depths > 0]
        mean_depth = np.mean(valid_depths)
        median_depth = np.median(valid_depths)
        
        # TUM fr1/desk has typical depths around 0.5-2.0m
        assert 0.3 < mean_depth < 5.0, \
            f"Mean depth {mean_depth:.2f}m seems implausible for TUM desk"
        
        print(f"  OK: {valid_depth_count}/{total_kpts} keypoints have valid depth")
        print(f"      Mean depth: {mean_depth:.2f}m, Median: {median_depth:.2f}m")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # Test 6: End-to-end on multiple consecutive frames
    # ------------------------------------------------------------------
    print("[Test 6] End-to-end: detect + match on 5 consecutive frames...")
    try:
        frames = []
        
        # Load and process 5 consecutive frames
        for i in range(5):
            result = _load_tum_frame(tum_path, index=i)
            if result is None:
                return False
            
            rgb, depth, timestamp = result
            frame = Frame.from_images(timestamp, rgb, depth, cam)
            tracker.detect_and_compute(frame)
            frames.append(frame)
        
        # Match consecutive pairs
        match_counts = []
        for i in range(len(frames) - 1):
            matches = tracker.match_frames(frames[i], frames[i+1])
            match_counts.append(len(matches))
        
        # All consecutive pairs should have good matches
        assert all(count > 50 for count in match_counts), \
            f"All pairs should have >50 matches: {match_counts}"
        
        avg_matches = np.mean(match_counts)
        print(f"  OK: Processed 5 frames, avg {avg_matches:.0f} matches per pair")
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
    print("VALIDATION CHECKPOINT 2B: Feature Tracker")
    print("=" * 70)
    print()

    # Check if we're in slam_ws
    if not Path("datasets").exists():
        print("ERROR: Must run from slam_ws root directory")
        print("  cd slam_ws")
        print("  source venv/bin/activate")
        print("  python3 tests/test_checkpoint_2b_feature_tracker.py")
        sys.exit(1)

    try:
        passed = _run_checkpoint()
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        passed = False

    print()
    print("=" * 70)
    if passed:
        print("CHECKPOINT 2B: PASSED")
        print("")
        print("✓ ORB2 detector works on synthetic and real images")
        print("✓ Descriptors are binary uint8, shape (N, 32)")
        print("✓ BruteForce matcher with ratio test finds good matches")
        print("✓ Depth association works correctly for RGBD")
        print("✓ End-to-end pipeline processes TUM frames successfully")
        print("")
        print("Feature tracking is working! You may proceed to Module 2C")
        print("(g2o optimizer wrappers).")
    else:
        print("CHECKPOINT 2B: FAILED")
        print("")
        print("Feature tracker is NOT working correctly.")
        print("Do NOT proceed to Module 2C. Fix the issues above first.")
        print("")
        print("Common issues:")
        print("  - TUM dataset not downloaded: run phase0_download_tum.sh")
        print("  - associations.txt missing: run associate.py")
        print("  - OpenCV without ORB: install opencv-contrib-python")
    print("=" * 70)

    sys.exit(0 if passed else 1)