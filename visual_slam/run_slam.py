#!/usr/bin/env python3
"""
=============================================================================
visual_slam/run_slam.py

CLI entry point for running Visual SLAM on datasets.

IMPORTANT: This is the main script for processing RGBD datasets.
---------------------------------------------------------------------------
Processes TUM RGBD datasets and outputs trajectory files.

Usage:
    python3 visual_slam/run_slam.py <dataset_path> [--output <path>]

Examples:
    python3 visual_slam/run_slam.py datasets/tum/rgbd_dataset_freiburg1_desk
    python3 visual_slam/run_slam.py datasets/tum/rgbd_dataset_freiburg1_room --output results/

References
----------
pyslam: main_slam.py
ORB-SLAM2: Examples/RGB-D/rgbd_tum.cc

=============================================================================
"""

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import cv2

from visual_slam.adapter import VisualSlamAdapter
from visual_slam.tracking import TrackingState
from slam_core.common.types3d import CameraIntrinsics


def load_tum_associations(dataset_path: Path) -> list:
    """
    Load TUM dataset associations file.
    
    Parameters
    ----------
    dataset_path : Path
        Path to TUM dataset directory.
    
    Returns
    -------
    list
        List of (timestamp, rgb_path, depth_path) tuples.
    """
    associations_file = dataset_path / "associations.txt"
    
    if not associations_file.exists():
        raise FileNotFoundError(f"Associations file not found: {associations_file}")
    
    associations = []
    with open(associations_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamp = float(parts[0])
                rgb_path = dataset_path / parts[1]
                depth_path = dataset_path / parts[3]
                associations.append((timestamp, rgb_path, depth_path))
    
    return associations


def save_trajectory_tum(trajectory: list, output_path: Path) -> None:
    """
    Save trajectory in TUM format.
    
    TUM format: timestamp tx ty tz qx qy qz qw
    
    Parameters
    ----------
    trajectory : list
        List of PoseEstimate objects.
    output_path : Path
        Output file path.
    """
    from scipy.spatial.transform import Rotation
    
    with open(output_path, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        for pose in trajectory:
            T = pose.matrix
            t = T[:3, 3]
            R = T[:3, :3]
            
            # Convert rotation matrix to quaternion
            rot = Rotation.from_matrix(R)
            q = rot.as_quat()  # [x, y, z, w]
            
            f.write(f"{pose.timestamp:.6f} "
                   f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                   f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def run_visual_slam(
    dataset_path: Path,
    output_dir: Path,
    max_frames: int = -1,
) -> None:
    """
    Run Visual SLAM on TUM RGBD dataset.
    
    Parameters
    ----------
    dataset_path : Path
        Path to TUM dataset directory.
    output_dir : Path
        Output directory for trajectory files.
    max_frames : int
        Maximum number of frames to process (-1 = all).
    """
    print("=" * 70)
    print("Visual SLAM - TUM RGBD Dataset Processing")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset associations...")
    associations = load_tum_associations(dataset_path)
    
    if max_frames > 0:
        associations = associations[:max_frames]
    
    print(f"  Found {len(associations)} frames")
    print()
    
    # TUM fr1/desk camera parameters
    camera = CameraIntrinsics(
        fx=517.3,
        fy=516.5,
        cx=318.6,
        cy=255.3,
        width=640,
        height=480,
        depth_scale=5000.0,  # TUM depth is in millimeters / 5
    )
    
    # Create SLAM system
    print("Initializing Visual SLAM...")
    slam = VisualSlamAdapter(camera)
    print()
    
    # Process frames
    print("Processing frames...")
    print()
    
    start_time = time.time()
    num_tracked = 0
    num_lost = 0
    
    for i, (timestamp, rgb_path, depth_path) in enumerate(associations):
        # Load images
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"  WARNING: Could not load RGB image: {rgb_path}")
            continue
        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"  WARNING: Could not load depth image: {depth_path}")
            continue
        
        # Process frame
        pose, state = slam.process_frame(rgb, depth, timestamp)
        
        if state == TrackingState.OK:
            num_tracked += 1
        elif state == TrackingState.LOST:
            num_lost += 1
        
        # Progress output
        if (i + 1) % 50 == 0:
            stats = slam.get_stats()
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            
            print(f"  Frame {i+1}/{len(associations)}: "
                  f"{stats['tracking_state']} | "
                  f"KFs: {stats['keyframes_created']} | "
                  f"MPs: {stats['map_points']} | "
                  f"FPS: {fps:.1f}")
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("Processing complete!")
    print("=" * 70)
    
    stats = slam.get_stats()
    print(f"  Frames processed:  {stats['frames_processed']}")
    print(f"  Frames tracked:    {num_tracked}")
    print(f"  Frames lost:       {num_lost}")
    print(f"  Keyframes created: {stats['keyframes_created']}")
    print(f"  Map points:        {stats['map_points']}")
    print(f"  Loops closed:      {stats['loops_closed']}")
    print(f"  Processing time:   {elapsed:.2f}s")
    print(f"  Average FPS:       {len(associations)/elapsed:.1f}")
    print()
    
    # Save trajectory
    trajectory = slam.get_trajectory()
    
    if len(trajectory) > 0:
        output_file = output_dir / f"trajectory_{dataset_path.name}.txt"
        save_trajectory_tum(trajectory, output_file)
        print(f"Trajectory saved: {output_file}")
        print(f"  {len(trajectory)} poses")
    else:
        print("WARNING: No trajectory to save (tracking failed)")
    
    print()
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Visual SLAM on TUM RGBD dataset"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to TUM dataset directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visual_slam_outputs"),
        help="Output directory for trajectory files (default: visual_slam_outputs/)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process (default: all)"
    )
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"ERROR: Dataset directory not found: {args.dataset}")
        sys.exit(1)
    
    try:
        run_visual_slam(args.dataset, args.output, args.max_frames)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()