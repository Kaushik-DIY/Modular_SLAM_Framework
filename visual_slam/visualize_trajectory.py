#!/usr/bin/env python3
"""
=============================================================================
visual_slam/visualize_trajectory.py

Visualize SLAM trajectory and compare with ground truth.

Usage:
    python3 visual_slam/visualize_trajectory.py <trajectory_file> [--ground-truth <gt_file>]

Examples:
    # Visualize SLAM trajectory only
    python3 visual_slam/visualize_trajectory.py visual_slam_outputs/trajectory_rgbd_dataset_freiburg1_desk.txt
    
    # Compare with ground truth
    python3 visual_slam/visualize_trajectory.py \
        visual_slam_outputs/trajectory_rgbd_dataset_freiburg1_desk.txt \
        --ground-truth datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt

=============================================================================
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


def load_trajectory_tum(filepath: Path) -> tuple:
    """
    Load trajectory from TUM format file.
    
    TUM format: timestamp tx ty tz qx qy qz qw
    
    Returns
    -------
    timestamps : np.ndarray
        Timestamps.
    positions : np.ndarray
        Positions (N, 3).
    orientations : np.ndarray
        Quaternions (N, 4) as [x, y, z, w].
    """
    timestamps = []
    positions = []
    orientations = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            if len(parts) >= 8:
                timestamps.append(float(parts[0]))
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                orientations.append([float(parts[4]), float(parts[5]), 
                                   float(parts[6]), float(parts[7])])
    
    return (np.array(timestamps), 
            np.array(positions), 
            np.array(orientations))


def align_trajectories(est_pos, gt_pos):
    """
    Align estimated trajectory to ground truth using Horn's method.
    
    Computes optimal rotation and translation to align est_pos to gt_pos.
    
    Parameters
    ----------
    est_pos : np.ndarray
        Estimated positions (N, 3).
    gt_pos : np.ndarray
        Ground truth positions (N, 3).
    
    Returns
    -------
    aligned_pos : np.ndarray
        Aligned estimated positions (N, 3).
    """
    # Center both trajectories
    est_center = np.mean(est_pos, axis=0)
    gt_center = np.mean(gt_pos, axis=0)
    
    est_centered = est_pos - est_center
    gt_centered = gt_pos - gt_center
    
    # Compute optimal rotation using SVD
    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply alignment
    aligned = (R @ est_centered.T).T + gt_center
    
    return aligned


def compute_metrics(est_pos, gt_pos, aligned_pos):
    """
    Compute trajectory error metrics.
    
    Parameters
    ----------
    est_pos : np.ndarray
        Estimated positions (N, 3).
    gt_pos : np.ndarray
        Ground truth positions (N, 3).
    aligned_pos : np.ndarray
        Aligned estimated positions (N, 3).
    
    Returns
    -------
    dict
        Error metrics.
    """
    # Absolute Trajectory Error (ATE)
    ate = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    
    metrics = {
        'ate_mean': np.mean(ate),
        'ate_median': np.median(ate),
        'ate_std': np.std(ate),
        'ate_min': np.min(ate),
        'ate_max': np.max(ate),
        'ate_rmse': np.sqrt(np.mean(ate**2)),
    }
    
    return metrics, ate


def plot_trajectory_3d(est_pos, gt_pos=None, aligned_pos=None, output_path=None):
    """
    Plot trajectory in 3D.
    
    Parameters
    ----------
    est_pos : np.ndarray
        Estimated positions (N, 3).
    gt_pos : np.ndarray, optional
        Ground truth positions (N, 3).
    aligned_pos : np.ndarray, optional
        Aligned estimated positions (N, 3).
    output_path : Path, optional
        If provided, save figure to this path.
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    
    if gt_pos is not None:
        ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
                'g-', label='Ground Truth', linewidth=2, alpha=0.7)
    
    if aligned_pos is not None:
        ax1.plot(aligned_pos[:, 0], aligned_pos[:, 1], aligned_pos[:, 2], 
                'b-', label='SLAM (aligned)', linewidth=2, alpha=0.7)
    else:
        ax1.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 
                'b-', label='SLAM', linewidth=2, alpha=0.7)
    
    # Mark start and end
    ax1.scatter([est_pos[0, 0]], [est_pos[0, 1]], [est_pos[0, 2]], 
               c='red', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter([est_pos[-1, 0]], [est_pos[-1, 1]], [est_pos[-1, 2]], 
               c='orange', s=100, marker='s', label='End', zorder=5)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Top-down view (X-Y)
    ax2 = fig.add_subplot(132)
    
    if gt_pos is not None:
        ax2.plot(gt_pos[:, 0], gt_pos[:, 1], 
                'g-', label='Ground Truth', linewidth=2, alpha=0.7)
    
    if aligned_pos is not None:
        ax2.plot(aligned_pos[:, 0], aligned_pos[:, 1], 
                'b-', label='SLAM (aligned)', linewidth=2, alpha=0.7)
    else:
        ax2.plot(est_pos[:, 0], est_pos[:, 1], 
                'b-', label='SLAM', linewidth=2, alpha=0.7)
    
    ax2.scatter([est_pos[0, 0]], [est_pos[0, 1]], 
               c='red', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter([est_pos[-1, 0]], [est_pos[-1, 1]], 
               c='orange', s=100, marker='s', label='End', zorder=5)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (X-Y)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Plot 3: Error over time (if ground truth available)
    ax3 = fig.add_subplot(133)
    
    if gt_pos is not None and aligned_pos is not None:
        ate = np.linalg.norm(aligned_pos - gt_pos, axis=1)
        ax3.plot(ate, 'r-', linewidth=1.5)
        ax3.axhline(np.mean(ate), color='b', linestyle='--', 
                   label=f'Mean: {np.mean(ate):.3f}m')
        ax3.axhline(np.median(ate), color='g', linestyle='--', 
                   label=f'Median: {np.median(ate):.3f}m')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Error (m)')
        ax3.set_title('Absolute Trajectory Error (ATE)')
        ax3.legend()
        ax3.grid(True)
    else:
        # Show trajectory length over time
        trajectory_length = np.zeros(len(est_pos))
        for i in range(1, len(est_pos)):
            trajectory_length[i] = trajectory_length[i-1] + \
                np.linalg.norm(est_pos[i] - est_pos[i-1])
        
        ax3.plot(trajectory_length, 'b-', linewidth=1.5)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Distance (m)')
        ax3.set_title(f'Cumulative Distance: {trajectory_length[-1]:.2f}m')
        ax3.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SLAM trajectory and compare with ground truth"
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to SLAM trajectory file (TUM format)"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Path to ground truth file (TUM format)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save figure to this path (e.g., trajectory.png)"
    )
    
    args = parser.parse_args()
    
    if not args.trajectory.exists():
        print(f"ERROR: Trajectory file not found: {args.trajectory}")
        return
    
    print("=" * 70)
    print("SLAM Trajectory Visualization")
    print("=" * 70)
    print(f"Trajectory: {args.trajectory}")
    
    # Load SLAM trajectory
    est_t, est_pos, est_ori = load_trajectory_tum(args.trajectory)
    print(f"  Loaded {len(est_pos)} poses")
    
    # Compute basic statistics
    total_dist = np.sum(np.linalg.norm(np.diff(est_pos, axis=0), axis=1))
    print(f"  Total distance: {total_dist:.2f}m")
    print(f"  Time span: {est_t[-1] - est_t[0]:.2f}s")
    
    bbox_min = np.min(est_pos, axis=0)
    bbox_max = np.max(est_pos, axis=0)
    print(f"  Bounding box: X[{bbox_min[0]:.2f}, {bbox_max[0]:.2f}] "
          f"Y[{bbox_min[1]:.2f}, {bbox_max[1]:.2f}] "
          f"Z[{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]")
    
    # Load ground truth if provided
    gt_pos = None
    aligned_pos = None
    
    if args.ground_truth:
        if not args.ground_truth.exists():
            print(f"WARNING: Ground truth file not found: {args.ground_truth}")
        else:
            print()
            print(f"Ground Truth: {args.ground_truth}")
            gt_t, gt_pos, gt_ori = load_trajectory_tum(args.ground_truth)
            print(f"  Loaded {len(gt_pos)} poses")
            
            # Find common timestamps (associate trajectories)
            min_len = min(len(est_pos), len(gt_pos))
            est_pos = est_pos[:min_len]
            gt_pos = gt_pos[:min_len]
            
            # Align trajectories
            print()
            print("Aligning trajectories...")
            aligned_pos = align_trajectories(est_pos, gt_pos)
            
            # Compute metrics
            metrics, ate = compute_metrics(est_pos, gt_pos, aligned_pos)
            
            print()
            print("=" * 70)
            print("Trajectory Error Metrics (ATE)")
            print("=" * 70)
            print(f"  RMSE:   {metrics['ate_rmse']:.4f} m")
            print(f"  Mean:   {metrics['ate_mean']:.4f} m")
            print(f"  Median: {metrics['ate_median']:.4f} m")
            print(f"  Std:    {metrics['ate_std']:.4f} m")
            print(f"  Min:    {metrics['ate_min']:.4f} m")
            print(f"  Max:    {metrics['ate_max']:.4f} m")
    
    print()
    print("=" * 70)
    print("Generating visualization...")
    
    # Plot
    plot_trajectory_3d(est_pos, gt_pos, aligned_pos, args.output)
    
    print("=" * 70)


if __name__ == "__main__":
    main()