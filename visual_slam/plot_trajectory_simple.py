#!/usr/bin/env python3
"""
Quick 2D trajectory plotter for fast visualization.

Usage:
    python3 visual_slam/plot_trajectory_simple.py <trajectory_file>
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_trajectory(filepath):
    """Load TUM format trajectory."""
    positions = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(positions)


def plot_simple(positions, title="SLAM Trajectory"):
    """Simple 2D plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top-down view (X-Y)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=200, 
               marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, 
               marker='s', label='End', zorder=5, edgecolors='black', linewidths=2)
    
    # Add arrows to show direction
    step = max(1, len(positions) // 20)
    for i in range(0, len(positions) - step, step):
        dx = positions[i + step, 0] - positions[i, 0]
        dy = positions[i + step, 1] - positions[i, 1]
        ax1.arrow(positions[i, 0], positions[i, 1], dx * 0.5, dy * 0.5,
                 head_width=0.05, head_length=0.05, fc='gray', ec='gray', alpha=0.3)
    
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('Top-Down View (X-Y Plane)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Side view (X-Z)
    ax2.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax2.scatter(positions[0, 0], positions[0, 2], c='green', s=200, 
               marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax2.scatter(positions[-1, 0], positions[-1, 2], c='red', s=200, 
               marker='s', label='End', zorder=5, edgecolors='black', linewidths=2)
    
    ax2.set_xlabel('X (meters)', fontsize=12)
    ax2.set_ylabel('Z (meters)', fontsize=12)
    ax2.set_title('Side View (X-Z Plane)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Stats
    total_dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    bbox = np.max(positions, axis=0) - np.min(positions, axis=0)
    
    stats_text = f"Total Distance: {total_dist:.2f}m\n"
    stats_text += f"Bounding Box: {bbox[0]:.2f}m × {bbox[1]:.2f}m × {bbox[2]:.2f}m\n"
    stats_text += f"Total Poses: {len(positions)}"
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_trajectory_simple.py <trajectory_file>")
        sys.exit(1)
    
    traj_file = Path(sys.argv[1])
    if not traj_file.exists():
        print(f"ERROR: File not found: {traj_file}")
        sys.exit(1)
    
    print(f"Loading trajectory: {traj_file}")
    positions = load_trajectory(traj_file)
    print(f"Loaded {len(positions)} poses")
    
    plot_simple(positions, f"SLAM Trajectory: {traj_file.name}")