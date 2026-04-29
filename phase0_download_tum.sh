#!/bin/bash
# =============================================================================
# Phase 0, Step 2: Download TUM RGBD Dataset Sequences
# =============================================================================
#
# Purpose:
#   Download TUM RGBD benchmark sequences for offline Visual SLAM testing.
#   These datasets provide synchronized RGB + Depth images with ground truth
#   camera trajectories, which we use to validate our Visual SLAM pipeline.
#
# Directory structure:
#   slam_ws/
#     datasets/
#       fr079/              <- existing Lidar dataset
#       intel/              <- existing Lidar dataset  
#       lab_run_2/          <- existing Lidar dataset
#       tum/                <- NEW: TUM RGBD sequences go here
#         freiburg1_desk/
#         freiburg1_room/
#         freiburg2_desk/
#
# Sequences downloaded:
#   - fr1/desk  : Desktop scene, moderate motion, ~600 frames (primary test)
#   - fr1/room  : Room-scale loop, ~1400 frames (loop closure test)
#   - fr2/desk  : Desktop with slower motion, ~2900 frames (accuracy test)
#
# Each sequence contains:
#   rgb/          — 640x480 PNG color images (~30Hz)
#   depth/        — 640x480 16-bit PNG depth images (scale: 5000 = 1 meter)
#   groundtruth.txt  — timestamp tx ty tz qx qy qz qw
#   rgb.txt       — timestamp filename
#   depth.txt     — timestamp filename
#   accelerometer.txt — IMU data (if available)
#
# Usage (from slam_ws root, with venv activated):
#   chmod +x phase0_download_tum.sh
#   ./phase0_download_tum.sh
#
# The script auto-detects slam_ws root from its location.
# =============================================================================

set -e

# ------------------------------------------------------------------
# Auto-detect slam_ws root directory
# ------------------------------------------------------------------
# Assumes this script is placed in slam_ws root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If you placed the script in slam_ws root, use it directly
# Otherwise, navigate up to find the datasets/ folder
if [ -d "$SCRIPT_DIR/datasets" ]; then
    SLAM_WS_ROOT="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../datasets" ]; then
    SLAM_WS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "ERROR: Cannot find datasets/ folder."
    echo "Please run this script from slam_ws root, or place it there."
    exit 1
fi

DATASETS_ROOT="$SLAM_WS_ROOT/datasets"
TUM_DIR="$DATASETS_ROOT/tum"

mkdir -p "$TUM_DIR"
cd "$TUM_DIR"

echo "============================================================"
echo "Phase 0, Step 2: Download TUM RGBD Dataset"
echo ""
echo "slam_ws root:    $SLAM_WS_ROOT"
echo "datasets root:   $DATASETS_ROOT"
echo "TUM target dir:  $TUM_DIR"
echo "============================================================"

# Verify existing datasets
echo ""
echo "Existing Lidar datasets:"
for ds in fr079 intel lab_run_2; do
    if [ -d "$DATASETS_ROOT/$ds" ]; then
        echo "  ✓ $ds/"
    else
        echo "  ✗ $ds/ (not found)"
    fi
done

# ------------------------------------------------------------------
# Check if running in virtual environment
# ------------------------------------------------------------------
echo ""
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment detected:"
    echo "  VIRTUAL_ENV = $VIRTUAL_ENV"
    echo "  Python: $(which python3)"
else
    echo "WARNING: No virtual environment detected."
    echo "  If you're using a venv, activate it first:"
    echo "  source venv/bin/activate"
    echo ""
fi

# ------------------------------------------------------------------
# TUM RGBD base URL
# ------------------------------------------------------------------
TUM_BASE="https://cvg.cit.tum.de/rgbd/dataset/freiburg"

# ------------------------------------------------------------------
# Sequence definitions: name, URL suffix
# ------------------------------------------------------------------
declare -A SEQUENCES
SEQUENCES=(
    ["freiburg1_desk"]="1/rgbd_dataset_freiburg1_desk.tgz"
    ["freiburg1_room"]="1/rgbd_dataset_freiburg1_room.tgz"
    ["freiburg2_desk"]="2/rgbd_dataset_freiburg2_desk.tgz"
)

# ------------------------------------------------------------------
# Download and extract each sequence
# ------------------------------------------------------------------
for name in "freiburg1_desk" "freiburg1_room" "freiburg2_desk"; do
    url_suffix="${SEQUENCES[$name]}"
    url="${TUM_BASE}${url_suffix}"
    tarfile="$(basename "$url_suffix")"
    # TUM tarballs extract to rgbd_dataset_freiburgX_Y/
    dirname="rgbd_dataset_${name}"

    echo ""
    echo "--- $name ---"

    if [ -d "$dirname" ]; then
        echo "  Already exists: $dirname/ — skipping download"
    else
        echo "  Downloading: $url"
        wget -q --show-progress "$url" -O "$tarfile"

        echo "  Extracting..."
        tar xzf "$tarfile"

        echo "  Cleaning up tarball..."
        rm -f "$tarfile"

        echo "  Done: $dirname/"
    fi

    # Count frames
    if [ -d "$dirname/rgb" ]; then
        rgb_count=$(ls "$dirname/rgb/" 2>/dev/null | wc -l)
        echo "  RGB frames: $rgb_count"
    fi
    if [ -d "$dirname/depth" ]; then
        depth_count=$(ls "$dirname/depth/" 2>/dev/null | wc -l)
        echo "  Depth frames: $depth_count"
    fi
done

# ------------------------------------------------------------------
# Download the association script from TUM (Python 3 compatible)
# ------------------------------------------------------------------
echo ""
echo "--- Downloading association script ---"
ASSOC_SCRIPT="$TUM_DIR/associate.py"

if [ ! -f "$ASSOC_SCRIPT" ]; then
    echo "  Creating associate.py (Python 3 compatible)..."
    cat > "$ASSOC_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
Associate two timestamp-filename lists (from TUM RGBD benchmark).
Finds pairs with the smallest time difference below a given threshold.

Usage:
    python3 associate.py rgb.txt depth.txt > associations.txt
    python3 associate.py rgb.txt depth.txt --max_difference 0.02
"""
import argparse
import sys

def read_file_list(filename):
    """Read timestamp-filename pairs from TUM format file."""
    file_list = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                file_list[float(parts[0])] = ' '.join(parts[1:])
    return file_list

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate entries from two timestamp-filename dictionaries.
    
    Returns list of (timestamp_a, timestamp_b) tuples where
    abs(timestamp_b - (timestamp_a + offset)) < max_difference.
    """
    first_keys = sorted(first_list.keys())
    second_keys = sorted(second_list.keys())
    matches = []
    
    for a in first_keys:
        best_diff = max_difference
        best_b = None
        
        for b in second_keys:
            diff = abs(b - (a + offset))
            if diff < best_diff:
                best_diff = diff
                best_b = b
            # Early exit: timestamps are sorted
            if b - (a + offset) > max_difference:
                break
                
        if best_b is not None:
            matches.append((a, best_b))
    
    return matches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Associate timestamp-filename lists from TUM RGBD benchmark'
    )
    parser.add_argument('first_file', help='First file (e.g., rgb.txt)')
    parser.add_argument('second_file', help='Second file (e.g., depth.txt)')
    parser.add_argument('--max_difference', default=0.02, type=float,
                        help='Maximum time difference (default: 0.02s)')
    parser.add_argument('--offset', default=0.0, type=float,
                        help='Time offset between files (default: 0.0s)')
    args = parser.parse_args()

    first = read_file_list(args.first_file)
    second = read_file_list(args.second_file)
    matches = associate(first, second, args.offset, args.max_difference)

    for a, b in matches:
        print(f"{a:.6f} {first[a]} {b:.6f} {second[b]}")
PYEOF
    chmod +x "$ASSOC_SCRIPT"
    echo "  Created: associate.py"
else
    echo "  Already exists: associate.py"
fi

# ------------------------------------------------------------------
# Generate association files for each sequence
# ------------------------------------------------------------------
echo ""
echo "--- Generating RGB-Depth associations ---"

for seq in freiburg1_desk freiburg1_room freiburg2_desk; do
    dirname="rgbd_dataset_${seq}"
    
    if [ -d "$dirname" ]; then
        assoc_file="$dirname/associations.txt"
        
        if [ -f "$assoc_file" ]; then
            echo "  $seq: associations.txt already exists"
        else
            echo "  $seq: generating associations.txt..."
            
            # Use the Python from the venv if active, otherwise python3
            PYTHON_CMD="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/}python3"
            
            $PYTHON_CMD "$ASSOC_SCRIPT" \
                "$dirname/rgb.txt" \
                "$dirname/depth.txt" \
                > "$assoc_file"
            
            assoc_count=$(wc -l < "$assoc_file")
            echo "  $seq: $assoc_count associations created"
        fi
    fi
done

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Phase 0, Step 2: COMPLETE"
echo ""
echo "Downloaded TUM RGBD sequences to:"
echo "  $TUM_DIR/"
echo ""
for seq in freiburg1_desk freiburg1_room freiburg2_desk; do
    dirname="rgbd_dataset_${seq}"
    if [ -d "$dirname" ]; then
        rgb_count=$(ls "$dirname/rgb/" 2>/dev/null | wc -l)
        assoc_count=0
        [ -f "$dirname/associations.txt" ] && assoc_count=$(wc -l < "$dirname/associations.txt")
        printf "  %-35s %4d RGB, %4d depth pairs\n" "$seq:" "$rgb_count" "$assoc_count"
    fi
done
echo ""
echo "TUM fr1/desk camera intrinsics (for reference):"
echo "  fx=517.3  fy=516.5  cx=318.6  cy=255.3"
echo "  depth_scale=5000 (i.e., pixel_value / 5000 = meters)"
echo "  image size: 640 x 480"
echo ""
echo "Your dataset folder structure:"
echo "  $DATASETS_ROOT/"
echo "  ├── fr079/              (Lidar)"
echo "  ├── intel/              (Lidar)"
echo "  ├── lab_run_2/          (Lidar)"
echo "  └── tum/                (RGBD - NEW)"
echo "      ├── rgbd_dataset_freiburg1_desk/"
echo "      ├── rgbd_dataset_freiburg1_room/"
echo "      └── rgbd_dataset_freiburg2_desk/"
echo ""
echo "Next step: Run validation checkpoint:"
echo "  python3 tests/test_checkpoint_0_g2o.py"
echo "============================================================"
