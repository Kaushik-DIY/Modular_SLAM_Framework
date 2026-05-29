#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_ROOT="${1:-${REPO_ROOT}/PY-ORB}"

echo "Staging PY-ORB into: ${DEST_ROOT}"
rm -rf "${DEST_ROOT}"
mkdir -p "${DEST_ROOT}"

copy_file() {
    local rel="$1"
    mkdir -p "${DEST_ROOT}/$(dirname "${rel}")"
    cp "${REPO_ROOT}/${rel}" "${DEST_ROOT}/${rel}"
}

copy_tree_filtered() {
    local src_rel="$1"
    local dest_rel="$2"
    mkdir -p "${DEST_ROOT}/${dest_rel}"
    rsync -a \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '*.pyo' \
        --exclude '*.pyd' \
        --exclude '.DS_Store' \
        "${REPO_ROOT}/${src_rel}/" "${DEST_ROOT}/${dest_rel}/"
}

copy_file "visual_slam/g2o_compat.py"
copy_tree_filtered "visual_slam/orbslam" "visual_slam/orbslam"
rm -f "${DEST_ROOT}/visual_slam/orbslam/run_tum_rgbd_smoke.py"

copy_file "tools/export_orbslam_map.py"
copy_file "tools/run_fr1_room_full_evaluation.py"
copy_file "tools/evaluate_tum_trajectory.py"
copy_file "tools/build_tum_reference_cloud.py"
copy_file "tools/plot_fr1_room_evaluation.py"

copy_file "third_party/local/pydbow3/pydbow3.cpython-311-x86_64-linux-gnu.so"
copy_file "third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so"
copy_file "third_party/vocabs/ORBvoc.dbow3"

cat > "${DEST_ROOT}/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.venv/
venv/
datasets/
visual_slam_outputs/
outputs/
carto_outputs/
hector_outputs/
*.log
*.zip
*.npy
*.npz
*.ply
*.jsonl
EOF

cat > "${DEST_ROOT}/MANIFEST_PY_ORB.md" <<'EOF'
# PY-ORB Manifest

This staged repository includes the visual ORB-SLAM pipeline and the direct helper files it imports at runtime.

Included:
- `visual_slam/g2o_compat.py`
- `visual_slam/orbslam/` except `run_tum_rgbd_smoke.py`
- `tools/export_orbslam_map.py`
- `tools/run_fr1_room_full_evaluation.py`
- `tools/evaluate_tum_trajectory.py`
- `tools/build_tum_reference_cloud.py`
- `tools/plot_fr1_room_evaluation.py`
- `third_party/local/pydbow3/pydbow3.cpython-311-x86_64-linux-gnu.so`
- `third_party/local/orbslam2_features/orbslam2_features.cpython-311-x86_64-linux-gnu.so`
- `third_party/vocabs/ORBvoc.dbow3`

Excluded on purpose:
- `datasets/`
- `visual_slam/reference_audit/`
- `visual_slam/orbslam/run_tum_rgbd_smoke.py`
- `third_party/pyslam_reference/`
- `third_party/g2opy/`
- build directories, caches, outputs, logs, and zip archives

Notes:
- `g2o` itself is not bundled here. Install it separately in the target environment.
- `orbslam2_features` is optional if you run with the OpenCV ORB backend instead of `pyslam_orb2`.
- The included `.so` files are platform-specific build artifacts for this environment.
EOF

echo "Done."
echo "Next steps:"
echo "  1. cd \"${DEST_ROOT}\""
echo "  2. git init"
echo "  3. git add ."
echo "  4. git commit -m 'Initial PY-ORB import'"
