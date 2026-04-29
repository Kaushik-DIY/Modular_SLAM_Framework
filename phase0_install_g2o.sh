#!/bin/bash
# =============================================================================
# Phase 0, Step 1: Install g2o Python Bindings (g2opy)
# =============================================================================
# 
# Purpose:
#   Install g2o graph optimization library with Python bindings.
#   g2o is required by the Visual SLAM pipeline for:
#     - Motion-only Bundle Adjustment (tracking front-end)
#     - Local Bundle Adjustment (local mapping back-end)
#     - Pose Graph Optimization (loop closing)
#     - Global Bundle Adjustment (post loop closure)
#
# Prerequisites:
#   - Ubuntu 20.04 / 22.04 / 24.04
#   - Python 3.8+ with pip
#   - cmake, build-essential
#
# Usage:
#   chmod +x phase0_install_g2o.sh
#   ./phase0_install_g2o.sh
#
# If this script fails, see the "Manual Fallback" section at the bottom.
# =============================================================================

set -e  # Exit on any error

echo "============================================================"
echo "Phase 0, Step 1: Installing g2o Python Bindings"
echo "============================================================"

# ------------------------------------------------------------------
# 0. Remove any existing g2opy installation
# ------------------------------------------------------------------
echo ""
echo "[0/5] Checking for existing g2o installations..."
if pip show g2opy &>/dev/null || pip show g2o-python &>/dev/null; then
    echo "      Found existing g2o package(s). Uninstalling..."
    pip uninstall -y g2opy g2o-python 2>&1 | grep -i "successfully\|not found" || true
    echo "      Cleanup complete."
else
    echo "      No existing installation found."
fi

# ------------------------------------------------------------------
# 1. Install system dependencies
# ------------------------------------------------------------------
echo ""
echo "[1/6] Installing system dependencies..."
echo "      (requires sudo — enter password if prompted)"
echo ""

sudo apt-get update -qq
sudo apt-get install -y \
    cmake \
    build-essential \
    libeigen3-dev \
    libsuitesparse-dev \
    libspdlog-dev \
    libfmt-dev \
    python3-dev \
    2>&1 | tail -5

echo "      System dependencies installed."

# ------------------------------------------------------------------
# 2. Clone g2opy (uoip fork with complete Python bindings)
# ------------------------------------------------------------------
echo ""
echo "[2/6] Cloning g2opy repository (uoip fork)..."
echo "      This fork has complete SE2/SE3 bindings for SLAM."
echo ""

# Clone into a temporary build directory
G2O_BUILD_DIR="/tmp/g2opy_build_$(date +%s)"
mkdir -p "$G2O_BUILD_DIR"
cd "$G2O_BUILD_DIR"

git clone --depth 1 https://github.com/uoip/g2opy.git
cd g2opy

echo "      Cloned to: $G2O_BUILD_DIR/g2opy"

# ------------------------------------------------------------------
# 3. Build g2opy
# ------------------------------------------------------------------
echo ""
echo "[3/6] Building g2opy (this may take 5-10 minutes)..."
echo ""

mkdir -p build
cd build

cmake .. 2>&1 | tail -10
make -j$(nproc) 2>&1 | tail -20

echo "      Build complete."

# ------------------------------------------------------------------
# 4. Install Python bindings
# ------------------------------------------------------------------
echo ""
echo "[4/6] Installing Python bindings..."
echo ""

cd "$G2O_BUILD_DIR/g2opy"
pip install . 2>&1 | tail -5

echo "      Python bindings installed."

# ------------------------------------------------------------------
# 5. Verify installation
# ------------------------------------------------------------------
echo ""
echo "[5/6] Verifying g2o Python import..."
echo ""

python3 -c "
import g2o
print('  g2o module loaded successfully')

# Check for required SE3 classes
required = ['SparseOptimizer', 'VertexSE3', 'EdgeSE3', 'BlockSolverSE3']
available = dir(g2o)
missing = [cls for cls in required if cls not in available]

if missing:
    print(f'  ERROR: Missing required classes: {missing}')
    print(f'  Available classes: {[x for x in available if not x.startswith(\"_\")][:15]}')
    exit(1)

print(f'  Required SE3 classes present: {required}')

# Quick sanity: can we create an optimizer?
opt = g2o.SparseOptimizer()
print('  SparseOptimizer created successfully')

# Can we create SE3 vertex?
v = g2o.VertexSE3()
print('  VertexSE3 created successfully')

print()
print('  g2o installation VERIFIED.')
"

echo ""
echo "============================================================"
echo "Phase 0, Step 1: COMPLETE"
echo ""
echo "Next step: Run the validation checkpoint:"
echo "  python3 tests/test_checkpoint_0_g2o.py"
echo "============================================================"

# ------------------------------------------------------------------
# 6. Cleanup (optional — uncomment to remove build files)
# ------------------------------------------------------------------
echo ""
echo "[6/6] Cleaning up build directory..."
rm -rf "$G2O_BUILD_DIR"
echo "      Build files removed."

exit 0

# =============================================================================
# MANUAL FALLBACK (if the above fails)
# =============================================================================
#
# Option A: Try the older uoip/g2opy fork
#
#   git clone https://github.com/uoip/g2opy.git
#   cd g2opy
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#   cd ..
#   python setup.py install
#
# Option B: Try pip install g2o-python (pre-built wheels, may not have all types)
#
#   pip install g2o-python
#
# Option C: Use pyslam's built-in g2o build
#
#   git clone --recursive https://github.com/luigifreda/pyslam.git
#   cd pyslam
#   ./install_all.sh
#   # g2o is built as part of pyslam's installation
#   # Then copy the g2o .so file to your site-packages
#
# =============================================================================
