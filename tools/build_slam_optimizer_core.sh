#!/bin/bash
# Build slam_optimizer_core — GIL-free C++ g2o BA module.
# Run from workspace root with the venv active.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$SCRIPT_DIR/.."
BUILD_DIR="$WORKSPACE/third_party/build/slam_optimizer_core"

cd "$BUILD_DIR"
cmake . -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

SITE_PACKAGES="$WORKSPACE/.venv/lib/python3.11/site-packages"
cp slam_optimizer_core.cpython-311-x86_64-linux-gnu.so "$SITE_PACKAGES/"

echo ""
echo "Build complete. Testing..."
"$WORKSPACE/.venv/bin/python" -c "import slam_optimizer_core; print(slam_optimizer_core.hello())"
echo "Done. Module installed to $SITE_PACKAGES/"
