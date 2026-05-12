#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$SCRIPT_DIR/.."
BUILD_DIR="$WORKSPACE/third_party/cpp_slam_core/build"
VENV_SITELIB="$WORKSPACE/.venv/lib/python3.11/site-packages"

cd "$WORKSPACE"
source .venv/bin/activate

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPython3_EXECUTABLE="$WORKSPACE/.venv/bin/python" \
    "$@"

make -j"$(nproc)" cpp_slam_core

SO_FILE=$(find . -name "cpp_slam_core*.so" | head -1)
if [ -z "$SO_FILE" ]; then
    echo "ERROR: cpp_slam_core*.so not found after build"
    exit 1
fi

cp "$SO_FILE" "$VENV_SITELIB/"
echo ""
echo "Installed: $VENV_SITELIB/$(basename $SO_FILE)"
echo ""
echo "Verify:"
python -c "import cpp_slam_core; print('  hello():', cpp_slam_core.hello()); print('  USE_CPP_CORE:', cpp_slam_core.USE_CPP_CORE)"
