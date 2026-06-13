#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$SCRIPT_DIR/.."
BUILD_DIR="$WORKSPACE/third_party/fusion_core/build"
VENV_SITELIB="$WORKSPACE/.venv/lib/python3.11/site-packages"

cd "$WORKSPACE"
source .venv/bin/activate

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPython3_EXECUTABLE="$WORKSPACE/.venv/bin/python" \
    "$@"

make -j"$(nproc)" fusion_core

SO_FILE=$(find . -name "fusion_core*.so" | head -1)
if [ -z "$SO_FILE" ]; then
    echo "ERROR: fusion_core*.so not found after build"
    exit 1
fi

cp "$SO_FILE" "$VENV_SITELIB/"
echo ""
echo "Installed: $VENV_SITELIB/$(basename $SO_FILE)"
echo ""
echo "Verify:"
python -c "import fusion_core; print('  hello():', fusion_core.hello())"
