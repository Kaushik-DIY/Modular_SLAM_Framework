#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
VOCAB_DIR="$REPO_ROOT/third_party/vocabs"
TARGET_DBOW3="$VOCAB_DIR/ORBvoc.dbow3"
SOURCE_DBOW3="$REPO_ROOT/third_party/pyslam_reference/thirdparty/pydbow3/modules/dbow3/orbvoc.dbow3"
PYSLAM_DBOW3_URL="https://drive.google.com/uc?id=13xmRtop_ow3aPtv3qCT5beG19_mlogqI"

usage() {
    echo "Usage: bash tools/install_pyslam_vocabulary_local.sh [--clean]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--clean" ]]; then
    mkdir -p "$VOCAB_DIR"
    rm -f "$VOCAB_DIR/ORBvoc.dbow3" "$VOCAB_DIR/ORBvoc.dbow2" "$VOCAB_DIR/ORBvoc.txt"
    rm -f "$VOCAB_DIR/ORBvoc.dbow3.tmp" "$VOCAB_DIR/ORBvoc.dbow2.tmp" "$VOCAB_DIR/ORBvoc.txt.tmp"
    echo "Cleaned local pySLAM vocabulary artifacts from $VOCAB_DIR"
    exit 0
fi

if [[ $# -gt 0 ]]; then
    usage
    exit 2
fi

if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "ERROR: required venv Python not found: $PYTHON_EXE" >&2
    exit 1
fi

VENV_PATH="$("$PYTHON_EXE" -c 'import sys; print(sys.executable)')"
if [[ "$VENV_PATH" != "$PYTHON_EXE" ]]; then
    echo "ERROR: unexpected Python executable: $VENV_PATH" >&2
    exit 1
fi

mkdir -p "$VOCAB_DIR"

echo "pySLAM DBOW3 vocabulary local installer"
echo "Repository: $REPO_ROOT"
echo "Python:     $VENV_PATH"
echo "Target:     $TARGET_DBOW3"
echo "Source URL: $PYSLAM_DBOW3_URL"

if [[ -f "$TARGET_DBOW3" ]]; then
    echo "Vocabulary already installed."
elif [[ -f "$SOURCE_DBOW3" ]]; then
    echo "Copying bundled pySLAM vocabulary:"
    echo "  $SOURCE_DBOW3"
    cp "$SOURCE_DBOW3" "$TARGET_DBOW3.tmp"
    mv "$TARGET_DBOW3.tmp" "$TARGET_DBOW3"
else
    echo "ERROR: bundled pySLAM vocabulary was not found." >&2
    echo "pySLAM source URL is: $PYSLAM_DBOW3_URL" >&2
    echo "Download was not attempted automatically because Google Drive downloads need confirmation handling." >&2
    exit 1
fi

if [[ ! -s "$TARGET_DBOW3" ]]; then
    echo "ERROR: installed vocabulary is empty: $TARGET_DBOW3" >&2
    exit 1
fi

SIZE_BYTES="$(wc -c < "$TARGET_DBOW3")"
echo "Installed vocabulary: $TARGET_DBOW3"
echo "Size bytes: $SIZE_BYTES"
