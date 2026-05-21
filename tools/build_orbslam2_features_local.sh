#!/usr/bin/env bash
set -euo pipefail

EXPECTED_REPO_ROOT="/home/kaushik/slam_ws"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(readlink -f "${SCRIPT_DIR}/..")"

usage() {
    cat <<'EOF'
Usage: bash tools/build_orbslam2_features_local.sh [--clean] [--help]

Safely rebuild pySLAM's orbslam2_features module using only local project paths.

Options:
  --clean   Remove only the local orbslam2_features build/install artifacts and venv .pth file.
  --help    Show this help text.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 || ( $# -eq 1 && "${1}" != "--clean" ) ]]; then
    usage >&2
    exit 2
fi

if [[ "$(readlink -f "${REPO_ROOT}")" != "${EXPECTED_REPO_ROOT}" ]]; then
    echo "ERROR: expected repo root ${EXPECTED_REPO_ROOT}, got ${REPO_ROOT}" >&2
    exit 2
fi

if [[ "${EUID}" -eq 0 ]]; then
    echo "ERROR: do not run this local build script as root." >&2
    exit 2
fi

VENV_PY="${REPO_ROOT}/.venv/bin/python"
PYSLAM_ORB_SOURCE="${REPO_ROOT}/third_party/pyslam_reference/thirdparty/orbslam2_features"
PYSLAM_PYBIND11="${REPO_ROOT}/third_party/pyslam_reference/thirdparty/pybind11"
PYSLAM_CASTER="${REPO_ROOT}/third_party/pyslam_reference/cpp/casters/opencv_type_casters.h"
BUILD_ROOT="${REPO_ROOT}/third_party/build/orbslam2_features"
SOURCE_COPY="${BUILD_ROOT}/source"
CMAKE_BUILD_DIR="${BUILD_ROOT}/cmake-build"
LOCAL_INSTALL="${REPO_ROOT}/third_party/local/orbslam2_features"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "ERROR: missing executable venv Python: ${VENV_PY}" >&2
    exit 2
fi

PYTHON_EXECUTABLE="$("${VENV_PY}" -c 'import sys; print(sys.executable)')"
EXPECTED_PYTHON="${VENV_PY}"
if [[ "${PYTHON_EXECUTABLE}" != "${EXPECTED_PYTHON}" ]]; then
    echo "ERROR: venv Python mismatch: expected ${EXPECTED_PYTHON}, got ${PYTHON_EXECUTABLE}" >&2
    exit 2
fi

SITE_PACKAGES="$("${VENV_PY}" -c 'import site; print(site.getsitepackages()[0])')"
PTH_FILE="${SITE_PACKAGES}/orbslam2_features_local.pth"
case "$(readlink -f "${SITE_PACKAGES}")" in
    "${REPO_ROOT}/.venv"/*) ;;
    *)
        echo "ERROR: site-packages is outside repo venv: ${SITE_PACKAGES}" >&2
        exit 2
        ;;
esac

clean_local_artifacts() {
    echo "Removing local orbslam2_features artifacts:"
    echo "  ${BUILD_ROOT}"
    echo "  ${LOCAL_INSTALL}"
    echo "  ${PTH_FILE}"
    rm -rf "${BUILD_ROOT}"
    rm -rf "${LOCAL_INSTALL}"
    rm -f "${PTH_FILE}"
}

if [[ "${1:-}" == "--clean" ]]; then
    clean_local_artifacts
    exit 0
fi

if [[ ! -d "${PYSLAM_ORB_SOURCE}" ]]; then
    echo "ERROR: missing pySLAM orbslam2_features source: ${PYSLAM_ORB_SOURCE}" >&2
    exit 2
fi

if [[ ! -d "${PYSLAM_PYBIND11}" ]]; then
    echo "ERROR: missing pySLAM pybind11 source: ${PYSLAM_PYBIND11}" >&2
    exit 2
fi

if [[ ! -f "${PYSLAM_CASTER}" ]]; then
    echo "ERROR: missing pySLAM OpenCV caster header: ${PYSLAM_CASTER}" >&2
    exit 2
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake is not available. Install a venv-local cmake package or provide cmake locally." >&2
    exit 2
fi

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_EXECUTABLE}"
echo "Source: ${PYSLAM_ORB_SOURCE}"
echo "Build root: ${BUILD_ROOT}"
echo "Local install: ${LOCAL_INSTALL}"

rm -rf "${SOURCE_COPY}" "${CMAKE_BUILD_DIR}"
mkdir -p "${BUILD_ROOT}" "${LOCAL_INSTALL}"

cp -a "${PYSLAM_ORB_SOURCE}" "${SOURCE_COPY}"
rm -rf "${BUILD_ROOT}/pybind11"
ln -s "${REPO_ROOT}/third_party/pyslam_reference/thirdparty/pybind11" "${BUILD_ROOT}/pybind11"

# The pySLAM source uses a relative symlink that is valid only in its original tree.
# The local build copy needs the real header so the copied source is self-contained.
rm -f "${SOURCE_COPY}/opencv_type_casters.h"
cp "${PYSLAM_CASTER}" "${SOURCE_COPY}/opencv_type_casters.h"

cmake -S "${SOURCE_COPY}" \
    -B "${CMAKE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="${VENV_PY}" \
    -DPython_EXECUTABLE="${VENV_PY}" \
    -DPYTHON_EXECUTABLE="${VENV_PY}" \
    -DCMAKE_INSTALL_PREFIX="${LOCAL_INSTALL}"

cmake --build "${CMAKE_BUILD_DIR}" --config Release -j"$(nproc)"

mapfile -t BUILT_MODULES < <(find "${SOURCE_COPY}/lib" -maxdepth 1 -type f -name 'orbslam2_features*.so' | sort)
if [[ "${#BUILT_MODULES[@]}" -ne 1 ]]; then
    echo "ERROR: expected exactly one built orbslam2_features module, found ${#BUILT_MODULES[@]}" >&2
    printf '%s\n' "${BUILT_MODULES[@]}" >&2
    exit 2
fi

rm -f "${LOCAL_INSTALL}"/orbslam2_features*.so
cp "${BUILT_MODULES[0]}" "${LOCAL_INSTALL}/"
printf '%s\n' "${LOCAL_INSTALL}" > "${PTH_FILE}"

"${VENV_PY}" - <<'PY'
import sys
print("python:", sys.executable)
import orbslam2_features
print("orbslam2_features:", orbslam2_features.__file__)
print("symbols:", [s for s in dir(orbslam2_features) if "ORB" in s or "Extractor" in s])
PY
