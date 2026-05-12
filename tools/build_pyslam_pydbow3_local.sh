#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
SOURCE_REF="$REPO_ROOT/third_party/pyslam_reference/thirdparty/pydbow3"
PYBIND11_REF="$REPO_ROOT/third_party/pyslam_reference/thirdparty/pybind11"
BUILD_ROOT="$REPO_ROOT/third_party/build/pydbow3"
SOURCE_COPY="$BUILD_ROOT/source"
LOCAL_DIR="$REPO_ROOT/third_party/local/pydbow3"

usage() {
    echo "Usage: bash tools/build_pyslam_pydbow3_local.sh [--clean]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf "$BUILD_ROOT" "$LOCAL_DIR"
    echo "Cleaned local pydbow3 build/install artifacts."
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

if [[ ! -f "$SOURCE_REF/CMakeLists.txt" ]]; then
    echo "ERROR: pySLAM pydbow3 source not found: $SOURCE_REF" >&2
    exit 1
fi

if [[ ! -f "$PYBIND11_REF/CMakeLists.txt" ]]; then
    echo "ERROR: pySLAM pybind11 source not found: $PYBIND11_REF" >&2
    exit 1
fi

mkdir -p "$BUILD_ROOT" "$LOCAL_DIR"
rm -rf "$SOURCE_COPY" "$BUILD_ROOT/pybind11"
cp -a "$SOURCE_REF" "$SOURCE_COPY"
cp -a "$PYBIND11_REF" "$BUILD_ROOT/pybind11"
rm -rf "$SOURCE_COPY/build" "$SOURCE_COPY/lib" "$SOURCE_COPY/modules/dbow3/build" "$SOURCE_COPY/modules/dbow3/install"

"$PYTHON_EXE" - "$SOURCE_COPY/src/py_dbow3.cpp" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

if "struct TransformResult" not in text:
    text = text.replace(
        "namespace py = pybind11;\n",
        """namespace py = pybind11;

namespace DBoW3
{
struct TransformResult
{
\tDBoW3::BowVector bowVector;
\tDBoW3::FeatureVector featureVector;
};
}

static std::vector<cv::Mat> rows_to_feature_vector(const cv::Mat &features)
{
\tstd::vector<cv::Mat> rows;
\trows.reserve(features.rows);
\tfor (int r = 0; r < features.rows; ++r)
\t\trows.push_back(features.row(r));
\treturn rows;
}
""",
    )

if "transform_with_feature_vector" not in text:
    text = text.replace(
        "\tdouble score(const DBoW3::BowVector &A, const DBoW3::BowVector &B)\n",
        """\tDBoW3::TransformResult transform_with_feature_vector(const cv::Mat &features, int levelsup = 4)
\t{
\t\tDBoW3::TransformResult result;
\t\tstd::vector<cv::Mat> rows = rows_to_feature_vector(features);
\t\tvocabulary->transform(rows, result.bowVector, result.featureVector, levelsup);
\t\treturn result;
\t}

\tdouble score(const DBoW3::BowVector &A, const DBoW3::BowVector &B)\n""",
    )

if 'py::class_<DBoW3::FeatureVector>(m, "FeatureVector")' not in text:
    text = text.replace(
        '\tpy::class_<DBoW3::BowVector>(m, "BowVector")\n',
        """\tpy::class_<DBoW3::FeatureVector>(m, "FeatureVector")
\t\t.def(py::init<>())
\t\t.def("to_dict", [](const DBoW3::FeatureVector &obj) {
\t\t\tstd::map<DBoW3::NodeId, std::vector<unsigned int>> out;
\t\t\tfor (const auto &item : obj)
\t\t\t\tout[item.first] = item.second;
\t\t\treturn out;
\t\t})
\t\t.def("__len__", [](const DBoW3::FeatureVector &obj) { return obj.size(); })
\t\t.def("__repr__", [](const DBoW3::FeatureVector &obj) {
\t\t\tstd::ostringstream os;
\t\t\tos << obj;
\t\t\treturn os.str();
\t\t})
\t\t.def("__str__", [](const DBoW3::FeatureVector &obj) {
\t\t\tstd::ostringstream os;
\t\t\tos << obj;
\t\t\treturn os.str();
\t\t});

\tpy::class_<DBoW3::TransformResult>(m, "TransformResult")
\t\t.def(py::init<>())
\t\t.def_readwrite("bowVector", &DBoW3::TransformResult::bowVector)
\t\t.def_readwrite("featureVector", &DBoW3::TransformResult::featureVector);

\tpy::class_<DBoW3::BowVector>(m, "BowVector")\n""",
    )

if '.def("transform_with_feature_vector", &Vocabulary::transform_with_feature_vector' not in text:
    text = text.replace(
        '\t\t.def("transform", &Vocabulary::transform)\n',
        '\t\t.def("transform", &Vocabulary::transform)\n'
        '\t\t.def("transform_with_feature_vector", &Vocabulary::transform_with_feature_vector, py::arg("features"), py::arg("levelsup") = 4)\n',
    )

path.write_text(text)
PY

echo "Building pySLAM pydbow3 locally"
echo "Repository: $REPO_ROOT"
echo "Python:     $VENV_PATH"
echo "Build root: $BUILD_ROOT"
echo "Install:    $LOCAL_DIR"

cmake -S "$SOURCE_COPY/modules/dbow3" -B "$BUILD_ROOT/dbow3-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$SOURCE_COPY/modules/dbow3/install" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_UTILS=OFF \
    -DBUILD_TESTS=OFF
cmake --build "$BUILD_ROOT/dbow3-build" --parallel 2
cmake --install "$BUILD_ROOT/dbow3-build"

cmake -S "$SOURCE_COPY" -B "$BUILD_ROOT/bindings-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$PYTHON_EXE" \
    -DPython_EXECUTABLE="$PYTHON_EXE" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXE" \
    -DPYBIND11_FINDPYTHON=ON \
    -DBUILD_WITH_MARCH_NATIVE=OFF
cmake --build "$BUILD_ROOT/bindings-build" --parallel 2

PYDBOW3_SO="$(find "$SOURCE_COPY/lib" -maxdepth 1 -type f -name 'pydbow3*.so' | sort | tail -1)"
if [[ -z "$PYDBOW3_SO" || ! -f "$PYDBOW3_SO" ]]; then
    echo "ERROR: pydbow3 shared object was not produced." >&2
    exit 1
fi

cp "$PYDBOW3_SO" "$LOCAL_DIR/"

SITE_PACKAGES="$("$PYTHON_EXE" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0])
PY
)"
mkdir -p "$SITE_PACKAGES"
printf '%s\n' "$LOCAL_DIR" > "$SITE_PACKAGES/slam_ws_pydbow3_local.pth"

PYTHONPATH="$LOCAL_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_EXE" - <<'PY'
import pydbow3
print("pydbow3 import ok:", pydbow3.__file__)
print("pydbow3 feature-vector API:", hasattr(pydbow3.Vocabulary(), "transform_with_feature_vector"))
PY

echo "Installed pydbow3 binding:"
find "$LOCAL_DIR" -maxdepth 1 -type f -name 'pydbow3*.so' -printf '  %p %s bytes\n'
echo "Installed venv path file: $SITE_PACKAGES/slam_ws_pydbow3_local.pth"
