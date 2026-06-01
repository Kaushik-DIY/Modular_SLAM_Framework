cmake_minimum_required(VERSION 3.14)
project(slam_optimizer_core CXX)

# Custom g2o-based bundle-adjustment / pose-optimization core (SOC) used by
# visual_slam (optimizer_g2o -> slam_optimizer_core). The source is owned by this
# repo; it links the *prebuilt* g2o static libraries from the vendored g2opy tree
# (../g2opy/lib) and uses its generated headers (../g2opy + ../g2opy/build).
#
# Build:  cmake -S . -B build && cmake --build build -j
# Install: copy build/slam_optimizer_core.cpython-311-*.so into the venv
#          site-packages (same as cpp_slam_core).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the venv Python so pybind11 finds the right interpreter
set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python" CACHE FILEPATH "")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11 3.x from venv — compatible with Python 3.11
find_package(pybind11 REQUIRED
    HINTS "$ENV{VIRTUAL_ENV}/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
)

find_package(Eigen3 REQUIRED)

# Prebuilt g2o artifacts from the vendored g2opy tree (sibling directory)
set(G2O_SRC_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/../g2opy")
set(G2O_LIB_DIR   "${G2O_SRC_DIR}/lib")
set(G2O_BUILD_DIR "${G2O_SRC_DIR}/build")  # for g2o/config.h

# Compile with the same flags as g2opy to match ABI
add_compile_options(-O3 -march=native -fPIC)

pybind11_add_module(slam_optimizer_core
    "${CMAKE_CURRENT_SOURCE_DIR}/slam_optimizer_core.cpp"
)

target_include_directories(slam_optimizer_core PRIVATE
    ${G2O_SRC_DIR}                # g2o/ headers are under ../g2opy/g2o/
    ${G2O_BUILD_DIR}              # g2o/config.h generated here
    ${EIGEN3_INCLUDE_DIR}
    "$ENV{VIRTUAL_ENV}/lib/python3.11/site-packages/numpy/core/include"
)

# Link the prebuilt g2o static libraries
target_link_libraries(slam_optimizer_core PRIVATE
    "${G2O_LIB_DIR}/libg2o_core.a"
    "${G2O_LIB_DIR}/libg2o_types_sba.a"
    "${G2O_LIB_DIR}/libg2o_solver_eigen.a"
    "${G2O_LIB_DIR}/libg2o_stuff.a"
    cholmod
    lapack
    blas
)
