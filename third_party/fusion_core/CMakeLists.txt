cmake_minimum_required(VERSION 3.16)
project(fusion_core CXX)
# C++17 (was 20): the sources use no C++20-only features, and C++17 builds with
# the Jetson/Ubuntu-18.04 stock GCC 7.5 — no gcc-10/PPA needed for aarch64.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# fusion_core — C++ core for the RTAB-inspired multi-modal shared map (v2).
# Owns: Signature store, STM/WM/LTM memory tiers, SE(2) fusion graph (g2o),
# neighborhood retrieval + local occupancy grids, correlative B&B verifier.
# Python (slam_core/fusion2/) is UI/orchestration only.
# Build/binding patterns mirror ../cpp_slam_core (proven in production here).
# ---------------------------------------------------------------------------

# Python + pybind11 from venv
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pb11_ret
)
if(NOT _pb11_ret EQUAL 0)
    message(FATAL_ERROR "Could not locate pybind11 cmake dir via Python")
endif()
find_package(pybind11 CONFIG REQUIRED HINTS ${PYBIND11_CMAKE_DIR})

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# Prebuilt g2o static libs + headers from the vendored g2opy tree
set(G2O_SRC_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/../g2opy")
set(G2O_LIB_DIR   "${G2O_SRC_DIR}/lib")
set(G2O_BUILD_DIR "${G2O_SRC_DIR}/build")  # for generated g2o/config.h

message(STATUS "Python3: ${Python3_EXECUTABLE}")
message(STATUS "pybind11: ${PYBIND11_CMAKE_DIR}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_VERSION_STRING}")

# Source list — grows per phase (C0 skeleton .. C5 B&B)
set(FUSION_CORE_SOURCES
    src/bindings.cpp           # C0
    src/signature_bindings.cpp # C1
    src/memory_bindings.cpp    # C2
    src/fusion_graph.cpp       # C3
    src/graph_bindings.cpp     # C3
    src/map_ops_bindings.cpp   # C4
    src/scan_match_common.cpp  # V4.4-P0 shared GN refine / grid scoring
    src/bnb_matcher.cpp        # C5
    src/bnb_bindings.cpp       # C5
    src/vo_frontend.cpp        # V3.1
    src/vo_bindings.cpp        # V3.1
    src/scan_to_submap_2d.cpp  # V4.4-P2 scan_to_submap port
    src/scan_to_map_2d.cpp     # V4.4-P3 scan_to_map port
    src/lidar_frontend.cpp     # V4.4 native LiDAR front-end
    src/lidar_bindings.cpp     # V4.4
)

pybind11_add_module(fusion_core ${FUSION_CORE_SOURCES})

target_include_directories(fusion_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${G2O_SRC_DIR}
    ${G2O_BUILD_DIR}
)

# Static g2o linked from C0 so the multi-module symbol-coexistence risk
# (fusion_core + slam_optimizer_core + g2o pybind in one process) is retired
# immediately; -fvisibility=hidden keeps symbols module-private.
# The vendored g2o static libs were built with OpenGL draw actions enabled, so
# types_slam2d pulls GL symbols — link the helper + system GL to satisfy them.
find_package(OpenGL REQUIRED)
target_link_libraries(fusion_core PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    "${G2O_LIB_DIR}/libg2o_solver_eigen.a"
    "${G2O_LIB_DIR}/libg2o_types_slam2d.a"
    "${G2O_LIB_DIR}/libg2o_core.a"
    "${G2O_LIB_DIR}/libg2o_stuff.a"
    "${G2O_LIB_DIR}/libg2o_opengl_helper.a"
    OpenGL::GL
    OpenGL::GLU
)

target_compile_options(fusion_core PRIVATE
    -O3 -march=native -fvisibility=hidden
)

install(TARGETS fusion_core
    DESTINATION ${Python3_SITELIB}
)
