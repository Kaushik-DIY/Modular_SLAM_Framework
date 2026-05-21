cmake_minimum_required(VERSION 3.16)
project(sim3solver_standalone LANGUAGES CXX)

# Standalone CMake build for pyslam's C++ sim3solver pybind11 module.
# Source files are symlinked from third_party/pyslam_reference/cpp/ and are
# NOT modified. We only add a thin standalone build script here.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Match pyslam's optimization flags.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ---- Dependencies ---------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core features2d)
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Eigen3 include:  ${EIGEN3_INCLUDE_DIR}")
message(STATUS "OpenCV version:  ${OpenCV_VERSION}")
message(STATUS "OpenCV libs:     ${OpenCV_LIBS}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers
    ${CMAKE_CURRENT_SOURCE_DIR}/casters
)

# ---- Static lib with the solver implementation ----------------------------
add_library(sim3solver_core STATIC
    solvers/Sim3Solver.cpp
    solvers/Sim3PointRegistrationSolver.cpp
    solvers/Random.cpp
)
target_link_libraries(sim3solver_core PRIVATE ${OpenCV_LIBS})

# ---- pybind11 module ------------------------------------------------------
pybind11_add_module(sim3solver solvers/sim3solver_module.cpp)
target_link_libraries(sim3solver PRIVATE sim3solver_core ${OpenCV_LIBS})
