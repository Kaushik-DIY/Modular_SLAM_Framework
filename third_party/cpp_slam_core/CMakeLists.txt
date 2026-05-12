cmake_minimum_required(VERSION 3.16)
project(cpp_slam_core CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python + pybind11 from venv
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# pybind11 from venv
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
find_package(TBB REQUIRED)

# g2o headers (for LocalMappingCore → slam_optimizer_core dispatch later)
set(G2O_ROOT ${CMAKE_SOURCE_DIR}/../g2opy)
set(G2O_INCLUDE_DIR ${G2O_ROOT})

message(STATUS "Python3: ${Python3_EXECUTABLE}")
message(STATUS "pybind11: ${PYBIND11_CMAKE_DIR}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_VERSION_STRING}")
message(STATUS "TBB: ${TBB_VERSION}")
message(STATUS "NumPy include: ${Python3_NumPy_INCLUDE_DIRS}")

# Source list — grows per phase
set(CPP_SLAM_CORE_SOURCES
    src/bindings.cpp
    src/map_point.cpp          # Phase 1
    src/frame.cpp              # Phase 2
    src/keyframe.cpp           # Phase 3
    src/local_mapping_core.cpp # Phase 4
)

pybind11_add_module(cpp_slam_core ${CPP_SLAM_CORE_SOURCES})

target_include_directories(cpp_slam_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
)

target_link_libraries(cpp_slam_core PRIVATE
    ${OpenCV_LIBS}
    TBB::tbb
    Eigen3::Eigen
)

target_compile_options(cpp_slam_core PRIVATE
    -O3 -march=native -fvisibility=hidden
)

install(TARGETS cpp_slam_core
    DESTINATION ${Python3_SITELIB}
)
