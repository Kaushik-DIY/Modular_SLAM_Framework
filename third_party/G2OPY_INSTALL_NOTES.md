# g2opy installation notes for Modular SLAM framework

Workspace:
- ~/slam_ws

Python environment:
- ~/slam_ws/.venv
- Python 3.11.0rc1 observed during build

Important dependency constraint:
- NumPy must be 1.26.4 for this patched g2opy build.
- NumPy 2.x caused segmentation faults in Eigen/Numpy conversions.

Source:
- uoip/g2opy cloned into ~/slam_ws/third_party/g2opy

Manual patches applied:
1. Replaced bundled old pybind11 with newer pybind11.
2. Updated python/CMakeLists.txt to use add_subdirectory() for pybind11.
3. Patched python/core/eigen_types.h quaternion bindings:
   - vec()
   - x()
   - y()
   - z()
   - w()
4. Added missing static symbol definitions in g2o/core/hyper_graph.cpp:
   - const int HyperGraph::UnassignedId;
   - const int HyperGraph::InvalidId;

Build/install method:
- Built using CMake and make.
- pip install . failed because the old repository uses a flat-layout incompatible with modern setuptools.
- Installed manually by copying:
  third_party/g2opy/lib/g2o.cpython-311-x86_64-linux-gnu.so
  into:
  .venv/lib/python3.11/site-packages/

Validation passed:
- import g2o
- SparseOptimizer / VertexSE3 / EdgeSE3 creation
- SE3Quat construction with NumPy vectors
- SE3Expmap pose graph optimization
- Isometry3d / VertexSE3 / EdgeSE3 runtime construction
