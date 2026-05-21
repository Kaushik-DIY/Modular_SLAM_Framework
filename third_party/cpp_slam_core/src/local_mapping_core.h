#pragma once

#include "keyframe.h"
#include "map_point.h"

#include <pybind11/pybind11.h>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
namespace slam {

// Mirrors Python SensorType enum values.
enum class SensorTypeEnum : int { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

class LocalMappingCore {
  public:
    // ---- Public state (mirrors Python LocalMappingCore) ----------------------
    py::object map;
    py::object kf_cur;
    int        sensor_type;    // SensorTypeEnum value
    int        kid_last_BA{-1};

    // Recently added map points (keyed by Python object identity)
    std::unordered_set<py::object, PyObjHash, PyObjEqual> recently_added;

    // Abort flags (py::object for Python compatibility)
    py::object opt_abort_flag;
    py::object mp_opt_abort_flag;

    // ---- Construction --------------------------------------------------------
    explicit LocalMappingCore(py::object map_obj, int stype);

    // ---- Lifecycle -----------------------------------------------------------
    void reset();
    void add_points(py::object pts);     // accepts list or iterable
    void remove_points(py::object pts);
    void set_opt_abort_flag(bool value);

    // ---- Core LM methods (same names as Python LocalMappingCore) -------------
    void         process_new_keyframe();
    int          cull_map_points();
    int          cull_keyframes();
    // pm_cls: Python ProjectionMatcher class (pass None to auto-import)
    int          fuse_map_points(float desc_dist_sigma,
                                 py::object pm_cls = py::none());
    py::tuple    local_BA();
    py::tuple    large_window_BA();

  private:
    // Return the first kid that observed point p (for cull heuristic).
    int _point_first_kid(py::object p, int fallback_kid) const;

    // Collect neighbor KFs for fuse_map_points.
    std::vector<py::object> _get_neighbor_keyframes(int num_neighbors) const;
};

} // namespace slam
