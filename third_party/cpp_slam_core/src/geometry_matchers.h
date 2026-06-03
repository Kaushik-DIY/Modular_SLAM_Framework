#pragma once
// C++ port of the projection matchers (the visual-SLAM hot path). Mirrors the
// semantics of visual_slam/orbslam/slam/geometry_matchers.py
// (_search_map_by_projection / _search_frame_by_projection) exactly, but runs on
// C++ Frame/MapPoint objects with the native kd-tree + projection (no Python in
// the inner loop). Parameters and scale_factors are passed in from the Python
// dispatcher so this stays free of the Python Parameters object.
//
// Not derived from pySLAM's GPL geometry_matchers.cpp — ported from OUR Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <utility>
#include <vector>

namespace py = pybind11;

namespace cppcore {

// Tunables forwarded from Python (Parameters.*), so the C++ matcher reproduces
// the Python thresholds exactly.
struct MatchParams {
    float max_reproj_distance;
    float max_descriptor_distance;
    float ratio_test;
    float viewing_cos_limit;
    float min_depth;
    float far_points_threshold;  // +inf if disabled
    double log_scale_factor;     // = log(scale_factor); matches Python predict_scale
    int num_levels;              // ORB pyramid levels
};

// Port of _search_map_by_projection. `points` = local map points (py::object,
// each a cpp_slam_core.MapPoint). `frame_obj` = the current Frame (py::object so
// add_frame_view receives it). `scale_factors` = per-octave scale factors.
// Returns (found_count, matched_current_feature_indices), and mutates state
// (increase_visible/found + add_frame_view) exactly like the Python version.
std::pair<int, std::vector<int>> search_map_by_projection(
    const py::list &points, py::object frame_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast> scale_factors,
    const MatchParams &p);

// Port of _search_frame_by_projection. `ref_points` = the reference frame's
// matched map points (pre-filtered in Python: non-outlier, not already-matched);
// `ref_idxs`/`ref_octaves` are the parallel actual f_ref feature indices and
// octaves. `f_cur_obj` = the current Frame. Each ref point is projected into
// f_cur and matched to the best descriptor among kd neighbours within +/-1 of
// the REF octave (simple argmin + threshold — NO ratio test), with an optional
// stereo right-u consistency check. Returns the matched (idxs_ref, idxs_cur)
// pairs that passed the descriptor threshold, in reference iteration order,
// BEFORE add_frame_view and the rotation-histogram filter — both of which the
// Python dispatcher applies, so add_frame_view's bool-gate and the rotation
// filter stay bit-identical to the Python path. Unlike search_map this does NOT
// call increase_visible(), has no last_frame_id_seen gate, and uses the REF
// octave (not a distance-predicted level) for the radius + window.
std::pair<std::vector<int>, std::vector<int>> search_frame_by_projection(
    const py::list &ref_points, const std::vector<int> &ref_idxs,
    const std::vector<int> &ref_octaves, py::object f_cur_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast> scale_factors,
    float max_reproj_distance, float max_descriptor_distance,
    float viewing_cos_limit, float min_depth, bool do_stereo_check);

}  // namespace cppcore
