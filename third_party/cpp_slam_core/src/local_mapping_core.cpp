#include "local_mapping_core.h"

#include <algorithm>

namespace slam {

// ---- Construction ----------------------------------------------------------
LocalMappingCore::LocalMappingCore(py::object map_obj, int stype)
    : map(map_obj),
      kf_cur(py::none()),
      sensor_type(stype) {
    // Create abort flags matching Python's _make_flag() pattern.
    // We store them as simple namespaces with a .value attribute.
    try {
        py::object g2o = py::module_::import("g2o");
        opt_abort_flag = g2o.attr("Flag")(false);
    } catch (...) {
        py::object ns = py::module_::import("types").attr("SimpleNamespace");
        opt_abort_flag = ns(py::arg("value") = false);
    }
    py::object ns = py::module_::import("types").attr("SimpleNamespace");
    mp_opt_abort_flag = ns(py::arg("value") = false);
}

// ---- Lifecycle -------------------------------------------------------------
void LocalMappingCore::reset() {
    recently_added.clear();
}

void LocalMappingCore::add_points(py::object pts) {
    for (auto item : pts) {
        recently_added.insert(py::reinterpret_borrow<py::object>(item));
    }
}

void LocalMappingCore::remove_points(py::object pts) {
    for (auto item : pts) {
        recently_added.erase(py::reinterpret_borrow<py::object>(item));
    }
}

void LocalMappingCore::set_opt_abort_flag(bool value) {
    try { opt_abort_flag.attr("value") = value; } catch (...) {}
    try { mp_opt_abort_flag.attr("value") = value; } catch (...) {}
}

// ---- process_new_keyframe ---------------------------------------------------
void LocalMappingCore::process_new_keyframe() {
    // Fast path: if kf_cur is a C++ KeyFrame, use typed access.
    if (py::isinstance<KeyFrame>(kf_cur)) {
        KeyFrame &kf = kf_cur.cast<KeyFrame &>();
        auto pts_idxs = kf.get_matched_good_points_and_idxs();
        for (auto &[p, idx] : pts_idxs) {
            bool added = p.attr("add_observation")(kf_cur, idx).cast<bool>();
            if (added) {
                p.attr("update_info")();
            } else {
                recently_added.insert(p);
            }
        }
        kf.update_connections();
    } else {
        // Python KeyFrame fallback: call via attribute dispatch.
        auto pts_idxs = kf_cur.attr("get_matched_good_points_and_idxs")();
        for (auto item : pts_idxs) {
            py::tuple tup = item.cast<py::tuple>();
            py::object p  = tup[0];
            int        idx = tup[1].cast<int>();
            bool added = p.attr("add_observation")(kf_cur, idx).cast<bool>();
            if (added) {
                p.attr("update_info")();
            } else {
                recently_added.insert(p);
            }
        }
        kf_cur.attr("update_connections")();
    }
}

// ---- _point_first_kid -------------------------------------------------------
int LocalMappingCore::_point_first_kid(py::object p, int fallback_kid) const {
    // Check for pre-computed first_kid attribute.
    if (py::hasattr(p, "first_kid")) {
        try { return p.attr("first_kid").cast<int>(); } catch (...) {}
    }
    // Compute from observations.
    try {
        auto obs = p.attr("observations")();
        int min_kid = fallback_kid;
        bool found_any = false;
        for (auto item : obs) {
            py::tuple tup = item.cast<py::tuple>();
            py::object kf  = tup[0];
            int kid = kf.attr("kid").cast<int>();
            if (!found_any || kid < min_kid) { min_kid = kid; found_any = true; }
        }
        return min_kid;
    } catch (...) {}
    return fallback_kid;
}

// ---- cull_map_points --------------------------------------------------------
int LocalMappingCore::cull_map_points() {
    int th_obs = (sensor_type != static_cast<int>(SensorTypeEnum::MONOCULAR)) ? 3 : 2;
    constexpr float kMinFoundRatio = 0.25f;
    int current_kid = kf_cur.attr("kid").cast<int>();

    std::vector<py::object> to_keep;
    int n_removed = 0;

    for (const auto &p : recently_added) {
        try {
            if (p.attr("is_bad")().cast<bool>()) {
                ++n_removed;
                continue;
            }
            float fr = p.attr("get_found_ratio")().cast<float>();
            if (fr < kMinFoundRatio) {
                p.attr("set_bad")();
                map.attr("remove_point")(p);
                ++n_removed;
                continue;
            }
            int first_kid = _point_first_kid(p, current_kid);
            int n_obs = p.attr("num_observations")().cast<int>();
            if (current_kid - first_kid >= 2 && n_obs <= th_obs) {
                p.attr("set_bad")();
                map.attr("remove_point")(p);
                ++n_removed;
                continue;
            }
            if (current_kid - first_kid >= 3) {
                ++n_removed;   // graduate out of recently_added without deleting
                continue;
            }
            to_keep.push_back(p);
        } catch (...) {
            // Skip problematic points.
        }
    }

    recently_added.clear();
    for (auto &p : to_keep) recently_added.insert(p);
    return n_removed;
}

// ---- _get_neighbor_keyframes ------------------------------------------------
std::vector<py::object> LocalMappingCore::_get_neighbor_keyframes(int num_neighbors) const {
    std::vector<py::object> result;
    try {
        // Prefer local_map.get_best_neighbors if available.
        if (py::hasattr(map, "local_map")) {
            py::object lm = map.attr("local_map");
            if (py::hasattr(lm, "get_best_neighbors")) {
                auto raw = lm.attr("get_best_neighbors")(kf_cur, py::arg("N") = num_neighbors);
                for (auto item : raw) result.push_back(py::reinterpret_borrow<py::object>(item));
            } else {
                auto raw = kf_cur.attr("get_best_covisible_keyframes")(num_neighbors);
                for (auto item : raw) result.push_back(py::reinterpret_borrow<py::object>(item));
            }
        } else {
            auto raw = kf_cur.attr("get_best_covisible_keyframes")(num_neighbors);
            for (auto item : raw) result.push_back(py::reinterpret_borrow<py::object>(item));
        }
    } catch (...) {}

    // Filter: exclude kf_cur and bad KFs.
    std::vector<py::object> filtered;
    for (auto &kf : result) {
        try {
            if (kf.is_none()) continue;
            bool same = (kf.ptr() == kf_cur.ptr());
            bool bad  = kf.attr("is_bad")().cast<bool>();
            if (!same && !bad) filtered.push_back(kf);
        } catch (...) {}
    }
    return filtered;
}

// ---- fuse_map_points --------------------------------------------------------
int LocalMappingCore::fuse_map_points(float desc_dist_sigma, py::object pm_cls) {
    // Import ProjectionMatcher if not provided.
    if (pm_cls.is_none()) {
        try {
            pm_cls = py::module_::import(
                "visual_slam.orbslam.slam.geometry_matchers").attr("ProjectionMatcher");
        } catch (...) { return 0; }
    }

    // Read parameters from Python.
    float max_reproj_dist = 3.0f, ratio_test = 0.9f;
    try {
        py::object params = py::module_::import(
            "visual_slam.orbslam.slam.config_parameters").attr("Parameters");
        max_reproj_dist = params.attr("kMaxReprojectionDistanceFuse").cast<float>();
        ratio_test      = params.attr("kMatchRatioTestMap").cast<float>();
    } catch (...) {}

    // Determine neighbor count from Parameters.
    int num_neighbors = 10;
    try {
        py::object params = py::module_::import(
            "visual_slam.orbslam.slam.config_parameters").attr("Parameters");
        if (sensor_type == static_cast<int>(SensorTypeEnum::MONOCULAR)) {
            num_neighbors = params.attr("kLocalMappingNumNeighborKeyFramesMonocular").cast<int>();
        } else {
            num_neighbors = params.attr("kLocalMappingNumNeighborKeyFramesStereo").cast<int>();
        }
    } catch (...) {}

    std::vector<py::object> local_kfs = _get_neighbor_keyframes(num_neighbors);
    int total = 0;

    // 1. Fuse current KF points into each neighbor.
    auto cur_pts = kf_cur.attr("get_matched_good_points")();
    for (const auto &kf : local_kfs) {
        try {
            int n = pm_cls.attr("search_and_fuse")(
                cur_pts, kf,
                py::arg("max_reproj_distance") = max_reproj_dist,
                py::arg("max_descriptor_distance") = desc_dist_sigma,
                py::arg("ratio_test") = ratio_test
            ).cast<int>();
            total += n;
        } catch (...) {}
    }

    // 2. Collect neighbor points not already in kf_cur, then fuse into kf_cur.
    std::unordered_set<py::object, PyObjHash, PyObjEqual> seen;
    std::vector<py::object> fuse_candidates;

    for (const auto &kf : local_kfs) {
        try {
            auto pts = kf.attr("get_matched_good_points")();
            for (auto item : pts) {
                py::object p = py::reinterpret_borrow<py::object>(item);
                if (p.is_none()) continue;
                if (p.attr("is_bad")().cast<bool>()) continue;
                if (seen.count(p)) continue;
                if (p.attr("is_in_keyframe")(kf_cur).cast<bool>()) continue;
                seen.insert(p);
                fuse_candidates.push_back(p);
            }
        } catch (...) {}
    }

    try {
        py::list cand_list;
        for (auto &p : fuse_candidates) cand_list.append(p);
        int n = pm_cls.attr("search_and_fuse")(
            cand_list, kf_cur,
            py::arg("max_reproj_distance") = max_reproj_dist,
            py::arg("max_descriptor_distance") = desc_dist_sigma,
            py::arg("ratio_test") = ratio_test
        ).cast<int>();
        total += n;
    } catch (...) {}

    // 3. Update info for all current KF points after fusion.
    try {
        auto pts = kf_cur.attr("get_matched_good_points")();
        for (auto item : pts) {
            py::object p = py::reinterpret_borrow<py::object>(item);
            if (!p.is_none() && !p.attr("is_bad")().cast<bool>()) {
                p.attr("update_info")();
            }
        }
    } catch (...) {}

    kf_cur.attr("update_connections")();
    return total;
}

// ---- cull_keyframes ---------------------------------------------------------
int LocalMappingCore::cull_keyframes() {
    int th_obs = 3;
    float cull_ratio = 0.9f;
    int   min_pts = 0;
    float max_time_dist = 1e9f;

    try {
        py::object params = py::module_::import(
            "visual_slam.orbslam.slam.config_parameters").attr("Parameters");
        cull_ratio    = params.attr("kKeyframeCullingRedundantObsRatio").cast<float>();
        min_pts       = params.attr("kKeyframeCullingMinNumPoints").cast<int>();
        max_time_dist = params.attr("kKeyframeMaxTimeDistanceInSecForCulling").cast<float>();
    } catch (...) {}

    auto covisible = kf_cur.attr("get_covisible_keyframes")();
    int n_culled = 0;

    for (auto item : covisible) {
        py::object kf = py::reinterpret_borrow<py::object>(item);
        try {
            if (kf.attr("kid").cast<int>() == 0) continue;
            if (kf.attr("is_bad")().cast<bool>()) continue;

            // Get valid points for this KF.
            py::object kf_points = kf.attr("get_points")();
            // depths array for depth thresholding
            py::object depths_arr = py::none();
            float depth_threshold = 1e9f;
            if (sensor_type != static_cast<int>(SensorTypeEnum::MONOCULAR)) {
                try {
                    depths_arr = kf.attr("depths");
                    depth_threshold = kf.attr("camera").attr("depth_threshold").cast<float>();
                } catch (...) {}
            }

            int kf_num_pts = 0, kf_num_redundant = 0;
            int n_kps = static_cast<int>(py::len(kf_points));

            // Also get octaves for this KF.
            py::object kf_octaves = py::none();
            try { kf_octaves = kf.attr("octaves"); } catch (...) {}

            for (int i = 0; i < n_kps; i++) {
                py::object p = kf_points.attr("__getitem__")(i);
                if (p.is_none()) continue;
                if (p.attr("is_bad")().cast<bool>()) continue;

                // Depth check for RGBD/stereo.
                if (!depths_arr.is_none()) {
                    try {
                        float d = depths_arr.attr("__getitem__")(i).cast<float>();
                        if (d > depth_threshold || d < 0.0f) continue;
                    } catch (...) {}
                }

                kf_num_pts++;

                int n_obs = p.attr("num_observations")().cast<int>();
                if (n_obs <= th_obs) continue;

                // Get scale level of this observation.
                int scale_i = 0;
                if (!kf_octaves.is_none()) {
                    try { scale_i = kf_octaves.attr("__getitem__")(i).cast<int>(); } catch (...) {}
                }

                // Count other KFs observing this point at similar or finer scale.
                int p_num_obs = 0;
                auto obs = p.attr("observations")();
                for (auto obs_item : obs) {
                    py::tuple obs_tup = obs_item.cast<py::tuple>();
                    py::object kf_j = obs_tup[0];
                    int idx_j       = obs_tup[1].cast<int>();
                    if (kf_j.ptr() == kf.ptr()) continue;
                    if (kf_j.attr("is_bad")().cast<bool>()) continue;

                    int scale_j = 0;
                    try {
                        py::object oct_j = kf_j.attr("octaves");
                        scale_j = oct_j.attr("__getitem__")(idx_j).cast<int>();
                    } catch (...) {}

                    if (scale_j <= scale_i + 1) {
                        p_num_obs++;
                        if (p_num_obs >= th_obs) break;
                    }
                }
                if (p_num_obs >= th_obs) kf_num_redundant++;
            }

            bool remove = (kf_num_redundant >
                           cull_ratio * std::max(kf_num_pts, 1)) &&
                          (kf_num_pts > min_pts);

            if (remove) {
                // Don't remove if parent is too close in time.
                try {
                    py::object parent = kf.attr("get_parent")();
                    if (!parent.is_none()) {
                        float dt = std::abs(kf.attr("timestamp").cast<double>() -
                                            parent.attr("timestamp").cast<double>());
                        if (dt < max_time_dist) remove = false;
                    }
                } catch (...) {}
            }

            if (remove) {
                kf.attr("set_bad")();
                n_culled++;
            }
        } catch (...) {}
    }
    return n_culled;
}

// ---- local_BA ---------------------------------------------------------------
py::tuple LocalMappingCore::local_BA() {
    float err = map.attr("locally_optimize")(
        kf_cur,
        py::arg("abort_flag")    = opt_abort_flag,
        py::arg("mp_abort_flag") = mp_opt_abort_flag
    ).cast<float>();

    int tracked = 0;
    if (py::isinstance<KeyFrame>(kf_cur)) {
        tracked = kf_cur.cast<KeyFrame &>().num_tracked_points(3);
    } else {
        try { tracked = kf_cur.attr("num_tracked_points")(3).cast<int>(); } catch (...) {}
    }
    return py::make_tuple(err, tracked);
}

// ---- large_window_BA --------------------------------------------------------
py::tuple LocalMappingCore::large_window_BA() {
    kid_last_BA = kf_cur.attr("kid").cast<int>();

    // Read window size from Parameters.
    int window_size = 10;
    try {
        py::object params = py::module_::import(
            "visual_slam.orbslam.slam.config_parameters").attr("Parameters");
        window_size = params.attr("kLargeBAWindowSize").cast<int>();
    } catch (...) {}

    py::tuple result = map.attr("optimize")(
        py::arg("local_window_size") = window_size,
        py::arg("abort_flag")        = opt_abort_flag
    ).cast<py::tuple>();
    return result;
}

} // namespace slam
