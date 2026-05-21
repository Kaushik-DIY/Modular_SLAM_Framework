#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "map_point.h"
#include "frame.h"
#include "keyframe.h"
#include "local_mapping_core.h"

namespace py = pybind11;
using namespace slam;

// ---------------------------------------------------------------------------
// Helper: numpy (N,3) double → Eigen::Vector3d
// ---------------------------------------------------------------------------
static Eigen::Vector3d np_to_vec3(py::array_t<double, py::array::c_style> arr) {
    auto r = arr.unchecked<1>();
    return Eigen::Vector3d(r(0), r(1), r(2));
}

// ---------------------------------------------------------------------------
// MapPoint binding
// ---------------------------------------------------------------------------
static void bind_map_point(py::module_ &m) {
    py::class_<MapPoint, MapPointPtr>(m, "MapPoint", py::dynamic_attr())
        // ---- Construction --------------------------------------------------
        .def(py::init([](py::array_t<double> pos, py::object kf, int idx,
                         py::object map_obj, int given_id) {
            auto mp = std::make_shared<MapPoint>(given_id);
            if (pos.size() == 3) {
                auto r = pos.unchecked<1>();
                mp->update_position(Eigen::Vector3d(r(0), r(1), r(2)));
            }
            mp->map = map_obj;
            mp->kf_ref = kf;
            mp->rgb = py::none();
            mp->replacement = py::none();
            if (!kf.is_none() && idx >= 0) {
                mp->add_observation(kf, idx);
            }
            return mp;
        }),
            py::arg("pos"),
            py::arg("kf") = py::none(),
            py::arg("idx") = -1,
            py::arg("map") = py::none(),
            py::arg("id") = -1)

        // ---- Identity ------------------------------------------------------
        .def_readwrite("id",    &MapPoint::id)
        .def_readwrite("map",   &MapPoint::map)

        // ---- Position ------------------------------------------------------
        .def("get_position", [](const MapPoint &mp) {
            py::gil_scoped_release rel;
            Eigen::Vector3d p = mp.get_position();
            py::gil_scoped_acquire acq;
            return py::array_t<double>({3}, {sizeof(double)}, p.data());
        })
        .def("update_position", [](MapPoint &mp, py::array_t<double> pos) {
            auto r = pos.unchecked<1>();
            py::gil_scoped_release rel;
            mp.update_position(Eigen::Vector3d(r(0), r(1), r(2)));
        })
        // Alias used by slam_optimizer_bridge
        .def("set_position", [](MapPoint &mp, py::array_t<double> pos) {
            auto r = pos.unchecked<1>();
            py::gil_scoped_release rel;
            mp.update_position(Eigen::Vector3d(r(0), r(1), r(2)));
        })
        .def_property_readonly("min_distance", &MapPoint::min_distance)
        .def_property_readonly("max_distance", &MapPoint::max_distance)
        .def_property("normal",
            [](const MapPoint &mp) -> py::array_t<double> {
                return py::array_t<double>({3}, {sizeof(double)}, mp.normal.data());
            },
            [](MapPoint &mp, py::array_t<double> arr) {
                auto r = arr.unchecked<1>();
                mp.normal = Eigen::Vector3d(r(0), r(1), r(2));
            })

        // ---- Descriptor ----------------------------------------------------
        .def("get_descriptor", [](const MapPoint &mp) -> py::object {
            cv::Mat d = mp.get_descriptor();
            if (d.empty()) return py::none();
            auto arr = py::array_t<uint8_t>({32});  // 1-D (32,)
            std::memcpy(arr.mutable_data(), d.data, 32);
            return arr;
        })
        .def("min_des_distance", [](const MapPoint &mp, py::array_t<uint8_t> q) {
            cv::Mat qm(1, 32, CV_8U, q.mutable_data());
            py::gil_scoped_release rel;
            return mp.min_des_distance(qm);
        })

        // ---- Observations --------------------------------------------------
        .def("add_observation",      &MapPoint::add_observation)
        .def("remove_observation",   &MapPoint::remove_observation,
             py::arg("kf"), py::arg("idx") = -1, py::arg("map_no_lock") = false)
        .def("observations",         &MapPoint::observations)
        .def("keyframes",            &MapPoint::keyframes)
        .def("is_in_keyframe",       &MapPoint::is_in_keyframe)
        .def("get_observation_idx",  &MapPoint::get_observation_idx)
        .def_property_readonly("n_obs", &MapPoint::num_observations)
        .def("num_observations",       &MapPoint::num_observations)

        // ---- Frame views ---------------------------------------------------
        .def("add_frame_view",       &MapPoint::add_frame_view)
        .def("remove_frame_view",    &MapPoint::remove_frame_view,
             py::arg("frame"), py::arg("idx") = -1)
        .def("is_in_frame",          &MapPoint::is_in_frame)
        .def("get_frame_view_idx",   &MapPoint::get_frame_view_idx)
        .def("frame_views",          &MapPoint::frame_views)
        .def("frames",               &MapPoint::frames)

        // ---- Combined status helpers ---------------------------------------
        .def("is_bad_or_is_in_keyframe",  &MapPoint::is_bad_or_is_in_keyframe)
        .def("is_good_with_min_obs",      &MapPoint::is_good_with_min_obs)

        // ---- Descriptor write ----------------------------------------------
        .def("set_des",  &MapPoint::set_des)

        // ---- position property (alias for get/update_position) -------------
        .def_property("position",
            [](const MapPoint &mp) -> py::array_t<double> {
                Eigen::Vector3d p = mp.get_position();
                return py::array_t<double>({3}, {sizeof(double)}, p.data());
            },
            [](MapPoint &mp, py::array_t<double> arr) {
                auto r = arr.unchecked<1>();
                mp.update_position(Eigen::Vector3d(r(0), r(1), r(2)));
            })
        .def_property("position_world",
            [](const MapPoint &mp) -> py::array_t<double> {
                Eigen::Vector3d p = mp.get_position();
                return py::array_t<double>({3}, {sizeof(double)}, p.data());
            },
            [](MapPoint &mp, py::array_t<double> arr) {
                auto r = arr.unchecked<1>();
                mp.update_position(Eigen::Vector3d(r(0), r(1), r(2)));
            })

        // ---- Status --------------------------------------------------------
        .def_property_readonly("_is_bad", &MapPoint::is_bad)
        .def("is_bad",               &MapPoint::is_bad)
        .def("set_bad",              &MapPoint::set_bad, py::arg("map_no_lock") = false)
        .def_property("to_be_erased",
            [](const MapPoint &mp) { return mp.to_be_erased.load(); },
            [](MapPoint &mp, bool v) { mp.to_be_erased.store(v); })
        .def("replace_with",         &MapPoint::replace_with)
        .def("get_replacement",      &MapPoint::get_replacement)

        // ---- Info update ---------------------------------------------------
        .def("update_info",                 &MapPoint::update_info)
        .def("update_normal_and_depth",     &MapPoint::update_normal_and_depth,
             py::arg("force") = false)
        .def("update_best_descriptor",      &MapPoint::update_best_descriptor,
             py::arg("force") = false)

        // ---- Other references ----------------------------------------------
        .def_readwrite("kf_ref",           &MapPoint::kf_ref)
        .def_readwrite("replacement",      &MapPoint::replacement)
        .def_readwrite("rgb",              &MapPoint::rgb)
        .def_readwrite("lba_count",        &MapPoint::lba_count)
        .def_readwrite("num_times_visible",&MapPoint::num_times_visible)
        .def_readwrite("num_times_found",  &MapPoint::num_times_found)
        .def_readwrite("last_frame_id_seen",&MapPoint::last_frame_id_seen)
        .def_readwrite("corrected_by_kf",  &MapPoint::corrected_by_kf)
        .def_readwrite("corrected_reference", &MapPoint::corrected_reference)
        .def_readwrite("pt_GBA",           &MapPoint::pt_GBA)
        .def_readwrite("is_pt_GBA_valid",  &MapPoint::is_pt_GBA_valid)
        .def_readwrite("GBA_kf_id",        &MapPoint::GBA_kf_id)

        // ---- Statistics ----------------------------------------------------
        .def("increase_visible",   &MapPoint::increase_visible, py::arg("n") = 1)
        .def("increase_found",     &MapPoint::increase_found,   py::arg("n") = 1)
        .def("get_found_ratio",    [](const MapPoint &mp) -> double {
            // Return double to avoid float32 precision issues in Python tests
            if (mp.num_times_visible <= 0) return 0.0;
            return double(mp.num_times_found) / double(mp.num_times_visible);
        })
        .def("predict_detection_level", &MapPoint::predict_detection_level)

        // ---- Repr / hash ---------------------------------------------------
        .def("__repr__",  &MapPoint::__repr__)
        .def("__str__",   &MapPoint::__repr__)
        .def("__hash__",  [](const MapPoint &mp) { return mp.id; })
        .def("__eq__",    [](const MapPoint &a, const MapPoint &b) {
            return a.id == b.id;
        })
        .def("__lt__",    [](const MapPoint &a, const MapPoint &b) {
            return a.id < b.id;
        })

        // ---- Static helpers ------------------------------------------------
        .def_static("next_id", &MapPoint::next_id)
        .def_static("set_id",  &MapPoint::set_id)
        ;
}

// ---------------------------------------------------------------------------
// Frame binding
// ---------------------------------------------------------------------------
static void bind_frame(py::module_ &m) {
    using namespace slam;
    py::class_<Frame, FramePtr>(m, "Frame", py::dynamic_attr())
        // ---- Construction --------------------------------------------------
        .def(py::init([](py::object camera, int given_id) {
            auto f = std::make_shared<Frame>(given_id);
            f->camera = camera;
            return f;
        }), py::arg("camera") = py::none(), py::arg("id") = -1)

        // ---- Identity ------------------------------------------------------
        .def_readwrite("id",        &Frame::id)
        .def_readwrite("img_id",    &Frame::img_id)
        .def_readwrite("timestamp", &Frame::timestamp)
        .def_readwrite("camera",    &Frame::camera)

        // ---- Feature array initialization ----------------------------------
        .def("init_feature_arrays", &Frame::init_feature_arrays,
             py::arg("kps"), py::arg("des"), py::arg("kps_ur"),
             py::arg("octaves"), py::arg("n_features"))

        // ---- Feature arrays as numpy views ---------------------------------
        .def_property("kpsu",
            [](const Frame &f) -> py::array_t<float> {
                int n = f.kpsu.rows();
                if (n == 0) {
                    std::vector<ssize_t> shape{0, 2};
                    return py::array_t<float>(shape);
                }
                std::vector<ssize_t> shape{n, 2};
                std::vector<ssize_t> strides{2 * (ssize_t)sizeof(float), (ssize_t)sizeof(float)};
                return py::array_t<float>(shape, strides, f.kpsu.data());
            },
            [](Frame &f, py::array_t<float> arr) {
                auto r = arr.unchecked<2>();
                int n = static_cast<int>(r.shape(0));
                f.kpsu.resize(n, 2);
                for (int i = 0; i < n; ++i) {
                    f.kpsu(i, 0) = r(i, 0);
                    f.kpsu(i, 1) = r(i, 1);
                }
            })
        // Alias for compatibility with code using kpsu directly as array
        .def_property_readonly("kps_ur",
            [](const Frame &f) -> py::array_t<float> {
                int n = static_cast<int>(f.kps_ur.size());
                if (n == 0) return py::array_t<float>({0});
                return py::array_t<float>({n}, {sizeof(float)}, f.kps_ur.data());
            })
        .def_property_readonly("uRs",
            [](const Frame &f) -> py::array_t<float> {
                int n = static_cast<int>(f.kps_ur.size());
                if (n == 0) return py::array_t<float>({0});
                return py::array_t<float>({n}, {sizeof(float)}, f.kps_ur.data());
            })
        .def_property_readonly("octaves",
            [](const Frame &f) -> py::array_t<int32_t> {
                int n = static_cast<int>(f.octaves.size());
                if (n == 0) return py::array_t<int32_t>({0});
                return py::array_t<int32_t>({n}, {sizeof(int32_t)}, f.octaves.data());
            })
        .def_property("des",
            [](const Frame &f) -> py::object {
                if (f.des.empty()) return py::none();
                int rows = f.des.rows, cols = f.des.cols;
                auto arr = py::array_t<uint8_t>({rows, cols});
                std::memcpy(arr.mutable_data(), f.des.data, rows * cols);
                return arr;
            },
            [](Frame &f, py::array_t<uint8_t> arr) {
                int rows = static_cast<int>(arr.shape(0));
                int cols = static_cast<int>(arr.shape(1));
                f.des = cv::Mat(rows, cols, CV_8U);
                std::memcpy(f.des.data, arr.data(), rows * cols);
            })
        .def_property("kps",
            [](const Frame &f) { return f.kps; },
            [](Frame &f, py::list kps) { f.kps = kps; })

        // ---- Point associations (list of Python objects) -------------------
        .def_property("points",
            [](const Frame &f) {
                py::list lst;
                for (const auto &p : f.points) lst.append(p);
                return lst;
            },
            [](Frame &f, py::list lst) {
                f.points.clear();
                for (auto &item : lst) f.points.push_back(py::reinterpret_borrow<py::object>(item));
            })
        .def_property("outliers",
            [](const Frame &f) -> py::array_t<bool> {
                int n = static_cast<int>(f.outliers.size());
                auto arr = py::array_t<bool>({n});
                for (int i = 0; i < n; ++i) arr.mutable_data()[i] = f.outliers[i];
                return arr;
            },
            [](Frame &f, py::array_t<bool> arr) {
                int n = static_cast<int>(arr.size());
                f.outliers.resize(n);
                for (int i = 0; i < n; ++i) f.outliers[i] = arr.data()[i];
            })

        // ---- Pose ----------------------------------------------------------
        .def("Tcw", [](const Frame &f) {
            // Eigen is column-major; convert to row-major for numpy
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T = f.Tcw();
            return py::array_t<double>({4, 4}, {4*sizeof(double), sizeof(double)}, T.data());
        })
        .def("Twc", [](const Frame &f) {
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T = f.Twc();
            return py::array_t<double>({4, 4}, {4*sizeof(double), sizeof(double)}, T.data());
        })
        .def("Ow", [](const Frame &f) {
            Eigen::Vector3d ow = f.Ow();
            return py::array_t<double>({3}, {sizeof(double)}, ow.data());
        })
        .def("update_pose", [](Frame &f, py::object pose_or_mat) {
            // Accept Eigen::Matrix4d array or g2o.Isometry3d
            try {
                auto arr = pose_or_mat.cast<py::array_t<double>>();
                auto r = arr.unchecked<2>();
                Eigen::Matrix4d T;
                for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) T(i, j) = r(i, j);
                f.update_pose(T);
            } catch (...) {
                f.update_pose_from_g2o(pose_or_mat);
            }
        })

        // ---- Point match management ----------------------------------------
        .def("get_point_match", [](const Frame &f, int idx) -> py::object {
            if (idx < 0 || idx >= static_cast<int>(f.points.size()))
                return py::none();
            // Explicitly return a new reference so the return value is valid even
            // after pytest assertion cleanup decrements the bound-method reference.
            PyObject *raw = f.points[idx].ptr();
            Py_XINCREF(raw);
            return py::reinterpret_steal<py::object>(raw);
        })
        .def("set_point_match",     &Frame::set_point_match)
        .def("remove_point_match",  &Frame::remove_point_match)
        .def("remove_point",        &Frame::remove_point)
        .def("replace_point_match", &Frame::replace_point_match)
        .def("reset_points",        &Frame::reset_points)
        .def("num_kps",             &Frame::num_kps)

        // ---- Depth array (RGBD/stereo — numpy array or None) ---------------
        .def_readwrite("depths",      &Frame::depths)

        // ---- BoW -----------------------------------------------------------
        .def_readwrite("bow_vector",  &Frame::bow_vector)
        .def_readwrite("feat_vector", &Frame::feat_vector)
        .def_readwrite("g_des",       &Frame::g_des)
        .def_readwrite("f_des",       &Frame::f_des)

        // ---- Static --------------------------------------------------------
        .def_static("next_id", &Frame::next_id)
        .def_static("set_id",  &Frame::set_id)

        // ---- Repr ----------------------------------------------------------
        .def("__repr__", &Frame::__repr__)
        .def("__str__",  &Frame::__repr__)
        .def("__hash__", [](const Frame &f) { return f.id; })
        .def("__eq__",   [](const Frame &a, const Frame &b) { return a.id == b.id; })
        .def("__lt__",   [](const Frame &a, const Frame &b) { return a.id < b.id; })
        ;
}

// ---------------------------------------------------------------------------
// KeyFrame binding
// ---------------------------------------------------------------------------
static void bind_keyframe(py::module_ &m) {
    using namespace slam;
    py::class_<KeyFrame, KeyFramePtr, Frame>(m, "KeyFrame", py::dynamic_attr())
        // ---- Construction --------------------------------------------------
        .def(py::init([](int kid, int frame_id, py::object camera) {
            auto kf = std::make_shared<KeyFrame>(kid, frame_id);
            kf->camera = camera;
            return kf;
        }), py::arg("kid") = -1, py::arg("frame_id") = -1,
            py::arg("camera") = py::none())

        // ---- Identity ------------------------------------------------------
        .def_readwrite("kid",       &KeyFrame::kid)
        .def_readwrite("map",       &KeyFrame::map)

        // ---- Status --------------------------------------------------------
        .def_property_readonly("_is_bad", &KeyFrame::is_bad)
        .def("is_bad",      &KeyFrame::is_bad)
        .def("set_bad",     &KeyFrame::set_bad)
        .def_property("to_be_erased",
            [](const KeyFrame &kf) { return kf.to_be_erased_kf; },
            [](KeyFrame &kf, bool v) { kf.to_be_erased_kf = v; })
        .def_readwrite("not_to_erase",  &KeyFrame::not_to_erase)
        .def_readwrite("is_keyframe",   &KeyFrame::is_keyframe)
        .def("set_not_erase",   &KeyFrame::set_not_erase)
        .def("set_erase",       &KeyFrame::set_erase)

        // ---- Covisibility --------------------------------------------------
        .def("add_connection",          &KeyFrame::add_connection)
        .def("add_connection_no_lock_", &KeyFrame::add_connection_no_lock_)
        .def("erase_connection",        &KeyFrame::erase_connection)
        .def("erase_connection_no_lock_", &KeyFrame::erase_connection_no_lock_)
        .def("get_connected_keyframes",     &KeyFrame::get_connected_keyframes)
        .def("get_covisible_keyframes",     &KeyFrame::get_covisible_keyframes)
        .def("get_best_covisibles",         &KeyFrame::get_best_covisible_keyframes)
        .def("get_best_covisible_keyframes",&KeyFrame::get_best_covisible_keyframes)
        .def("get_covisible_by_weight",     &KeyFrame::get_covisible_by_weight)
        // Alias matching Python KeyFrame.get_connected_by_weight
        .def("get_connected_by_weight",     &KeyFrame::get_covisible_by_weight)
        .def("get_weight",                  &KeyFrame::get_weight)
        .def("reset_covisibility",          &KeyFrame::reset_covisibility)
        .def("update_connections",          &KeyFrame::update_connections)

        // Compat: Python code reads connected_keyframes_weights as dict
        .def_property_readonly("connected_keyframes_weights",
            [](const KeyFrame &kf) {
                py::dict d;
                for (const auto &[k, v] : kf._covis_weights) d[k] = v;
                // Note: _covis_weights is private; expose via friend or provide accessor
                // This binding accesses the field directly in bindings.cpp
                return d;
            })

        // ---- Spanning tree -------------------------------------------------
        .def("set_parent",          &KeyFrame::set_parent)
        .def("set_parent_no_lock_", &KeyFrame::set_parent_no_lock_)
        .def("get_parent",          &KeyFrame::get_parent)
        .def("add_child",           &KeyFrame::add_child)
        .def("add_child_no_lock_",  &KeyFrame::add_child_no_lock_)
        .def("erase_child",         &KeyFrame::erase_child)
        .def("erase_child_no_lock_",&KeyFrame::erase_child_no_lock_)
        .def("get_children",        &KeyFrame::get_children)
        .def("has_child",           &KeyFrame::has_child)

        // ---- Loop edges ----------------------------------------------------
        .def("add_loop_edge",       &KeyFrame::add_loop_edge)
        .def("get_loop_edges",      &KeyFrame::get_loop_edges)

        // ---- GBA fields ----------------------------------------------------
        .def_readwrite("GBA_kf_id",         &KeyFrame::GBA_kf_id)
        .def_readwrite("is_Tcw_GBA_valid",  &KeyFrame::is_Tcw_GBA_valid)
        .def_readwrite("Tcw_GBA",           &KeyFrame::Tcw_GBA)
        .def_readwrite("Tcw_before_GBA",    &KeyFrame::Tcw_before_GBA)

        // ---- BoW / reloc fields --------------------------------------------
        .def_readwrite("loop_query_id",     &KeyFrame::loop_query_id)
        .def_readwrite("num_loop_words",    &KeyFrame::num_loop_words)
        .def_readwrite("loop_score",        &KeyFrame::loop_score)
        .def_readwrite("reloc_query_id",    &KeyFrame::reloc_query_id)
        .def_readwrite("num_reloc_words",   &KeyFrame::num_reloc_words)
        .def_readwrite("reloc_score",       &KeyFrame::reloc_score)

        // ---- Helpers -------------------------------------------------------
        .def("get_matched_good_points",          &KeyFrame::get_matched_good_points)
        .def("get_matched_good_points_and_idxs", &KeyFrame::get_matched_good_points_and_idxs)
        .def("get_points",                       &KeyFrame::get_points)
        .def("num_tracked_points",               &KeyFrame::num_tracked_points,
             py::arg("min_num_observations") = 0)

        // ---- Repr / hash ---------------------------------------------------
        .def("__repr__", &KeyFrame::__repr__)
        .def("__str__",  &KeyFrame::__repr__)
        .def("__hash__", [](const KeyFrame &kf) { return kf.kid; })
        .def("__eq__",   [](const KeyFrame &a, const KeyFrame &b) { return a.kid == b.kid; })
        .def("__lt__",   [](const KeyFrame &a, const KeyFrame &b) { return a.kid < b.kid; })
        ;
}

// ---------------------------------------------------------------------------
// LocalMappingCore binding
// ---------------------------------------------------------------------------
static void bind_local_mapping_core(py::module_ &m) {
    using namespace slam;
    py::class_<LocalMappingCore>(m, "LocalMappingCore")
        // ---- Construction --------------------------------------------------
        .def(py::init<py::object, int>(),
             py::arg("map"), py::arg("sensor_type") = 2)  // 2 = RGBD

        // ---- Public state --------------------------------------------------
        .def_readwrite("map",               &LocalMappingCore::map)
        .def_readwrite("kf_cur",            &LocalMappingCore::kf_cur)
        .def_readwrite("sensor_type",       &LocalMappingCore::sensor_type)
        .def_readwrite("kid_last_BA",       &LocalMappingCore::kid_last_BA)
        .def_readwrite("opt_abort_flag",    &LocalMappingCore::opt_abort_flag)
        .def_readwrite("mp_opt_abort_flag", &LocalMappingCore::mp_opt_abort_flag)

        // recently_added exposed as Python set for inspection
        .def_property("recently_added",
            [](const LocalMappingCore &lmc) {
                py::set s;
                for (const auto &p : lmc.recently_added) s.add(p);
                return s;
            },
            [](LocalMappingCore &lmc, py::object items) {
                lmc.recently_added.clear();
                for (auto item : items) {
                    lmc.recently_added.insert(
                        py::reinterpret_borrow<py::object>(item));
                }
            })

        // ---- Lifecycle -----------------------------------------------------
        .def("reset",                &LocalMappingCore::reset)
        .def("add_points",           &LocalMappingCore::add_points)
        .def("remove_points",        &LocalMappingCore::remove_points)
        .def("set_opt_abort_flag",   &LocalMappingCore::set_opt_abort_flag)

        // ---- Core LM methods -----------------------------------------------
        .def("process_new_keyframe", &LocalMappingCore::process_new_keyframe)
        .def("cull_map_points",      &LocalMappingCore::cull_map_points)
        .def("cull_keyframes",       &LocalMappingCore::cull_keyframes)
        .def("fuse_map_points",      &LocalMappingCore::fuse_map_points,
             py::arg("descriptor_distance_sigma"),
             py::arg("pm_cls") = py::none())
        .def("local_BA",             &LocalMappingCore::local_BA)
        .def("large_window_BA",      &LocalMappingCore::large_window_BA)
        ;
}

// ---------------------------------------------------------------------------
// Module entry
// ---------------------------------------------------------------------------
PYBIND11_MODULE(cpp_slam_core, m) {
    m.doc() = "cpp_slam_core: C++ SLAM data structures (MapPoint, Frame, KeyFrame, LocalMappingCore)";
    m.attr("USE_CPP_CORE") = true;

    m.def("hello", []() -> std::string { return "cpp_slam_core ok"; });

    bind_map_point(m);
    bind_frame(m);
    bind_keyframe(m);
    bind_local_mapping_core(m);

    // Python 3.11 adaptive interpreter specialization bug: calling a pybind11
    // instancemethod exactly 8 times in a tight for loop triggers a segfault
    // (CALL opcode quickening selects the C-extension fast path which crashes).
    // Wrapping ALL non-dunder instance methods with pure Python functions forces
    // the safe CALL_PY_EXACT_ARGS specialization path instead.
    py::exec(R"(
import types as _t
def _wrap_cpp_class(cls):
    for _n in list(dir(cls)):
        if _n.startswith('__'):
            continue
        try:
            _a = getattr(cls, _n)
        except Exception:
            continue
        if not callable(_a) or isinstance(_a, (_t.FunctionType, type, property)):
            continue
        def _mk(f, n):
            def _w(self, *a, **kw):
                return f(self, *a, **kw)
            _w.__name__ = n
            return _w
        try:
            setattr(cls, _n, _mk(_a, _n))
        except Exception:
            pass
for _c in [MapPoint, Frame, KeyFrame, LocalMappingCore]:
    _wrap_cpp_class(_c)
del _wrap_cpp_class, _c, _t
)", m.attr("__dict__"));
}
