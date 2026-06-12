// Native LiDAR front-end bindings (V4.4). Grows per phase:
// P1 voxel filters + extrapolator; P2/P3 add the LidarFrontend class.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lidar_frontend.h"
#include "pose_extrapolator_cv.h"
#include "scan_match_common.h"
#include "scan_to_submap_2d.h"

namespace py = pybind11;
using namespace fusion;

namespace {

MatX2d to_matx2d(const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
    if (a.ndim() != 2 || a.shape(1) != 2)
        throw std::invalid_argument("expected (N,2) float64 array");
    MatX2d m(a.shape(0), 2);
    std::memcpy(m.data(), a.data(), sizeof(double) * a.shape(0) * 2);
    return m;
}

py::array_t<double> from_matx2d(const MatX2d& m) {
    py::array_t<double> out({static_cast<ssize_t>(m.rows()), static_cast<ssize_t>(2)});
    std::memcpy(out.mutable_data(), m.data(), sizeof(double) * m.rows() * 2);
    return out;
}

}  // namespace

void bind_lidar(py::module_& m) {
    py::class_<VoxelFilterConfig>(m, "VoxelFilterConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &VoxelFilterConfig::enabled)
        .def_readwrite("fixed_size", &VoxelFilterConfig::fixed_size)
        .def_readwrite("adaptive_max_size", &VoxelFilterConfig::adaptive_max_size)
        .def_readwrite("adaptive_min_points", &VoxelFilterConfig::adaptive_min_points)
        .def_readwrite("adaptive_iters", &VoxelFilterConfig::adaptive_iters);

    m.def("fixed_voxel_filter",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> pts,
             double voxel_size) { return from_matx2d(fixed_voxel_filter(to_matx2d(pts), voxel_size)); },
          py::arg("pts"), py::arg("voxel_size"));
    m.def("adaptive_voxel_filter",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> pts,
             double max_voxel_size, int min_num_points, int num_iterations) {
              return from_matx2d(adaptive_voxel_filter(to_matx2d(pts), max_voxel_size,
                                                       min_num_points, num_iterations));
          },
          py::arg("pts"), py::arg("max_voxel_size"), py::arg("min_num_points"),
          py::arg("num_iterations"));
    m.def("voxel_preprocess",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> pts,
             const VoxelFilterConfig& cfg) { return from_matx2d(voxel_preprocess(to_matx2d(pts), cfg)); },
          py::arg("pts"), py::arg("config"));

    py::class_<ExtrapolatorConfig>(m, "ExtrapolatorConfig")
        .def(py::init<>())
        .def_readwrite("max_dt", &ExtrapolatorConfig::max_dt)
        .def_readwrite("init_vxy", &ExtrapolatorConfig::init_vxy)
        .def_readwrite("init_wz", &ExtrapolatorConfig::init_wz)
        .def_readwrite("pose_queue_duration_s", &ExtrapolatorConfig::pose_queue_duration_s)
        .def_readwrite("odom_queue_duration_s", &ExtrapolatorConfig::odom_queue_duration_s)
        .def_readwrite("odom_trust", &ExtrapolatorConfig::odom_trust)
        .def_readwrite("max_linear_speed_mps", &ExtrapolatorConfig::max_linear_speed_mps)
        .def_readwrite("max_angular_speed_rps", &ExtrapolatorConfig::max_angular_speed_rps)
        .def_readwrite("use_imu", &ExtrapolatorConfig::use_imu)
        .def_readwrite("imu_yaw_correction_alpha",
                       &ExtrapolatorConfig::imu_yaw_correction_alpha);

    py::class_<PoseExtrapolatorCV>(m, "PoseExtrapolatorCV")
        .def(py::init<ExtrapolatorConfig>(), py::arg("config") = ExtrapolatorConfig{})
        .def("has_state", &PoseExtrapolatorCV::has_state)
        .def("add_pose", &PoseExtrapolatorCV::add_pose, py::arg("t"), py::arg("pose"))
        .def("add_odometry", &PoseExtrapolatorCV::add_odometry, py::arg("t"), py::arg("pose"))
        .def("add_imu",
             [](PoseExtrapolatorCV& e, double t, double wz, py::object yaw) {
                 if (yaw.is_none()) e.add_imu(t, wz, false, 0.0);
                 else e.add_imu(t, wz, true, py::cast<double>(yaw));
             },
             py::arg("t"), py::arg("wz"), py::arg("yaw") = py::none())
        .def("predict", &PoseExtrapolatorCV::predict, py::arg("t"))
        .def_property_readonly("vx", &PoseExtrapolatorCV::vx)
        .def_property_readonly("vy", &PoseExtrapolatorCV::vy)
        .def_property_readonly("wz", &PoseExtrapolatorCV::wz);

    // ---- scan_to_submap (V4.4-P2) ----------------------------------------

    py::class_<SubmapBuilderConfig>(m, "SubmapBuilderConfig")
        .def(py::init<>())
        .def_readwrite("submap_size_m", &SubmapBuilderConfig::submap_size_m)
        .def_readwrite("resolution", &SubmapBuilderConfig::resolution)
        .def_readwrite("scans_per_submap", &SubmapBuilderConfig::scans_per_submap)
        .def_readwrite("l0", &SubmapBuilderConfig::l0)
        .def_readwrite("l_occ", &SubmapBuilderConfig::l_occ)
        .def_readwrite("l_free", &SubmapBuilderConfig::l_free)
        .def_readwrite("l_min", &SubmapBuilderConfig::l_min)
        .def_readwrite("l_max", &SubmapBuilderConfig::l_max);

    py::class_<SubmapBuilder2D, std::shared_ptr<SubmapBuilder2D>>(m, "NativeSubmapBuilder2D")
        .def(py::init<SubmapBuilderConfig>(), py::arg("config"))
        .def("insert_scan",
             [](SubmapBuilder2D& b, const Pose2& pose,
                py::array_t<double, py::array::c_style | py::array::forcecast> scan) {
                 return b.insert_scan(pose, to_matx2d(scan));
             },
             py::arg("pose_world"), py::arg("scan_local"))
        .def_property_readonly("num_active",
                               [](const SubmapBuilder2D& b) { return b.active().size(); })
        .def_property_readonly("num_finished",
                               [](const SubmapBuilder2D& b) { return b.finished().size(); })
        .def("active_info",
             [](const SubmapBuilder2D& b) {
                 py::list out;
                 for (const auto& sm : b.active())
                     out.append(py::make_tuple(sm->id, sm->num_inserted, sm->finished));
                 return out;
             })
        .def("active_log_odds",
             [](const SubmapBuilder2D& b, int idx) {
                 const auto& sm = b.active().at(idx);
                 py::array_t<float> out({static_cast<ssize_t>(sm->grid.h),
                                         static_cast<ssize_t>(sm->grid.w)});
                 std::memcpy(out.mutable_data(), sm->grid.L.data(),
                             sizeof(float) * sm->grid.L.size());
                 return out;
             },
             py::arg("idx"));

    py::class_<SearchWindow>(m, "SearchWindow")
        .def(py::init<>())
        .def_readwrite("xy_window", &SearchWindow::xy_window)
        .def_readwrite("theta_window", &SearchWindow::theta_window)
        .def_readwrite("xy_step", &SearchWindow::xy_step)
        .def_readwrite("theta_step", &SearchWindow::theta_step)
        .def_readwrite("level", &SearchWindow::level);

    py::class_<SubmapMatcherConfig>(m, "SubmapMatcherConfig")
        .def(py::init<>())
        .def_readwrite("min_score", &SubmapMatcherConfig::min_score)
        .def_readwrite("reject_below_min_score", &SubmapMatcherConfig::reject_below_min_score)
        .def_readwrite("min_valid", &SubmapMatcherConfig::min_valid)
        .def_readwrite("precomp_levels", &SubmapMatcherConfig::precomp_levels)
        .def_readwrite("max_match_points", &SubmapMatcherConfig::max_match_points)
        .def_readwrite("max_refine_points", &SubmapMatcherConfig::max_refine_points)
        .def_readwrite("refine_min_points", &SubmapMatcherConfig::refine_min_points)
        .def_readwrite("refine_w_trans", &SubmapMatcherConfig::refine_w_trans)
        .def_readwrite("refine_w_rot", &SubmapMatcherConfig::refine_w_rot)
        .def_readwrite("refine_iters", &SubmapMatcherConfig::refine_iters)
        .def_readwrite("refine_damping", &SubmapMatcherConfig::refine_damping)
        .def_readwrite("refine_eps_stop", &SubmapMatcherConfig::refine_eps_stop)
        .def_readwrite("refine_step_clip_xy", &SubmapMatcherConfig::refine_step_clip_xy)
        .def_readwrite("refine_step_clip_th", &SubmapMatcherConfig::refine_step_clip_th)
        .def_readwrite("coarse", &SubmapMatcherConfig::coarse)
        .def_readwrite("fine", &SubmapMatcherConfig::fine);

    py::class_<SubmapMatchResult>(m, "SubmapMatchResult")
        .def_readonly("success", &SubmapMatchResult::success)
        .def_readonly("score", &SubmapMatchResult::score)
        .def_readonly("refined_score", &SubmapMatchResult::refined_score)
        .def_readonly("pose_world", &SubmapMatchResult::pose_world);

    m.def("submap_match",
          [](const SubmapBuilder2D& b,
             py::array_t<double, py::array::c_style | py::array::forcecast> scan,
             const Pose2& predicted, const SubmapMatcherConfig& cfg) {
              const MatX2d pts = to_matx2d(scan);
              py::gil_scoped_release rel;
              return submap_match(b, pts, predicted, cfg);
          },
          py::arg("builder"), py::arg("scan_local"), py::arg("predicted_world"),
          py::arg("config"));

    // ---- scan_to_map (V4.4-P3) -------------------------------------------

    py::class_<MapMatcherConfig>(m, "MapMatcherConfig")
        .def(py::init<>())
        .def_readwrite("base_res", &MapMatcherConfig::base_res)
        .def_readwrite("size_m", &MapMatcherConfig::size_m)
        .def_readwrite("num_levels", &MapMatcherConfig::num_levels)
        .def_readwrite("l0", &MapMatcherConfig::l0)
        .def_readwrite("l_occ", &MapMatcherConfig::l_occ)
        .def_readwrite("l_free", &MapMatcherConfig::l_free)
        .def_readwrite("l_min", &MapMatcherConfig::l_min)
        .def_readwrite("l_max", &MapMatcherConfig::l_max)
        .def_readwrite("ray_steps", &MapMatcherConfig::ray_steps)
        .def_readwrite("bootstrap_scans", &MapMatcherConfig::bootstrap_scans)
        .def_readwrite("gn_iters_per_level", &MapMatcherConfig::gn_iters_per_level)
        .def_readwrite("gn_damping", &MapMatcherConfig::gn_damping)
        .def_readwrite("min_points", &MapMatcherConfig::min_points)
        .def_readwrite("min_score", &MapMatcherConfig::min_score)
        .def_readwrite("step_clip_xy", &MapMatcherConfig::step_clip_xy)
        .def_readwrite("step_clip_th", &MapMatcherConfig::step_clip_th)
        .def_readwrite("map_update_every", &MapMatcherConfig::map_update_every);

    py::class_<MapMatcher, std::shared_ptr<MapMatcher>>(m, "NativeMapMatcher")
        .def(py::init<MapMatcherConfig>(), py::arg("config"))
        .def("match",
             [](MapMatcher& mm,
                py::array_t<double, py::array::c_style | py::array::forcecast> scan,
                const Pose2& pred) {
                 auto r = mm.match(to_matx2d(scan), pred);
                 return py::make_tuple(r.success, r.bootstrap, r.score, r.inliers,
                                       r.pose_world);
             },
             py::arg("scan_local"), py::arg("predicted"))
        .def("insert",
             [](MapMatcher& mm, const Pose2& pose,
                py::array_t<double, py::array::c_style | py::array::forcecast> scan) {
                 mm.insert(pose, to_matx2d(scan));
             },
             py::arg("pose_world"), py::arg("scan_local"))
        .def("finest_log_odds", [](const MapMatcher& mm) {
            const auto& g = mm.finest();
            py::array_t<float> out({static_cast<ssize_t>(g.size()),
                                    static_cast<ssize_t>(g.size())});
            std::memcpy(out.mutable_data(), g.log_odds().data(),
                        sizeof(float) * g.log_odds().size());
            return out;
        });

    // ---- LidarFrontend orchestrator ---------------------------------------

    py::enum_<LidarMatcherKind>(m, "LidarMatcherKind")
        .value("SCAN_TO_SUBMAP", LidarMatcherKind::SCAN_TO_SUBMAP)
        .value("SCAN_TO_MAP", LidarMatcherKind::SCAN_TO_MAP);

    py::class_<MotionFilterConfig>(m, "MotionFilterConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &MotionFilterConfig::enabled)
        .def_readwrite("skip_below", &MotionFilterConfig::skip_below)
        .def_readwrite("max_time_s", &MotionFilterConfig::max_time_s)
        .def_readwrite("max_dist_m", &MotionFilterConfig::max_dist_m)
        .def_readwrite("max_angle_rad", &MotionFilterConfig::max_angle_rad);

    py::class_<KeyframeConfig>(m, "KeyframeConfig")
        .def(py::init<>())
        .def_readwrite("min_dist_m", &KeyframeConfig::min_dist_m)
        .def_readwrite("min_angle_rad", &KeyframeConfig::min_angle_rad)
        .def_readwrite("min_dt_s", &KeyframeConfig::min_dt_s);

    py::class_<LidarFrontendConfig>(m, "LidarFrontendConfig")
        .def(py::init<>())
        .def_readwrite("matcher", &LidarFrontendConfig::matcher)
        .def_readwrite("voxel", &LidarFrontendConfig::voxel)
        .def_readwrite("extrap", &LidarFrontendConfig::extrap)
        .def_readwrite("submap_builder", &LidarFrontendConfig::submap_builder)
        .def_readwrite("submap_matcher", &LidarFrontendConfig::submap_matcher)
        .def_readwrite("map_matcher", &LidarFrontendConfig::map_matcher)
        .def_readwrite("insert_filter", &LidarFrontendConfig::insert_filter)
        .def_readwrite("keyframe", &LidarFrontendConfig::keyframe);

    py::class_<LidarScanResult>(m, "LidarScanResult")
        .def_readonly("pose", &LidarScanResult::pose)
        .def_readonly("score", &LidarScanResult::score)
        .def_readonly("refined_score", &LidarScanResult::refined_score)
        .def_readonly("matched", &LidarScanResult::matched)
        .def_readonly("fallback", &LidarScanResult::fallback)
        .def_readonly("skipped", &LidarScanResult::skipped)
        .def_readonly("inserted", &LidarScanResult::inserted)
        .def_readonly("is_keyframe", &LidarScanResult::is_keyframe)
        .def_readonly("num_points", &LidarScanResult::num_points)
        .def_readonly("process_ms", &LidarScanResult::process_ms)
        .def("__repr__", [](const LidarScanResult& r) {
            return "LidarScanResult(matched=" + std::to_string(r.matched) +
                   ", score=" + std::to_string(r.score) +
                   ", kf=" + std::to_string(r.is_keyframe) +
                   ", ms=" + std::to_string(r.process_ms) + ")";
        });

    py::class_<LidarFrontend, std::shared_ptr<LidarFrontend>>(m, "NativeLidarFrontend")
        .def(py::init<LidarFrontendConfig>(), py::arg("config"))
        .def("add_imu",
             [](LidarFrontend& fe, double t, double wz, py::object yaw) {
                 if (yaw.is_none()) fe.add_imu(t, wz, false, 0.0);
                 else fe.add_imu(t, wz, true, py::cast<double>(yaw));
             },
             py::arg("t"), py::arg("wz"), py::arg("yaw") = py::none())
        .def("process",
             [](LidarFrontend& fe,
                py::array_t<double, py::array::c_style | py::array::forcecast> scan,
                double t,
                py::object imu_samples) {
                 const MatX2d pts = to_matx2d(scan);
                 std::vector<std::array<double, 3>> imu;
                 if (!imu_samples.is_none()) {
                     auto a = py::cast<py::array_t<double, py::array::c_style |
                                                           py::array::forcecast>>(imu_samples);
                     if (a.ndim() != 2 || a.shape(1) != 3)
                         throw std::invalid_argument("imu_samples must be (M,3): t,wz,yaw");
                     imu.resize(a.shape(0));
                     for (ssize_t i = 0; i < a.shape(0); ++i)
                         imu[i] = {a.at(i, 0), a.at(i, 1), a.at(i, 2)};
                 }
                 py::gil_scoped_release rel;
                 for (const auto& s : imu) fe.add_imu(s[0], s[1], true, s[2]);
                 return fe.process(pts, t);
             },
             py::arg("scan_xy"), py::arg("t"), py::arg("imu_samples") = py::none())
        .def("last_filtered_points", [](const LidarFrontend& fe) {
            return from_matx2d(fe.last_filtered_points());
        });

    // test hooks: correlative search + refine on a raw probability image
    m.def("bruteforce_search_on_image",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> img,
             double ox, double oy, double res, int precomp_levels, int level,
             py::array_t<double, py::array::c_style | py::array::forcecast> pts,
             const Pose2& center, const SearchWindow& win, int min_valid) {
              if (img.ndim() != 2) throw std::invalid_argument("img must be 2-D");
              std::vector<float> prob(img.data(), img.data() + img.size());
              MaxPoolStack stack(prob, static_cast<int>(img.shape(0)),
                                 static_cast<int>(img.shape(1)), precomp_levels);
              auto [pose, score] = bruteforce_search(stack, level, ox, oy, res,
                                                     to_matx2d(pts), center, win,
                                                     min_valid);
              return py::make_tuple(pose, score);
          });
    m.def("refine_on_image",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> img,
             double ox, double oy, double res,
             py::array_t<double, py::array::c_style | py::array::forcecast> pts,
             const Pose2& initial, const Pose2& prior, int iters, double damping,
             double eps_stop, double clip_xy, double clip_th, double w_trans,
             double w_rot, int min_points) {
              if (img.ndim() != 2) throw std::invalid_argument("img must be 2-D");
              RefineParams rp;
              rp.iters = iters; rp.damping = damping; rp.eps_stop = eps_stop;
              rp.step_clip_xy = clip_xy; rp.step_clip_th = clip_th;
              rp.w_trans = w_trans; rp.w_rot = w_rot; rp.min_points = min_points;
              return refine_pose_lm(img.data(), static_cast<int>(img.shape(1)),
                                    static_cast<int>(img.shape(0)), ox, oy, res,
                                    to_matx2d(pts), initial, prior, rp);
          });
}
