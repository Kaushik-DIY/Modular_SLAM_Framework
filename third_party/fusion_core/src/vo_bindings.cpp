// VoFrontend pybind11 bindings (V3.1).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>

#include "vo_frontend.h"

namespace py = pybind11;
using namespace fusion;

namespace {

cv::Mat mat_u8_view(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& a) {
    if (a.ndim() != 2) throw std::invalid_argument("expected 2-D uint8 image");
    return cv::Mat(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
                   CV_8U, const_cast<uint8_t*>(a.data()),
                   static_cast<size_t>(a.strides(0)));
}

cv::Mat mat_u16_view(const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>& a) {
    if (a.ndim() != 2) throw std::invalid_argument("expected 2-D uint16 depth");
    return cv::Mat(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
                   CV_16U, const_cast<uint16_t*>(a.data()),
                   static_cast<size_t>(a.strides(0)));
}

py::array f32_copy(const float* data, ssize_t rows, ssize_t cols) {
    py::array_t<float> out({rows, cols});
    std::memcpy(out.mutable_data(), data, sizeof(float) * rows * cols);
    return out;
}

}  // namespace

void bind_vo(py::module_& m) {
    py::class_<VoConfig>(m, "VoConfig")
        .def(py::init<>())
        .def_readwrite("n_features", &VoConfig::n_features)
        .def_readwrite("scale_factor", &VoConfig::scale_factor)
        .def_readwrite("n_levels", &VoConfig::n_levels)
        .def_readwrite("grid_cell_px", &VoConfig::grid_cell_px)
        .def_readwrite("max_per_cell", &VoConfig::max_per_cell)
        .def_readwrite("fx", &VoConfig::fx)
        .def_readwrite("fy", &VoConfig::fy)
        .def_readwrite("cx", &VoConfig::cx)
        .def_readwrite("cy", &VoConfig::cy)
        .def_readwrite("depth_factor", &VoConfig::depth_factor)
        .def_readwrite("depth_min", &VoConfig::depth_min)
        .def_readwrite("depth_max", &VoConfig::depth_max)
        .def_readwrite("window_size", &VoConfig::window_size)
        .def_readwrite("max_points_per_kf", &VoConfig::max_points_per_kf)
        .def_readwrite("match_radius_px", &VoConfig::match_radius_px)
        .def_readwrite("match_radius_fallback_px", &VoConfig::match_radius_fallback_px)
        .def_readwrite("max_hamming", &VoConfig::max_hamming)
        .def_readwrite("ratio_test", &VoConfig::ratio_test)
        .def_readwrite("enable_local_ba", &VoConfig::enable_local_ba)
        .def_readwrite("local_ba_iters", &VoConfig::local_ba_iters)
        .def_readwrite("min_obs_for_ba", &VoConfig::min_obs_for_ba)
        .def_readwrite("local_ba_max_points", &VoConfig::local_ba_max_points)
        .def_readwrite("min_inliers_ok", &VoConfig::min_inliers_ok)
        .def_readwrite("min_inliers_reinit", &VoConfig::min_inliers_reinit)
        .def_readwrite("reinit_patience", &VoConfig::reinit_patience)
        .def_readwrite("kf_min_frames", &VoConfig::kf_min_frames)
        .def_readwrite("kf_max_frames", &VoConfig::kf_max_frames)
        .def_readwrite("kf_ref_ratio", &VoConfig::kf_ref_ratio)
        .def_readwrite("kf_min_close_points", &VoConfig::kf_min_close_points);

    py::enum_<VoState>(m, "VoState")
        .value("INIT", VoState::INIT)
        .value("OK", VoState::OK)
        .value("WEAK", VoState::WEAK)
        .value("REINIT", VoState::REINIT);

    py::class_<VoResult>(m, "VoResult")
        .def_property_readonly("Twc", [](const VoResult& r) {
            py::array_t<double> out({static_cast<ssize_t>(4), static_cast<ssize_t>(4)});
            Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(out.mutable_data()) = r.Twc;
            return out;
        })
        .def_property_readonly("prev_Twc", [](const VoResult& r) {
            py::array_t<double> out({static_cast<ssize_t>(4), static_cast<ssize_t>(4)});
            Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(out.mutable_data()) = r.prev_Twc;
            return out;
        })
        .def_readonly("has_prev", &VoResult::has_prev)
        .def_readonly("state", &VoResult::state)
        .def_readonly("new_keyframe", &VoResult::new_keyframe)
        .def_readonly("kf_id", &VoResult::kf_id)
        .def_readonly("n_matches", &VoResult::n_matches)
        .def_readonly("n_inliers", &VoResult::n_inliers)
        .def_readonly("track_ms", &VoResult::track_ms)
        .def("__repr__", [](const VoResult& r) {
            return "VoResult(state=" + std::to_string(static_cast<int>(r.state)) +
                   ", inliers=" + std::to_string(r.n_inliers) +
                   ", new_kf=" + (r.new_keyframe ? std::string("True") : std::string("False")) +
                   ", ms=" + std::to_string(r.track_ms) + ")";
        });

    py::class_<VoFrontend, std::shared_ptr<VoFrontend>>(m, "VoFrontend")
        .def(py::init<VoConfig>(), py::arg("config"))
        .def("track",
             [](VoFrontend& vo,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> gray,
                py::array_t<uint16_t, py::array::c_style | py::array::forcecast> depth,
                double stamp, py::object prior_Twc) {
                 cv::Mat g = mat_u8_view(gray);
                 cv::Mat d = mat_u16_view(depth);
                 Eigen::Matrix4d prior = Eigen::Matrix4d::Identity();
                 bool has_prior = false;
                 if (!prior_Twc.is_none()) {
                     auto a = py::cast<py::array_t<double, py::array::c_style |
                                                            py::array::forcecast>>(prior_Twc);
                     if (a.ndim() != 2 || a.shape(0) != 4 || a.shape(1) != 4)
                         throw std::invalid_argument("prior_Twc must be 4x4");
                     prior = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(
                         a.data());
                     has_prior = true;
                 }
                 py::gil_scoped_release rel;
                 return vo.track(g, d, stamp, prior, has_prior);
             },
             py::arg("gray"), py::arg("depth"), py::arg("stamp"),
             py::arg("prior_Twc") = py::none())
        .def("keyframe_payload",
             [](const VoFrontend& vo, int kf_id) -> py::object {
                 const VoKeyframe* kf = vo.keyframe(kf_id);
                 if (kf == nullptr) return py::none();
                 const ssize_t n = kf->kpts.rows();
                 py::array_t<uint8_t> des({n, static_cast<ssize_t>(32)});
                 std::memcpy(des.mutable_data(), kf->des.data, static_cast<size_t>(n) * 32);
                 return py::make_tuple(
                     f32_copy(kf->kpts.data(), n, 2), des,
                     f32_copy(kf->pts3d_cam.data(), n, 3));
             },
             "Returns (kpts (N,2) f32, des (N,32) u8, pts3d_cam (N,3) f32 with NaN "
             "for no-depth) for a window keyframe, or None if evicted.")
        .def_property_readonly("window_count", &VoFrontend::window_count)
        .def_property_readonly("window_bytes", &VoFrontend::window_bytes);
}
