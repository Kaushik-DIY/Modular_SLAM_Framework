// B&B matcher bindings (C5).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstring>

#include "bnb_matcher.h"

namespace py = pybind11;
using namespace fusion;

void bind_bnb(py::module_& m) {
    py::class_<BnbConfig>(m, "BnbConfig")
        .def(py::init<>())
        .def_readwrite("linear_search_window", &BnbConfig::linear_search_window)
        .def_readwrite("angular_search_window", &BnbConfig::angular_search_window)
        .def_readwrite("depth", &BnbConfig::depth)
        .def_readwrite("min_rotational_step", &BnbConfig::min_rotational_step)
        .def_readwrite("max_match_points", &BnbConfig::max_match_points)
        .def_readwrite("do_refine", &BnbConfig::do_refine)
        .def_readwrite("max_refine_points", &BnbConfig::max_refine_points)
        .def_readwrite("refine_min_points", &BnbConfig::refine_min_points)
        .def_readwrite("refine_w_trans", &BnbConfig::refine_w_trans)
        .def_readwrite("refine_w_rot", &BnbConfig::refine_w_rot)
        .def_readwrite("refine_iters", &BnbConfig::refine_iters)
        .def_readwrite("refine_damping", &BnbConfig::refine_damping)
        .def_readwrite("refine_eps_stop", &BnbConfig::refine_eps_stop)
        .def_readwrite("refine_step_clip_xy", &BnbConfig::refine_step_clip_xy)
        .def_readwrite("refine_step_clip_th", &BnbConfig::refine_step_clip_th);

    py::class_<BnbResult>(m, "BnbResult")
        .def_readonly("success", &BnbResult::success)
        .def_readonly("coarse_score", &BnbResult::coarse_score)
        .def_readonly("refined_score", &BnbResult::refined_score)
        .def_readonly("pose", &BnbResult::pose)
        .def_readonly("refined", &BnbResult::refined)
        .def_readonly("num_rotations", &BnbResult::num_rotations)
        .def_readonly("num_points_match", &BnbResult::num_points_match)
        .def("__repr__", [](const BnbResult& r) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "BnbResult(success=%s, coarse=%.3f, refined=%.3f, "
                          "pose=(%.3f, %.3f, %.3f))",
                          r.success ? "True" : "False", r.coarse_score, r.refined_score,
                          r.pose.x, r.pose.y, r.pose.theta);
            return std::string(buf);
        });

    m.def("bnb_match",
          [](const LocalGridPtr& grid,
             py::array_t<float, py::array::c_style | py::array::forcecast> scan,
             const Pose2& predicted_pose, const BnbConfig& cfg) {
              MatX2f pts;
              if (scan.size() > 0) {
                  if (scan.ndim() != 2 || scan.shape(1) != 2)
                      throw std::invalid_argument("scan must be (N,2) float");
                  pts.resize(scan.shape(0), 2);
                  std::memcpy(pts.data(), scan.data(), sizeof(float) * scan.size());
              }
              py::gil_scoped_release rel;
              return bnb_match(*grid, pts, predicted_pose, cfg);
          },
          py::arg("grid"), py::arg("scan_local"), py::arg("predicted_pose"),
          py::arg("config") = BnbConfig{});
}
