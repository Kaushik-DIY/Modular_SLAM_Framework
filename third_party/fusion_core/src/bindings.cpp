// fusion_core pybind11 module entry. Pattern follows ../cpp_slam_core/src/bindings.cpp.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <g2o/types/slam2d/se2.h>

#include "common.h"

namespace py = pybind11;
using fusion::Pose2;

#define FUSION_CORE_VERSION "0.2.0-v3"

// Force-links g2o types_slam2d and proves SE2 math works inside this module
// while g2o's own pybind module / slam_optimizer_core may be loaded in the same
// process (the multi-module coexistence risk retired at C0).
static Pose2 g2o_se2_roundtrip(double x, double y, double theta) {
    const g2o::SE2 a(x, y, theta);
    const g2o::SE2 b = a * a.inverse();   // identity
    const g2o::SE2 c = b * a;             // back to a
    return Pose2(c.translation().x(), c.translation().y(), c.rotation().angle());
}

void bind_signature(py::module_& m);  // signature_bindings.cpp (C1)
void bind_memory(py::module_& m);     // memory_bindings.cpp (C2)
void bind_graph(py::module_& m);      // graph_bindings.cpp (C3)
void bind_map_ops(py::module_& m);    // map_ops_bindings.cpp (C4)
void bind_bnb(py::module_& m);        // bnb_bindings.cpp (C5)
void bind_vo(py::module_& m);         // vo_bindings.cpp (V3.1)
void bind_lidar(py::module_& m);      // lidar_bindings.cpp (V4.4)

PYBIND11_MODULE(fusion_core, m) {
    m.doc() = "C++ core for the RTAB-inspired multi-modal shared map (fusion v2)";
    m.attr("__version__") = FUSION_CORE_VERSION;
    bind_signature(m);
    bind_memory(m);
    bind_graph(m);
    bind_map_ops(m);
    bind_bnb(m);
    bind_vo(m);
    bind_lidar(m);

    m.def("hello", [] { return std::string("fusion_core ") + FUSION_CORE_VERSION; });

    py::class_<Pose2>(m, "Pose2")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("theta"))
        .def_readwrite("x", &Pose2::x)
        .def_readwrite("y", &Pose2::y)
        .def_readwrite("theta", &Pose2::theta)
        .def("compose", &Pose2::compose)
        .def("inverse", &Pose2::inverse)
        .def("__repr__", [](const Pose2& p) {
            char buf[80];
            std::snprintf(buf, sizeof(buf), "Pose2(%.6f, %.6f, %.6f)", p.x, p.y, p.theta);
            return std::string(buf);
        });

    m.def("g2o_se2_roundtrip", &g2o_se2_roundtrip,
          py::arg("x"), py::arg("y"), py::arg("theta"),
          "SE2 compose/inverse round-trip through statically-linked g2o (C0 link check)");
}
