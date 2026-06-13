// FusionGraph2D pybind11 bindings (C3).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "fusion_graph.h"

namespace py = pybind11;
using namespace fusion;

void bind_graph(py::module_& m) {
    py::class_<GraphConfig>(m, "GraphConfig")
        .def(py::init<>())
        .def_readwrite("huber_scale", &GraphConfig::huber_scale)
        .def_readwrite("spine_trans_weight", &GraphConfig::spine_trans_weight)
        .def_readwrite("spine_rot_weight", &GraphConfig::spine_rot_weight)
        .def_readwrite("max_iterations", &GraphConfig::max_iterations);

    py::class_<FusionGraph2D, std::shared_ptr<FusionGraph2D>>(m, "FusionGraph2D")
        .def(py::init<GraphConfig>(), py::arg("config") = GraphConfig{})
        .def("add_node", &FusionGraph2D::add_node, py::arg("id"), py::arg("pose"))
        .def("set_node_pose", &FusionGraph2D::set_node_pose)
        .def("set_fixed", &FusionGraph2D::set_fixed)
        .def("add_spine_edge", &FusionGraph2D::add_spine_edge,
             py::arg("from_id"), py::arg("to_id"), py::arg("rel_pose"),
             py::arg("trans_weight") = -1.0, py::arg("rot_weight") = -1.0)
        .def("add_loop_edge", &FusionGraph2D::add_loop_edge,
             py::arg("from_id"), py::arg("to_id"), py::arg("rel_pose"),
             py::arg("trans_weight"), py::arg("rot_weight"))
        .def("optimize", &FusionGraph2D::optimize, py::arg("iterations") = -1,
             py::call_guard<py::gil_scoped_release>())
        .def("get_pose", &FusionGraph2D::get_pose)
        .def("has_node", &FusionGraph2D::has_node)
        .def("poses", [](const FusionGraph2D& g) {
            auto snap = g.poses_snapshot();
            py::array_t<double> out({static_cast<ssize_t>(snap.size()), static_cast<ssize_t>(4)});
            auto r = out.mutable_unchecked<2>();
            for (ssize_t i = 0; i < static_cast<ssize_t>(snap.size()); ++i)
                for (ssize_t j = 0; j < 4; ++j) r(i, j) = snap[i][j];
            return out;
        }, "Sorted-by-id (N,4) array of (id, x, y, theta)")
        .def_property_readonly("num_nodes", &FusionGraph2D::num_nodes)
        .def_property_readonly("num_edges", &FusionGraph2D::num_edges)
        .def_property_readonly("num_loop_edges", &FusionGraph2D::num_loop_edges)
        .def_property_readonly("fixed_id", &FusionGraph2D::fixed_id);
}
