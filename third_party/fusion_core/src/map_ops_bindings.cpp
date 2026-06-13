// LocalGrid + neighborhood retrieval bindings (C4).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "local_grid.h"
#include "retrieval.h"

namespace py = pybind11;
using namespace fusion;

void bind_map_ops(py::module_& m) {
    py::class_<GridConfig>(m, "GridConfig")
        .def(py::init<>())
        .def_readwrite("resolution", &GridConfig::resolution)
        .def_readwrite("l_occ", &GridConfig::l_occ)
        .def_readwrite("l_free", &GridConfig::l_free)
        .def_readwrite("l_min", &GridConfig::l_min)
        .def_readwrite("l_max", &GridConfig::l_max)
        .def_readwrite("margin", &GridConfig::margin);

    py::class_<LocalGrid, LocalGridPtr>(m, "LocalGrid")
        .def_property_readonly("width", &LocalGrid::width)
        .def_property_readonly("height", &LocalGrid::height)
        .def_property_readonly("origin_x", &LocalGrid::origin_x)
        .def_property_readonly("origin_y", &LocalGrid::origin_y)
        .def_property_readonly("resolution", &LocalGrid::resolution)
        .def_property_readonly("log_odds", [](py::object self) {
            auto& g = py::cast<LocalGrid&>(self);
            return py::array_t<float>(
                {static_cast<ssize_t>(g.height()), static_cast<ssize_t>(g.width())},
                {static_cast<ssize_t>(sizeof(float)) * g.width(),
                 static_cast<ssize_t>(sizeof(float))},
                g.data(), self);   // zero-copy view, grid kept alive
        })
        .def("probability", [](const LocalGrid& g) {
            auto p = g.probability();
            py::array_t<float> out({static_cast<ssize_t>(g.height()),
                                    static_cast<ssize_t>(g.width())});
            std::memcpy(out.mutable_data(), p.data(), p.size() * sizeof(float));
            return out;
        });

    py::class_<Neighborhood>(m, "Neighborhood")
        .def_readonly("signatures", &Neighborhood::signatures)
        .def_readonly("poses", &Neighborhood::poses)
        .def_readonly("reactivated", &Neighborhood::reactivated)
        .def("__len__", [](const Neighborhood& n) { return n.signatures.size(); });

    m.def("retrieve_neighborhood", &retrieve_neighborhood,
          py::arg("memory"), py::arg("graph"), py::arg("candidate_id"),
          py::arg("graph_depth") = 2, py::arg("metric_radius") = 5.0,
          py::arg("scans_only") = true,
          py::call_guard<py::gil_scoped_release>());

    m.def("assemble_local_grid", &assemble_local_grid,
          py::arg("signatures"), py::arg("poses"), py::arg("config") = GridConfig{},
          py::call_guard<py::gil_scoped_release>());

    // Convenience: retrieval + assembly in one GIL-released call.
    m.def("candidate_local_grid",
          [](MemoryManager& mem, FusionGraph2D& graph, int32_t candidate_id,
             int graph_depth, double metric_radius, GridConfig cfg) {
              auto nb = retrieve_neighborhood(mem, graph, candidate_id, graph_depth,
                                              metric_radius, true);
              return std::make_pair(assemble_local_grid(nb.signatures, nb.poses, cfg), nb);
          },
          py::arg("memory"), py::arg("graph"), py::arg("candidate_id"),
          py::arg("graph_depth") = 2, py::arg("metric_radius") = 5.0,
          py::arg("config") = GridConfig{},
          py::call_guard<py::gil_scoped_release>());
}
