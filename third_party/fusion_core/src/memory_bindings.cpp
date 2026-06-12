// MemoryManager pybind11 bindings (C2).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "memory_manager.h"

namespace py = pybind11;
using namespace fusion;

void bind_memory(py::module_& m) {
    py::enum_<Tier>(m, "Tier")
        .value("STM", Tier::STM)
        .value("WM", Tier::WM)
        .value("LTM", Tier::LTM)
        .value("NONE", Tier::NONE);

    py::class_<MemoryConfig>(m, "MemoryConfig")
        .def(py::init<>())
        .def_readwrite("stm_size", &MemoryConfig::stm_size)
        .def_readwrite("wm_cap", &MemoryConfig::wm_cap)
        .def_readwrite("rehearsal_similarity", &MemoryConfig::rehearsal_similarity)
        .def_readwrite("rehearsal_enabled", &MemoryConfig::rehearsal_enabled)
        .def_readwrite("reactivation_neighbor_depth", &MemoryConfig::reactivation_neighbor_depth);

    py::class_<InsertResult>(m, "InsertResult")
        .def_readonly("id", &InsertResult::id)
        .def_readonly("similarity", &InsertResult::similarity)
        .def_readonly("rehearsal_merged", &InsertResult::rehearsal_merged)
        .def_readonly("merged_predecessor_id", &InsertResult::merged_predecessor_id)
        .def_readonly("moved_to_wm", &InsertResult::moved_to_wm)
        .def_readonly("transferred_to_ltm", &InsertResult::transferred_to_ltm)
        .def("__repr__", [](const InsertResult& r) {
            return "InsertResult(id=" + std::to_string(r.id) +
                   ", merged=" + (r.rehearsal_merged ? "True" : "False") +
                   ", ->wm=" + std::to_string(r.moved_to_wm.size()) +
                   ", ->ltm=" + std::to_string(r.transferred_to_ltm.size()) + ")";
        });

    py::class_<MemoryManager, std::shared_ptr<MemoryManager>>(m, "MemoryManager")
        .def(py::init<MemoryConfig, std::shared_ptr<InRamLtmStore>>(),
             py::arg("config"), py::arg("ltm_store"))
        .def("insert", &MemoryManager::insert,
             py::arg("signature"), py::arg("similarity") = -1.0,
             py::call_guard<py::gil_scoped_release>())
        .def("on_loop_confirmed", &MemoryManager::on_loop_confirmed,
             py::arg("query_id"), py::arg("matched_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("reactivate", &MemoryManager::reactivate,
             py::arg("id"), py::arg("neighbor_depth") = -1,
             py::call_guard<py::gil_scoped_release>())
        .def("touch", &MemoryManager::touch, py::call_guard<py::gil_scoped_release>())
        .def("tier", &MemoryManager::tier)
        .def("get", &MemoryManager::get)
        .def("stm_ids", &MemoryManager::stm_ids)
        .def("wm_ids", &MemoryManager::wm_ids)
        .def("stm_count", &MemoryManager::stm_count)
        .def("wm_count", &MemoryManager::wm_count)
        .def("ltm_count", &MemoryManager::ltm_count)
        .def("payload_bytes", &MemoryManager::payload_bytes)
        .def_static("scan_overlap_similarity",
                    [](const SignaturePtr& a, const SignaturePtr& b) {
                        return MemoryManager::scan_overlap_similarity(*a, *b);
                    });
}
