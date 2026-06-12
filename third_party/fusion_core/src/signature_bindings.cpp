// Signature + LtmStore pybind11 bindings (C1).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>

#include "signature.h"
#include "ltm_store.h"

namespace py = pybind11;
using namespace fusion;

namespace {

// Copy an (N,2)/(N,3) float numpy array into a C++-owned row-major Eigen matrix.
template <typename Mat>
void copy_into(Mat& dst, py::array_t<float, py::array::c_style | py::array::forcecast> arr,
               int expected_cols) {
    if (arr.ndim() == 0 || arr.size() == 0) { dst.resize(0, expected_cols); return; }
    if (arr.ndim() != 2 || arr.shape(1) != expected_cols)
        throw std::invalid_argument("expected (N, " + std::to_string(expected_cols) + ") float array");
    dst.resize(arr.shape(0), expected_cols);
    std::memcpy(dst.data(), arr.data(), sizeof(float) * arr.size());
}

// Zero-copy numpy view over C++-owned data; `base` keeps the Signature alive.
py::array view_f32(py::object base, const float* data, ssize_t rows, ssize_t cols) {
    return py::array_t<float>({rows, cols},
                              {static_cast<ssize_t>(sizeof(float)) * cols,
                               static_cast<ssize_t>(sizeof(float))},
                              data, base);
}

}  // namespace

void bind_signature(py::module_& m) {
    py::enum_<LinkType>(m, "LinkType")
        .value("NEIGHBOR", LinkType::NEIGHBOR)
        .value("LOOP", LinkType::LOOP)
        .value("MERGED", LinkType::MERGED);

    py::class_<Link>(m, "Link")
        .def_readonly("to_id", &Link::to_id)
        .def_readonly("type", &Link::type)
        .def_readonly("rel_pose", &Link::rel_pose)
        .def_readonly("trans_info", &Link::trans_info)
        .def_readonly("rot_info", &Link::rot_info)
        .def("__repr__", [](const Link& l) {
            return "Link(to=" + std::to_string(l.to_id) +
                   ", type=" + std::to_string(static_cast<int>(l.type)) + ")";
        });

    py::class_<Signature, SignaturePtr>(m, "Signature")
        .def(py::init([](int32_t id, double stamp,
                         py::array_t<float, py::array::c_style | py::array::forcecast> kpts,
                         py::object des,
                         py::array_t<float, py::array::c_style | py::array::forcecast> pts3d,
                         py::array_t<float, py::array::c_style | py::array::forcecast> scan_xy) {
                 auto sig = std::make_shared<Signature>();
                 sig->id = id;
                 sig->stamp = stamp;
                 copy_into(sig->kpts, kpts, 2);
                 copy_into(sig->pts3d, pts3d, 3);
                 copy_into(sig->scan_xy, scan_xy, 2);
                 if (!des.is_none()) {
                     auto d = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(des);
                     if (d.size() > 0) {
                         if (d.ndim() != 2 || d.shape(1) != 32)
                             throw std::invalid_argument("descriptors must be (N, 32) uint8");
                         sig->des = cv::Mat(static_cast<int>(d.shape(0)), 32, CV_8U);
                         std::memcpy(sig->des.data, d.data(), d.size());
                     }
                 }
                 if (sig->des.rows > 0 && sig->kpts.rows() != sig->des.rows)
                     throw std::invalid_argument("kpts and des row counts differ");
                 return sig;
             }),
             py::arg("id"), py::arg("stamp"),
             py::arg("kpts") = py::array_t<float>(),
             py::arg("des") = py::none(),
             py::arg("pts3d") = py::array_t<float>(),
             py::arg("scan_xy") = py::array_t<float>(),
             "Payloads are COPIED into C++-owned storage (map lifetime independent "
             "of front-end objects).")
        .def_readonly("id", &Signature::id)
        .def_readonly("stamp", &Signature::stamp)
        .def_readwrite("weight", &Signature::weight)
        .def_readwrite("pose", &Signature::pose)
        .def_property_readonly("kpts", [](py::object self) {
            auto& s = py::cast<Signature&>(self);
            return view_f32(self, s.kpts.data(), s.kpts.rows(), 2);
        })
        .def_property_readonly("des", [](py::object self) -> py::object {
            auto& s = py::cast<Signature&>(self);
            if (s.des.rows == 0) return py::none();
            return py::array_t<uint8_t>(
                {static_cast<ssize_t>(s.des.rows), static_cast<ssize_t>(32)},
                {static_cast<ssize_t>(s.des.step[0]), static_cast<ssize_t>(1)},
                s.des.data, self);
        })
        .def_property_readonly("pts3d", [](py::object self) {
            auto& s = py::cast<Signature&>(self);
            return view_f32(self, s.pts3d.data(), s.pts3d.rows(), 3);
        })
        .def_property_readonly("scan_xy", [](py::object self) {
            auto& s = py::cast<Signature&>(self);
            return view_f32(self, s.scan_xy.data(), s.scan_xy.rows(), 2);
        })
        .def_property_readonly("has_visual", &Signature::has_visual)
        .def_property_readonly("has_scan", &Signature::has_scan)
        .def_property_readonly("num_kpts", &Signature::num_kpts)
        .def_property_readonly("num_scan_points", &Signature::num_scan_points)
        .def_property_readonly("payload_bytes", &Signature::payload_bytes)
        .def("add_link", [](Signature& s, int32_t to_id, LinkType type, const Pose2& rel,
                            double trans_info, double rot_info) {
                 s.links.push_back(Link{to_id, type, rel, trans_info, rot_info});
             },
             py::arg("to_id"), py::arg("type"), py::arg("rel_pose"),
             py::arg("trans_info") = 0.0, py::arg("rot_info") = 0.0)
        .def_property_readonly("links", [](const Signature& s) { return s.links; })
        .def("__repr__", [](const Signature& s) {
            return "Signature(id=" + std::to_string(s.id) +
                   ", kpts=" + std::to_string(s.num_kpts()) +
                   ", scan=" + std::to_string(s.num_scan_points()) +
                   ", w=" + std::to_string(s.weight) + ")";
        });

    py::class_<InRamLtmStore, std::shared_ptr<InRamLtmStore>>(m, "InRamLtmStore")
        .def(py::init<>())
        .def("store", &InRamLtmStore::store)
        .def("load", &InRamLtmStore::load)
        .def("contains", &InRamLtmStore::contains)
        .def("erase", &InRamLtmStore::erase)
        .def("size", &InRamLtmStore::size)
        .def("ids", &InRamLtmStore::ids)
        .def("payload_bytes", &InRamLtmStore::payload_bytes);
}
