#include "frame.h"

#include <sstream>

namespace slam {

// ---- Static members --------------------------------------------------------
std::atomic<int> Frame::_next_id{0};
std::mutex Frame::_id_lock;

int Frame::next_id() {
    std::lock_guard<std::mutex> lk(_id_lock);
    return _next_id.load();
}

void Frame::set_id(int id) {
    _next_id.store(id);
}

// ---- Construction ----------------------------------------------------------
Frame::Frame(int given_id)
    : id(given_id < 0 ? _next_id.fetch_add(1) : given_id),
      _Tcw(Eigen::Matrix4d::Identity()),
      bow_vector(py::none()),
      feat_vector(py::none()),
      g_des(py::none()),
      f_des(py::none()),
      depths(py::none()) {}

// ---- Feature array normalization -------------------------------------------
MatNx2f Frame::_normalize_kpsu(py::object kps_in, int n) {
    MatNx2f result(n, 2);
    result.setZero();
    if (n == 0) return result;

    // Try as numpy array first (fast path)
    try {
        auto arr = kps_in.cast<py::array_t<float>>();
        auto r = arr.unchecked<2>();
        for (int i = 0; i < n && i < static_cast<int>(r.shape(0)); ++i) {
            result(i, 0) = r(i, 0);
            result(i, 1) = r(i, 1);
        }
        return result;
    } catch (...) {}

    // Try as numpy float64 array
    try {
        auto arr = kps_in.cast<py::array_t<double>>();
        auto r = arr.unchecked<2>();
        for (int i = 0; i < n && i < static_cast<int>(r.shape(0)); ++i) {
            result(i, 0) = static_cast<float>(r(i, 0));
            result(i, 1) = static_cast<float>(r(i, 1));
        }
        return result;
    } catch (...) {}

    // Try as list of cv2.KeyPoint or (x,y) tuples/arrays
    try {
        auto lst = kps_in.cast<py::list>();
        for (int i = 0; i < n && i < static_cast<int>(lst.size()); ++i) {
            py::object kp = lst[i];
            if (py::hasattr(kp, "pt")) {
                // cv2.KeyPoint
                auto pt = kp.attr("pt");
                result(i, 0) = pt.attr("__getitem__")(0).cast<float>();
                result(i, 1) = pt.attr("__getitem__")(1).cast<float>();
            } else {
                // Tuple or array (x, y)
                result(i, 0) = kp.cast<py::sequence>()[0].cast<float>();
                result(i, 1) = kp.cast<py::sequence>()[1].cast<float>();
            }
        }
        return result;
    } catch (...) {}

    return result;  // zeros on failure
}

Eigen::VectorXf Frame::_normalize_kps_ur(py::object kps_ur_in, int n) {
    Eigen::VectorXf result = Eigen::VectorXf::Constant(n, -1.0f);
    if (n == 0 || kps_ur_in.is_none()) return result;
    try {
        auto arr = kps_ur_in.cast<py::array_t<float>>();
        auto r = arr.unchecked<1>();
        int sz = std::min(n, static_cast<int>(r.shape(0)));
        for (int i = 0; i < sz; ++i) result(i) = r(i);
    } catch (...) {
        try {
            auto arr = kps_ur_in.cast<py::array_t<double>>();
            auto r = arr.unchecked<1>();
            int sz = std::min(n, static_cast<int>(r.shape(0)));
            for (int i = 0; i < sz; ++i) result(i) = static_cast<float>(r(i));
        } catch (...) {}
    }
    return result;
}

Eigen::VectorXi Frame::_extract_octaves_from_kps(py::object kps_in, int n) {
    Eigen::VectorXi result = Eigen::VectorXi::Zero(n);
    if (n == 0 || kps_in.is_none()) return result;
    try {
        auto lst = kps_in.cast<py::list>();
        int sz = std::min(n, static_cast<int>(lst.size()));
        for (int i = 0; i < sz; ++i) {
            py::object kp = lst[i];
            if (py::hasattr(kp, "octave")) {
                result(i) = std::max(0, kp.attr("octave").cast<int>());
            }
        }
    } catch (...) {}
    return result;
}

Eigen::VectorXi Frame::_normalize_octaves(py::object octaves_in, int n) {
    Eigen::VectorXi result = Eigen::VectorXi::Zero(n);
    if (n == 0 || octaves_in.is_none()) return result;
    try {
        auto arr = octaves_in.cast<py::array_t<int32_t>>();
        auto r = arr.unchecked<1>();
        int sz = std::min(n, static_cast<int>(r.shape(0)));
        for (int i = 0; i < sz; ++i) result(i) = r(i);
    } catch (...) {
        try {
            auto lst = octaves_in.cast<py::list>();
            int sz = std::min(n, static_cast<int>(lst.size()));
            for (int i = 0; i < sz; ++i) result(i) = lst[i].cast<int>();
        } catch (...) {}
    }
    return result;
}

void Frame::init_feature_arrays(py::object kps_in, py::array_t<uint8_t> des_in,
                                  py::object kps_ur_in, py::object octaves_in,
                                  int n_features) {
    // Normalize kpsu (handles cv2.KeyPoint, np array, or list of tuples)
    kpsu = _normalize_kpsu(kps_in, n_features);
    kps_ur = _normalize_kps_ur(kps_ur_in, n_features);

    // Octaves: if not provided explicitly, try extracting from kps list
    if (octaves_in.is_none()) {
        octaves = _extract_octaves_from_kps(kps_in, n_features);
    } else {
        octaves = _normalize_octaves(octaves_in, n_features);
    }

    // Copy descriptors to cv::Mat
    if (des_in.size() > 0) {
        int rows = static_cast<int>(des_in.shape(0));
        int cols = static_cast<int>(des_in.shape(1));
        des = cv::Mat(rows, cols, CV_8U);
        std::memcpy(des.data, des_in.data(), rows * cols);
    }

    // Initialize point associations
    points.assign(n_features, py::none());
    outliers.assign(n_features, false);
}

// ---- Pose ------------------------------------------------------------------
Eigen::Matrix4d Frame::Tcw() const {
    std::lock_guard<std::mutex> lk(_lock_pose);
    return _Tcw;
}

Eigen::Matrix4d Frame::Twc() const {
    std::lock_guard<std::mutex> lk(_lock_pose);
    return _Tcw.inverse();
}

Eigen::Vector3d Frame::Ow() const {
    std::lock_guard<std::mutex> lk(_lock_pose);
    Eigen::Matrix3d Rwc = _Tcw.block<3,3>(0,0).transpose();
    Eigen::Vector3d tcw = _Tcw.block<3,1>(0,3);
    return -Rwc * tcw;
}

void Frame::update_pose(const Eigen::Matrix4d &Tcw_new) {
    std::lock_guard<std::mutex> lk(_lock_pose);
    _Tcw = Tcw_new;
}

void Frame::update_pose_from_g2o(py::object isometry3d) {
    // Accepts g2o.Isometry3d — extract 4x4 matrix
    try {
        auto mat_obj = isometry3d.attr("matrix")();
        auto arr = mat_obj.cast<py::array_t<double>>();
        auto r = arr.unchecked<2>();
        Eigen::Matrix4d T;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                T(i, j) = r(i, j);
        std::lock_guard<std::mutex> lk(_lock_pose);
        _Tcw = T;
    } catch (...) {}
}

// ---- Point match management ------------------------------------------------
py::object Frame::get_point_match(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(points.size())) return py::none();
    return points[idx];
}

void Frame::set_point_match(py::object mp, int idx) {
    if (idx < 0 || idx >= static_cast<int>(points.size())) return;
    points[idx] = mp;
}

void Frame::remove_point_match(int idx) {
    if (idx < 0 || idx >= static_cast<int>(points.size())) return;
    points[idx] = py::none();
}

void Frame::remove_point(py::object mp) {
    for (auto &p : points) {
        if (!p.is_none() && p.ptr() == mp.ptr()) p = py::none();
    }
}

void Frame::replace_point_match(py::object old_mp, py::object new_mp) {
    for (auto &p : points) {
        if (!p.is_none() && p.ptr() == old_mp.ptr()) p = new_mp;
    }
}

void Frame::reset_points() {
    py::object none = py::none();
    size_t n = points.size();
    for (size_t i = 0; i < n; ++i) points[i] = none;
    std::fill(outliers.begin(), outliers.end(), false);
}

// ---- Projection (delegate to Python camera) --------------------------------
py::object Frame::project_points_py(py::array_t<double> pts3d) const {
    if (camera.is_none()) return py::none();
    try {
        Eigen::Matrix4d Tcw_ = Tcw();
        // Transform to camera frame
        // Camera.project() expects (N, 3) array
        return camera.attr("project")(pts3d);
    } catch (...) { return py::none(); }
}

// ---- Repr ------------------------------------------------------------------
std::string Frame::__repr__() const {
    std::ostringstream ss;
    ss << "Frame(id=" << id << ", n_kps=" << num_kps() << ")";
    return ss.str();
}

} // namespace slam
