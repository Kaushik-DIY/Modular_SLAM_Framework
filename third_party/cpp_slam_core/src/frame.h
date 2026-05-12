#pragma once

#include <Eigen/Dense>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/core.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace slam {

class Frame;
using FramePtr = std::shared_ptr<Frame>;

// Row-major (N,2) float32 — matches numpy's default memory layout for kpsu.
using MatNx2f = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;

class Frame : public std::enable_shared_from_this<Frame> {
  public:
    // ---- Global ID management ----------------------------------------------
    static std::atomic<int> _next_id;
    static std::mutex _id_lock;
    static int next_id();
    static void set_id(int id);

    // ---- Identity ----------------------------------------------------------
    int id;
    int img_id = -1;
    double timestamp = 0.0;

    // ---- Camera (Python object) -------------------------------------------
    py::object camera;   // Python Camera (fx, fy, cx, cy, bf, width, height)

    // ---- Feature arrays (C++, normalized at construction) -----------------
    // kpsu: undistorted keypoint (x,y) — always float32 (N, 2) row-major
    MatNx2f kpsu;
    // kps_ur: right stereo coord (or -1 for monocular) — float32 (N,)
    Eigen::VectorXf kps_ur;
    // octaves: scale level — int32 (N,)
    Eigen::VectorXi octaves;
    // des: descriptors — uint8 (N, 32)
    cv::Mat des;
    // kps: original cv2.KeyPoint list (kept for compatibility, Python)
    py::list kps;

    // ---- Map associations --------------------------------------------------
    // points[i] = matched MapPoint (py::object — Python or C++ MapPoint)
    std::vector<py::object> points;
    // outliers[i] = True if feature i is an outlier during tracking
    std::vector<bool> outliers;

    // ---- Pose (Tcw convention: pc = Tcw * pw) ------------------------------
    // Protected by _lock_pose for thread safety.
    mutable std::mutex _lock_pose;
    Eigen::Matrix4d _Tcw;

    // ---- Depth array (RGBD / stereo) — py::object holding np.ndarray or None
    py::object depths;

    // ---- BoW / global descriptors (Python) ---------------------------------
    py::object bow_vector;
    py::object feat_vector;
    py::object g_des;
    py::object f_des;

  public:
    // ---- Construction ------------------------------------------------------
    explicit Frame(int given_id = -1);

    // Virtual destructor (required for polymorphism + pybind11 inheritance)
    virtual ~Frame() = default;

    // Delete copy/move — use shared_ptr
    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;

    // ---- Feature array initialization (call after setting kps, des, etc.) --
    // Normalizes kpsu: accepts cv2.KeyPoint list or numpy (N,2) array.
    // This permanently fixes the cv2.KeyPoint subscriptability issue.
    void init_feature_arrays(py::object kps_in, py::array_t<uint8_t> des_in,
                              py::object kps_ur_in, py::object octaves_in,
                              int n_features);

    // ---- kpsu helpers (GIL-free after init) --------------------------------
    int num_kps() const { return static_cast<int>(kpsu.rows()); }

    // ---- Pose access (GIL-free) --------------------------------------------
    Eigen::Matrix4d Tcw() const;
    Eigen::Matrix4d Twc() const;
    Eigen::Vector3d Ow() const;   // camera center in world

    void update_pose(const Eigen::Matrix4d &Tcw_new);
    void update_pose_from_g2o(py::object isometry3d);  // accepts g2o.Isometry3d

    // ---- Point match management (GIL needed — py::object) -----------------
    py::object get_point_match(int idx) const;
    void set_point_match(py::object mp, int idx);
    void remove_point_match(int idx);
    void remove_point(py::object mp);
    void replace_point_match(py::object old_mp, py::object new_mp);
    void reset_points();

    // ---- Projection (GIL needed for camera intrinsics if camera is Python) -
    // These delegate to Python camera object for now.
    py::object project_points_py(py::array_t<double> pts3d) const;

    // ---- Repr --------------------------------------------------------------
    std::string __repr__() const;

  private:
    static MatNx2f _normalize_kpsu(py::object kps_in, int n);
    static Eigen::VectorXf _normalize_kps_ur(py::object kps_ur_in, int n);
    static Eigen::VectorXi _normalize_octaves(py::object octaves_in, int n);
    static Eigen::VectorXi _extract_octaves_from_kps(py::object kps_in, int n);
};

} // namespace slam
