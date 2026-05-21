#pragma once

#include <Eigen/Dense>
#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace slam {

class MapPoint;
using MapPointPtr = std::shared_ptr<MapPoint>;

// Comparator for py::object keys using Python's id() (memory address).
// Safe for KeyFrame refs because KF objects live for the entire SLAM run.
struct PyObjCompare {
    bool operator()(const py::object &a, const py::object &b) const {
        return reinterpret_cast<std::uintptr_t>(a.ptr()) <
               reinterpret_cast<std::uintptr_t>(b.ptr());
    }
};

class MapPoint : public std::enable_shared_from_this<MapPoint> {
  public:
    // ---- Global ID management (thread-safe) --------------------------------
    static std::atomic<int> _next_id;
    static std::mutex _id_lock;
    static int next_id();
    static void set_id(int id);

    // ---- Identity ----------------------------------------------------------
    int id;

    // ---- Geometric data (C++, GIL-free) ------------------------------------
    mutable std::mutex _lock_pos;
    Eigen::Vector3d _pos;
    Eigen::Vector3d normal;
    float _min_distance;
    float _max_distance;

    // ---- Visual data -------------------------------------------------------
    mutable std::mutex _lock_features;
    cv::Mat des;  // best descriptor (ORB: 32 bytes)

    // ---- Status (atomic, GIL-free) -----------------------------------------
    std::atomic<bool> _is_bad{false};
    std::atomic<bool> to_be_erased{false};

    // ---- Statistics --------------------------------------------------------
    int _num_observations;
    int num_times_visible;
    int num_times_found;
    int last_frame_id_seen;
    int lba_count;

    // ---- Observations: py::object KF key → feature index ------------------
    // Phase-1: KF refs are Python objects. Phase-3 upgrade: KeyFramePtr.
    // Keyed by Python object pointer (stable over object lifetime).
    std::map<py::object, int, PyObjCompare> _observations;

    // ---- Other Python references -------------------------------------------
    py::object map;         // Python Map
    py::object kf_ref;      // reference KeyFrame (Python)
    py::object replacement; // MapPoint replacing this (Python or None)
    int corrected_by_kf;
    int corrected_reference;
    py::object rgb;         // (r, g, b) tuple or None

    // ---- GBA support -------------------------------------------------------
    Eigen::Vector3d pt_GBA;
    bool is_pt_GBA_valid;
    int GBA_kf_id;

  public:
    // ---- Construction ------------------------------------------------------
    explicit MapPoint(int given_id = -1);

    // Delete copy/move to enforce shared_ptr ownership
    MapPoint(const MapPoint &) = delete;
    MapPoint &operator=(const MapPoint &) = delete;

    // ---- Position access (GIL-free) ----------------------------------------
    Eigen::Vector3d get_position() const;
    void update_position(const Eigen::Vector3d &pos);

    float min_distance() const;
    float max_distance() const;

    // ---- Descriptor (GIL-free) ---------------------------------------------
    cv::Mat get_descriptor() const;
    float min_des_distance(const cv::Mat &query_des) const;

    // ---- Observation management (GIL needed — touches Python KFs) ----------
    bool add_observation(py::object kf, int idx);
    void remove_observation(py::object kf, int idx = -1, bool map_no_lock = false);
    std::vector<std::pair<py::object, int>> observations() const;
    std::vector<py::object> keyframes() const;
    bool is_in_keyframe(py::object kf) const;
    int get_observation_idx(py::object kf) const;
    int num_observations() const;

    // ---- Frame views (transient, not persisted) ----------------------------
    std::map<py::object, int, PyObjCompare> _frame_views;
    bool add_frame_view(py::object frame, int idx);
    void remove_frame_view(py::object frame, int idx = -1);
    bool is_in_frame(py::object frame) const;
    int get_frame_view_idx(py::object frame) const;
    std::vector<std::pair<py::object, int>> frame_views() const;
    std::vector<py::object> frames() const;

    // ---- Combined status helpers (GIL held) --------------------------------
    bool is_bad_or_is_in_keyframe(py::object kf) const;
    bool is_good_with_min_obs(int min_obs) const;

    // ---- Descriptor write --------------------------------------------------
    void set_des(const py::array_t<uint8_t> &arr);

    // ---- Status (GIL-free reads, GIL needed for set_bad callbacks) ---------
    bool is_bad() const { return _is_bad.load(); }
    void set_bad(bool map_no_lock = false);
    void replace_with(MapPointPtr p);
    MapPointPtr get_replacement() const;

    // ---- Info update (GIL needed — accesses kf.kpsu, kf.des, kf.octaves) --
    void update_info();
    void update_normal_and_depth(bool force = false);
    void update_best_descriptor(bool force = false);

    // ---- Scale prediction --------------------------------------------------
    int predict_detection_level(float dist) const;

    // ---- Statistics --------------------------------------------------------
    void increase_visible(int n = 1);
    void increase_found(int n = 1);
    float get_found_ratio() const;

    // ---- Repr --------------------------------------------------------------
    std::string __repr__() const;

  private:
    mutable std::mutex _lock_replacement;
    MapPointPtr _replacement_cpp; // nullptr except during replace_with

    static float _hamming_distance(const cv::Mat &a, const cv::Mat &b);
};

} // namespace slam
