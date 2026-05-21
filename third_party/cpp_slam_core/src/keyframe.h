#pragma once

#include "frame.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace slam {

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;

// Python object hash by pointer identity (same as Python id()).
// Safe for KF references because KF objects live for the entire SLAM run.
struct PyObjHash {
    std::size_t operator()(const py::object &o) const noexcept {
        return std::hash<void *>{}(o.ptr());
    }
};
struct PyObjEqual {
    bool operator()(const py::object &a, const py::object &b) const noexcept {
        return a.ptr() == b.ptr();
    }
};

// C++ KeyFrame — inherits Frame + adds covisibility graph.
// Cross-KF references (covisibility entries, parent, children, loop edges)
// are stored as py::object (Phase 3: all KFs are C++ wrapped in Python).
class KeyFrame : public Frame {
  public:
    // ---- Identity ----------------------------------------------------------
    int kid;          // KeyFrame-specific ID (typically = Frame::id at creation)

    // ---- Status ------------------------------------------------------------
    std::atomic<bool> _kf_is_bad{false};
    bool to_be_erased_kf{false};
    bool not_to_erase{false};
    bool is_keyframe{true};

    // ---- Reference to Python Map object ------------------------------------
    py::object map;

    // ---- GBA fields --------------------------------------------------------
    int GBA_kf_id{0};
    bool is_Tcw_GBA_valid{false};
    py::object Tcw_GBA;           // np.ndarray or None
    py::object Tcw_before_GBA;   // np.ndarray or None

    // ---- Loop / reloc scoring fields ---------------------------------------
    int loop_query_id{-1};
    int num_loop_words{0};
    float loop_score{0.0f};
    int reloc_query_id{-1};
    int num_reloc_words{0};
    float reloc_score{0.0f};

    // ---- BoW fields (Python) -----------------------------------------------
    // bow_vector and feat_vector inherited from Frame (py::object).

  public:
    mutable std::mutex _lock_connections;

    // Covisibility graph: other_kf_pyobj → co-visibility weight
    std::unordered_map<py::object, int, PyObjHash, PyObjEqual> _covis_weights;

    // Ordered covisibility (sorted descending by weight), updated on change
    std::vector<std::pair<py::object, int>> _ordered_covis;

  private:

    // Spanning tree
    bool _init_parent{false};
    bool _is_first_connection{true};
    py::object _parent;      // Python KF or None
    std::vector<py::object> _children;

    // Loop edges
    std::vector<py::object> _loop_edges;

  public:
    // ---- Construction ------------------------------------------------------
    explicit KeyFrame(int given_kid = -1, int given_frame_id = -1);

    KeyFrame(const KeyFrame &) = delete;
    KeyFrame &operator=(const KeyFrame &) = delete;

    // ---- Status ------------------------------------------------------------
    bool is_bad() const { return _kf_is_bad.load(); }
    void set_bad();
    void set_not_erase();
    void set_erase();

    // ---- Covisibility graph ------------------------------------------------
    void add_connection(py::object other_kf, int weight);
    void add_connection_no_lock_(py::object other_kf, int weight);
    void erase_connection(py::object other_kf);
    void erase_connection_no_lock_(py::object other_kf);

    std::vector<py::object> get_connected_keyframes() const;
    std::vector<py::object> get_covisible_keyframes() const;
    std::vector<py::object> get_best_covisible_keyframes(int N) const;
    std::vector<py::object> get_covisible_by_weight(int min_weight) const;
    int get_weight(py::object other_kf) const;

    void reset_covisibility();
    void update_connections();    // main covisibility rebuild — called from 8 sites

    // ---- Spanning tree -----------------------------------------------------
    void set_parent(py::object kf);
    void set_parent_no_lock_(py::object kf);
    py::object get_parent() const;
    void add_child_no_lock_(py::object kf);
    void add_child(py::object kf);
    void erase_child(py::object kf);
    void erase_child_no_lock_(py::object kf);
    std::vector<py::object> get_children() const;
    bool has_child(py::object kf) const;

    // ---- Loop edges --------------------------------------------------------
    void add_loop_edge(py::object kf);
    std::vector<py::object> get_loop_edges() const;

    // ---- Helper -----------------------------------------------------------
    std::vector<py::object> get_matched_good_points() const;
    // Returns (point, idx) pairs for all non-None, non-bad matched points.
    std::vector<std::pair<py::object, int>> get_matched_good_points_and_idxs() const;
    // Returns self.points as a Python list (compatibility with Python KeyFrame.get_points()).
    std::vector<py::object> get_points() const;
    // Count matched points with at least min_obs observations.
    int num_tracked_points(int min_obs = 0) const;

    // ---- Repr --------------------------------------------------------------
    std::string __repr__() const;

  private:
    void _rebuild_ordered_covis_no_lock_();
};

} // namespace slam
