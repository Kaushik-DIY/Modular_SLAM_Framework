#pragma once
// Minimal 2D radius-search kd-tree over a frame's undistorted keypoints (kpsu).
//
// This is OUR thin wrapper around the vendored, BSD-licensed nanoflann library
// (src/utils/nanoflann.hpp). It is NOT derived from pySLAM's GPL ckdtree_eigen.h
// — only the small, standard nanoflann usage pattern is reused. It replaces the
// per-frame scipy cKDTree used by the Python projection matcher, providing the
// same query_ball_point(x, y, r) -> sorted indices semantics so the C++ matcher
// produces results identical to the Python path.
#include "utils/nanoflann.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace cppcore {

// nanoflann point-cloud adaptor over an owned row-major (N,2) float buffer.
struct Kpsu2DAdaptor {
    std::vector<float> coords;  // x0,y0,x1,y1,... (owned, stable)

    inline std::size_t kdtree_get_point_count() const { return coords.size() / 2; }
    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const {
        return coords[idx * 2 + dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
};

class KdTree2D {
  public:
    using Metric = nanoflann::L2_Simple_Adaptor<float, Kpsu2DAdaptor>;
    using Index = nanoflann::KDTreeSingleIndexAdaptor<Metric, Kpsu2DAdaptor, 2, std::size_t>;

    KdTree2D() = default;

    // Build from a contiguous row-major (N,2) float buffer (e.g. Frame.kpsu).
    void build(const float *xy, std::size_t n, std::size_t leaf_max_size = 10) {
        adaptor_.coords.assign(xy, xy + n * 2);
        n_ = n;
        index_ = std::make_unique<Index>(2, adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index_->buildIndex();
    }

    bool ready() const { return static_cast<bool>(index_); }
    std::size_t size() const { return n_; }

    // Indices of all keypoints within radius r of (x, y), sorted ascending —
    // matching scipy cKDTree.query_ball_point ordering the Python matcher relies on.
    std::vector<int> query_ball_point(float x, float y, float r) const {
        std::vector<int> out;
        if (!index_ || n_ == 0) return out;
        const float q[2] = {x, y};
        const float r2 = r * r;  // L2_Simple_Adaptor works in squared distance
        std::vector<nanoflann::ResultItem<std::size_t, float>> matches;
        index_->radiusSearch(q, r2, matches, nanoflann::SearchParameters());
        out.reserve(matches.size());
        for (const auto &m : matches) out.push_back(static_cast<int>(m.first));
        std::sort(out.begin(), out.end());
        return out;
    }

  private:
    std::size_t n_ = 0;
    Kpsu2DAdaptor adaptor_;
    std::unique_ptr<Index> index_;
};

}  // namespace cppcore
