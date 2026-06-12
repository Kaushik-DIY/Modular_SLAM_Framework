#pragma once
// Signature — one fused keyframe in the shared multi-modal map.
//
// Design rules (fusion v2):
//  * ALL payload data is C++-owned in packed arrays (Eigen row-major / cv::Mat).
//    No per-point objects, no per-object Python dicts — this is the deliberate
//    antidote to the Python map's ~10 GB object-graph overhead.
//  * Payloads are COPIED IN at construction (from the front-ends' numpy arrays)
//    so map lifetime is independent of front-end keyframe culling.
//  * Python receives zero-copy numpy VIEWS (base = the owning Signature object).

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "common.h"

namespace fusion {

enum class LinkType : uint8_t {
    NEIGHBOR = 0,  // consecutive keyframes (graph spine)
    LOOP = 1,      // accepted loop closure
    MERGED = 2,    // rehearsal merge record (kept for traceability)
};

struct Link {
    int32_t to_id = -1;
    LinkType type = LinkType::NEIGHBOR;
    Pose2 rel_pose;            // T_from^{-1} * T_to
    double trans_info = 0.0;   // translation information (weight)
    double rot_info = 0.0;     // rotation information (weight)
};

class Signature {
public:
    int32_t id = -1;
    double stamp = 0.0;
    int32_t weight = 0;        // RTAB weight (rehearsal / loop bumps)
    Pose2 pose;                // current optimized SE(2) pose

    // Visual payload (empty when frame had no synced camera data)
    MatX2f kpts;               // N x 2 float32 — undistorted pixel coords
    cv::Mat des;               // N x 32 uint8  — ORB descriptors
    MatX3f pts3d;              // N x 3 float32 — camera/world 3D points for PnP

    // LiDAR payload (empty when no scan inside the sync window)
    MatX2f scan_xy;            // M x 2 float32 — scan points in sensor frame

    std::vector<Link> links;

    bool has_visual() const { return des.rows > 0; }
    bool has_scan() const { return scan_xy.rows() > 0; }
    int num_kpts() const { return static_cast<int>(kpts.rows()); }
    int num_scan_points() const { return static_cast<int>(scan_xy.rows()); }

    // Payload footprint in bytes (diagnostics / memory accounting).
    size_t payload_bytes() const {
        return static_cast<size_t>(kpts.size()) * sizeof(float) +
               static_cast<size_t>(des.total()) * des.elemSize() +
               static_cast<size_t>(pts3d.size()) * sizeof(float) +
               static_cast<size_t>(scan_xy.size()) * sizeof(float) +
               links.size() * sizeof(Link);
    }
};

using SignaturePtr = std::shared_ptr<Signature>;

}  // namespace fusion
