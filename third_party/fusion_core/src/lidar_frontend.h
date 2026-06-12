#pragma once
// Native LiDAR front-end (V4.4): per-scan odometry with BOTH local-mapping
// variants selectable — scan_to_submap (two-stage correlative + GN refine on
// rotating Cartographer-style submaps) and scan_to_map (coarse-to-fine GN on a
// single growing map pyramid). Port of the Python hector stack
// (slam_core/matching/{scan_to_submap,scan_to_map}.py + hector/adapter.py +
// slam_core/fusion2/lidar_frontend.py); parity-critical semantics are ported
// verbatim — see the V4 plan's findings list before "improving" anything.
//
// P1: voxel filters + extrapolator. P2: scan_to_submap. P3: scan_to_map.

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "common.h"
#include "pose_extrapolator_cv.h"

namespace fusion {

using MatX2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

// ---------------------------------------------------------------------------
// Voxel preprocessing (port of slam_core/matching/preprocessing.py).
// Math in double; centroids emitted in SORTED voxel-key order (np.unique
// lexicographic parity — downstream stride subsampling depends on row order).
// ---------------------------------------------------------------------------

struct VoxelFilterConfig {
    bool enabled = true;
    double fixed_size = 0.03;          // VOXEL_FIXED_SIZE (lab)
    double adaptive_max_size = 0.10;   // VOXEL_ADAPTIVE_MAX_SIZE
    int adaptive_min_points = 200;     // VOXEL_ADAPTIVE_MIN_POINTS
    int adaptive_iters = 6;            // VOXEL_ADAPTIVE_ITERS
};

MatX2d fixed_voxel_filter(const MatX2d& pts, double voxel_size);
MatX2d adaptive_voxel_filter(const MatX2d& pts, double max_voxel_size,
                             int min_num_points, int num_iterations);
// fixed then adaptive (PointCloudProcessor.process semantics)
MatX2d voxel_preprocess(const MatX2d& pts, const VoxelFilterConfig& cfg);

// ---------------------------------------------------------------------------
// Front-end orchestrator (port of hector/adapter.py process_scan +
// fusion2 LidarFrontend keyframe logic).
// ---------------------------------------------------------------------------

enum class LidarMatcherKind : int { SCAN_TO_SUBMAP = 0, SCAN_TO_MAP = 1 };

// Adapter insert filter. DORMANT by default — the fusion2 wiring passes
// motion_params=None (match + insert every scan), which is the parity baseline.
struct MotionFilterConfig {
    bool enabled = false;
    bool skip_below = false;            // motion_filter_skip (dead-reckon path)
    double max_time_s = 0.5;
    double max_dist_m = 0.10;
    double max_angle_rad = 0.0349;      // 2 deg
};

struct KeyframeConfig {                 // fusion2 keyframe decision
    double min_dist_m = 0.25;
    double min_angle_rad = 0.2094;      // 12 deg
    double min_dt_s = 2.0;
};

struct LidarFrontendConfig;             // fwd (defined below with both matchers)

struct LidarScanResult {
    Pose2 pose;                         // final world pose (matched or fallback)
    double score = -1.0;                // s2s: coarse correlative confidence
    double refined_score = -1.0;
    bool matched = false;
    bool fallback = false;              // matcher failed -> IMU/CV prediction
    bool skipped = false;               // motion-filter skip path
    bool inserted = false;
    bool is_keyframe = false;
    int num_points = 0;                 // after voxel filter
    double process_ms = 0.0;
};

}  // namespace fusion

#include "scan_to_map_2d.h"
#include "scan_to_submap_2d.h"

namespace fusion {

struct LidarFrontendConfig {
    LidarMatcherKind matcher = LidarMatcherKind::SCAN_TO_SUBMAP;
    VoxelFilterConfig voxel;
    ExtrapolatorConfig extrap;
    SubmapBuilderConfig submap_builder;
    SubmapMatcherConfig submap_matcher;
    MapMatcherConfig map_matcher;       // scan_to_map variant (P3)
    MotionFilterConfig insert_filter;
    KeyframeConfig keyframe;
};

class LidarFrontend {
public:
    explicit LidarFrontend(LidarFrontendConfig cfg);

    void add_imu(double t, double wz, bool has_yaw, double yaw) {
        _extrap.add_imu(t, wz, has_yaw, yaw);
    }
    LidarScanResult process(const MatX2d& scan_xy, double t);
    const MatX2d& last_filtered_points() const { return _last_pts; }

private:
    LidarScanResult process_s2s(const MatX2d& pts, double t, const Pose2& pred,
                                bool do_insert);
    LidarScanResult process_s2m(const MatX2d& pts, double t, const Pose2& pred,
                                bool do_insert);
    bool keyframe_decision(const Pose2& pose, double t);

    LidarFrontendConfig _cfg;
    PoseExtrapolatorCV _extrap;
    std::unique_ptr<SubmapBuilder2D> _submaps;
    std::unique_ptr<MapMatcher> _s2m;
    MatX2d _last_pts;
    Pose2 _last_kf_pose;
    double _last_kf_t = 0.0;
    bool _has_kf = false;
    Pose2 _last_insert_pose;
    double _last_insert_t = 0.0;
    bool _has_insert = false;
    long _k = -1;
};

}  // namespace fusion
