#pragma once
// VoFrontend — lean windowed RGB-D visual odometry (fusion v3).
//
// Purpose-built for the fusion architecture: the front-end provides
// drift-bounded odometry + keyframe payloads; the shared map owns global
// memory and loop closure. There is no global map here — just a sliding
// window of K keyframes and the map points they observe. Window eviction IS
// the long-term memory policy (~1-2 MB resident).
//
// v3.5: the window runs a local sliding-window BUNDLE ADJUSTMENT after each
// keyframe (jointly refining the windowed camera poses + the map points they
// co-observe, oldest KF fixed as gauge). This is the missing ingredient the
// reference ORB-SLAM2 used to make a coherent map with loops/global-BA OFF
// (1361 local-BA runs): motion-only tracking alone is NOT drift-bounded over
// hundreds of keyframes. Map points are now shared across keyframes (a point
// gains an observation whenever a later frame matches it) so BA has the
// multi-view constraints that tie consecutive poses together.
//
// Parameter values port the proven Python stack
// (visual_slam/orbslam/slam/config_parameters.py): ORB 2000 feats / 8 levels /
// 1.2 scale; projection radius 7 px (fallback 15); ratio 0.9; KF gates >=10
// frames AND (matched < 0.75*ref OR close-point starvation), hard cap 30
// frames; <=150 depth points per KF within [0.01, depth_max].

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "common.h"

namespace fusion {

struct VoConfig {
    // extraction
    int n_features = 2000;
    float scale_factor = 1.2f;
    int n_levels = 8;
    int grid_cell_px = 48;          // bucketing cell for spatial distribution
    int max_per_cell = 6;
    // camera (set from Python sensor_config)
    double fx = 600, fy = 600, cx = 320, cy = 240;
    double depth_factor = 1000.0;   // u16 -> metres
    double depth_min = 0.01;
    double depth_max = 5.0;         // ~ camera.depth_threshold
    // window / points
    int window_size = 7;
    int max_points_per_kf = 150;
    // matching
    double match_radius_px = 7.0;
    double match_radius_fallback_px = 15.0;
    int max_hamming = 100;          // TH_HIGH
    double ratio_test = 0.9;
    // pose-free brute-force recovery (strict: no spatial gating)
    int bf_max_hamming = 50;        // TH_LOW
    double bf_ratio_test = 0.75;
    double bf_pnp_reproj_px = 4.0;
    // pose optimization
    int opt_rounds = 2;
    int opt_iters_per_round = 10;
    double huber_delta_px = 2.447;  // sqrt(5.991)
    double chi2_outlier = 5.991;    // 2-DoF 95%
    // local bundle adjustment
    bool enable_local_ba = true;
    int local_ba_iters = 5;
    int min_obs_for_ba = 2;         // a point constrains poses only if seen >=2x
    int local_ba_max_points = 300;  // cap active points (most-observed kept) -> bounded cost
    // state thresholds
    int min_inliers_ok = 20;
    int min_inliers_reinit = 15;
    // brief tracking failures (blur/occlusion) coast on the prediction and keep
    // the window alive; only after this many consecutive failures do we hard
    // depth-reinit (which discards the window and injects a discontinuity).
    int reinit_patience = 3;
    // keyframe policy
    int kf_min_frames = 10;
    int kf_max_frames = 30;
    double kf_ref_ratio = 0.75;
    int kf_min_close_points = 100;
};

enum class VoState : uint8_t { INIT = 0, OK = 1, WEAK = 2, REINIT = 3 };

struct VoKeyframe {
    int id = -1;
    double stamp = 0.0;
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();  // refined by local BA
    MatX2f kpts;                 // all extracted keypoints (payload)
    cv::Mat des;                 // N x 32
    MatX3f pts3d_cam;            // per-kpt camera-frame 3D (NaN = no depth)
    std::vector<int> octave;     // per-kpt pyramid level
};
using VoKeyframePtr = std::shared_ptr<VoKeyframe>;

// One observation of a window map point: keyframe id + keypoint index in it.
struct VoObs { int kf_id; int kp_idx; };

// A window map point: world position (refined by BA) + the keyframes that see
// it. Created from depth at its owner KF; a later frame matching it adds an obs.
struct VoMapPoint {
    Eigen::Vector3d pw = Eigen::Vector3d::Zero();
    std::array<uint8_t, 32> des{};   // representative descriptor (owner's); no alloc
    int octave = 0;
    std::vector<VoObs> obs;
};

struct VoResult {
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prev_Twc = Eigen::Matrix4d::Identity();  // prev emitted KF, refined
    bool has_prev = false;       // prev_Twc valid (false on the first/reinit KF)
    VoState state = VoState::INIT;
    bool new_keyframe = false;
    int kf_id = -1;              // valid when new_keyframe
    int n_matches = 0;
    int n_inliers = 0;
    double track_ms = 0.0;
};

class VoFrontend {
public:
    explicit VoFrontend(VoConfig cfg);

    // gray: 8UC1, depth: 16UC1 (raw, depth_factor applies). has_prior selects
    // whether prior_Twc seeds the pose (else constant-velocity from history).
    VoResult track(const cv::Mat& gray, const cv::Mat& depth, double stamp,
                   const Eigen::Matrix4d& prior_Twc, bool has_prior);

    const VoKeyframe* keyframe(int id) const;   // payload access for Python
    int window_count() const { return static_cast<int>(_window.size()); }
    size_t window_bytes() const;

private:
    void extract(const cv::Mat& gray, std::vector<cv::KeyPoint>& kps, cv::Mat& des) const;
    // Creates a keyframe, attaches new-frame observations to matched window
    // points (inlier matches into the gathered arrays) and spawns new map
    // points for unmatched depth keypoints. Slides the window + prunes points.
    VoKeyframePtr make_keyframe(double stamp, const Eigen::Matrix4d& Twc,
                                const std::vector<cv::KeyPoint>& kps,
                                const cv::Mat& des, const cv::Mat& depth,
                                const std::vector<std::pair<int, int>>& matches,
                                const std::vector<bool>& inlier_mask);
    // returns matches as (map-point idx into gathered arrays, kp idx)
    int match_window(const std::vector<cv::KeyPoint>& kps, const cv::Mat& des,
                     const Eigen::Matrix4d& Twc_pred, double radius_px,
                     std::vector<std::pair<int, int>>& matches) const;
    // pose-FREE fallback when projection matching fails (wrong motion model
    // during fast turns): strict brute-force descriptor matching against all
    // window points — no search window, so it survives any prediction error.
    int match_bruteforce(const std::vector<cv::KeyPoint>& kps, const cv::Mat& des,
                         std::vector<std::pair<int, int>>& matches) const;
    // PnP RANSAC over brute-force matches -> Tcw (no prior needed)
    bool recover_pose(const std::vector<cv::KeyPoint>& kps,
                      const std::vector<std::pair<int, int>>& matches,
                      Eigen::Matrix4d& Tcw) const;
    // motion-only GN; returns inliers, updates Tcw
    int optimize_pose(const std::vector<cv::KeyPoint>& kps,
                      const std::vector<std::pair<int, int>>& matches,
                      Eigen::Matrix4d& Tcw, std::vector<bool>& inlier_mask) const;
    // local sliding-window BA: refine window KF poses (oldest fixed) + the
    // points they co-observe; writes back kf->Twc and _points[].pw.
    void local_ba();
    void rebuild_gathered();     // gathered arrays mirror _points 1:1 (row==idx)

    VoConfig _cfg;
    cv::Ptr<cv::ORB> _orb;
    std::deque<VoKeyframePtr> _window;
    std::vector<VoMapPoint> _points;          // window local map
    std::vector<float> _level_sigma2;         // (1.2^2)^octave
    // gathered point caches (rebuilt on window/point change); row p == _points[p]
    MatX3f _g_world;
    cv::Mat _g_des;
    std::vector<int> _g_octave;

    Eigen::Matrix4d _last_Twc = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d _velocity = Eigen::Matrix4d::Identity();  // Twc_prev^-1 * Twc
    bool _has_pose = false;
    int _next_kf_id = 0;
    int _frames_since_kf = 0;
    int _consec_fail = 0;                     // consecutive sub-threshold frames
    int _ref_matched = 0;                     // matches at last KF creation
    int _last_emit_kf_id = -1;                // previously emitted keyframe id
    Eigen::Matrix4d _prev_emit_Twc = Eigen::Matrix4d::Identity();  // last emitted KF pose
    bool _have_prev_emit = false;             // _prev_emit_Twc valid (across reinit)
};

}  // namespace fusion
