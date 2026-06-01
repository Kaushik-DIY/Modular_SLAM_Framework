/*
 * slam_optimizer_core — GIL-free C++ g2o bundle adjustment for the Modular SLAM Framework.
 *
 * Adapted from pySLAM's optimizer_g2o.cpp (LGPL-3.0).
 * All g2o API calls are unchanged from pySLAM; only data source changes
 * from C++ KeyFrame/MapPoint objects to flat numpy arrays.
 *
 * Exported functions:
 *   run_local_ba(kf_poses, kf_ids, kf_fixed, point_pos, observations, camera,
 *                rounds, use_robust_kernel, prune_outliers)
 *   run_global_ba(kf_poses, kf_ids, point_pos, observations, camera,
 *                 rounds, use_robust_kernel, loop_kf_id)
 *   set_abort(bool) / get_abort() — module-level abort flag
 *   hello()        — smoke test
 *
 * Array interface (float64 row-major unless noted):
 *   kf_poses       (N, 16)   row-major 4×4 Tcw matrices
 *   kf_ids         (N,)      int64    KF IDs — vertex id = kid*2
 *   kf_fixed       (N,)      uint8    1 = fixed (do not optimise)
 *   point_pos      (M, 3)    float64
 *   observations   (K, 8)    float64  [kf_row, pt_row, u, v, ur, octave, inv_sigma2, is_stereo]
 *   camera         (5,)      float64  [fx, fy, cx, cy, bf]
 *
 * Returns py::dict:
 *   updated_poses    (N, 16)  float64  — input values unchanged for fixed KFs
 *   updated_points   (M, 3)   float64
 *   outlier_mask     (K,)     uint8    1 = outlier after final pass
 *   initial_mse      float    — activeChi2/n_edges before optimisation
 *   mse              float    — activeChi2/n_inliers after optimisation (-1 if aborted)
 *   n_bad_edges      int
 */

#include <cstring>
#include <iostream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Constants — identical to pySLAM config_parameters.h
// ---------------------------------------------------------------------------
static constexpr double kChi2Mono      = 5.991;
static constexpr double kChi2Stereo    = 7.815;
static constexpr double kThHuberMono   = 2.447;   // sqrt(kChi2Mono)
static constexpr double kThHuberStereo = 2.796;   // sqrt(kChi2Stereo)

// Module-level abort flag.
// Plain bool (not atomic) because setForceStopFlag needs a bool*.
// Python side must call set_abort() before launching BA; thread-safety is
// provided by the GIL being held during Python's set_abort() call.
static bool g_abort_flag = false;

// ---------------------------------------------------------------------------
// Helper: row-major 16-elem array → Eigen Matrix4d
// ---------------------------------------------------------------------------
static inline Eigen::Matrix4d mat4(const double *p) {
    Eigen::Matrix4d T;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            T(r, c) = p[r * 4 + c];
    return T;
}

// ---------------------------------------------------------------------------
// Helper: build empty result dict on early abort
// ---------------------------------------------------------------------------
static py::dict abort_result(
    py::array_t<double> kf_poses,
    py::array_t<double> point_pos,
    int K, double initial_mse)
{
    auto out_mask = py::array_t<uint8_t>(K);
    std::memset(out_mask.mutable_data(), 0, K);
    py::dict r;
    r["updated_poses"]  = kf_poses;
    r["updated_points"] = point_pos;
    r["outlier_mask"]   = out_mask;
    r["initial_mse"]    = initial_mse;
    r["mse"]            = -1.0;
    r["n_bad_edges"]    = 0;
    return r;
}

// ---------------------------------------------------------------------------
// run_local_ba — core implementation
// ---------------------------------------------------------------------------
py::dict run_local_ba(
    py::array_t<double,  py::array::c_style | py::array::forcecast> kf_poses,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> kf_ids,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> kf_fixed,
    py::array_t<double,  py::array::c_style | py::array::forcecast> point_pos,
    py::array_t<double,  py::array::c_style | py::array::forcecast> observations,
    py::array_t<double,  py::array::c_style | py::array::forcecast> camera,
    int  rounds,
    bool use_robust_kernel,
    bool prune_outliers)
{
    const int N = (int)kf_poses.shape(0);
    const int M = (int)point_pos.shape(0);
    const int K = (int)observations.shape(0);

    const double  *poses_ptr = kf_poses.data();
    const int64_t *ids_ptr   = kf_ids.data();
    const uint8_t *fixed_ptr = kf_fixed.data();
    const double  *pts_ptr   = point_pos.data();
    const double  *obs_ptr   = observations.data();
    const double  *cam       = camera.data();

    const double fx = cam[0], fy = cam[1], cx = cam[2], cy = cam[3], bf = cam[4];

    // ------------------------------------------------------------------
    // Build g2o optimizer — same setup as pySLAM local_bundle_adjustment
    // ------------------------------------------------------------------
    g2o::SparseOptimizer optimizer;
    {
        auto linearSolver =
            std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
        auto blockSolver =
            std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
        optimizer.setAlgorithm(
            new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));
    }
    // Wire module-level abort flag
    optimizer.setForceStopFlag(&g_abort_flag);

    // ------------------------------------------------------------------
    // KF vertices — vertex id = kid * 2  (even — pySLAM convention)
    // ------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix4d Tcw = mat4(poses_ptr + i * 16);
        auto *v = new g2o::VertexSE3Expmap();
        v->setEstimate(g2o::SE3Quat(Tcw.block<3,3>(0,0), Tcw.block<3,1>(0,3)));
        v->setId((int)(ids_ptr[i] * 2));
        v->setFixed(fixed_ptr[i] != 0);
        optimizer.addVertex(v);
    }

    // ------------------------------------------------------------------
    // Point vertices — vertex id = pt_row * 2 + 1  (odd)
    // ------------------------------------------------------------------
    for (int j = 0; j < M; ++j) {
        auto *v = new g2o::VertexSBAPointXYZ();
        v->setId(j * 2 + 1);
        v->setEstimate(Eigen::Vector3d(pts_ptr[j*3], pts_ptr[j*3+1], pts_ptr[j*3+2]));
        v->setMarginalized(true);
        v->setFixed(false);
        optimizer.addVertex(v);
    }

    // ------------------------------------------------------------------
    // Edges
    // obs row: [kf_row(0), pt_row(1), u(2), v(3), ur(4), octave(5), inv_sigma2(6), is_stereo(7)]
    // ------------------------------------------------------------------
    struct EdgeMono   { g2o::EdgeSE3ProjectXYZ       *edge; int k; };
    struct EdgeStereo { g2o::EdgeStereoSE3ProjectXYZ *edge; int k; };
    std::vector<EdgeMono>   edges_mono;
    std::vector<EdgeStereo> edges_stereo;
    edges_mono.reserve(K);
    edges_stereo.reserve(K);

    for (int k = 0; k < K; ++k) {
        const double *row     = obs_ptr + k * 8;
        const int    kf_row   = (int)row[0];
        const int    pt_row   = (int)row[1];
        const double u        = row[2], v_ = row[3], ur = row[4];
        const double inv_s2   = row[6];
        const bool   is_ste   = (row[7] > 0.5);

        auto *v_kf = static_cast<g2o::VertexSE3Expmap *>(
            optimizer.vertex((int)(ids_ptr[kf_row] * 2)));
        auto *v_pt = static_cast<g2o::VertexSBAPointXYZ *>(
            optimizer.vertex(pt_row * 2 + 1));
        if (!v_kf || !v_pt) continue;

        if (is_ste) {
            auto *edge = new g2o::EdgeStereoSE3ProjectXYZ();
            edge->setVertex(0, v_pt);
            edge->setVertex(1, v_kf);
            edge->setMeasurement(Eigen::Vector3d(u, v_, ur));
            edge->setInformation(Eigen::Matrix3d::Identity() * inv_s2);
            if (use_robust_kernel) {
                auto *rk = new g2o::RobustKernelHuber();
                rk->setDelta(kThHuberStereo);
                edge->setRobustKernel(rk);
            }
            edge->fx = fx; edge->fy = fy; edge->cx = cx; edge->cy = cy; edge->bf = bf;
            optimizer.addEdge(edge);
            edges_stereo.push_back({edge, k});
        } else {
            auto *edge = new g2o::EdgeSE3ProjectXYZ();
            edge->setVertex(0, v_pt);
            edge->setVertex(1, v_kf);
            edge->setMeasurement(Eigen::Vector2d(u, v_));
            edge->setInformation(Eigen::Matrix2d::Identity() * inv_s2);
            if (use_robust_kernel) {
                auto *rk = new g2o::RobustKernelHuber();
                rk->setDelta(kThHuberMono);
                edge->setRobustKernel(rk);
            }
            edge->fx = fx; edge->fy = fy; edge->cx = cx; edge->cy = cy;
            optimizer.addEdge(edge);
            edges_mono.push_back({edge, k});
        }
    }

    const int n_total = (int)(edges_mono.size() + edges_stereo.size());

    // ------------------------------------------------------------------
    // Compute initial chi2 (returned as initial_mse for diagnostics)
    // ------------------------------------------------------------------
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    const double initial_mse = optimizer.activeChi2() / std::max(n_total, 1);

    if (g_abort_flag)
        return abort_result(kf_poses, point_pos, K, initial_mse);

    // ------------------------------------------------------------------
    // Two-pass optimisation — identical to pySLAM local_bundle_adjustment:
    //   Pass 1: 5 iters with robust kernels → detect outliers
    //   Pass 2: 'rounds' iters on inliers only
    // ------------------------------------------------------------------
    int n_bad = 0;
    {
        // Release the GIL for the pure-C++ g2o solve. optimize()/chi2()/setLevel()
        // touch only g2o objects built from the packed input arrays — no Python
        // objects — so this is safe, and it lets the tracking thread run
        // concurrently. Without it the LM thread held the GIL for the entire BA,
        // freezing tracking (a slow/large BA -> multi-minute stall) in threaded mode.
        py::gil_scoped_release release;
        optimizer.optimize(5);

        if (!g_abort_flag) {
            for (auto &em : edges_mono) {
                if (em.edge->chi2() > kChi2Mono || !em.edge->isDepthPositive()) {
                    em.edge->setLevel(1); ++n_bad;
                }
                em.edge->setRobustKernel(nullptr);
            }
            for (auto &es : edges_stereo) {
                if (es.edge->chi2() > kChi2Stereo || !es.edge->isDepthPositive()) {
                    es.edge->setLevel(1); ++n_bad;
                }
                es.edge->setRobustKernel(nullptr);
            }
            optimizer.initializeOptimization(0);
            optimizer.optimize(rounds);
        }
    }

    // ------------------------------------------------------------------
    // Build output arrays
    // ------------------------------------------------------------------
    auto out_poses  = py::array_t<double>({N, 16});
    auto out_points = py::array_t<double>({M, 3});
    auto out_mask   = py::array_t<uint8_t>(K);

    double  *op  = out_poses.mutable_data();
    double  *opp = out_points.mutable_data();
    uint8_t *om  = out_mask.mutable_data();

    // Initialise with input values (fixed KFs / unchanged points stay as-is)
    std::memcpy(op,  poses_ptr, N * 16 * sizeof(double));
    std::memcpy(opp, pts_ptr,   M *  3 * sizeof(double));
    std::memset(om, 0, K);

    // Updated KF poses (non-fixed only)
    for (int i = 0; i < N; ++i) {
        if (fixed_ptr[i]) continue;
        auto *v = static_cast<g2o::VertexSE3Expmap *>(
            optimizer.vertex((int)(ids_ptr[i] * 2)));
        if (!v) continue;
        g2o::SE3Quat est = v->estimate();
        Eigen::Matrix3d R = est.rotation().matrix();
        Eigen::Vector3d t = est.translation();
        double *row = op + i * 16;
        // row-major 4x4: R in columns 0-2, t in column 3
        for (int r = 0; r < 3; ++r) {
            row[r*4+0] = R(r,0);
            row[r*4+1] = R(r,1);
            row[r*4+2] = R(r,2);
            row[r*4+3] = t(r);
        }
        row[12] = 0.0; row[13] = 0.0; row[14] = 0.0; row[15] = 1.0;
    }

    // Updated point positions
    for (int j = 0; j < M; ++j) {
        auto *v = static_cast<g2o::VertexSBAPointXYZ *>(
            optimizer.vertex(j * 2 + 1));
        if (!v) continue;
        Eigen::Vector3d pt = v->estimate();
        opp[j*3]=pt.x(); opp[j*3+1]=pt.y(); opp[j*3+2]=pt.z();
    }

    // Outlier mask (re-evaluate after final pass)
    for (auto &em : edges_mono)
        om[em.k] = (em.edge->chi2() > kChi2Mono || !em.edge->isDepthPositive()) ? 1 : 0;
    for (auto &es : edges_stereo)
        om[es.k] = (es.edge->chi2() > kChi2Stereo || !es.edge->isDepthPositive()) ? 1 : 0;

    const int n_active = n_total - n_bad;
    const double mse   = optimizer.activeChi2() / std::max(n_active, 1);

    py::dict result;
    result["updated_poses"]  = out_poses;
    result["updated_points"] = out_points;
    result["outlier_mask"]   = out_mask;
    result["initial_mse"]    = initial_mse;
    result["mse"]            = mse;
    result["n_bad_edges"]    = n_bad;
    return result;
}

// ---------------------------------------------------------------------------
// run_pose_optimization — motion-only BA: fixed map points, optimise pose only.
//
// Array interface:
//   frame_pose    (16,)   float64  row-major Tcw
//   observations  (K, 8)  float64  [u, v, ur, inv_sigma2, is_stereo, px, py, pz]
//   camera        (5,)    float64  [fx, fy, cx, cy, bf]
//   rounds        int     number of outer rounds (default 4, pySLAM uses 4)
//   iters_per_round int   g2o iterations per round (default 10)
//
// Returns py::dict:
//   updated_pose  (16,)  float64
//   outlier_mask  (K,)   uint8    1 = outlier
//   num_inliers   int
//   mse           float
// ---------------------------------------------------------------------------
py::dict run_pose_optimization(
    py::array_t<double,  py::array::c_style | py::array::forcecast> frame_pose,
    py::array_t<double,  py::array::c_style | py::array::forcecast> observations,
    py::array_t<double,  py::array::c_style | py::array::forcecast> camera,
    int rounds,
    int iters_per_round)
{
    const int K = (int)observations.shape(0);
    const double *pose_ptr = frame_pose.data();
    const double *obs_ptr  = observations.data();
    const double *cam      = camera.data();

    const double fx = cam[0], fy = cam[1], cx = cam[2], cy = cam[3], bf = cam[4];

    // Parse initial pose
    Eigen::Matrix4d Tcw = mat4(pose_ptr);
    Eigen::Matrix3d Rcw = Tcw.block<3,3>(0,0);
    Eigen::Vector3d tcw = Tcw.block<3,1>(0,3);

    // ------------------------------------------------------------------
    // Build optimizer — same solver as local BA (pySLAM uses BlockSolverSE3)
    // ------------------------------------------------------------------
    g2o::SparseOptimizer optimizer;
    {
        auto linearSolver =
            std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
        auto blockSolver =
            std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
        optimizer.setAlgorithm(
            new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));
    }

    // Single pose vertex (id = 0)
    auto *vertex_se3 = new g2o::VertexSE3Expmap();
    vertex_se3->setEstimate(g2o::SE3Quat(Rcw, tcw));
    vertex_se3->setId(0);
    vertex_se3->setFixed(false);
    optimizer.addVertex(vertex_se3);

    // ------------------------------------------------------------------
    // Edges — EdgeSE3ProjectXYZOnlyPose / EdgeStereoSE3ProjectXYZOnlyPose
    // obs row: [u(0), v(1), ur(2), inv_sigma2(3), is_stereo(4), px(5), py(6), pz(7)]
    // ------------------------------------------------------------------
    struct EdgeMono   { g2o::EdgeSE3ProjectXYZOnlyPose       *edge; int k; };
    struct EdgeStereo { g2o::EdgeStereoSE3ProjectXYZOnlyPose *edge; int k; };
    std::vector<EdgeMono>   edges_mono;
    std::vector<EdgeStereo> edges_stereo;
    edges_mono.reserve(K);
    edges_stereo.reserve(K);

    for (int k = 0; k < K; ++k) {
        const double *row  = obs_ptr + k * 8;
        const double u     = row[0], v_ = row[1], ur = row[2];
        const double inv_s2 = row[3];
        const bool   is_ste = (row[4] > 0.5);
        const Eigen::Vector3d Xw(row[5], row[6], row[7]);

        if (is_ste) {
            auto *edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
            edge->setVertex(0, vertex_se3);
            edge->setMeasurement(Eigen::Vector3d(u, v_, ur));
            edge->setInformation(Eigen::Matrix3d::Identity() * inv_s2);
            auto *rk = new g2o::RobustKernelHuber();
            rk->setDelta(kThHuberStereo);
            edge->setRobustKernel(rk);
            edge->fx = fx; edge->fy = fy; edge->cx = cx; edge->cy = cy; edge->bf = bf;
            edge->Xw = Xw;
            optimizer.addEdge(edge);
            edges_stereo.push_back({edge, k});
        } else {
            auto *edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
            edge->setVertex(0, vertex_se3);
            edge->setMeasurement(Eigen::Vector2d(u, v_));
            edge->setInformation(Eigen::Matrix2d::Identity() * inv_s2);
            auto *rk = new g2o::RobustKernelHuber();
            rk->setDelta(kThHuberMono);
            edge->setRobustKernel(rk);
            edge->fx = fx; edge->fy = fy; edge->cx = cx; edge->cy = cy;
            edge->Xw = Xw;
            optimizer.addEdge(edge);
            edges_mono.push_back({edge, k});
        }
    }

    // Outlier flags — mirrors frame->outliers[] in pySLAM
    std::vector<bool> outliers(K, false);

    // ------------------------------------------------------------------
    // 4-round optimisation with outlier detection — identical to pySLAM
    // ------------------------------------------------------------------
    int num_bad = 0;

    for (int it = 0; it < rounds; ++it) {
        // Reset to initial pose at start of each round (pySLAM convention)
        vertex_se3->setEstimate(g2o::SE3Quat(Rcw, tcw));
        optimizer.initializeOptimization();
        optimizer.optimize(iters_per_round);

        num_bad = 0;

        for (auto &em : edges_mono) {
            if (outliers[em.k]) em.edge->computeError();
            double chi2 = em.edge->chi2();
            if (chi2 > kChi2Mono) {
                outliers[em.k] = true;
                em.edge->setLevel(1);
                ++num_bad;
            } else {
                outliers[em.k] = false;
                em.edge->setLevel(0);
            }
            if (it == 2) em.edge->setRobustKernel(nullptr);
        }

        for (auto &es : edges_stereo) {
            if (outliers[es.k]) es.edge->computeError();
            double chi2 = es.edge->chi2();
            if (chi2 > kChi2Stereo) {
                outliers[es.k] = true;
                es.edge->setLevel(1);
                ++num_bad;
            } else {
                outliers[es.k] = false;
                es.edge->setLevel(0);
            }
            if (it == 2) es.edge->setRobustKernel(nullptr);
        }

        // Stop early if too few edges remain (pySLAM: < 10)
        if ((int)optimizer.edges().size() < 10) break;
    }

    // ------------------------------------------------------------------
    // Extract results
    // ------------------------------------------------------------------
    const int num_inliers = K - num_bad;

    // Updated pose
    g2o::SE3Quat est = vertex_se3->estimate();
    Eigen::Matrix3d R_opt = est.rotation().matrix();
    Eigen::Vector3d t_opt = est.translation();

    auto out_pose = py::array_t<double>(16);
    double *op = out_pose.mutable_data();
    std::memcpy(op, pose_ptr, 16 * sizeof(double));  // default = input
    for (int r = 0; r < 3; ++r) {
        op[r*4+0] = R_opt(r,0); op[r*4+1] = R_opt(r,1);
        op[r*4+2] = R_opt(r,2); op[r*4+3] = t_opt(r);
    }
    op[12] = 0.0; op[13] = 0.0; op[14] = 0.0; op[15] = 1.0;

    // Outlier mask
    auto out_mask = py::array_t<uint8_t>(K);
    uint8_t *om = out_mask.mutable_data();
    for (int k = 0; k < K; ++k)
        om[k] = outliers[k] ? 1 : 0;

    const double mse = optimizer.activeChi2() / std::max(num_inliers, 1);

    py::dict result;
    result["updated_pose"] = out_pose;
    result["outlier_mask"] = out_mask;
    result["num_inliers"]  = num_inliers;
    result["mse"]          = mse;
    return result;
}


// ---------------------------------------------------------------------------
// run_global_ba — all KFs free except kid==0  (ORB-SLAM2 gauge fix)
// ---------------------------------------------------------------------------
py::dict run_global_ba(
    py::array_t<double,  py::array::c_style | py::array::forcecast> kf_poses,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> kf_ids,
    py::array_t<double,  py::array::c_style | py::array::forcecast> point_pos,
    py::array_t<double,  py::array::c_style | py::array::forcecast> observations,
    py::array_t<double,  py::array::c_style | py::array::forcecast> camera,
    int  rounds,
    bool use_robust_kernel,
    int  loop_kf_id)
{
    const int N = (int)kf_poses.shape(0);
    const int64_t *ids_ptr = kf_ids.data();

    auto kf_fixed = py::array_t<uint8_t>(N);
    uint8_t *fp = kf_fixed.mutable_data();
    for (int i = 0; i < N; ++i)
        fp[i] = (ids_ptr[i] == 0) ? 1 : 0;

    return run_local_ba(kf_poses, kf_ids, kf_fixed, point_pos, observations, camera,
                        rounds, use_robust_kernel, /*prune_outliers=*/false);
}

// ---------------------------------------------------------------------------
// Abort flag
// ---------------------------------------------------------------------------
static void set_abort(bool val) { g_abort_flag = val; }
static bool get_abort()         { return g_abort_flag; }

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(slam_optimizer_core, m) {
    m.doc() = "GIL-free C++ g2o bundle adjustment — Modular SLAM Framework";

    m.def("hello", []() { return "slam_optimizer_core: g2o ok"; });

    m.def("run_local_ba", &run_local_ba,
          py::arg("kf_poses"),
          py::arg("kf_ids"),
          py::arg("kf_fixed"),
          py::arg("point_pos"),
          py::arg("observations"),
          py::arg("camera"),
          py::arg("rounds") = 10,
          py::arg("use_robust_kernel") = true,
          py::arg("prune_outliers") = true,
          "Local BA. Dict: updated_poses(N,16), updated_points(M,3), "
          "outlier_mask(K,), initial_mse, mse, n_bad_edges.");

    m.def("run_global_ba", &run_global_ba,
          py::arg("kf_poses"),
          py::arg("kf_ids"),
          py::arg("point_pos"),
          py::arg("observations"),
          py::arg("camera"),
          py::arg("rounds") = 20,
          py::arg("use_robust_kernel") = true,
          py::arg("loop_kf_id") = 0,
          "Global BA (only kid==0 fixed). Same return dict as run_local_ba.");

    m.def("run_pose_optimization", &run_pose_optimization,
          py::arg("frame_pose"),
          py::arg("observations"),
          py::arg("camera"),
          py::arg("rounds") = 4,
          py::arg("iters_per_round") = 10,
          "Motion-only BA. "
          "observations (K,8): [u,v,ur,inv_sigma2,is_stereo,px,py,pz]. "
          "Dict: updated_pose(16,), outlier_mask(K,), num_inliers, mse.");

    m.def("set_abort", &set_abort, py::arg("val"));
    m.def("get_abort", &get_abort);
}
