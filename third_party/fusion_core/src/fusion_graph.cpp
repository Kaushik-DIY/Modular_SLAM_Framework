#include "fusion_graph.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/types/slam2d/vertex_se2.h>

namespace fusion {

double FusionGraph2D::optimize(int iterations) {
    std::lock_guard<std::mutex> lk(_m);
    if (_nodes.empty() || _edges.empty()) return 0.0;
    if (iterations <= 0) iterations = _cfg.max_iterations;

    using BlockSolverSE2 = g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>>;
    auto linear = std::make_unique<g2o::LinearSolverEigen<BlockSolverSE2::PoseMatrixType>>();
    auto block = std::make_unique<BlockSolverSE2>(std::move(linear));
    auto* algo = new g2o::OptimizationAlgorithmLevenberg(std::move(block));

    g2o::SparseOptimizer opt;
    opt.setAlgorithm(algo);
    opt.setVerbose(false);

    for (const auto& kv : _nodes) {
        auto* v = new g2o::VertexSE2();
        v->setId(kv.first);
        v->setEstimate(g2o::SE2(kv.second.x, kv.second.y, kv.second.theta));
        if (kv.first == _fixed_id) v->setFixed(true);
        opt.addVertex(v);
    }

    for (const auto& e : _edges) {
        auto itf = _nodes.find(e.from_id);
        auto itt = _nodes.find(e.to_id);
        if (itf == _nodes.end() || itt == _nodes.end()) continue;
        auto* edge = new g2o::EdgeSE2();
        edge->setVertex(0, opt.vertex(e.from_id));
        edge->setVertex(1, opt.vertex(e.to_id));
        edge->setMeasurement(g2o::SE2(e.rel_pose.x, e.rel_pose.y, e.rel_pose.theta));
        Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
        info(0, 0) = std::max(e.trans_weight, 1e-12);
        info(1, 1) = std::max(e.trans_weight, 1e-12);
        info(2, 2) = std::max(e.rot_weight, 1e-12);
        edge->setInformation(info);
        if (e.is_loop && _cfg.huber_scale > 0.0) {
            auto* rk = new g2o::RobustKernelHuber();
            rk->setDelta(_cfg.huber_scale);
            edge->setRobustKernel(rk);
        }
        opt.addEdge(edge);
    }

    opt.initializeOptimization();
    opt.optimize(iterations);

    for (auto& kv : _nodes) {
        if (kv.first == _fixed_id) continue;
        auto* v = static_cast<g2o::VertexSE2*>(opt.vertex(kv.first));
        if (v == nullptr) continue;
        const auto est = v->estimate();
        kv.second = Pose2(est.translation().x(), est.translation().y(),
                          est.rotation().angle());
    }
    return opt.chi2();
}

}  // namespace fusion
