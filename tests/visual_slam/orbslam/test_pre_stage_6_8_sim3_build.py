"""Smoke tests for the pre-Stage 6/8 build:

  * g2o `VertexSim3Expmap` exposes the camera-intrinsics members
    (`_principle_point1/2`, `_focal_length1/2`, `_fix_scale`) and the
    `is_depth_positive` method on `EdgeSim3ProjectXYZ` /
    `EdgeInverseSim3ProjectXYZ` — the prerequisites for Stage 8
    (port of pyslam's `optimize_sim3`).
  * pyslam's C++ `sim3solver` module imports, accepts numpy-fed input, and
    recovers a known synthetic Sim3 transform — the prerequisite for Stage 6.

These tests do NOT touch loop-closure runtime behavior.  They only validate the
build artifacts.  See
`CHECKPOINT_PRE_STAGE_6_8_G2OPY_SIM3_AND_SIM3SOLVER_BUILD_AUDIT.md`.
"""

from __future__ import annotations

import g2o
import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# g2o Sim3 binding parity                                                     #
# --------------------------------------------------------------------------- #

def test_vertex_sim3_intrinsics_attributes_are_settable():
    v = g2o.VertexSim3Expmap()
    v._fix_scale = True
    v._principle_point1 = np.array([320.0, 240.0])
    v._focal_length1 = np.array([500.0, 500.0])
    v._principle_point2 = np.array([318.6, 255.3])
    v._focal_length2 = np.array([517.3, 516.5])

    assert v._fix_scale is True
    np.testing.assert_allclose(v._principle_point1, [320.0, 240.0])
    np.testing.assert_allclose(v._focal_length1, [500.0, 500.0])
    np.testing.assert_allclose(v._principle_point2, [318.6, 255.3])
    np.testing.assert_allclose(v._focal_length2, [517.3, 516.5])


def test_cam_map1_applies_configured_intrinsics():
    """cam_map1 must use the configured fx/fy/cx/cy, not identity."""
    v = g2o.VertexSim3Expmap()
    v._principle_point1 = np.array([320.0, 240.0])
    v._focal_length1 = np.array([500.0, 500.0])
    v.set_estimate(g2o.Sim3())  # identity

    # Already-normalized (x/z, y/z); cam_map1 must scale by fx/fy and add cx/cy.
    pt = np.array([0.2, 0.4])
    projection = v.cam_map1(pt)

    np.testing.assert_allclose(projection, [500 * 0.2 + 320, 500 * 0.4 + 240])


def test_cam_map2_applies_configured_intrinsics():
    v = g2o.VertexSim3Expmap()
    v._principle_point2 = np.array([100.0, 200.0])
    v._focal_length2 = np.array([300.0, 400.0])
    v.set_estimate(g2o.Sim3())

    pt = np.array([0.5, 0.25])
    projection = v.cam_map2(pt)

    np.testing.assert_allclose(projection, [300 * 0.5 + 100, 400 * 0.25 + 200])


def test_sim3_projection_edges_expose_is_depth_positive():
    e_forward = g2o.EdgeSim3ProjectXYZ()
    e_inverse = g2o.EdgeInverseSim3ProjectXYZ()
    assert hasattr(e_forward, "is_depth_positive")
    assert hasattr(e_inverse, "is_depth_positive")
    assert callable(e_forward.is_depth_positive)
    assert callable(e_inverse.is_depth_positive)


def test_optimizer_stack_for_sim3_is_constructible():
    """Stage 8 needs SparseOptimizer + BlockSolverX + Levenberg + Sim3 vertex."""
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    sim3 = g2o.Sim3()
    vertex = g2o.VertexSim3Expmap()
    vertex.set_estimate(sim3)
    vertex.set_id(0)
    vertex.set_fixed(False)
    vertex._fix_scale = True
    optimizer.add_vertex(vertex)

    assert optimizer.vertex(0) is not None


# --------------------------------------------------------------------------- #
# sim3solver C++ module                                                       #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def sim3solver():
    return pytest.importorskip("sim3solver")


def _make_camera_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def _rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)).astype(
        np.float32
    )


def test_sim3solver_module_imports_and_exposes_expected_api(sim3solver):
    for sym in (
        "Sim3Solver",
        "Sim3SolverInput",
        "Sim3SolverInput2",
        "Sim3PointRegistrationSolver",
        "Sim3PointRegistrationSolverInput",
    ):
        assert hasattr(sim3solver, sym), f"missing symbol: {sym}"

    inp = sim3solver.Sim3SolverInput()
    for attr in (
        "K1",
        "K2",
        "Rcw1",
        "Rcw2",
        "tcw1",
        "tcw2",
        "points_3d_w1",
        "points_3d_w2",
        "sigmas2_1",
        "sigmas2_2",
        "fix_scale",
    ):
        assert hasattr(inp, attr), f"Sim3SolverInput missing attr: {attr}"


def test_sim3solver_recovers_known_ground_truth_transform(sim3solver):
    """RGB-D / fix_scale=True regime: recover a known (R, t) from 80 noiseless
    3D correspondences seen by two cameras with the same K."""
    rng = np.random.default_rng(seed=20260515)
    n_points = 80

    # World points around the second camera centre at z=2 m.
    points_w1 = rng.uniform(low=[-0.5, -0.5, 1.5], high=[0.5, 0.5, 2.5], size=(n_points, 3))

    # Ground-truth transform from kf1 to kf2 (kf2's pose relative to kf1):
    R_gt = _rodrigues([0.0, 1.0, 0.0], np.deg2rad(15.0))  # 15-degree yaw
    t_gt = np.array([0.30, -0.05, 0.10], dtype=np.float32)

    # Same physical points observed by both cameras → same world coords.
    points_w2 = points_w1.astype(np.float32)
    points_w1 = points_w1.astype(np.float32)

    K = _make_camera_K()
    inp = sim3solver.Sim3SolverInput()
    inp.K1 = K
    inp.K2 = K
    inp.Rcw1 = np.eye(3, dtype=np.float32)              # KF1 at world origin
    inp.tcw1 = np.zeros(3, dtype=np.float32)
    inp.Rcw2 = R_gt                                      # KF2 at (R_gt, t_gt)
    inp.tcw2 = t_gt
    inp.points_3d_w1 = points_w1
    inp.points_3d_w2 = points_w2
    inp.sigmas2_1 = np.ones(n_points, dtype=np.float32)
    inp.sigmas2_2 = np.ones(n_points, dtype=np.float32)
    inp.fix_scale = True                                 # RGB-D

    solver = sim3solver.Sim3Solver(inp)
    solver.set_ransac_parameters(0.99, 20, 300)

    converged = False
    num_inliers = 0
    for _ in range(60):
        _, is_no_more, _, num_inliers, is_converged = solver.iterate(5)
        if is_converged:
            converged = True
            break
        if is_no_more:
            break

    assert converged, f"Sim3Solver did not converge (num_inliers={num_inliers})"
    assert num_inliers > 60, f"too few inliers: {num_inliers}"

    R_recovered = np.asarray(solver.get_estimated_rotation())
    t_recovered = np.asarray(solver.get_estimated_translation())
    scale_recovered = solver.get_estimated_scale()

    # The solver returns the transform from kf2 to kf1 (T12); since both cameras
    # observe the SAME world points, the recovered (R12, t12) maps kf2's world
    # to kf1's world, i.e. R12 = Rcw1 @ Rcw2^T and t12 = -R12 @ tcw2 (with tcw1=0).
    R_expected = inp.Rcw1 @ inp.Rcw2.T
    t_expected = inp.tcw1 - R_expected @ inp.tcw2

    np.testing.assert_allclose(R_recovered, R_expected, atol=1e-3)
    np.testing.assert_allclose(t_recovered, t_expected, atol=1e-2)
    assert abs(scale_recovered - 1.0) < 1e-3, f"scale not fixed: {scale_recovered}"


def test_sim3solver_rejects_unrelated_point_clouds(sim3solver):
    """When the two clouds are unrelated noise, the solver must NOT converge
    with many inliers."""
    rng = np.random.default_rng(seed=20260516)
    n_points = 80

    points_w1 = rng.uniform(low=[-0.5, -0.5, 1.5], high=[0.5, 0.5, 2.5], size=(n_points, 3)).astype(np.float32)
    points_w2 = rng.uniform(low=[-0.5, -0.5, 1.5], high=[0.5, 0.5, 2.5], size=(n_points, 3)).astype(np.float32)

    K = _make_camera_K()
    inp = sim3solver.Sim3SolverInput()
    inp.K1 = K
    inp.K2 = K
    inp.Rcw1 = np.eye(3, dtype=np.float32)
    inp.tcw1 = np.zeros(3, dtype=np.float32)
    inp.Rcw2 = np.eye(3, dtype=np.float32)
    inp.tcw2 = np.zeros(3, dtype=np.float32)
    inp.points_3d_w1 = points_w1
    inp.points_3d_w2 = points_w2
    inp.sigmas2_1 = np.ones(n_points, dtype=np.float32)
    inp.sigmas2_2 = np.ones(n_points, dtype=np.float32)
    inp.fix_scale = True

    solver = sim3solver.Sim3Solver(inp)
    solver.set_ransac_parameters(0.99, 20, 300)

    # Run plenty of iterations to give noise every chance to "converge" — it
    # must not.
    converged = False
    for _ in range(60):
        _, is_no_more, _, _, is_converged = solver.iterate(5)
        if is_converged:
            converged = True
            break
        if is_no_more:
            break

    assert not converged, "Sim3Solver wrongly accepted unrelated clouds"
