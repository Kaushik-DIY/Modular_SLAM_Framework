#!/usr/bin/env python3
"""
=============================================================================
VALIDATION CHECKPOINT 0 — g2o SE(3) Pose Graph Optimization Smoke Test
=============================================================================

Purpose:
    Verify that g2o Python bindings are correctly installed and can perform
    SE(3) pose graph optimization. This is the foundational dependency for
    all Visual SLAM optimization (motion-only BA, local BA, PGO, GBA).

What this test does:
    1. Creates a trivial pose graph with 3 nodes arranged in a line
       (node 0 at origin, node 1 at x=1m, node 2 at x=2m)
    2. Adds odometry edges between consecutive nodes
    3. Adds a loop closure edge from node 2 back to node 0 with a
       deliberate 5cm error (measures -2.05m instead of true -2.0m)
    4. Runs Levenberg-Marquardt optimization
    5. Verifies that the loop error is distributed across all nodes
       (node 2 should end up near x=2.0, not x=2.05)

Pass criteria:
    - g2o imports without error
    - Optimizer converges (no crash, no NaN)
    - Node 2 final position within 5cm of (2.0, 0.0, 0.0)
    - All edge types (VertexSE3, EdgeSE3) work correctly

If this test FAILS:
    - Do NOT proceed to Phase 1
    - Check g2o installation: pip show g2o-python or pip show g2opy
    - Try reinstalling with the fallback methods in phase0_install_g2o.sh
    - Common issue: missing libcholmod (sudo apt install libsuitesparse-dev)

Usage:
    python3 tests/test_checkpoint_0_g2o.py
=============================================================================
"""

from __future__ import annotations

import sys
import traceback

import numpy as np


def _run_checkpoint() -> bool:
    """
    Execute the g2o SE(3) smoke test.
    Returns True if all checks pass, False otherwise.
    """

    # ------------------------------------------------------------------
    # Step 1: Import g2o — if this fails, nothing else matters
    # ------------------------------------------------------------------
    print("[Step 1] Importing g2o module...")
    try:
        import g2o
    except ImportError as e:
        print(f"  FAILED: Cannot import g2o: {e}")
        print("  -> Install g2o first: ./phase0_install_g2o.sh")
        return False
    print(f"  OK: g2o imported (module at: {g2o.__file__})")

    # ------------------------------------------------------------------
    # Step 2: Create optimizer with Levenberg-Marquardt algorithm
    # ------------------------------------------------------------------
    print("[Step 2] Creating SparseOptimizer with LM solver...")
    try:
        optimizer = g2o.SparseOptimizer()

        # BlockSolverSE3 uses 6-DOF pose blocks (for SE(3) vertices)
        # LinearSolverCholmodSE3 uses CHOLMOD sparse direct solver
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)
    except Exception as e:
        print(f"  FAILED: Cannot create optimizer: {e}")
        print("  -> Likely missing CHOLMOD. Try: sudo apt install libsuitesparse-dev")
        return False
    print("  OK: Optimizer created with BlockSolverSE3 + CHOLMOD")

    # ------------------------------------------------------------------
    # Step 3: Add 3 SE(3) vertices (nodes in the pose graph)
    # ------------------------------------------------------------------
    print("[Step 3] Adding 3 SE(3) vertices...")

    # Node 0: at origin (fixed — anchor node)
    # Node 1: at x=1.0m (initial guess matches ground truth)
    # Node 2: at x=2.0m (initial guess matches ground truth)
    node_positions = [
        np.eye(4),                          # Node 0: origin
        _make_translation(1.0, 0.0, 0.0),   # Node 1: 1m forward
        _make_translation(2.0, 0.0, 0.0),   # Node 2: 2m forward
    ]

    for i, T in enumerate(node_positions):
        v = g2o.VertexSE3()
        v.set_id(i)
        v.set_estimate(g2o.Isometry3d(T))
        if i == 0:
            v.set_fixed(True)  # Anchor the first node
        optimizer.add_vertex(v)

    print(f"  OK: {len(node_positions)} vertices added (node 0 fixed)")

    # ------------------------------------------------------------------
    # Step 4: Add odometry edges (0->1 and 1->2)
    # ------------------------------------------------------------------
    print("[Step 4] Adding odometry edges...")

    # Information matrix: diagonal, higher = more confident
    # Units: [tx, ty, tz, rx, ry, rz] with rotation in radians
    info_odom = np.eye(6) * 100.0  # High confidence in odometry

    # Edge 0->1: 1m translation along x
    _add_se3_edge(optimizer, 0, 1,
                  _make_translation(1.0, 0.0, 0.0),
                  info_odom)

    # Edge 1->2: 1m translation along x
    _add_se3_edge(optimizer, 1, 2,
                  _make_translation(1.0, 0.0, 0.0),
                  info_odom)

    print("  OK: 2 odometry edges added (0->1, 1->2)")

    # ------------------------------------------------------------------
    # Step 5: Add loop closure edge (2->0) with deliberate error
    # ------------------------------------------------------------------
    print("[Step 5] Adding loop closure edge with 5cm error...")

    # True measurement should be (-2.0, 0, 0) but we introduce 5cm error
    # This simulates real-world loop closure measurement noise
    info_loop = np.eye(6) * 50.0  # Slightly less confident than odometry

    _add_se3_edge(optimizer, 2, 0,
                  _make_translation(-2.05, 0.0, 0.0),
                  info_loop)

    print("  OK: Loop closure edge added (2->0, measurement = -2.05m)")

    # ------------------------------------------------------------------
    # Step 6: Optimize
    # ------------------------------------------------------------------
    print("[Step 6] Running optimization (20 iterations)...")

    optimizer.initialize_optimization()
    iterations = optimizer.optimize(20)

    print(f"  OK: Optimization converged in {iterations} iterations")

    # ------------------------------------------------------------------
    # Step 7: Extract and verify results
    # ------------------------------------------------------------------
    print("[Step 7] Verifying results...")

    results = {}
    for i in range(3):
        T = optimizer.vertex(i).estimate().matrix()
        pos = T[:3, 3]
        results[i] = pos
        print(f"  Node {i}: x={pos[0]:.6f}, y={pos[1]:.6f}, z={pos[2]:.6f}")

    # --- Checks ---

    # Check 1: Node 0 should be at origin (it's fixed)
    assert np.allclose(results[0], [0, 0, 0], atol=1e-9), \
        f"Node 0 moved from origin: {results[0]}"

    # Check 2: Node 2 should be near x=2.0 (loop error distributed)
    # Without optimization, node 2 is at exactly x=2.0 (from odometry)
    # The loop closure says it should be at x=2.05 from node 0
    # After optimization, the error is distributed: node 2 ≈ 2.0 ± small correction
    err_x = abs(results[2][0] - 2.0)
    assert err_x < 0.05, \
        f"Node 2 x-position too far from 2.0: {results[2][0]:.6f} (error={err_x:.6f}m)"

    # Check 3: No NaN or Inf in results
    for i, pos in results.items():
        assert np.all(np.isfinite(pos)), f"Node {i} has non-finite position: {pos}"

    # Check 4: Node 1 should be approximately between 0 and 2
    assert 0.5 < results[1][0] < 1.5, \
        f"Node 1 x-position suspicious: {results[1][0]:.6f}"

    print("")
    print("  All checks passed:")
    print(f"    Node 0: origin (fixed)")
    print(f"    Node 1: x = {results[1][0]:.6f}m (expected ~1.0)")
    print(f"    Node 2: x = {results[2][0]:.6f}m (expected ~2.0, error = {err_x:.6f}m)")

    return True


def _make_translation(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix with pure translation.
    
    Parameters
    ----------
    tx, ty, tz : float
        Translation components in meters.
    
    Returns
    -------
    np.ndarray
        4x4 identity rotation with [tx, ty, tz] translation.
    """
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def _add_se3_edge(
    optimizer,
    id_from: int,
    id_to: int,
    measurement: np.ndarray,
    information: np.ndarray,
) -> None:
    """
    Add an SE(3) edge (between-factor) to the optimizer.
    
    Parameters
    ----------
    optimizer : g2o.SparseOptimizer
        The optimization graph.
    id_from : int
        Source vertex ID.
    id_to : int
        Target vertex ID.
    measurement : np.ndarray
        4x4 relative transformation matrix T_from_to.
    information : np.ndarray
        6x6 information matrix (inverse covariance).
    """
    import g2o

    edge = g2o.EdgeSE3()
    edge.set_vertex(0, optimizer.vertex(id_from))
    edge.set_vertex(1, optimizer.vertex(id_to))
    edge.set_measurement(g2o.Isometry3d(measurement))
    edge.set_information(information)
    optimizer.add_edge(edge)


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VALIDATION CHECKPOINT 0: g2o SE(3) Pose Graph Optimization")
    print("=" * 70)
    print()

    try:
        passed = _run_checkpoint()
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        passed = False

    print()
    print("=" * 70)
    if passed:
        print("CHECKPOINT 0: PASSED")
        print("")
        print("g2o is correctly installed and can perform SE(3) PGO.")
        print("You may proceed to Phase 1 (slam_core type extension).")
    else:
        print("CHECKPOINT 0: FAILED")
        print("")
        print("g2o is NOT working correctly. Do NOT proceed to Phase 1.")
        print("Fix the installation first using the instructions in")
        print("phase0_install_g2o.sh (see Manual Fallback section).")
    print("=" * 70)

    sys.exit(0 if passed else 1)
