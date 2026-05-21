#!/usr/bin/env python3
"""
Diagnostic: test Sim3Solver on synthetic data matching what we see in loop closure.
Checks if the solver works with float32 data and various inlier ratios.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_sim3solver_basics():
    try:
        import sim3solver as _sim3s
    except ImportError:
        print("sim3solver not available")
        return

    print("=== Sim3Solver diagnostic ===")

    # Create synthetic scenario:
    # KF1 at origin looking down Z axis
    # KF2 at (0.2, 0, 0) with slight rotation
    # 40 map points in world frame

    np.random.seed(42)
    n = 40

    # Ground truth transform between cam1 and cam2
    t_true_12 = np.array([0.2, 0.0, 0.0], dtype=np.float64)  # cam2 is 0.2m to the right of cam1
    angle = 0.05  # 5 degrees rotation
    R_true_12 = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ], dtype=np.float64)
    s_true = 1.0  # RGB-D: fixed scale

    # Camera poses (world to camera)
    Rcw1 = np.eye(3, dtype=np.float64)
    tcw1 = np.array([0, 0, 0], dtype=np.float64)

    Rcw2 = R_true_12 @ Rcw1
    tcw2 = R_true_12 @ tcw1 + t_true_12

    # Calibration (TUM fr1 approx)
    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Random 3D world points visible from both cameras
    pts_w = np.random.uniform([-1.5, -1.0, 2.0], [1.5, 1.0, 4.0], (n, 3))

    # These are the SAME physical points, so p3d_w1 == p3d_w2 (perfect case, no drift)
    p3d_w1 = pts_w.copy().astype(np.float32)
    p3d_w2 = pts_w.copy().astype(np.float32)

    sigmas2 = np.ones(n, dtype=np.float32)

    si = _sim3s.Sim3SolverInput()
    si.fix_scale = True  # RGB-D
    si.K1 = K
    si.Rcw1 = Rcw1.astype(np.float32)
    si.tcw1 = tcw1.astype(np.float32)
    si.K2 = K
    si.Rcw2 = Rcw2.astype(np.float32)
    si.tcw2 = tcw2.astype(np.float32)
    si.points_3d_w1 = p3d_w1
    si.points_3d_w2 = p3d_w2
    si.sigmas2_1 = sigmas2
    si.sigmas2_2 = sigmas2

    solver = _sim3s.Sim3Solver(si)
    solver.set_ransac_parameters(0.99, 20, 300)

    is_converged = False
    is_no_more = False
    num_inliers = 0
    iter_count = 0
    while not (is_converged or is_no_more):
        _, is_no_more, inlier_flags, num_inliers, is_converged = solver.iterate(5)
        iter_count += 5

    print(f"[PERFECT CASE] n={n}, converged={is_converged}, is_no_more={is_no_more}, "
          f"num_inliers={num_inliers}, iters={iter_count}")

    if is_converged:
        R12 = solver.get_estimated_rotation()
        t12 = solver.get_estimated_translation()
        s12 = solver.get_estimated_scale()
        t_err = np.linalg.norm(np.array(t12) - t_true_12)
        print(f"  t12={np.array(t12)}, t_err={t_err:.4f}")
        print(f"  scale={s12:.4f}")

    # Now test with drift: add Gaussian noise to p3d_w2 to simulate map drift
    print()
    for drift_std in [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]:
        # Simulate drift as a consistent rigid offset + small per-point noise
        drift_offset = np.array([drift_std * 5, 0, 0], dtype=np.float64)  # systematic drift
        noise = np.random.normal(0, drift_std * 0.1, (n, 3))
        p3d_w2_noisy = (pts_w + drift_offset + noise).astype(np.float32)

        si2 = _sim3s.Sim3SolverInput()
        si2.fix_scale = True
        si2.K1 = K
        si2.Rcw1 = Rcw1.astype(np.float32)
        si2.tcw1 = tcw1.astype(np.float32)
        si2.K2 = K
        si2.Rcw2 = Rcw2.astype(np.float32)
        si2.tcw2 = tcw2.astype(np.float32)
        si2.points_3d_w1 = p3d_w1
        si2.points_3d_w2 = p3d_w2_noisy
        si2.sigmas2_1 = sigmas2
        si2.sigmas2_2 = sigmas2

        s2 = _sim3s.Sim3Solver(si2)
        s2.set_ransac_parameters(0.99, 20, 300)

        is_converged = False
        is_no_more = False
        while not (is_converged or is_no_more):
            _, is_no_more, _, num_inliers, is_converged = s2.iterate(5)

        print(f"[DRIFT drift_std={drift_std:.2f} offset={drift_offset[0]:.3f}m] "
              f"converged={is_converged}, inliers={num_inliers}")

    # Test with visual aliasing: completely random p3d_w2 (worst case)
    print()
    p3d_w2_random = np.random.uniform([-1.5, -1.0, 2.0], [1.5, 1.0, 4.0], (n, 3)).astype(np.float32)
    si3 = _sim3s.Sim3SolverInput()
    si3.fix_scale = True
    si3.K1 = K
    si3.Rcw1 = Rcw1.astype(np.float32)
    si3.tcw1 = tcw1.astype(np.float32)
    si3.K2 = K
    si3.Rcw2 = Rcw2.astype(np.float32)
    si3.tcw2 = tcw2.astype(np.float32)
    si3.points_3d_w1 = p3d_w1
    si3.points_3d_w2 = p3d_w2_random
    si3.sigmas2_1 = sigmas2
    si3.sigmas2_2 = sigmas2

    s3 = _sim3s.Sim3Solver(si3)
    s3.set_ransac_parameters(0.99, 20, 300)

    is_converged = False
    is_no_more = False
    while not (is_converged or is_no_more):
        _, is_no_more, _, num_inliers, is_converged = s3.iterate(5)
    print(f"[RANDOM ALIASING] converged={is_converged}, inliers={num_inliers}")

    # Test with mixed: 30% genuine + 70% aliased matches
    print()
    n_genuine = int(n * 0.3)
    p3d_w2_mixed = p3d_w2_random.copy()
    p3d_w2_mixed[:n_genuine] = p3d_w1[:n_genuine]  # first n_genuine are genuine

    si4 = _sim3s.Sim3SolverInput()
    si4.fix_scale = True
    si4.K1 = K
    si4.Rcw1 = Rcw1.astype(np.float32)
    si4.tcw1 = tcw1.astype(np.float32)
    si4.K2 = K
    si4.Rcw2 = Rcw2.astype(np.float32)
    si4.tcw2 = tcw2.astype(np.float32)
    si4.points_3d_w1 = p3d_w1
    si4.points_3d_w2 = p3d_w2_mixed
    si4.sigmas2_1 = sigmas2
    si4.sigmas2_2 = sigmas2

    s4 = _sim3s.Sim3Solver(si4)
    s4.set_ransac_parameters(0.99, 20, 300)

    is_converged = False
    is_no_more = False
    while not (is_converged or is_no_more):
        _, is_no_more, _, num_inliers, is_converged = s4.iterate(5)
    print(f"[MIXED {n_genuine}/{n} genuine] converged={is_converged}, inliers={num_inliers}")


def analyze_json_diagnostics():
    """Analyze any saved diagnostic JSON files."""
    import json, glob
    files = sorted(glob.glob('/tmp/sim3_diag_*.json'))
    if not files:
        print("\n=== No diagnostic JSON files found (need to run SLAM first) ===")
        return

    print(f"\n=== Analyzing {len(files)} diagnostic JSON files ===")
    for path in files:
        with open(path) as f:
            d = json.load(f)

        print(f"\n--- {path} ---")
        print(f"  cur_kf={d['cur_kid']}, cand_kf={d['cand_kid']}, n3d={d['n3d']}")

        p1 = np.array(d['p3d_w1'])
        p2 = np.array(d['p3d_w2'])
        cur_Tcw = np.array(d['cur_Tcw'])
        cand_Tcw = np.array(d['cand_Tcw'])

        print(f"  p3d_w1 range: x=[{p1[:,0].min():.2f},{p1[:,0].max():.2f}] "
              f"y=[{p1[:,1].min():.2f},{p1[:,1].max():.2f}] "
              f"z=[{p1[:,2].min():.2f},{p1[:,2].max():.2f}]")
        print(f"  p3d_w2 range: x=[{p2[:,0].min():.2f},{p2[:,0].max():.2f}] "
              f"y=[{p2[:,1].min():.2f},{p2[:,1].max():.2f}] "
              f"z=[{p2[:,2].min():.2f},{p2[:,2].max():.2f}]")

        # Check if p1 and p2 overlap spatially
        diff = p1 - p2
        print(f"  p3d_w1-p3d_w2 diff: mean={np.abs(diff).mean():.3f}, "
              f"std={diff.std():.3f}, max={np.abs(diff).max():.3f}")

        # Camera positions
        Rcw1 = cur_Tcw[:3, :3]
        tcw1 = cur_Tcw[:3, 3]
        Rcw2 = cand_Tcw[:3, :3]
        tcw2 = cand_Tcw[:3, 3]

        cam1_pos = -Rcw1.T @ tcw1  # camera center in world
        cam2_pos = -Rcw2.T @ tcw2
        cam_dist = np.linalg.norm(cam1_pos - cam2_pos)
        print(f"  cam1_pos={cam1_pos}, cam2_pos={cam2_pos}, cam_dist={cam_dist:.3f}m")

        # Check Tcw validity (should be rotation matrix)
        det1 = np.linalg.det(Rcw1)
        det2 = np.linalg.det(Rcw2)
        print(f"  Rcw1 det={det1:.4f} (should be ~1.0), Rcw2 det={det2:.4f}")

        # Check if points are in front of cameras
        pts_c1 = (Rcw1 @ p1.T + tcw1.reshape(3,1)).T
        pts_c2 = (Rcw2 @ p2.T + tcw2.reshape(3,1)).T
        behind_c1 = (pts_c1[:, 2] <= 0).sum()
        behind_c2 = (pts_c2[:, 2] <= 0).sum()
        print(f"  Points behind cam1: {behind_c1}/{len(p1)}, behind cam2: {behind_c2}/{len(p2)}")

        # Check sigmas
        s2_1 = np.array(d['s2_1'])
        print(f"  sigma2 range: [{s2_1.min():.3f}, {s2_1.max():.3f}]")

        # Try Sim3Solver on this actual data
        try:
            import sim3solver as _sim3s
            si = _sim3s.Sim3SolverInput()
            si.fix_scale = True
            si.K1 = np.array(d['K1'], dtype=np.float32)
            si.Rcw1 = Rcw1.astype(np.float32)
            si.tcw1 = tcw1.astype(np.float32)
            si.K2 = si.K1  # same camera
            si.Rcw2 = Rcw2.astype(np.float32)
            si.tcw2 = tcw2.astype(np.float32)
            si.points_3d_w1 = p1.astype(np.float32)
            si.points_3d_w2 = p2.astype(np.float32)
            si.sigmas2_1 = s2_1.astype(np.float32)
            si.sigmas2_2 = np.array(d['s2_2'], dtype=np.float32)

            solver = _sim3s.Sim3Solver(si)
            solver.set_ransac_parameters(0.99, 20, 300)

            is_converged = False
            is_no_more = False
            n_inliers = 0
            while not (is_converged or is_no_more):
                _, is_no_more, _, n_inliers, is_converged = solver.iterate(5)
            print(f"  Sim3Solver replay: converged={is_converged}, inliers={n_inliers}")

            # Also try identity Sim3 (as sanity check - if p1==p2, identity should give ~40 inliers)
            si_id = _sim3s.Sim3SolverInput()
            si_id.fix_scale = True
            si_id.K1 = si.K1
            si_id.Rcw1 = Rcw1.astype(np.float32)
            si_id.tcw1 = tcw1.astype(np.float32)
            si_id.K2 = si.K1
            si_id.Rcw2 = Rcw2.astype(np.float32)
            si_id.tcw2 = tcw2.astype(np.float32)
            si_id.points_3d_w1 = p1.astype(np.float32)
            si_id.points_3d_w2 = p1.copy().astype(np.float32)  # p2 = p1 (same point!)
            si_id.sigmas2_1 = s2_1.astype(np.float32)
            si_id.sigmas2_2 = np.array(d['s2_2'], dtype=np.float32)

            s_id = _sim3s.Sim3Solver(si_id)
            s_id.set_ransac_parameters(0.99, 20, 300)

            is_converged = False
            is_no_more = False
            n_inliers = 0
            while not (is_converged or is_no_more):
                _, is_no_more, _, n_inliers, is_converged = s_id.iterate(5)
            print(f"  Sim3Solver IDENTITY sanity (p2=p1): converged={is_converged}, inliers={n_inliers}")

        except ImportError:
            print("  sim3solver not available for replay")


if __name__ == '__main__':
    test_sim3solver_basics()
    analyze_json_diagnostics()
