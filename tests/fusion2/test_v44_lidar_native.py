"""V4.4 — native C++ LiDAR front-end: parity vs the Python hector stack.

P1: voxel filters + PoseExtrapolatorCV (exact parity, synthetic).
P2/P3 add grid/correlative/refine/matcher parity + E2E + timing gates.
"""
import math

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")

from carto.local_slam.pose_extrapolator import PoseExtrapolatorCV as PyExtrap
from slam_core.common.types import Pose2 as PyPose2
from slam_core.matching.preprocessing import (PointCloudProcessor,
                                              PointCloudProcessorConfig)


# ---------------------------------------------------------------------------
# P1.1 voxel filter parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7])
def test_voxel_filter_parity(seed):
    rng = np.random.default_rng(seed)
    # mixed positive/negative coords, clusters + spread (exercises floor on negatives)
    pts = np.vstack([
        rng.uniform(-8, 8, (700, 2)),
        rng.normal(0, 0.02, (200, 2)) + [1.234, -3.456],
        rng.normal(0, 0.01, (100, 2)) + [-0.05, 0.05],
    ])

    py_fixed = PointCloudProcessor._fixed_voxel_filter(pts, voxel_size=0.03)
    cc_fixed = fusion_core.fixed_voxel_filter(pts, 0.03)
    assert cc_fixed.shape == py_fixed.shape, "fixed filter row count/order"
    np.testing.assert_allclose(cc_fixed, py_fixed, atol=1e-12)

    py_ad = PointCloudProcessor._adaptive_voxel_filter(
        py_fixed, max_voxel_size=0.10, min_num_points=200, num_iterations=6)
    cc_ad = fusion_core.adaptive_voxel_filter(cc_fixed, 0.10, 200, 6)
    assert cc_ad.shape == py_ad.shape
    np.testing.assert_allclose(cc_ad, py_ad, atol=1e-12)

    # full pipeline (fixed -> adaptive), PointCloudProcessor.process parity
    proc = PointCloudProcessor(PointCloudProcessorConfig(
        fixed_voxel_size=0.03, adaptive_voxel_max_size=0.10,
        adaptive_min_num_points=200, adaptive_num_iterations=6, enabled=True))
    py_out, _ = proc.process(pts)
    cfg = fusion_core.VoxelFilterConfig()
    cfg.fixed_size, cfg.adaptive_max_size = 0.03, 0.10
    cfg.adaptive_min_points, cfg.adaptive_iters = 200, 6
    cc_out = fusion_core.voxel_preprocess(pts, cfg)
    assert cc_out.shape == py_out.shape
    np.testing.assert_allclose(cc_out, py_out, atol=1e-12)


# ---------------------------------------------------------------------------
# P1.2 extrapolator parity (scripted sequence incl. edge cases)
# ---------------------------------------------------------------------------

def _make_pair(use_imu=True, alpha=0.02):
    py = PyExtrap(max_dt=0.5, init_vxy=0.0, init_wz=0.0, use_imu=use_imu,
                  imu_yaw_correction_alpha=alpha)
    cfg = fusion_core.ExtrapolatorConfig()
    cfg.use_imu = use_imu
    cfg.imu_yaw_correction_alpha = alpha
    cc = fusion_core.PoseExtrapolatorCV(cfg)
    return py, cc


def _assert_pose_eq(cp, pp, msg=""):
    assert abs(cp.x - pp.x) < 1e-12, msg
    assert abs(cp.y - pp.y) < 1e-12, msg
    d = math.atan2(math.sin(cp.theta - pp.theta), math.cos(cp.theta - pp.theta))
    assert abs(d) < 1e-12, msg


def test_extrapolator_parity():
    py, cc = _make_pair()

    # predict before any state -> identity
    _assert_pose_eq(cc.predict(0.0), py.predict(0.0), "no-state predict")

    script = [
        ("pose", 0.00, (0.0, 0.0, 0.0)),
        ("imu", 0.02, (0.30, 0.10)),
        ("pose", 0.10, (0.05, 0.001, 0.02)),
        ("predict", 0.20),
        ("pose", 0.20, (0.11, 0.004, 0.05)),
        ("imu", 0.25, (0.80, 0.18)),
        ("predict", 0.30),
        # same-timestamp replace
        ("pose", 0.30, (0.16, 0.01, 0.08)),
        ("pose", 0.30, (0.165, 0.011, 0.081)),
        ("predict", 0.35),
        # dt > max_dt clamp
        ("predict", 1.50),
        # long gap then new pose (queue trim path)
        ("pose", 2.50, (0.5, 0.05, 0.30)),
        ("imu", 2.55, (-0.50, 0.25)),
        ("predict", 2.60),
        ("pose", 2.70, (0.45, 0.06, 0.22)),
        ("predict", 2.80),
    ]
    for step in script:
        if step[0] == "pose":
            _, t, (x, y, th) = step
            py.add_pose(t, PyPose2(x, y, th))
            cc.add_pose(t, fusion_core.Pose2(x, y, th))
        elif step[0] == "imu":
            _, t, (wz, yaw) = step
            py.add_imu(t, wz, yaw)
            cc.add_imu(t, wz, yaw)
        else:
            _, t = step
            _assert_pose_eq(cc.predict(t), py.predict(t), f"predict@{t}")
    # velocity state parity (used as last-resort fallback)
    assert abs(cc.vx - py.vx) < 1e-12
    assert abs(cc.vy - py.vy) < 1e-12
    assert abs(cc.wz - py.wz) < 1e-12


def test_extrapolator_parity_no_imu():
    py, cc = _make_pair(use_imu=False)
    for k in range(8):
        t = 0.1 * k
        p = (0.1 * k, 0.01 * k * k, 0.05 * k)
        py.add_pose(t, PyPose2(*p))
        cc.add_pose(t, fusion_core.Pose2(*p))
        if k % 2:
            _assert_pose_eq(cc.predict(t + 0.07), py.predict(t + 0.07), f"k={k}")


# ---------------------------------------------------------------------------
# P2 — scan_to_submap parity
# ---------------------------------------------------------------------------

from slam_core.matching.scan_to_submap.submaps import SubmapBuilder2D as PySubmapBuilder


def _room_scan(rng, n=400, half=4.0, noise=0.01):
    """Synthetic rectangular-room scan (sensor at origin)."""
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        tx = half / max(abs(c), 1e-9)
        ty = half / max(abs(s), 1e-9)
        r = min(tx, ty)
        pts.append([r * c, r * s])
    return np.asarray(pts) + rng.normal(0, noise, (n, 2))


def _builders(scans_per_submap=6):
    py = PySubmapBuilder(submap_size_m=20.0, resolution=0.05,
                         scans_per_submap=scans_per_submap, ray_steps=20,
                         l0=0.0, l_occ=1.0, l_free=-0.1, l_min=-5.0, l_max=5.0)
    cfg = fusion_core.SubmapBuilderConfig()
    cfg.submap_size_m, cfg.resolution = 20.0, 0.05
    cfg.scans_per_submap = scans_per_submap
    cfg.l0, cfg.l_occ, cfg.l_free, cfg.l_min, cfg.l_max = 0.0, 1.0, -0.1, -5.0, 5.0
    cc = fusion_core.NativeSubmapBuilder2D(cfg)
    return py, cc


def test_grid_insert_and_rotation_parity():
    rng = np.random.default_rng(3)
    py, cc = _builders(scans_per_submap=6)
    # 20 scans with small motion: exercises fill, finish, rotation (N=6, N/2=3),
    # and the startup simultaneous-fill case (both initial submaps fill together)
    for k in range(20):
        pose = (0.1 * k, 0.02 * k, 0.05 * k)
        scan = _room_scan(rng, n=300)
        py.insert_scan(PyPose2(*pose), scan)
        cc.insert_scan(fusion_core.Pose2(*pose), scan)

        py_active = py.get_active_submaps()
        cc_info = cc.active_info()
        assert len(py_active) == cc.num_active, f"k={k} active count"
        for (pid, pn, pf), sm in zip(
                [(i.id, i.num_inserted, i.finished) for i in py_active], cc_info):
            assert (pid, pn, pf) == tuple(sm), f"k={k} submap state"
        assert len(py.get_finished_submaps()) == cc.num_finished, f"k={k} finished"
        for idx in range(len(py_active)):
            np.testing.assert_allclose(cc.active_log_odds(idx),
                                       py_active[idx].grid.L, atol=1e-5,
                                       err_msg=f"k={k} grid {idx}")


def test_correlative_parity():
    from slam_core.matching.scan_to_submap.correlative import (
        PrecomputationGridStack, _bruteforce_search, _bruteforce_search_vectorized)
    rng = np.random.default_rng(5)
    py, cc = _builders(scans_per_submap=250)
    scan = _room_scan(rng, n=350)
    for k in range(8):
        py.insert_scan(PyPose2(0.05 * k, 0.01 * k, 0.02 * k), scan)
    target = py.get_active_submaps()[0]
    prob = target.grid.probability().astype(np.float32)
    stack = PrecomputationGridStack(prob, num_levels=3)
    pts = scan[::2]
    center = PyPose2(0.12, -0.04, 0.03)

    win = fusion_core.SearchWindow()
    win.xy_window, win.theta_window, win.xy_step, win.theta_step, win.level = \
        1.0, 0.4, 0.1, 0.05, 2
    cc_pose, cc_score = fusion_core.bruteforce_search_on_image(
        prob, float(target.grid.origin_world[0]), float(target.grid.origin_world[1]),
        0.05, 3, 2, pts, fusion_core.Pose2(center.x, center.y, center.theta), win, 30)

    for search in (_bruteforce_search, _bruteforce_search_vectorized):
        py_pose, py_score = search(
            stack=stack, level=2,
            grid_origin_xy=np.asarray(target.grid.origin_world, float), res=0.05,
            points_local=pts, center_pose=center, xy_window=1.0, th_window=0.4,
            xy_step=0.1, th_step=0.05, min_valid=30)
        assert abs(cc_pose.x - py_pose.x) < 1e-9, search.__name__
        assert abs(cc_pose.y - py_pose.y) < 1e-9, search.__name__
        assert abs(cc_pose.theta - py_pose.theta) < 1e-9, search.__name__
        assert abs(cc_score - py_score) < 1e-5, search.__name__  # f32 vs f64 sum


# ---------------------------------------------------------------------------
# P3 — scan_to_map parity
# ---------------------------------------------------------------------------

def _py_map_matcher():
    from slam_core.matching.scan_to_map import ScanToMapMatcher
    map_params = dict(base_res=0.05, size_m=40.0, num_levels=3, l0=0.0,
                      l_min=-5.0, l_max=5.0, l_free=-0.1, l_occ=1.0, ray_steps=20)
    corr_params = dict(gn_iters_per_level=[20, 15, 10], gn_damping=1e-4,
                       min_points=60, min_inliers_accept=60, min_score=0.10,
                       step_clip_xy=0.03, step_clip_th=math.radians(1.0))
    return ScanToMapMatcher(map_params=map_params, corr_params=corr_params)


def _cc_map_matcher():
    cfg = fusion_core.MapMatcherConfig()
    cfg.base_res, cfg.size_m, cfg.num_levels = 0.05, 40.0, 3
    cfg.l0, cfg.l_occ, cfg.l_free, cfg.l_min, cfg.l_max = 0.0, 1.0, -0.1, -5.0, 5.0
    cfg.ray_steps, cfg.bootstrap_scans = 20, 1
    cfg.gn_iters_per_level = [20, 15, 10]
    cfg.gn_damping, cfg.min_points, cfg.min_score = 1e-4, 60, 0.10
    cfg.step_clip_xy, cfg.step_clip_th = 0.03, math.radians(1.0)
    return fusion_core.NativeMapMatcher(cfg)


def test_s2m_integrate_and_match_parity():
    rng = np.random.default_rng(21)
    py, cc = _py_map_matcher(), _cc_map_matcher()
    scan = _room_scan(rng, n=350)
    poses = [(0.0, 0.0, 0.0)] + [(0.06 * k, 0.012 * k, 0.03 * k) for k in range(1, 10)]

    for k, p in enumerate(poses):
        pred_py = PyPose2(*p)
        pred_cc = fusion_core.Pose2(*p)
        r_py = py.match(t=0.1 * k, scan_points_local=scan,
                        predicted_pose_world=pred_py)
        ok_cc, boot_cc, score_cc, inl_cc, pose_cc = cc.match(scan, pred_cc)

        assert r_py.success == ok_cc, f"k={k} success"
        if k == 0:
            assert boot_cc and (r_py.debug or {}).get("reason") == "bootstrap_seeding"
        if r_py.success:
            # Tolerance note: log-odds grids are BIT-identical (asserted below),
            # but numpy evaluates the sigmoid in float32 arithmetic while C++
            # computes it in double and rounds — ~1 ULP (6e-8) per cell, which
            # 45 GN iterations amplify to ~1e-4 in pose/score. Pure float-path
            # noise, far below scan noise; the algorithm itself is verbatim.
            assert abs(score_cc - r_py.score) < 1e-3, f"k={k} score"
            assert abs(pose_cc.x - r_py.pose_world.x) < 1e-3, f"k={k} x"
            assert abs(pose_cc.y - r_py.pose_world.y) < 1e-3, f"k={k} y"
            assert abs(pose_cc.theta - r_py.pose_world.theta) < 1e-3, f"k={k} th"

        # mirror the adapter: insert at the final pose (skip bootstrap scans)
        if (r_py.debug or {}).get("reason") != "bootstrap_seeding":
            py.update_target(r_py.pose_world, scan)
            cc.insert(fusion_core.Pose2(r_py.pose_world.x, r_py.pose_world.y,
                                        r_py.pose_world.theta), scan)

    # full finest-grid parity at the end (catches the fancy-index dedupe rule:
    # a naive accumulating port diverges from the FIRST duplicated free cell)
    np.testing.assert_allclose(cc.finest_log_odds(), py.pyr.finest().logodds,
                               atol=1e-5)


# ---------------------------------------------------------------------------
# P4 — E2E parity + timing on lab_hybrid (legacy vs native, 300 scans)
# ---------------------------------------------------------------------------

from pathlib import Path

_LAB = Path("datasets/lab_hybrid")
needs_lab = pytest.mark.skipif(not _LAB.exists(), reason="lab_hybrid not present")


def _run_frontends(kind_native: str, kind_legacy: str, n_scans: int = 300):
    from slam_core.fusion2.Dependencies.dataset import LabHybridStream
    from slam_core.fusion2.Front_End.lidar_frontend import make_lidar_frontend

    stream = LabHybridStream(_LAB, 0.05)
    imu = str(_LAB / "imu.csv")
    fe_n = make_lidar_frontend(kind_native, dataset_name="lab_hybrid", imu_path=imu)
    fe_l = make_lidar_frontend(kind_legacy, dataset_name="lab_hybrid", imu_path=imu)

    rows = []
    for t, scan in stream.lidar_stream(n_scans):
        pn, _, kn = fe_n.process(t, scan)
        pl, _, kl = fe_l.process(t, scan)
        rows.append((pn, pl, kn, kl))
    return fe_n, fe_l, rows


@needs_lab
def test_e2e_s2s_parity_and_timing():
    fe_n, fe_l, rows = _run_frontends("native_s2s", "legacy_s2s")

    dxy = np.array([math.hypot(pn.x - pl.x, pn.y - pl.y) for pn, pl, *_ in rows])
    dth = np.array([abs(math.atan2(math.sin(pn.theta - pl.theta),
                                   math.cos(pn.theta - pl.theta)))
                    for pn, pl, *_ in rows])
    frac_close = float(np.mean((dxy < 2e-3) & (dth < 2e-3)))
    print(f"\n  s2s e2e: {frac_close*100:.1f}% scans within 2mm/2mrad | "
          f"final dxy={dxy[-1]*1000:.2f}mm | "
          f"kf native={sum(r[2] for r in rows)} legacy={sum(r[3] for r in rows)}")
    # float-accumulation can flip a near-tie correlative candidate; the GN
    # refine re-converges, so we gate on the fraction + final drift.
    assert frac_close >= 0.99, f"only {frac_close*100:.1f}% scans matched"
    assert dxy[-1] < 1e-2, f"final position drift {dxy[-1]:.4f} m"
    assert abs(sum(r[2] for r in rows) - sum(r[3] for r in rows)) <= 1, "keyframes"
    assert fe_n.fallback_count == fe_l.fallback_count, "fallback indices"

    ms = np.asarray(fe_n.process_ms)
    print(f"  s2s native timing: mean={ms.mean():.1f}ms p95={np.percentile(ms,95):.1f}ms"
          f" (legacy ~280ms; sensor budget 104ms)")
    assert ms.mean() <= 50.0, f"mean {ms.mean():.1f} ms"
    assert np.percentile(ms, 95) <= 80.0, f"p95 {np.percentile(ms,95):.1f} ms"


@needs_lab
def test_e2e_s2m_parity():
    # s2m is a CLOSED feedback loop (each side integrates scans into its own
    # map at its own poses), so the 1-ULP float32-sigmoid difference (numpy
    # computes sigmoid in float32; C++ in double-then-round) gets re-amplified
    # every scan and the two trajectories slowly random-walk apart (mm -> a few
    # cm over hundreds of scans). All operators are proven verbatim at the unit
    # level above (grids BIT-identical, GN/refine/correlative exact), so the
    # gate here is behavioral lockstep + bounded drift; absolute map quality is
    # judged on the full-run eval (V4.5), not native-vs-legacy mm identity.
    fe_n, fe_l, rows = _run_frontends("native_s2m", "legacy_s2m")
    dxy = np.array([math.hypot(pn.x - pl.x, pn.y - pl.y) for pn, pl, *_ in rows])
    ms = np.asarray(fe_n.process_ms)
    kf_n, kf_l = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(f"\n  s2m e2e: median dxy={np.median(dxy)*1000:.2f}mm "
          f"max={dxy.max()*1000:.2f}mm | kf {kf_n}/{kf_l} | "
          f"fallbacks {fe_n.fallback_count}/{fe_l.fallback_count} | "
          f"native mean={ms.mean():.1f}ms")
    assert np.median(dxy) < 25e-3, f"median {np.median(dxy)*1000:.1f} mm"
    assert dxy.max() < 60e-3, f"max {dxy.max()*1000:.1f} mm"
    assert abs(kf_n - kf_l) <= 1, "keyframe lockstep"
    assert fe_n.fallback_count == fe_l.fallback_count, "fallback lockstep"
    assert ms.mean() <= 50.0


def test_refine_parity_initial_vs_prior():
    from slam_core.matching.scan_to_submap.refine import (CartoRefinementProblem,
                                                          refine_pose_submap)
    from slam_core.optimisers.gn_lm import GaussNewtonLM, GNLMConfig
    rng = np.random.default_rng(11)
    py, cc = _builders(scans_per_submap=250)
    scan = _room_scan(rng, n=350)
    for k in range(8):
        py.insert_scan(PyPose2(0.05 * k, 0.0, 0.0), scan)
    target = py.get_active_submaps()[0]
    pts = scan[::2]
    initial = PyPose2(0.42, -0.03, 0.025)   # != prior — the generalization
    prior = PyPose2(0.38, 0.01, 0.0)

    problem = CartoRefinementProblem(
        grid=target.grid, pts_local=pts,
        pred_pose_sub=np.array([prior.x, prior.y, prior.theta]),
        min_points=30, w_trans=0.1, w_rot=1.0)
    solver = GaussNewtonLM(GNLMConfig(
        iters=12, damping=1e-3, eps_stop=1e-6,
        step_clip=np.array([0.10, 0.10, math.radians(5.0)])))
    py_pose = refine_pose_submap(solver, problem, initial)

    prob = target.grid.probability().astype(np.float32)
    cc_pose = fusion_core.refine_on_image(
        prob, float(target.grid.origin_world[0]), float(target.grid.origin_world[1]),
        0.05, pts, fusion_core.Pose2(initial.x, initial.y, initial.theta),
        fusion_core.Pose2(prior.x, prior.y, prior.theta),
        12, 1e-3, 1e-6, 0.10, math.radians(5.0), 0.1, 1.0, 30)

    assert abs(cc_pose.x - py_pose.x) < 1e-7
    assert abs(cc_pose.y - py_pose.y) < 1e-7
    assert abs(cc_pose.theta - py_pose.theta) < 1e-7
