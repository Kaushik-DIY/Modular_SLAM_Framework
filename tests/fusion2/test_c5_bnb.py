"""C5 — C++ correlative B&B: self-recovery, parity vs Python backend, speed gate."""
import math
import time

import numpy as np
import pytest

fusion_core = pytest.importorskip("fusion_core")

from slam_core.common.types import Pose2 as PyPose2
from slam_core.matching.scan_to_submap.branch_and_bound_backend import (
    BranchAndBoundSubmapBackend,
)
from slam_core.matching.scan_to_submap.submaps import ProbabilityGrid
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapMatchRequest,
    SubmapSearchWindow,
)
from slam_core.optimisers.gn_lm import GaussNewtonLM, GNLMConfig

WINDOW_XY = 2.0
WINDOW_TH = math.radians(15.0)
DEPTH = 6


def _room_scan(seed=0, n=400):
    """Synthetic 2D room: points on the walls of an 8x6 room seen from inside."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    side = rng.integers(0, 4, n)
    pts = np.zeros((n, 2), np.float32)
    pts[side == 0] = np.c_[t[side == 0] * 8 - 4, np.full((side == 0).sum(), -3.0)]
    pts[side == 1] = np.c_[t[side == 1] * 8 - 4, np.full((side == 1).sum(), 3.0)]
    pts[side == 2] = np.c_[np.full((side == 2).sum(), -4.0), t[side == 2] * 6 - 3]
    pts[side == 3] = np.c_[np.full((side == 3).sum(), 4.0), t[side == 3] * 6 - 3]
    pts += rng.normal(0, 0.01, pts.shape).astype(np.float32)
    return pts


def _grid_from_scans(scans, poses, res=0.05):
    sigs = [fusion_core.Signature(i, float(i), scan_xy=s) for i, s in enumerate(scans)]
    cfg = fusion_core.GridConfig()
    cfg.resolution = res
    return fusion_core.assemble_local_grid(sigs, poses, cfg)


def _cpp_cfg():
    c = fusion_core.BnbConfig()
    c.linear_search_window = WINDOW_XY
    c.angular_search_window = WINDOW_TH
    c.depth = DEPTH
    c.max_match_points = 200
    return c


def test_self_loop_recovery():
    """Grid built from a scan at identity; query = same scan rotated/translated.
    The matcher must recover the inverse offset within 2 cm / 0.5 deg."""
    scan = _room_scan(0)
    grid = _grid_from_scans([scan], [fusion_core.Pose2(0, 0, 0)])

    true_offset = (0.45, -0.3, math.radians(8))
    c, s = math.cos(true_offset[2]), math.sin(true_offset[2])
    R = np.array([[c, -s], [s, c]], np.float32)
    # query scan = world points seen from the offset pose
    query = (scan - np.float32(true_offset[:2])) @ R  # inverse transform

    res = fusion_core.bnb_match(grid, query.astype(np.float32),
                                fusion_core.Pose2(0, 0, 0), _cpp_cfg())
    assert res.success and res.coarse_score > 0.6
    assert abs(res.pose.x - true_offset[0]) < 0.02
    assert abs(res.pose.y - true_offset[1]) < 0.02
    assert abs(res.pose.theta - true_offset[2]) < math.radians(0.5)


def test_negative_case_low_score():
    grid = _grid_from_scans([_room_scan(0)], [fusion_core.Pose2(0, 0, 0)])
    # a completely different environment (long corridor far away)
    rng = np.random.default_rng(9)
    corridor = np.c_[rng.uniform(-1, 1, 300), rng.choice([-0.8, 0.8], 300)].astype(np.float32)
    res = fusion_core.bnb_match(grid, corridor, fusion_core.Pose2(0, 0, 0), _cpp_cfg())
    assert res.refined_score < 0.55  # poor occupied-space consistency


class _FakeSubmap:
    def __init__(self, grid):
        self.id = 0
        self.finished = True
        self.grid = grid


def _python_backend_on_same_grid(cpp_grid, query, pred):
    """Run the Python B&B backend on EXACTLY the same grid content."""
    pg = ProbabilityGrid(size_m=1.0, res=cpp_grid.resolution)
    L = np.asarray(cpp_grid.log_odds, dtype=np.float32).copy()
    pg.h, pg.w = L.shape
    pg.L = L
    pg.origin_world = np.array([cpp_grid.origin_x, cpp_grid.origin_y], dtype=float)
    pg.size_m = max(L.shape) * cpp_grid.resolution

    cfg = ScanToSubmapBackendConfig(
        backend_type="branch_and_bound",
        coarse=SubmapSearchWindow(xy_window=WINDOW_XY, theta_window=WINDOW_TH,
                                  xy_step=0.05, theta_step=0.02, level=0),
        bnb_depth_limit=DEPTH,
        bnb_min_rotational_step=0.02,
        max_match_points=200,
        max_refine_points=180,
        do_refine=True,
    )
    solver = GaussNewtonLM(GNLMConfig(
        iters=8, damping=1e-3, eps_stop=1e-6,
        step_clip=np.array([0.10, 0.10, math.radians(5.0)])))
    backend = BranchAndBoundSubmapBackend(cfg, solver)
    req = SubmapMatchRequest(
        scan_points_local=np.asarray(query, dtype=float),
        predicted_pose_world=PyPose2(pred[0], pred[1], pred[2]),
        submap_pose_world=PyPose2(0.0, 0.0, 0.0),
        submap=_FakeSubmap(pg),
    )
    return backend, req


def test_parity_with_python_backend():
    scan = _room_scan(3)
    grid = _grid_from_scans([scan], [fusion_core.Pose2(0, 0, 0)])
    true_offset = (0.35, 0.2, math.radians(-6))
    c, s = math.cos(true_offset[2]), math.sin(true_offset[2])
    R = np.array([[c, -s], [s, c]], np.float32)
    query = (scan - np.float32(true_offset[:2])) @ R

    cpp = fusion_core.bnb_match(grid, query.astype(np.float32),
                                fusion_core.Pose2(0, 0, 0), _cpp_cfg())
    backend, req = _python_backend_on_same_grid(grid, query, (0.0, 0.0, 0.0))
    ref = backend.match(req)

    assert cpp.success == ref.success
    assert abs(cpp.coarse_score - ref.score) < 0.02
    assert abs(cpp.pose.x - ref.pose_world.x) < 0.02
    assert abs(cpp.pose.y - ref.pose_world.y) < 0.02
    dth = abs(cpp.pose.theta - ref.pose_world.theta)
    assert math.atan2(math.sin(dth), math.cos(dth)) < math.radians(0.5)


def test_speed_gate_50x():
    """lab_hybrid-sized verification: multi-scan local grid, 200-pt query."""
    rng = np.random.default_rng(5)
    scans = [(_room_scan(i) + rng.normal(0, 0.005, (400, 2)).astype(np.float32))
             for i in range(5)]
    poses = [fusion_core.Pose2(0.3 * i, 0.1 * i, 0.05 * i) for i in range(5)]
    grid = _grid_from_scans(scans, poses)
    query = _room_scan(99)

    cfg = _cpp_cfg()
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        cpp = fusion_core.bnb_match(grid, query, fusion_core.Pose2(0.1, 0.1, 0.02), cfg)
    cpp_ms = (time.perf_counter() - t0) / n_runs * 1000

    backend, req = _python_backend_on_same_grid(grid, query, (0.1, 0.1, 0.02))
    t0 = time.perf_counter()
    ref = backend.match(req)
    py_ms = (time.perf_counter() - t0) * 1000

    speedup = py_ms / cpp_ms
    print(f"\n  grid {grid.width}x{grid.height} @ {grid.resolution} m, "
          f"{cpp.num_rotations} rotations: C++ {cpp_ms:.1f} ms vs Python {py_ms:.0f} ms "
          f"-> {speedup:.0f}x  (cpp score {cpp.coarse_score:.3f} vs py {ref.score:.3f})")
    assert speedup >= 50.0
    assert cpp_ms < 50.0  # C7 per-candidate budget
