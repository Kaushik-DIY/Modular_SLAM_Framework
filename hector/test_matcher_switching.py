"""
Unit + integration tests for live front-end matcher switching
(scan_to_map <-> scan_to_submap) in the realtime runner.

Covers:
  - grace-window countdown (switch fires only AFTER N scans), both directions;
  - warm-start (new matcher is initialized + becomes active);
  - request edge cases (switch-to-active no-op; same-target ignore; last-wins);
  - pose continuity across a switch on real lab data (no trajectory jump).

Run: .venv/bin/python -m hector.test_matcher_switching
"""
from __future__ import annotations

import numpy as np

import hector.config as cfg
from slam_core.common.types import Pose2
from slam_core.matching.core import MatcherManager, BufferedScan
from hector.run_realtime_viz import _build_map_matcher, _build_submap_matcher
from slam_core.matching.scan_to_submap import SubmapBuilder2D


def _build_both():
    """Build both matchers exactly as the runner does (lab profile)."""
    cfg._apply_profile("lab_run_2")
    submaps = SubmapBuilder2D(
        submap_size_m=cfg.SUBMAP_SIZE_METERS, resolution=cfg.SUBMAP_RESOLUTION,
        scans_per_submap=cfg.SCANS_PER_SUBMAP, ray_steps=cfg.RAY_STEPS,
        l0=cfg.L0, l_occ=cfg.L_OCC, l_free=cfg.L_FREE, l_min=cfg.L_MIN, l_max=cfg.L_MAX,
    )
    submap_matcher = _build_submap_matcher(submaps, fast_match=True, local_solver="native")
    map_matcher = _build_map_matcher()
    return submap_matcher, map_matcher


def _synthetic_buffer(n: int, last_xy=(1.0, 0.5)) -> list[BufferedScan]:
    """A short stream of matched scans along a straight path, each carrying a small
    fixed box-shaped point cloud (enough geometry to warm-start either matcher)."""
    rng = np.random.default_rng(0)
    box = np.concatenate([
        np.column_stack([np.linspace(-2, 2, 40), np.full(40, 2.0)]),   # far wall
        np.column_stack([np.full(40, 2.0), np.linspace(-2, 2, 40)]),   # right wall
    ]).astype(float)
    out = []
    for i in range(n):
        frac = i / max(1, n - 1)
        x = frac * last_xy[0]
        y = frac * last_xy[1]
        pts = box + rng.normal(0, 0.005, box.shape)
        out.append(BufferedScan(t=float(i) * 0.1, scan_points_local=pts,
                                pose_world=Pose2(x, y, 0.0), score=0.9))
    return out


def test_grace_and_direction():
    for start_name in ("scan_to_submap", "scan_to_map"):
        submap_matcher, map_matcher = _build_both()
        active = submap_matcher if start_name == "scan_to_submap" else map_matcher
        other = map_matcher if start_name == "scan_to_submap" else submap_matcher

        mgr = MatcherManager(active_matcher=active, rolling_buffer_size=30,
                             min_buffer_for_switch=5)
        for bs in _synthetic_buffer(10):
            mgr.push_buffered_scan(bs.t, bs.scan_points_local, bs.pose_world, bs.score)

        N = 3
        mgr.request_switch(other, grace_scans=N)
        assert mgr.switch_requested and mgr.grace_remaining == N

        # First N ticks: original matcher keeps running.
        for tick in range(N):
            switched = mgr.maybe_activate_pending()
            assert switched is False, f"switched too early at tick {tick}"
            assert mgr.active_matcher is active

        # Next tick: switch fires.
        switched = mgr.maybe_activate_pending()
        assert switched is True, "switch did not fire after the grace window"
        assert mgr.active_matcher is other
        assert mgr.active_matcher.name == other.name
        assert other.is_initialized, "new matcher was not warm-started"
        assert not mgr.switch_requested
        print(f"  [ok] grace+direction {start_name} -> {other.name} "
              f"(fired exactly after {N} scans, warm-started)")


def test_request_edge_cases():
    submap_matcher, map_matcher = _build_both()
    mgr = MatcherManager(active_matcher=submap_matcher, rolling_buffer_size=30,
                         min_buffer_for_switch=5)
    for bs in _synthetic_buffer(10):
        mgr.push_buffered_scan(bs.t, bs.scan_points_local, bs.pose_world, bs.score)

    # 1) Switch to the already-active matcher = no-op (and cancels any pending).
    mgr.request_switch(submap_matcher, grace_scans=5)
    assert not mgr.switch_requested and mgr.grace_remaining == 0
    print("  [ok] switch-to-active is a no-op")

    # 2) Last-wins: request A, tick once, request B resets target+countdown.
    mgr.request_switch(map_matcher, grace_scans=5)
    mgr.maybe_activate_pending()                       # grace 5 -> 4
    assert mgr.grace_remaining == 4
    mgr.request_switch(submap_matcher, grace_scans=5)  # different target while pending
    # submap_matcher IS the active one -> this actually cancels (active==target).
    assert not mgr.switch_requested
    print("  [ok] re-request toward active cancels pending")

    # 3) Same-target re-request does NOT reset the countdown.
    mgr.request_switch(map_matcher, grace_scans=5)
    mgr.maybe_activate_pending()                       # grace 5 -> 4
    mgr.request_switch(map_matcher, grace_scans=5)     # same target -> ignored
    assert mgr.grace_remaining == 4, "same-target re-request should not reset countdown"
    print("  [ok] same-target re-request keeps the countdown")


def test_pose_continuity_on_lab():
    """Drive the real adapter pipeline on lab_run_2 and switch matcher at k=200;
    assert the trajectory does not jump at the activation scan."""
    import os
    from slam_core.dataio.dataset_catalog import load_dataset_scans
    from slam_core.matching.preprocessing import PointCloudProcessor, PointCloudProcessorConfig
    from carto.local_slam.range_to_points import ranges_to_points
    from carto.local_slam.pose_extrapolator import PoseExtrapolatorCV
    from hector.adapter import HectorLocalSlamAdapter, make_motion_filter_from_expected_velocity

    cfg._apply_profile("lab_run_2")
    profile, scans = load_dataset_scans("lab_run_2", scan_variant=cfg.DATASET_SCAN_VARIANT)
    scans = scans[:360]
    if not scans:
        print("  [skip] lab_run_2 scans not available")
        return

    submap_matcher, map_matcher = _build_both()
    # _build_both made its OWN submaps for submap_matcher; reuse that instance's builder.
    mgr = MatcherManager(active_matcher=submap_matcher, rolling_buffer_size=30,
                         min_buffer_for_switch=20)
    pp = PointCloudProcessor(PointCloudProcessorConfig(
        fixed_voxel_size=cfg.VOXEL_FIXED_SIZE, adaptive_voxel_max_size=cfg.VOXEL_ADAPTIVE_MAX_SIZE,
        adaptive_min_num_points=cfg.VOXEL_ADAPTIVE_MIN_POINTS,
        adaptive_num_iterations=cfg.VOXEL_ADAPTIVE_ITERS, enabled=cfg.VOXEL_FILTER_ENABLED))
    extrap = PoseExtrapolatorCV(max_dt=cfg.EXTRAP_MAX_DT, init_vxy=cfg.EXTRAP_INIT_VXY,
                                init_wz=cfg.EXTRAP_INIT_WZ)
    motion = make_motion_filter_from_expected_velocity(
        cfg.TARGET_INSERT_PERIOD_S, cfg.V_EXPECTED_MPS, cfg.W_EXPECTED_RPS)
    adapter = HectorLocalSlamAdapter(matcher_manager=mgr, extrapolator=extrap,
                                     motion_params=motion,
                                     use_extrapolator=getattr(cfg, "USE_EXTRAPOLATOR", True))
    first = scans[0]
    p0 = Pose2(*first["odom"]) if (profile.has_odom and first.get("odom")) else \
        Pose2(cfg.INITIAL_POSE_X, cfg.INITIAL_POSE_Y, cfg.INITIAL_POSE_THETA)
    adapter.initialize_extrapolator(float(first["t"]), p0)

    rmin = max(cfg.LIDAR_MIN_RANGE, profile.range_min)
    poses = []
    names = []
    switch_k = 200
    for k, s in enumerate(scans):
        if k == switch_k:
            mgr.request_switch(map_matcher, grace_scans=15)
        pts_raw = ranges_to_points(s["ranges"], profile.angle_min, profile.angle_inc,
                                   rmin, profile.range_max, stride=cfg.BEAM_STRIDE)
        pts, _ = pp.process(pts_raw)
        name_before = mgr.active_matcher.name
        odom = Pose2(*s["odom"]) if s.get("odom") is not None else None
        pose, result, _, _ = adapter.process_scan(
            k=k, t=float(s["t"]), scan_points_local=pts, odom_pose_world=odom,
            odom_alpha=(cfg.ODOM_ALPHA if odom is not None else 0.0))
        poses.append([pose.x, pose.y, pose.theta])
        names.append(mgr.active_matcher.name)

    poses = np.asarray(poses)
    # Find the activation index (first scan where the active name changed to scan_to_map).
    act = next((i for i in range(1, len(names)) if names[i] != names[i - 1]), None)
    assert act is not None, "switch never activated"
    assert names[act] == "scan_to_map"
    jump = float(np.hypot(*(poses[act, :2] - poses[act - 1, :2])))
    # Compare against the typical per-scan step in the surrounding window.
    steps = np.hypot(np.diff(poses[max(0, act - 20):act + 20, 0]),
                     np.diff(poses[max(0, act - 20):act + 20, 1]))
    typical = float(np.median(steps)) if len(steps) else 0.0
    print(f"  activation at k={act}; jump={jump * 100:.1f}cm, "
          f"typical step={typical * 100:.1f}cm")
    assert jump < max(0.30, 8.0 * typical + 0.05), \
        f"trajectory jumped {jump * 100:.1f}cm at the switch (typical {typical * 100:.1f}cm)"
    print(f"  [ok] pose continuous across the switch (grace warm-start)")


if __name__ == "__main__":
    print("test_grace_and_direction:")
    test_grace_and_direction()
    print("test_request_edge_cases:")
    test_request_edge_cases()
    print("test_pose_continuity_on_lab:")
    test_pose_continuity_on_lab()
    print("\nALL SWITCHING TESTS PASSED")
