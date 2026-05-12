"""Checkpoint 2.28A — Loop final projection expansion validation.

Verifies that ``ProjectionMatcher.search_more_map_points_by_projection`` is
called with the correct *current-keyframe corrected Tcw* in
``LoopGeometryChecker.check_candidates`` (loop_closing.py), and that the
final loop-acceptance gate counts matches *after* the projection expansion,
not just the SE3 RANSAC seed inliers.

See ``visual_slam/reference_audit/checkpoint_2_28A`` for the audit
documenting the world-drift transform convention used by this codebase.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    KeyFrame,
    KeyFrameDatabase,
    Map,
    MapPoint,
    PinholeCamera,
    SensorType,
    load_default_vocabulary,
)
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.geometry_matchers import (
    ProjectionMatcher,
    _search_more_map_points_by_projection,
)
from visual_slam.orbslam.slam.loop_closing import LoopGeometryChecker


# ---------------------------------------------------------------------------
# Test-only synthetic-scene helpers (kept local so failures are self-contained)
# ---------------------------------------------------------------------------


def _setup_tracker():
    FeatureTrackerShared.reset()
    tracker = create_orb2_feature_tracker()
    FeatureTrackerShared.set_feature_tracker(tracker)
    return tracker


def _make_camera() -> PinholeCamera:
    return PinholeCamera.from_params(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
        th_depth=40.0,
    )


def _make_Tcw(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def _make_descriptors(n: int = 80, seed: int = 22801) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 32), dtype=np.uint8)


def _make_grid_points(n: int = 80, depth: float = 2.0) -> list[np.ndarray]:
    out = []
    for i in range(n):
        out.append(
            np.array(
                [
                    -0.45 + 0.09 * (i % 10),
                    -0.25 + 0.07 * ((i // 10) % 8),
                    depth + 0.04 * (i % 5),
                ],
                dtype=np.float64,
            )
        )
    return out


def _project(Tcw: np.ndarray, point_w: np.ndarray, cam: PinholeCamera) -> tuple[float, float, float]:
    pc = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    u = cam.fx * pc[0] / pc[2] + cam.cx
    v = cam.fy * pc[1] / pc[2] + cam.cy
    ur = u - cam.bf / pc[2]
    return float(u), float(v), float(ur)


def _build_keyframe(
    slam_map: Map,
    cam: PinholeCamera,
    frame_id: int,
    positions_world: list[np.ndarray],
    descriptors: np.ndarray,
    Tcw: np.ndarray,
    *,
    attach_map_points: bool,
) -> KeyFrame:
    """Build a KeyFrame whose keypoints are the projection of ``positions_world``
    through ``Tcw``.  If ``attach_map_points`` is True the KF owns one
    MapPoint per keypoint; otherwise its ``points`` list is all-None so the
    KF acts as the *current* KF where new projection matches can be added.
    """
    frame = Frame(
        camera=cam,
        img=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_img=None,
        pose=g2o.Isometry3d(Tcw),
        id=frame_id,
        timestamp=float(frame_id),
    )
    kps = []
    uRs = []
    for point in positions_world:
        u, v, ur = _project(Tcw, point, cam)
        kps.append(cv2.KeyPoint(float(u), float(v), 20.0, 0.0, 1.0, 0))
        uRs.append(ur)
    frame.kps = kps
    frame.kpsu = kps
    frame.des = np.asarray(descriptors, dtype=np.uint8)
    frame.depths = np.full(len(kps), 2.0, dtype=np.float32)
    frame.uRs = np.asarray(uRs, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.octaves = np.zeros(len(kps), dtype=np.int32)
    frame.angles = np.zeros(len(kps), dtype=np.float32)
    frame.sizes = np.full(len(kps), 20.0, dtype=np.float32)
    frame.points = [None] * len(kps)
    frame.outliers = np.zeros(len(kps), dtype=bool)

    keyframe = KeyFrame(frame, kid=frame_id)
    slam_map.add_keyframe(keyframe)

    if attach_map_points:
        for idx, (position, descriptor) in enumerate(zip(positions_world, descriptors)):
            point = MapPoint(position.copy())
            point.set_descriptor(descriptor)
            point.add_observation(keyframe, idx)
            point.update_info()
            keyframe.points[idx] = point
            frame.points[idx] = point
            slam_map.add_point(point)

    keyframe.update_connections()
    return keyframe


def _build_anchor_map_points(
    slam_map: Map,
    anchor_kf: KeyFrame,
    positions_world: list[np.ndarray],
    descriptors: np.ndarray,
) -> list[MapPoint]:
    """Build a MapPoint per world position, anchored on a separate KF so
    update_info() can compute valid normal / distance invariance ranges.
    Returned points are NOT attached to ``anchor_kf.points``.
    """
    points = []
    for idx, (position, descriptor) in enumerate(zip(positions_world, descriptors)):
        mp = MapPoint(position.copy())
        mp.set_descriptor(descriptor)
        mp.add_observation(anchor_kf, idx)
        mp.update_info()
        slam_map.add_point(mp)
        points.append(mp)
    return points


def _build_loop_scene(n: int = 80):
    """Two keyframes + matching MapPoints aligned for SE3 RANSAC.

    ``loop_kf`` observes ``base`` at world Tcw=I.
    ``current_kf`` observes ``drifted = base + (1, 0, 0)`` at Tcw = translate(-1,0,0).
    Both KFs project the *same* image-plane locations because the drift in
    the world coordinates is exactly cancelled by the translation in the
    keyframe pose.  This is the same setup used by checkpoint 2.21.
    """
    _setup_tracker()
    slam_map = Map()
    cam = _make_camera()
    descriptors = _make_descriptors(n)
    base = _make_grid_points(n)
    drifted = [point + np.array([1.0, 0.0, 0.0], dtype=np.float64) for point in base]
    loop_kf = _build_keyframe(
        slam_map, cam, 0, base, descriptors, _make_Tcw(), attach_map_points=True
    )
    current_kf = _build_keyframe(
        slam_map,
        cam,
        100,
        drifted,
        descriptors,
        _make_Tcw(-1.0, 0.0, 0.0),
        attach_map_points=True,
    )
    return slam_map, cam, loop_kf, current_kf, descriptors


def _make_slam_namespace(slam_map: Map, cam: PinholeCamera, database):
    return SimpleNamespace(
        camera=cam,
        sensor_type=SensorType.RGBD,
        map=slam_map,
        keyframe_database=database,
        feature_tracker=FeatureTrackerShared.feature_tracker,
        local_mapping=None,
    )


# ---------------------------------------------------------------------------
# 1. Direct tests on _search_more_map_points_by_projection
# ---------------------------------------------------------------------------


def test_search_more_projects_world_points_with_correct_current_tcw():
    """With Tcw_corrected = current_kf.Tcw() and matched descriptors, all
    world points project into the current camera and become matches."""
    _setup_tracker()
    slam_map = Map()
    cam = _make_camera()
    n = 40
    descriptors = _make_descriptors(n, seed=22802)
    world_points = _make_grid_points(n)

    anchor_kf = _build_keyframe(
        slam_map, cam, 0, world_points, descriptors, _make_Tcw(), attach_map_points=True
    )
    target_points = list(anchor_kf.points)

    current_kf = _build_keyframe(
        slam_map,
        cam,
        50,
        world_points,
        descriptors,
        _make_Tcw(),
        attach_map_points=False,
    )

    matched = [None] * len(current_kf.points)
    found, _, diag = _search_more_map_points_by_projection(
        target_points,
        current_kf,
        current_kf.Tcw(),
        matched,
        f_cur_matched_points_idxs=None,
        max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceMapSearch,
        return_diagnostics=True,
    )

    assert found > 0, "expected >0 matches when projecting with the correct current Tcw"
    assert diag["projected_visible_points"] >= found
    assert diag["candidate_unique_points"] == n


def test_search_more_rejects_projection_when_using_candidate_pose_wrongly():
    """Passing the *candidate* keyframe's Tcw (the result of the buggy
    composition T12 @ Tc2w in the no-drift case) projects the points into
    a frame that doesn't observe them, so search_more must produce no
    matches."""
    _setup_tracker()
    slam_map = Map()
    cam = _make_camera()
    n = 40
    descriptors = _make_descriptors(n, seed=22803)
    world_points = _make_grid_points(n)

    anchor_kf = _build_keyframe(
        slam_map, cam, 0, world_points, descriptors, _make_Tcw(), attach_map_points=True
    )
    target_points = list(anchor_kf.points)

    current_kf = _build_keyframe(
        slam_map,
        cam,
        50,
        world_points,
        descriptors,
        _make_Tcw(),
        attach_map_points=False,
    )

    candidate_Tcw = _make_Tcw(50.0, 0.0, 0.0)

    matched = [None] * len(current_kf.points)
    found, _, diag = _search_more_map_points_by_projection(
        target_points,
        current_kf,
        candidate_Tcw,
        matched,
        f_cur_matched_points_idxs=None,
        max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceMapSearch,
        return_diagnostics=True,
    )

    assert found == 0, (
        "world points must not project into the current camera when the "
        "passed Tcw is the candidate's pose; got %d matches" % found
    )
    assert diag["projected_visible_points"] == 0


# ---------------------------------------------------------------------------
# 2. End-to-end LoopGeometryChecker tests
# ---------------------------------------------------------------------------


def _last_report(checker: LoopGeometryChecker, candidate_kf: KeyFrame) -> dict:
    candidate_kid = int(getattr(candidate_kf, "kid", getattr(candidate_kf, "id", -1)))
    assert candidate_kid in checker.last_candidate_reports, (
        "candidate report missing for kid=%d" % candidate_kid
    )
    return checker.last_candidate_reports[candidate_kid]


def test_true_loop_projection_expansion_adds_or_preserves_matches():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf, _ = _build_loop_scene(n=80)
    database.add(loop_kf)
    database.compute_bow(current_kf)

    checker = LoopGeometryChecker(keyframe_database=database)

    accepted = checker.check_candidates(current_kf, [loop_kf])

    assert accepted
    report = _last_report(checker, loop_kf)
    assert report["seed_inliers"] > 0
    assert report["candidate_covisible_points"] >= report["seed_inliers"]
    # search_more must not lose matches: the total after expansion is
    # >= the seed inliers from RANSAC.
    assert report["total_final_matches"] >= report["seed_inliers"]
    assert report["accepted_or_rejected"] == "accepted"


def test_false_loop_projection_expansion_does_not_inflate_matches():
    """Two keyframes that don't actually share a scene must not be
    rescued by the projection expansion above the final gate threshold."""
    _setup_tracker()
    slam_map = Map()
    cam = _make_camera()
    n = 80
    descriptors_a = _make_descriptors(n, seed=22810)
    descriptors_b = _make_descriptors(n, seed=22811)
    points_a = _make_grid_points(n)
    points_b = [
        np.array([5.0 + 0.05 * i, 5.0 + 0.05 * i, 4.0], dtype=np.float64) for i in range(n)
    ]
    loop_kf = _build_keyframe(
        slam_map, cam, 0, points_a, descriptors_a, _make_Tcw(), attach_map_points=True
    )
    current_kf = _build_keyframe(
        slam_map,
        cam,
        100,
        points_b,
        descriptors_b,
        _make_Tcw(0.0, 0.0, 0.0),
        attach_map_points=True,
    )

    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    database.add(loop_kf)
    database.compute_bow(current_kf)

    checker = LoopGeometryChecker(keyframe_database=database)
    accepted = checker.check_candidates(current_kf, [loop_kf])

    assert not accepted
    report = _last_report(checker, loop_kf)
    # Either the seed RANSAC failed outright, or the final gate rejected
    # the candidate.  In either case the total_final_matches must be
    # below the threshold we explicitly record.
    if "total_final_matches" in report:
        assert (
            report["total_final_matches"] < Parameters.kLoopClosingMinNumMatchedMapPoints
        )
        assert report.get("accepted_or_rejected") == "rejected"


def test_final_loop_gate_uses_total_after_search_more_not_seed_only():
    """A true loop where the SE3-guided refinement already produces enough
    matches must still expose ``total_final_matches`` (the expanded total)
    in the candidate report — i.e. the gate is computed off the post-
    search-more total, not off the seed inliers alone."""
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf, _ = _build_loop_scene(n=80)
    database.add(loop_kf)
    database.compute_bow(current_kf)

    checker = LoopGeometryChecker(keyframe_database=database)
    assert checker.check_candidates(current_kf, [loop_kf])

    report = _last_report(checker, loop_kf)
    assert report["total_final_matches"] >= Parameters.kLoopClosingMinNumMatchedMapPoints
    assert report["final_gate_threshold"] == Parameters.kLoopClosingMinNumMatchedMapPoints
    # total_final_matches must be the full count, which is at least the
    # seed inliers and is the value compared against the gate.
    assert report["total_final_matches"] >= report["seed_inliers"]


def test_loop_diagnostics_report_seed_added_total_and_threshold():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    _, _, loop_kf, current_kf, _ = _build_loop_scene(n=80)
    database.add(loop_kf)
    database.compute_bow(current_kf)

    checker = LoopGeometryChecker(keyframe_database=database)
    checker.check_candidates(current_kf, [loop_kf])

    report = _last_report(checker, loop_kf)
    for key in (
        "seed_inliers",
        "candidate_covisible_points",
        "projected_visible_points",
        "new_projection_matches",
        "total_final_matches",
        "final_gate_threshold",
        "accepted_or_rejected",
    ):
        assert key in report, "missing diagnostic key %r in candidate report" % key
    assert isinstance(report["seed_inliers"], int)
    assert isinstance(report["new_projection_matches"], int)
    assert report["accepted_or_rejected"] in {"accepted", "rejected"}


def test_transform_convention_documented_in_function_docstring():
    doc = ProjectionMatcher.search_more_map_points_by_projection.__doc__
    if not doc:
        # The static wrapper is *args/**kwargs; fall back to the underlying.
        doc = _search_more_map_points_by_projection.__doc__
    assert doc, "search_more_map_points_by_projection must have a docstring"
    assert "current_keyframe_Tcw_corrected" in doc, (
        "docstring must name the explicit corrected current Tcw parameter"
    )
    assert "Tcw_current @ inv(T12)" in doc or "current_keyframe.Tcw() @ inv(T12)" in doc, (
        "docstring must spell out the corrected pose composition formula"
    )
    assert "candidate" in doc, (
        "docstring must warn against passing the candidate keyframe pose"
    )
