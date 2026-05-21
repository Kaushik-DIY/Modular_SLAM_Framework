import numpy as np

from visual_slam.orbslam.slam.geometry_matchers import ProjectionFuseDiagnostics, ProjectionMatcher
from visual_slam.orbslam.slam.loop_closing import LoopClosing
from visual_slam.orbslam.slam.map_point import MapPoint

from tests.visual_slam.orbslam.test_checkpoint_2_21_loop_closing import (
    build_keyframe,
    build_loop_scene,
    make_base_points,
    make_camera,
    make_descriptors,
    make_slam_namespace,
    make_Tcw,
    seed_consistency,
    setup_tracker,
)
from visual_slam.orbslam.slam import KeyFrameDatabase, Map, load_default_vocabulary


def test_loop_fusion_imports():
    assert ProjectionMatcher.search_and_fuse_for_loop_correction is not None
    assert ProjectionFuseDiagnostics().as_dict()["projected_points"] == 0
    assert hasattr(LoopClosing, "process_keyframe")


def test_projection_fuse_adds_missing_observations():
    setup_tracker()
    slam_map = Map()
    cam = make_camera()
    descriptors = make_descriptors(30)
    loop_kf = build_keyframe(slam_map, cam, 0, make_base_points(30), descriptors, make_Tcw())
    target = build_keyframe(slam_map, cam, 1, make_base_points(30), descriptors, make_Tcw())
    for point in list(target.points):
        point.set_bad()
    target.points = [None] * len(target.points)

    diagnostics = ProjectionFuseDiagnostics()
    replace_points = [None] * len(loop_kf.get_matched_good_points())
    ProjectionMatcher.search_and_fuse_for_loop_correction(
        target,
        target.Tcw(),
        loop_kf.get_matched_good_points(),
        replace_points,
        max_descriptor_distance=100,
        diagnostics=diagnostics,
    )

    assert diagnostics.added_observations >= 20
    assert sum(point is not None for point in target.points) >= 20


def test_projection_fuse_replaces_duplicate_map_points():
    setup_tracker()
    slam_map, _, loop_kf, current_kf = build_loop_scene(n=50)
    diagnostics = ProjectionFuseDiagnostics()
    loop_points = loop_kf.get_matched_good_points()
    replace_points = [None] * len(loop_points)

    ProjectionMatcher.search_and_fuse_for_loop_correction(
        current_kf,
        make_Tcw(0.0),
        loop_points,
        replace_points,
        max_descriptor_distance=100,
        diagnostics=diagnostics,
    )

    assert diagnostics.replaced_points >= 20
    duplicate = next(point for point in replace_points if point is not None)
    keep = loop_points[replace_points.index(duplicate)]
    duplicate.replace_with(keep)
    assert duplicate.is_bad()
    assert duplicate.get_replacement() is keep
    assert keep in slam_map.get_points()


def test_projection_fuse_preserves_observations():
    setup_tracker()
    slam_map, _, loop_kf, current_kf = build_loop_scene(n=30)
    old_point = current_kf.get_point_match(0)
    loop_point = loop_kf.get_point_match(0)

    old_point.replace_with(loop_point)

    assert current_kf.get_point_match(0) is loop_point
    assert loop_point.is_in_keyframe(current_kf)
    assert old_point.is_bad()


def test_projection_fuse_recomputes_descriptor_and_depth():
    setup_tracker()
    slam_map = Map()
    cam = make_camera()
    descriptors = make_descriptors(24)
    loop_kf = build_keyframe(slam_map, cam, 0, make_base_points(24), descriptors, make_Tcw())
    target = build_keyframe(slam_map, cam, 1, make_base_points(24), descriptors, make_Tcw())
    for point in list(target.points):
        point.set_bad()
    target.points = [None] * len(target.points)

    diagnostics = ProjectionFuseDiagnostics()
    ProjectionMatcher.search_and_fuse_for_loop_correction(
        target,
        target.Tcw(),
        loop_kf.get_matched_good_points(),
        [None] * 24,
        max_descriptor_distance=100,
        diagnostics=diagnostics,
    )
    fused_point = loop_kf.get_point_match(0)

    assert fused_point.get_descriptor() is not None
    assert np.isfinite(fused_point.get_normal()).all()
    assert fused_point.max_distance >= fused_point.min_distance >= 0.0


def test_projection_fuse_updates_covisibility():
    setup_tracker()
    slam_map = Map()
    cam = make_camera()
    descriptors = make_descriptors(30)
    loop_kf = build_keyframe(slam_map, cam, 0, make_base_points(30), descriptors, make_Tcw())
    target = build_keyframe(slam_map, cam, 1, make_base_points(30), descriptors, make_Tcw())
    for point in list(target.points):
        point.set_bad()
    target.points = [None] * len(target.points)

    ProjectionMatcher.search_and_fuse_for_loop_correction(
        target,
        target.Tcw(),
        loop_kf.get_matched_good_points(),
        [None] * 30,
        max_descriptor_distance=100,
    )
    target.update_connections()

    assert loop_kf in target.get_connected_keyframes()


def test_projection_fuse_rejects_bad_or_nonvisible_points():
    setup_tracker()
    slam_map = Map()
    cam = make_camera()
    descriptors = make_descriptors(4)
    target = build_keyframe(slam_map, cam, 1, make_base_points(4), descriptors, make_Tcw())
    bad = MapPoint(np.array([0.0, 0.0, 2.0], dtype=np.float64))
    bad.set_descriptor(descriptors[0])
    bad.set_bad()
    far = MapPoint(np.array([1000.0, 1000.0, -2.0], dtype=np.float64))
    far.set_descriptor(descriptors[1])

    diagnostics = ProjectionFuseDiagnostics()
    ProjectionMatcher.search_and_fuse_for_loop_correction(
        target,
        target.Tcw(),
        [bad, far],
        [None, None],
        max_descriptor_distance=100,
        diagnostics=diagnostics,
    )

    assert diagnostics.rejected_bad_point >= 1
    assert diagnostics.rejected_not_visible >= 1


def test_projection_fuse_diagnostics_are_reported():
    diagnostics = ProjectionFuseDiagnostics()
    diagnostics.projected_points = 1
    diagnostics.visible_projected_points = 1
    diagnostics.candidate_matches = 1

    assert set(diagnostics.as_dict()) == {
        "projected_points",
        "visible_projected_points",
        "candidate_matches",
        "added_observations",
        "fused_points",
        "replaced_points",
        "rejected_bad_point",
        "rejected_not_visible",
        "rejected_descriptor",
        "rejected_scale",
        "rejected_duplicate",
    }


def test_loop_correction_uses_wider_fusion():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=60)
    loop_neighbor = build_keyframe(slam_map, cam, 10, make_base_points(60), loop_kf.des, make_Tcw())
    current_neighbor = build_keyframe(slam_map, cam, 110, make_base_points(60), current_kf.des, make_Tcw())
    for point in list(current_neighbor.points):
        point.set_bad()
    current_neighbor.points = [None] * len(current_neighbor.points)
    loop_kf.add_connection(loop_neighbor, 60)
    current_kf.add_connection(current_neighbor, 60)
    slam = make_slam_namespace(slam_map, cam, database)
    closing = LoopClosing(slam, database, consistency_threshold=0)
    corrected = {current_kf: make_Tcw(0.0), current_neighbor: make_Tcw(0.0)}

    fused = closing.loop_corrector.search_and_fuse_corrected_keyframes(
        current_kf,
        loop_kf,
        corrected,
    )

    assert fused >= 20
    assert sum(point is not None for point in current_neighbor.points) >= 20


def test_tum_smoke_no_regression_after_fusion():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=80)
    database.add(loop_kf)
    closing = LoopClosing(make_slam_namespace(slam_map, cam, database), database, consistency_threshold=0)
    seed_consistency(closing, loop_kf)

    assert closing.process_keyframe(current_kf)
    fusion = closing.last_diagnostics.fusion_diagnostics
    assert fusion.projected_points + fusion.rejected_duplicate > 0
    assert closing.last_diagnostics.optimization_result.success
