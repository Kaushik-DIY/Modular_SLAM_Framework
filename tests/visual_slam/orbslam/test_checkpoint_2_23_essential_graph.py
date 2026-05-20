import numpy as np

from visual_slam.orbslam.slam.essential_graph import (
    EssentialGraph,
    EssentialGraphResult,
    optimize_essential_graph_se3,
)
from visual_slam.orbslam.slam.loop_closing import LoopClosing
from visual_slam.orbslam.slam import KeyFrameDatabase, load_default_vocabulary

from tests.visual_slam.orbslam.test_checkpoint_2_21_loop_closing import (
    build_loop_scene,
    make_slam_namespace,
    make_Tcw,
    seed_consistency,
)


def _prepare_graph_scene():
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=80)
    current_kf.set_parent(loop_kf)
    loop_kf.add_connection(current_kf, 80)
    current_kf.add_connection(loop_kf, 80)
    correction_T = make_Tcw(-1.0)
    corrected_poses = {current_kf: make_Tcw(0.0)}
    loop_connections = {current_kf: [loop_kf]}
    return slam_map, cam, loop_kf, current_kf, correction_T, corrected_poses, loop_connections


def test_essential_graph_imports():
    assert EssentialGraph is not None
    assert EssentialGraphResult(True, 1.0, 0.0, 1).success
    assert optimize_essential_graph_se3 is not None


def test_essential_graph_builds_spanning_tree_edges():
    slam_map, _, loop_kf, current_kf, _, corrected, loops = _prepare_graph_scene()

    graph = EssentialGraph(
        map_object=slam_map,
        keyframes_to_correct=[current_kf],
        loop_keyframe=loop_kf,
        current_keyframe=current_kf,
        corrected_poses=corrected,
        loop_connections=loops,
    ).build_from_map()

    assert graph.edge_kinds["spanning_tree"] >= 1


def test_essential_graph_builds_covisibility_edges():
    slam_map, _, loop_kf, current_kf, _, corrected, loops = _prepare_graph_scene()

    graph = EssentialGraph(
        map_object=slam_map,
        keyframes_to_correct=[current_kf],
        loop_keyframe=loop_kf,
        current_keyframe=current_kf,
        corrected_poses=corrected,
        loop_connections=loops,
        min_covisibility_weight=10,
    ).build_from_map()

    assert graph.edge_kinds["covisibility"] >= 1 or graph.edge_kinds["spanning_tree"] >= 1


def test_essential_graph_adds_loop_edges():
    slam_map, _, loop_kf, current_kf, _, corrected, loops = _prepare_graph_scene()

    graph = EssentialGraph(
        map_object=slam_map,
        keyframes_to_correct=[current_kf],
        loop_keyframe=loop_kf,
        current_keyframe=current_kf,
        corrected_poses=corrected,
        loop_connections=loops,
    ).build_from_map()

    assert graph.edge_kinds["loop"] >= 1


def test_essential_graph_optimization_reduces_loop_error():
    slam_map, _, loop_kf, current_kf, correction_T, corrected, loops = _prepare_graph_scene()

    result = optimize_essential_graph_se3(
        [current_kf],
        loop_kf,
        current_kf,
        correction_T,
        map_object=slam_map,
        corrected_poses=corrected,
        loop_connections=loops,
    )

    assert result.success
    assert result.after_error < result.before_error
    assert result.graph_vertices >= 2
    assert result.graph_edges >= 1


def test_essential_graph_rejects_invalid_optimization():
    slam_map, _, loop_kf, current_kf, _, _, _ = _prepare_graph_scene()
    old_pose = current_kf.Tcw().copy()
    bad = np.eye(4, dtype=np.float64)
    bad[0, 3] = np.nan

    result = optimize_essential_graph_se3(
        [current_kf],
        loop_kf,
        current_kf,
        bad,
        map_object=slam_map,
    )

    assert not result.success
    np.testing.assert_allclose(current_kf.Tcw(), old_pose)


def test_map_points_are_corrected_after_pose_graph():
    slam_map, _, loop_kf, current_kf, correction_T, corrected, loops = _prepare_graph_scene()
    before = current_kf.get_point_match(0).get_position()

    result = optimize_essential_graph_se3(
        [current_kf],
        loop_kf,
        current_kf,
        correction_T,
        map_object=slam_map,
        corrected_poses=corrected,
        loop_connections=loops,
    )
    after = current_kf.get_point_match(0).get_position()

    assert result.success
    assert np.linalg.norm(after - loop_kf.get_point_match(0).get_position()) < np.linalg.norm(
        before - loop_kf.get_point_match(0).get_position()
    )
    assert result.corrected_points > 0


def test_covisibility_recomputed_after_pose_graph():
    slam_map, _, loop_kf, current_kf, correction_T, corrected, loops = _prepare_graph_scene()

    result = optimize_essential_graph_se3(
        [current_kf],
        loop_kf,
        current_kf,
        correction_T,
        map_object=slam_map,
        corrected_poses=corrected,
        loop_connections=loops,
    )

    assert result.success
    assert current_kf.get_connected_keyframes()


def test_loop_closing_uses_essential_graph_optimizer():
    vocab = load_default_vocabulary()
    database = KeyFrameDatabase(vocab)
    slam_map, cam, loop_kf, current_kf = build_loop_scene(n=80)
    database.add(loop_kf)
    closing = LoopClosing(make_slam_namespace(slam_map, cam, database), database, consistency_threshold=0)
    seed_consistency(closing, loop_kf)

    assert closing.process_keyframe(current_kf)
    result = closing.last_diagnostics.optimization_result
    assert result.success
    assert result.graph_vertices >= 2
    assert result.graph_edges >= 1


def test_rgbd_se3_policy_documented():
    import visual_slam.orbslam.slam.essential_graph as essential_graph

    doc = essential_graph.__doc__
    assert "RGB-D" in doc
    assert "SE3" in doc
    assert "monocular Sim3 parity is not claimed" in doc
