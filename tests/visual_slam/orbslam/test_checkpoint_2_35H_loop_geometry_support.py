from __future__ import annotations

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.run_rgbd_slam import LOOP_GEOMETRY_TRACE_COLUMNS
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
from visual_slam.orbslam.slam.loop_closing import LoopClosing, LoopGeometryChecker
from visual_slam.orbslam.slam.sim3_solver import estimate_scale_fixed_sim3


def _setup_tracker():
    FeatureTrackerShared.reset()
    tracker = create_orb2_feature_tracker()
    FeatureTrackerShared.set_feature_tracker(tracker)
    return tracker


def _make_camera():
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


def _make_Tcw(tx=0.0, ty=0.0, tz=0.0):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def _make_descriptors(n=80, seed=23580):
    return np.random.default_rng(seed).integers(0, 256, size=(n, 32), dtype=np.uint8)


def _make_base_points(n=80):
    points = []
    for i in range(n):
        points.append(
            np.array(
                [
                    -0.45 + 0.09 * (i % 10),
                    -0.25 + 0.07 * ((i // 10) % 8),
                    2.0 + 0.04 * (i % 5),
                ],
                dtype=np.float64,
            )
        )
    return points


def _project(Tcw, point_w, cam):
    pc = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    u = cam.fx * pc[0] / pc[2] + cam.cx
    v = cam.fy * pc[1] / pc[2] + cam.cy
    ur = u - cam.bf / pc[2]
    return float(u), float(v), float(ur)


def _build_keyframe(slam_map, cam, frame_id, positions, descriptors, Tcw):
    frame = Frame(
        camera=cam,
        img=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_img=None,
        pose=g2o.Isometry3d(Tcw),
        id=frame_id,
        timestamp=float(frame_id),
    )
    kps = []
    urs = []
    for point in positions:
        u, v, ur = _project(Tcw, point, cam)
        kps.append(cv2.KeyPoint(float(u), float(v), 20.0, 0.0, 1.0, 0))
        urs.append(ur)
    frame.kps = kps
    frame.kpsu = kps
    frame.des = np.asarray(descriptors, dtype=np.uint8)
    frame.depths = np.full(len(kps), 2.0, dtype=np.float32)
    frame.uRs = np.asarray(urs, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.octaves = np.zeros(len(kps), dtype=np.int32)
    frame.angles = np.zeros(len(kps), dtype=np.float32)
    frame.sizes = np.full(len(kps), 20.0, dtype=np.float32)
    frame.points = [None] * len(kps)
    frame.outliers = np.zeros(len(kps), dtype=bool)

    keyframe = KeyFrame(frame, kid=frame_id)
    slam_map.add_keyframe(keyframe)
    for idx, (position, descriptor) in enumerate(zip(positions, descriptors)):
        point = MapPoint(position.copy())
        point.set_descriptor(descriptor)
        point.add_observation(keyframe, idx)
        point.update_info()
        keyframe.points[idx] = point
        frame.points[idx] = point
        slam_map.add_point(point)
    keyframe.update_connections()
    return keyframe


def _build_loop_scene(n=80):
    _setup_tracker()
    slam_map = Map()
    cam = _make_camera()
    descriptors = _make_descriptors(n)
    base = _make_base_points(n)
    loop_kf = _build_keyframe(slam_map, cam, 0, base, descriptors, _make_Tcw())
    drifted = [point + np.array([1.0, 0.0, 0.0], dtype=np.float64) for point in base]
    current_kf = _build_keyframe(slam_map, cam, 100, drifted, descriptors, _make_Tcw(-1.0, 0.0, 0.0))
    return slam_map, loop_kf, current_kf


def _last_report(checker: LoopGeometryChecker, candidate_kf: KeyFrame) -> dict:
    candidate_kid = int(getattr(candidate_kf, "kid", getattr(candidate_kf, "id", -1)))
    return checker.last_candidate_reports[candidate_kid]


def test_bow_guided_loop_matching_returns_valid_mappoint_pairs():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    _, loop_kf, current_kf = _build_loop_scene(n=80)
    database.add(loop_kf)
    database.compute_bow(current_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert checker.check_candidates(current_kf, [loop_kf])
    report = _last_report(checker, loop_kf)

    assert report["bow_matches_with_valid_mappoints"] >= checker.min_matches
    assert report["geometry_input_correspondences"] >= checker.min_matches


def test_se3_seed_rejects_false_geometry():
    base = np.asarray(_make_base_points(50), dtype=np.float64)
    rng = np.random.default_rng(23582)
    unrelated = rng.uniform(low=[3.0, 3.0, 5.0], high=[6.0, 6.0, 8.0], size=base.shape)

    estimate = estimate_scale_fixed_sim3(unrelated, base, max_error=0.001)

    assert (not estimate.success) or int(np.sum(estimate.inlier_mask)) < 10


def test_se3_seed_accepts_synthetic_true_geometry():
    base = np.asarray(_make_base_points(50), dtype=np.float64)
    drifted = base + np.array([1.0, 0.0, 0.0], dtype=np.float64)

    estimate = estimate_scale_fixed_sim3(drifted, base, max_error=0.01)

    assert estimate.success
    np.testing.assert_allclose(estimate.t, np.array([-1.0, 0.0, 0.0]), atol=1e-6)


def test_pose_distance_gate_does_not_reject_when_disabled_or_not_applicable():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    _, loop_kf, current_kf = _build_loop_scene(n=80)
    database.add(loop_kf)
    database.compute_bow(current_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert checker.check_candidates(current_kf, [loop_kf])
    report = _last_report(checker, loop_kf)

    assert report["passed_pose_distance_gate"] is True
    assert "pose distance" not in str(report.get("rejection_reason", "")).lower()
    assert Parameters.kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3 == 0.0


def test_projection_expansion_uses_candidate_covisibility_group():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    slam_map, loop_kf, current_kf = _build_loop_scene(n=80)
    neighbor = _build_keyframe(
        slam_map,
        _make_camera(),
        1,
        _make_base_points(20),
        _make_descriptors(20, seed=23581),
        _make_Tcw(0.1, 0.0, 0.0),
    )
    loop_kf.add_connection(neighbor, 20)
    neighbor.add_connection(loop_kf, 20)
    database.add(loop_kf)
    database.compute_bow(current_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert checker.check_candidates(current_kf, [loop_kf])
    report = _last_report(checker, loop_kf)

    assert report["candidate_group_size"] >= 2
    assert report["candidate_group_map_points"] >= report["seed_inliers"]


def test_final_support_gate_uses_configured_threshold():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error
    database = KeyFrameDatabase(vocab)
    _, loop_kf, current_kf = _build_loop_scene(n=80)
    database.add(loop_kf)
    database.compute_bow(current_kf)
    checker = LoopGeometryChecker(keyframe_database=database)

    assert checker.check_candidates(current_kf, [loop_kf])
    report = _last_report(checker, loop_kf)

    assert report["final_gate_threshold"] == Parameters.kLoopClosingMinNumMatchedMapPoints
    assert report["final_matched_map_points"] >= Parameters.kLoopClosingMinNumMatchedMapPoints


def test_loop_geometry_trace_has_required_columns():
    closing = LoopClosing.__new__(LoopClosing)
    rows = closing._build_loop_geometry_trace_rows(
        [
            {
                "current_kf_id": 100,
                "candidate_kf_id": 2,
                "gt_loop_like": True,
                "gt_translation_distance": 0.4,
                "gt_rotation_angle_deg": 6.0,
                "bow_matches_raw": 40,
                "bow_matches_with_valid_mappoints": 32,
                "seed_correspondences": 32,
                "seed_inliers": 24,
                "seed_inlier_ratio": 0.75,
                "initial_se3_translation_norm": 1.0,
                "initial_se3_rotation_deg": 0.0,
                "estimated_pose_distance_threshold": 0.0,
                "estimated_pose_rotation_threshold_deg": 0.0,
                "passed_pose_distance_gate": True,
                "guided_projection_matches": 12,
                "refined_correspondences": 36,
                "geometry_refined_inliers": 30,
                "candidate_group_size": 3,
                "candidate_group_map_points": 120,
                "visible_projected_group_points": 55,
                "final_matched_map_points": 65,
                "final_gate_threshold": 60,
                "accepted": True,
                "rejection_reason": "",
            }
        ]
    )

    assert set(LOOP_GEOMETRY_TRACE_COLUMNS).issubset(set(rows[0].keys()))
    assert rows[0]["pair_key"] == "2-100"
