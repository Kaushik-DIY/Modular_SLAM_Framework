from pathlib import Path

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    DBoW3Vocabulary,
    FeatureTrackerShared,
    Frame,
    KeyFrame,
    KeyFrameDatabase,
    Map,
    MapPoint,
    PinholeCamera,
    Relocalizer,
    SensorType,
    get_bow_backend_status,
    get_default_vocabulary_path,
    load_default_vocabulary,
)


def setup_tracker():
    FeatureTrackerShared.reset()
    tracker = create_orb2_feature_tracker()
    FeatureTrackerShared.set_feature_tracker(tracker)
    return tracker


def make_camera():
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


def make_Tcw(tx=0.0, ty=0.0, tz=0.0):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def project(Tcw, point_w, cam):
    pc = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    u = cam.fx * pc[0] / pc[2] + cam.cx
    v = cam.fy * pc[1] / pc[2] + cam.cy
    ur = u - cam.bf / pc[2]
    return float(u), float(v), float(ur)


def make_points_and_descriptors(n=80, seed=220):
    rng = np.random.default_rng(seed)
    points = []
    descriptors = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)

    for i in range(n):
        x = -0.45 + 0.09 * (i % 10)
        y = -0.20 + 0.08 * ((i // 10) % 8)
        z = 2.0 + 0.08 * (i % 5)
        point = MapPoint(np.array([x, y, z], dtype=np.float64))
        point.set_descriptor(descriptors[i])
        points.append(point)

    return points, descriptors


def make_frame(cam, frame_id, Tcw_obs, Tcw_pose, points, descriptors):
    frame = Frame(
        camera=cam,
        img=None,
        depth_img=None,
        pose=g2o.Isometry3d(Tcw_pose),
        id=frame_id,
        timestamp=float(frame_id),
    )

    kps = []
    uRs = []
    for point in points:
        u, v, ur = project(Tcw_obs, point.get_position(), cam)
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
    frame.idxs = np.arange(len(kps), dtype=np.int32)
    return frame


def build_keyframe(slam_map, cam, frame_id, points, descriptors, Tcw=None):
    frame = make_frame(cam, frame_id, Tcw or make_Tcw(), Tcw or make_Tcw(), points, descriptors)
    keyframe = KeyFrame(frame, kid=frame_id)
    slam_map.add_keyframe(keyframe)

    for idx, point in enumerate(points):
        frame.points[idx] = point
        keyframe.points[idx] = point
        point.add_observation(keyframe, idx)
        point.update_info()
        if point.map is None:
            slam_map.add_point(point)

    keyframe.update_connections()
    return keyframe


def test_vocabulary_installer_script_exists_and_is_safe():
    script = Path("tools/install_pyslam_vocabulary_local.sh")

    text = script.read_text()

    assert script.exists()
    assert "third_party/vocabs" in text
    assert "--clean" in text
    assert "sudo" not in text
    assert "apt " not in text
    assert "dnf " not in text
    assert "pacman" not in text


def test_vocabulary_path_discovery_and_backend_status_are_explicit():
    path = get_default_vocabulary_path()
    status = get_bow_backend_status()

    assert path.name == "ORBvoc.dbow3"
    assert "third_party/vocabs" in str(path)
    assert status.backend_name == "dbow3"
    assert isinstance(status.available, bool)
    if status.available:
        assert status.vocabulary_path.exists()
        assert status.pydbow3_path.exists()
    else:
        assert status.reason


def test_keyframe_can_compute_and_store_bow_when_backend_available():
    setup_tracker()
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error

    slam_map = Map()
    cam = make_camera()
    points, descriptors = make_points_and_descriptors(n=50)
    keyframe = build_keyframe(slam_map, cam, 0, points, descriptors)

    bow, feature_vector = keyframe.compute_bow(vocab)

    assert bow is keyframe.g_des
    assert keyframe.bow_vector is bow
    assert keyframe.f_des == feature_vector
    assert isinstance(keyframe.feature_vector, dict)
    assert len(vocab.bow_to_vec(bow)) > 0
    assert any(len(indices) > 0 for indices in keyframe.feature_vector.values())
    grouped_indices = sorted(idx for indices in keyframe.feature_vector.values() for idx in indices)
    assert grouped_indices
    assert all(0 <= idx < len(keyframe.des) for idx in grouped_indices)


def test_keyframe_database_add_erase_clear():
    setup_tracker()
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error

    slam_map = Map()
    cam = make_camera()
    points, descriptors = make_points_and_descriptors(n=50)
    keyframe = build_keyframe(slam_map, cam, 0, points, descriptors)
    database = KeyFrameDatabase(vocab)

    database.add(keyframe)
    assert any(keyframe in keyframes for keyframes in database.inverted_file.values())

    database.erase(keyframe)
    assert all(keyframe not in keyframes for keyframes in database.inverted_file.values())

    database.add(keyframe)
    database.clear()
    assert len(database.inverted_file) == 0


def test_relocalization_candidate_retrieval_returns_similar_keyframe():
    setup_tracker()
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error

    slam_map = Map()
    cam = make_camera()
    points, descriptors = make_points_and_descriptors(n=70)
    keyframe = build_keyframe(slam_map, cam, 0, points, descriptors)
    frame = make_frame(cam, 200, make_Tcw(0.1), make_Tcw(), points, descriptors)
    database = KeyFrameDatabase(vocab)
    database.add(keyframe)

    candidates = database.detect_relocalization_candidates(frame)

    assert keyframe in candidates
    assert frame.g_des is not None


def test_loop_candidate_retrieval_excludes_connected_and_recent_keyframes():
    setup_tracker()
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error

    slam_map = Map()
    cam = make_camera()
    points0, descriptors = make_points_and_descriptors(n=70)
    points_connected = [MapPoint(p.get_position()) for p in points0]
    points_recent = [MapPoint(p.get_position()) for p in points0]
    points_loop = [MapPoint(p.get_position()) for p in points0]
    for point_set in (points_connected, points_recent, points_loop):
        for p, d in zip(point_set, descriptors):
            p.set_descriptor(d)

    query = build_keyframe(slam_map, cam, 100, points0, descriptors)
    connected = build_keyframe(slam_map, cam, 0, points_connected, descriptors)
    recent = build_keyframe(slam_map, cam, 95, points_recent, descriptors)
    loop = build_keyframe(slam_map, cam, 20, points_loop, descriptors)
    query.add_connection(connected, 25)
    connected.add_connection(query, 25)

    database = KeyFrameDatabase(vocab)
    for keyframe in (connected, recent, loop):
        database.add(keyframe)

    candidates = database.detect_loop_candidates(query, min_score=0.0)

    assert loop in candidates
    assert connected not in candidates
    assert recent not in candidates


def test_missing_vocabulary_fails_clearly_without_breaking_imports(tmp_path):
    missing_path = tmp_path / "missing_ORBvoc.dbow3"
    vocab = DBoW3Vocabulary(missing_path, autoload=True)
    database = KeyFrameDatabase(vocab)

    assert not vocab.available
    assert "vocabulary file not found" in vocab.error
    assert not database.available
    assert database.detect_relocalization_candidates(Frame(camera=make_camera(), img=None)) == []


def test_relocalizer_uses_database_backed_retrieval_when_available():
    setup_tracker()
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error

    slam_map = Map()
    cam = make_camera()
    points, descriptors = make_points_and_descriptors(n=80)
    keyframe = build_keyframe(slam_map, cam, 0, points, descriptors)
    frame = make_frame(cam, 200, make_Tcw(0.10, 0.02, -0.03), make_Tcw(0.4, -0.1, 0.2), points, descriptors)
    database = KeyFrameDatabase(vocab)
    database.add(keyframe)
    relocalizer = Relocalizer(slam_map, keyframe_database=database)

    ok = relocalizer.relocalize(frame, keyframe_database=database, keyframes_map=slam_map.keyframes_map)

    assert ok
    assert relocalizer.num_relocalization_candidates == 1
    assert relocalizer.last_bow_guided_matching_available
    assert not relocalizer.last_fallback_descriptor_matching
    assert frame.kf_ref is keyframe
