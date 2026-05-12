from types import SimpleNamespace

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    BoWGuidedMatcher,
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


def make_frame(descriptors, feature_vector, angles=None):
    n = len(descriptors)
    keypoints = [
        cv2.KeyPoint(float(10 + i * 3), float(20 + i), 20.0, float(0 if angles is None else angles[i]), 1.0, 0)
        for i in range(n)
    ]
    frame = SimpleNamespace(
        kps=keypoints,
        kpsu=keypoints,
        des=np.asarray(descriptors, dtype=np.uint8),
        feature_vector=feature_vector,
        f_des=feature_vector,
        angles=np.asarray([kp.angle for kp in keypoints], dtype=np.float32),
        octaves=np.zeros(n, dtype=np.int32),
        kps_ur=np.full(n, -1.0, dtype=np.float32),
        kd=None,
    )
    return frame


def make_descriptors(n, seed=20):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 32), dtype=np.uint8)


def test_backend_feature_vector_contains_descriptor_index_groups():
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error

    descriptors = make_descriptors(16)
    result = vocab.transform_with_feature_vector(descriptors, levelsup=4)
    feature_vector = DBoW3Vocabulary.feature_vector_to_dict(result.featureVector)

    assert feature_vector
    assert sorted(idx for group in feature_vector.values() for idx in group) == list(range(len(descriptors)))


def test_bow_guided_matcher_only_compares_shared_visual_words():
    descriptors1 = make_descriptors(4)
    descriptors2 = make_descriptors(4, seed=30)
    descriptors2[0] = descriptors1[0]
    descriptors2[2] = descriptors1[2]

    frame1 = make_frame(descriptors1, {10: [0, 1], 20: [2], 40: [3]})
    frame2 = make_frame(descriptors2, {10: [0, 1], 30: [2], 50: [3]})

    result = BoWGuidedMatcher().match(frame1, frame2, max_descriptor_distance=0, ratio_test=0.75)

    assert result.available
    assert result.diagnostics.shared_words == 1
    assert result.idxs1.tolist() == [0]
    assert result.idxs2.tolist() == [0]


def test_bow_guided_matcher_rejects_when_no_visual_words_overlap():
    descriptors = make_descriptors(3)
    frame1 = make_frame(descriptors, {10: [0, 1, 2]})
    frame2 = make_frame(descriptors.copy(), {20: [0, 1, 2]})

    result = BoWGuidedMatcher().match(frame1, frame2, max_descriptor_distance=0)

    assert result.available
    assert result.diagnostics.shared_words == 0
    assert len(result.idxs1) == 0


def test_bow_guided_matcher_applies_orb_orientation_filter():
    descriptors = make_descriptors(14)
    feature_vector = {10: list(range(14))}
    angles1 = np.zeros(14, dtype=np.float32)
    angles2 = np.array([0] * 10 + [40, 100, 180, 270], dtype=np.float32)
    frame1 = make_frame(descriptors, feature_vector, angles=angles1)
    frame2 = make_frame(descriptors.copy(), feature_vector, angles=angles2)

    FeatureTrackerShared.oriented_features = True
    try:
        result = BoWGuidedMatcher().match(frame1, frame2, max_descriptor_distance=0)
    finally:
        FeatureTrackerShared.oriented_features = False

    assert result.available
    assert result.diagnostics.orientation_rejects > 0
    assert len(result.idxs1) < 14


def test_relocalizer_fallback_path_is_explicit_when_bow_unavailable():
    setup_tracker()
    slam_map = Map()
    cam = make_camera()
    descriptors = make_descriptors(20)
    points = [MapPoint(np.array([0.01 * i, 0.02 * (i % 5), 2.0], dtype=np.float64)) for i in range(20)]

    frame = Frame(camera=cam, img=None, pose=g2o.Isometry3d(np.eye(4)), id=0, timestamp=0.0)
    frame.kps = [cv2.KeyPoint(float(100 + i), 100.0, 20.0, 0.0, 1.0, 0) for i in range(20)]
    frame.kpsu = frame.kps
    frame.des = descriptors
    frame.depths = np.full(20, 2.0, dtype=np.float32)
    frame.uRs = np.full(20, -1.0, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.octaves = np.zeros(20, dtype=np.int32)
    frame.angles = np.zeros(20, dtype=np.float32)
    frame.sizes = np.full(20, 20.0, dtype=np.float32)
    frame.points = points.copy()
    keyframe = KeyFrame(frame, kid=0)
    slam_map.add_keyframe(keyframe)
    for idx, point in enumerate(points):
        point.set_descriptor(descriptors[idx])
        point.add_observation(keyframe, idx)
        slam_map.add_point(point)

    query = Frame(camera=cam, img=None, pose=g2o.Isometry3d(np.eye(4)), id=1, timestamp=1.0)
    query.kps = frame.kps
    query.kpsu = frame.kps
    query.des = descriptors.copy()
    query.octaves = frame.octaves
    query.angles = frame.angles
    query.uRs = frame.uRs
    query.kps_ur = frame.kps_ur
    query.points = [None] * 20

    relocalizer = Relocalizer(slam_map)
    idxs_frame, idxs_keyframe = relocalizer.match_frame_to_keyframe(query, keyframe)

    assert len(idxs_frame) > 0
    assert len(idxs_keyframe) > 0
    assert relocalizer.last_fallback_descriptor_matching
    assert not relocalizer.last_bow_guided_matching_available


def test_bow_matcher_uses_database_vocabulary_when_feature_vector_missing():
    setup_tracker()
    vocab = load_default_vocabulary()
    assert vocab.available, vocab.error

    slam_map = Map()
    cam = make_camera()
    descriptors = make_descriptors(40)
    points = [MapPoint(np.array([0.02 * (i % 8), 0.02 * (i // 8), 2.0], dtype=np.float64)) for i in range(40)]
    frame = Frame(camera=cam, img=None, pose=g2o.Isometry3d(np.eye(4)), id=0, timestamp=0.0)
    frame.kps = [cv2.KeyPoint(float(100 + i), 100.0, 20.0, 0.0, 1.0, 0) for i in range(40)]
    frame.kpsu = frame.kps
    frame.des = descriptors
    frame.depths = np.full(40, 2.0, dtype=np.float32)
    frame.uRs = np.full(40, -1.0, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.octaves = np.zeros(40, dtype=np.int32)
    frame.angles = np.zeros(40, dtype=np.float32)
    frame.points = points.copy()
    keyframe = KeyFrame(frame, kid=0)
    database = KeyFrameDatabase(vocab)
    database.add(keyframe)

    query = make_frame(descriptors.copy(), None)
    query.compute_bow = lambda vocabulary: __import__(
        "visual_slam.orbslam.slam.bow", fromlist=["compute_bow_for_frame"]
    ).compute_bow_for_frame(query, vocabulary)

    result = BoWGuidedMatcher(database.voc).match(query, keyframe, max_descriptor_distance=0)

    assert result.available
    assert len(result.idxs1) == 40
