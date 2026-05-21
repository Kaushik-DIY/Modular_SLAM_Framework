import cv2
import numpy as np

from visual_slam.orbslam.local_features import (
    FeatureDescriptorTypes,
    FeatureDetectorTypes,
    FeatureMatcherTypes,
    FeatureTrackerConfigs,
    FeatureTrackerTypes,
    create_orb2_feature_tracker,
)
from visual_slam.orbslam.slam import FeatureTrackerShared, Parameters


def make_feature_image(seed=7):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)

    # Add stable geometric structure.
    for x in range(40, 600, 80):
        cv2.circle(image, (x, 240), 20, 255, 2)
    for y in range(40, 440, 80):
        cv2.line(image, (60, y), (580, y), 180, 2)

    return image


def test_orb2_config_matches_pyslam_subset():
    cfg = FeatureTrackerConfigs.get_config_from_name("ORB2")

    assert cfg["num_features"] == Parameters.kNumFeatures
    assert cfg["num_levels"] == 8
    assert abs(cfg["scale_factor"] - 1.2) < 1e-12
    assert cfg["detector_type"] == FeatureDetectorTypes.ORB2
    assert cfg["descriptor_type"] == FeatureDescriptorTypes.ORB2
    assert cfg["tracker_type"] == FeatureTrackerTypes.DES_BF
    assert cfg["matcher_type"] == FeatureMatcherTypes.DES_BF
    assert cfg["deterministic"] is False


def test_orb2_feature_tracker_detects_binary_descriptors():
    tracker = create_orb2_feature_tracker()
    image = make_feature_image()

    keypoints, descriptors = tracker.detectAndCompute(image)

    assert len(keypoints) > 100
    assert descriptors is not None
    assert descriptors.dtype == np.uint8
    assert descriptors.ndim == 2
    assert descriptors.shape[1] == 32
    assert len(keypoints) == descriptors.shape[0]


def test_feature_manager_scale_statistics_match_orbslam_pattern():
    tracker = create_orb2_feature_tracker()
    fm = tracker.feature_manager

    assert fm.num_levels == 8
    assert abs(fm.scale_factor - 1.2) < 1e-12
    assert abs(fm.inv_scale_factor - (1.0 / 1.2)) < 1e-12

    expected = np.array([1.2 ** i for i in range(8)], dtype=np.float32)
    np.testing.assert_allclose(fm.scale_factors, expected, rtol=1e-6)
    np.testing.assert_allclose(fm.inv_scale_factors, 1.0 / expected, rtol=1e-6)
    np.testing.assert_allclose(fm.level_sigmas2, expected ** 2, rtol=1e-6)
    np.testing.assert_allclose(fm.inv_level_sigmas2, 1.0 / (expected ** 2), rtol=1e-6)


def test_feature_tracker_matching_self_image():
    tracker = create_orb2_feature_tracker()
    image = make_feature_image()

    kps, des = tracker.detectAndCompute(image)
    result = tracker.matcher.match(image, image, des, des, kps, kps, ratio_test=0.9)

    assert len(result.idxs1) > 100
    assert len(result.idxs1) == len(result.idxs2)
    assert len(result.distances) == len(result.idxs1)
    assert np.min(result.distances) == 0


def test_feature_tracker_shared_state():
    FeatureTrackerShared.reset()

    tracker = create_orb2_feature_tracker()
    FeatureTrackerShared.set_feature_tracker(tracker)

    assert FeatureTrackerShared.feature_tracker is tracker
    assert FeatureTrackerShared.feature_manager is tracker.feature_manager
    assert FeatureTrackerShared.feature_matcher is tracker.matcher
    assert FeatureTrackerShared.descriptor_distance is tracker.feature_manager.descriptor_distance
    assert FeatureTrackerShared.descriptor_distances is tracker.feature_manager.descriptor_distances
    assert FeatureTrackerShared.oriented_features is True
    assert Parameters.kMaxDescriptorDistance == 100

    FeatureTrackerShared.reset()
    assert FeatureTrackerShared.feature_tracker is None
