import cv2
import numpy as np

from visual_slam.orbslam.local_features import (
    FeatureExtractionResult,
    OpenCVORBBackend,
    create_orb2_feature_tracker,
)
from visual_slam.orbslam.slam import FeatureTrackerShared, Frame, PinholeCamera, SensorType


def make_camera():
    return PinholeCamera.from_params(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
        th_depth=40.0,
    )


def make_tum_like_image(seed=21):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 128, size=(480, 640), dtype=np.uint8)
    for x in range(60, 620, 70):
        cv2.circle(image, (x, 180), 18, 255, 2)
        cv2.rectangle(image, (x - 18, 300), (x + 18, 336), 190, 2)
    for y in range(60, 440, 70):
        cv2.line(image, (80, y), (560, y), 220, 2)
    return image


def test_opencv_backend_contract_and_descriptor_shape():
    backend = OpenCVORBBackend(num_features=800, num_levels=8, scale_factor=1.2)
    result = backend.extract(make_tum_like_image())

    assert isinstance(result, FeatureExtractionResult)
    assert result.backend_name == "opencv_orb"
    assert isinstance(result.keypoints, list)
    assert all(isinstance(kp, cv2.KeyPoint) for kp in result.keypoints)
    assert isinstance(result.descriptors, np.ndarray)
    assert result.descriptors.dtype == np.uint8
    assert result.descriptors.ndim == 2
    assert result.descriptors.shape[1] == 32
    assert result.descriptors.shape[0] == len(result.keypoints)
    assert len(result.octaves) == len(result.keypoints)
    assert len(result.angles) == len(result.keypoints)
    assert len(result.sizes) == len(result.keypoints)


def test_feature_tracker_uses_backend_contract():
    tracker = create_orb2_feature_tracker(extractor_backend="opencv_orb", num_features=800)
    image = make_tum_like_image()

    result = tracker.extract(image)
    keypoints, descriptors = tracker.detectAndCompute(image)

    assert result.backend_name == "opencv_orb"
    assert len(keypoints) == len(result.keypoints)
    assert descriptors.dtype == np.uint8
    assert descriptors.shape == result.descriptors.shape
    assert tracker.feature_manager.extractor_backend_name == "opencv_orb"


def test_frame_feature_metadata_populated_from_tracker():
    FeatureTrackerShared.reset()
    tracker = create_orb2_feature_tracker(extractor_backend="opencv_orb", num_features=800)
    FeatureTrackerShared.set_feature_tracker(tracker)

    frame = Frame(camera=make_camera(), img=make_tum_like_image(), timestamp=1.0)

    assert len(frame.kps) > 100
    assert len(frame.kpsu) == len(frame.kps)
    assert frame.des.dtype == np.uint8
    assert frame.des.shape == (len(frame.kps), 32)
    assert frame.octaves.shape == (len(frame.kps),)
    assert frame.angles.shape == (len(frame.kps),)
    assert frame.sizes.shape == (len(frame.kps),)

    FeatureTrackerShared.reset()


def test_repeated_extraction_feature_count_is_stable():
    backend = OpenCVORBBackend(num_features=800, num_levels=8, scale_factor=1.2)
    image = make_tum_like_image()

    counts = [len(backend.extract(image).keypoints) for _ in range(5)]
    assert max(counts) - min(counts) <= max(2, int(0.02 * max(counts)))
