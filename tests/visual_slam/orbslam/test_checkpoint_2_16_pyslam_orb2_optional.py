import cv2
import numpy as np
import pytest

from visual_slam.orbslam.local_features import (
    BackendUnavailableError,
    OpenCVORBBackend,
    PySLAMORB2Backend,
    create_extractor_backend,
    create_orb2_feature_tracker,
)


def make_feature_image():
    image = np.zeros((320, 420), dtype=np.uint8)
    for x in range(40, 390, 50):
        cv2.circle(image, (x, 110), 16, 255, 2)
        cv2.rectangle(image, (x - 14, 210), (x + 14, 238), 180, 2)
    for y in range(45, 300, 55):
        cv2.line(image, (35, y), (385, y), 220, 2)
    return image


def test_local_feature_imports_do_not_require_orbslam2_features():
    tracker = create_orb2_feature_tracker(extractor_backend="opencv_orb", num_features=500)
    kps, des = tracker.detectAndCompute(make_feature_image())

    assert len(kps) > 50
    assert des.dtype == np.uint8
    assert des.shape == (len(kps), 32)


def test_pyslam_orb2_availability_is_boolean():
    assert isinstance(PySLAMORB2Backend.is_available(), bool)


def test_explicit_pyslam_orb2_request_fails_clearly_when_unavailable():
    if PySLAMORB2Backend.is_available():
        pytest.skip("orbslam2_features is available in this environment")

    with pytest.raises(BackendUnavailableError, match="orbslam2_features"):
        create_extractor_backend("pyslam_orb2", num_features=500)


def test_opencv_backend_still_extracts_binary_descriptors():
    backend = OpenCVORBBackend(num_features=500)
    result = backend.extract(make_feature_image())

    assert result.backend_name == "opencv_orb"
    assert len(result.keypoints) > 50
    assert result.descriptors.dtype == np.uint8
    assert result.descriptors.shape == (len(result.keypoints), 32)


def test_pyslam_orb2_backend_contract_when_available():
    if not PySLAMORB2Backend.is_available():
        pytest.skip("orbslam2_features is not installed in the project venv")

    backend = create_extractor_backend("pyslam_orb2", num_features=500)
    result = backend.extract(make_feature_image())

    assert result.backend_name == "pyslam_orb2"
    assert all(isinstance(kp, cv2.KeyPoint) for kp in result.keypoints)
    assert result.descriptors.dtype == np.uint8
    assert result.descriptors.shape == (len(result.keypoints), 32)
    assert np.all(result.octaves >= 0)
