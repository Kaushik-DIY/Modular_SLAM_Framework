import math

import g2o
import numpy as np

from visual_slam.orbslam.slam import (
    CameraPose,
    CameraType,
    CameraUtils,
    PinholeCamera,
    SensorType,
    focal2fov,
    fov2focal,
)


def test_camera_pose_tcw_convention_and_center():
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, 3] = np.array([1.0, 2.0, 3.0])

    pose = CameraPose(g2o.Isometry3d(Tcw))

    np.testing.assert_allclose(pose.Tcw, Tcw)
    np.testing.assert_allclose(pose.Rcw, np.eye(3))
    np.testing.assert_allclose(pose.tcw, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(pose.Rwc, np.eye(3))
    np.testing.assert_allclose(pose.Ow, np.array([-1.0, -2.0, -3.0]))
    np.testing.assert_allclose(pose.get_inverse_matrix(), np.linalg.inv(Tcw))


def test_fov_focal_round_trip():
    focal = 500.0
    pixels = 640
    fov = focal2fov(focal, pixels)
    focal_back = fov2focal(fov, pixels)
    assert abs(focal_back - focal) < 1e-9


def test_pinhole_camera_from_params_rgbd():
    cam = PinholeCamera.from_params(
        width=640,
        height=480,
        fx=517.3,
        fy=516.5,
        cx=318.6,
        cy=255.3,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
        depth_map_factor=5000.0,
        th_depth=40.0,
    )

    assert cam.type == CameraType.PINHOLE
    assert cam.sensor_type == SensorType.RGBD
    assert cam.is_stereo()
    assert abs(cam.bf - 517.3 * 0.08) < 1e-12
    assert abs(cam.b - 0.08) < 1e-12
    assert abs(cam.depth_factor - 1.0 / 5000.0) < 1e-15
    assert abs(cam.depth_threshold - (cam.bf * 40.0 / cam.fx)) < 1e-12

    np.testing.assert_allclose(
        cam.K,
        np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(cam.K @ cam.Kinv, np.eye(3), atol=1e-12)


def test_project_unproject_and_stereo_projection():
    cam = PinholeCamera.from_params(
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

    xcs = np.array([[0.0, 0.0, 2.0], [0.2, -0.1, 4.0]], dtype=np.float64)

    uv, z = cam.project(xcs)
    np.testing.assert_allclose(z, np.array([2.0, 4.0]))
    np.testing.assert_allclose(uv[0], np.array([320.0, 240.0]))
    np.testing.assert_allclose(uv[1], np.array([345.0, 227.5]))

    xyz0 = cam.unproject_3d(uv[0], z[0])
    np.testing.assert_allclose(xyz0, xcs[0], atol=1e-12)

    uvu, z_st = cam.project_stereo(xcs)
    np.testing.assert_allclose(z_st, z)
    np.testing.assert_allclose(uvu[:, :2], uv)
    np.testing.assert_allclose(uvu[:, 2], uv[:, 0] - cam.bf / z)

    assert cam.is_in_image(uv[0], z[0])
    assert cam.are_in_image(uv, z).tolist() == [True, True]


def test_camera_utils_backproject_3d():
    cam = PinholeCamera.from_params(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        sensor_type=SensorType.RGBD,
        baseline=0.08,
    )

    uv = np.array([[320.0, 240.0], [345.0, 227.5]], dtype=np.float64)
    depth = np.array([2.0, 4.0], dtype=np.float64)

    xyz = CameraUtils.backproject_3d(uv, depth, cam.K)
    np.testing.assert_allclose(xyz[0], np.array([0.0, 0.0, 2.0]), atol=1e-12)
    np.testing.assert_allclose(xyz[1], np.array([0.2, -0.1, 4.0]), atol=1e-12)
