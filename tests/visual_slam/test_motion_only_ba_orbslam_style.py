from types import SimpleNamespace

import cv2
import g2o
import numpy as np

from slam_core.common.types3d import CameraIntrinsics
from visual_slam.optimizer import motion_only_ba_orbslam_style


def make_twc(tx=0.0, ty=0.0, tz=0.0):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def project_world_point(Twc, point_w, camera):
    Tcw = np.linalg.inv(Twc)
    p_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    assert p_c[2] > 0.0

    u = camera.fx * p_c[0] / p_c[2] + camera.cx
    v = camera.fy * p_c[1] / p_c[2] + camera.cy
    return float(u), float(v)


def mean_reprojection_error(Twc, points_w, keypoints, camera):
    errors = []
    for p_w, kp in zip(points_w, keypoints):
        u, v = project_world_point(Twc, p_w, camera)
        errors.append(np.linalg.norm(np.array([u, v]) - np.array(kp.pt)))
    return float(np.mean(errors))


def test_motion_only_ba_orbslam_style_converges_translation():
    camera = CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        depth_scale=5000.0,
    )

    Twc_true = make_twc(tx=0.10, ty=-0.04, tz=0.20)

    points_w = []
    keypoints = []
    map_points = []

    for iy in range(4):
        for ix in range(5):
            p_w = np.array(
                [
                    -0.4 + 0.2 * ix,
                    -0.3 + 0.2 * iy,
                    2.0 + 0.1 * ((ix + iy) % 3),
                ],
                dtype=np.float64,
            )
            u, v = project_world_point(Twc_true, p_w, camera)
            points_w.append(p_w)
            keypoints.append(cv2.KeyPoint(float(u), float(v), 20.0, -1, 1.0, 0))
            map_points.append(SimpleNamespace(position_world=p_w, is_bad=False))

    # Perturbed initial Twc.
    Twc_init = make_twc(tx=0.20, ty=-0.10, tz=0.35)

    frame = SimpleNamespace(
        pose_world=g2o.Isometry3d(Twc_init),
        keypoints=keypoints,
        map_point_matches=map_points,
    )

    err_before = mean_reprojection_error(Twc_init, points_w, keypoints, camera)

    pose_opt = motion_only_ba_orbslam_style(
        frame=frame,
        camera=camera,
        iterations=20,
        min_edges=10,
        remove_outliers=False,
    )

    assert pose_opt is not None

    Twc_opt = pose_opt.matrix()
    err_after = mean_reprojection_error(Twc_opt, points_w, keypoints, camera)

    assert err_after < err_before * 0.25
    assert np.linalg.norm(Twc_opt[:3, 3] - Twc_true[:3, 3]) < 1e-3


def test_motion_only_ba_orbslam_style_rejects_insufficient_edges():
    camera = CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        depth_scale=5000.0,
    )

    frame = SimpleNamespace(
        pose_world=g2o.Isometry3d(np.eye(4, dtype=np.float64)),
        keypoints=[],
        map_point_matches=[],
    )

    pose_opt = motion_only_ba_orbslam_style(
        frame=frame,
        camera=camera,
        iterations=5,
        min_edges=10,
    )

    assert pose_opt is None
