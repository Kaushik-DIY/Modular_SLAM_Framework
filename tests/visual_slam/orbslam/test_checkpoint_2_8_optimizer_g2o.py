import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    KeyFrame,
    MapPoint,
    PinholeCamera,
    SensorType,
    bundle_adjustment,
    global_bundle_adjustment,
    local_bundle_adjustment,
    pose_optimization,
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
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return Tcw


def project(Tcw, point_w, camera):
    p_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    assert p_c[2] > 0.0
    u = camera.fx * p_c[0] / p_c[2] + camera.cx
    v = camera.fy * p_c[1] / p_c[2] + camera.cy
    ur = u - camera.bf / p_c[2]
    return float(u), float(v), float(ur)


def make_synthetic_frame(Tcw_true, Tcw_init, n=25):
    camera = make_camera()
    frame = Frame(camera=camera, img=None, depth_img=None, pose=g2o.Isometry3d(Tcw_init), timestamp=1.0)

    keypoints = []
    points = []
    uRs = []

    for iy in range(5):
        for ix in range(5):
            if len(points) >= n:
                break
            p_w = np.array(
                [
                    -0.4 + 0.2 * ix,
                    -0.3 + 0.15 * iy,
                    2.0 + 0.05 * ((ix + iy) % 3),
                ],
                dtype=np.float64,
            )
            u, v, ur = project(Tcw_true, p_w, camera)
            keypoints.append(cv2.KeyPoint(float(u), float(v), 20.0, -1.0, 1.0, 0))
            uRs.append(ur)
            points.append(MapPoint(p_w))

    frame.kps = keypoints
    frame.kpsu = keypoints
    frame.des = np.zeros((len(keypoints), 32), dtype=np.uint8)
    frame.depths = np.full(len(keypoints), 2.0, dtype=np.float32)
    frame.uRs = np.asarray(uRs, dtype=np.float32)
    frame.points = points
    frame.outliers = np.zeros(len(keypoints), dtype=bool)
    frame.idxs = np.arange(len(keypoints), dtype=np.int32)

    return frame


def mean_reprojection_error(frame, Tcw_true=None):
    Tcw = frame.Tcw() if Tcw_true is None else Tcw_true
    errors = []

    for kp, point in zip(frame.kps, frame.points):
        u, v, _ = project(Tcw, point.get_position(), frame.camera)
        errors.append(np.linalg.norm(np.array([u, v]) - np.array(kp.pt)))

    return float(np.mean(errors))


def make_synthetic_keyframe(Tcw_true, Tcw_init, kid, timestamp):
    frame = make_synthetic_frame(Tcw_true=Tcw_true, Tcw_init=Tcw_init, n=25)
    frame.timestamp = timestamp
    kf = KeyFrame(frame, kid=kid)
    return kf


def attach_observations(kf, points, Tcw_true):
    kf.kps = []
    kf.kpsu = []
    kf.uRs = []
    kf.depths = []
    kf.des = []

    for i, point in enumerate(points):
        u, v, ur = project(Tcw_true, point.get_position(), kf.camera)
        kf.kps.append(cv2.KeyPoint(float(u), float(v), 20.0, -1.0, 1.0, 0))
        kf.kpsu.append(kf.kps[-1])
        kf.uRs.append(ur)
        kf.depths.append(2.0)
        kf.des.append(np.zeros(32, dtype=np.uint8))
        kf.points.append(point)
        point.add_observation(kf, i)

    kf.uRs = np.asarray(kf.uRs, dtype=np.float32)
    kf.kps_ur = kf.uRs  # keep kps_ur in sync — KeyFrame.__init__ copies uRs, but we rebuilt uRs here
    kf.depths = np.asarray(kf.depths, dtype=np.float32)
    kf.des = np.asarray(kf.des, dtype=np.uint8)
    kf.outliers = np.zeros(len(kf.kps), dtype=bool)


def test_pose_optimization_converges_tcw_translation():
    setup_tracker()

    Tcw_true = make_Tcw(tx=0.10, ty=-0.03, tz=0.15)
    Tcw_init = make_Tcw(tx=0.20, ty=-0.10, tz=0.30)

    frame = make_synthetic_frame(Tcw_true=Tcw_true, Tcw_init=Tcw_init, n=25)

    err_before = mean_reprojection_error(frame)

    num_inliers, mse = pose_optimization(frame, rounds=4, iterations_per_round=10)

    err_after = mean_reprojection_error(frame)

    assert num_inliers >= 20
    assert np.isfinite(mse)
    assert err_after < err_before * 0.1
    np.testing.assert_allclose(frame.Tcw()[:3, 3], Tcw_true[:3, 3], atol=1e-3)


def test_bundle_adjustment_with_fixed_points_refines_second_keyframe_pose():
    setup_tracker()

    Tcw0_true = make_Tcw(0.0, 0.0, 0.0)
    Tcw1_true = make_Tcw(0.10, 0.0, 0.0)
    Tcw1_init = make_Tcw(0.20, -0.05, 0.10)

    kf0 = make_synthetic_keyframe(Tcw0_true, Tcw0_true, kid=0, timestamp=1.0)
    kf1 = make_synthetic_keyframe(Tcw1_true, Tcw1_init, kid=1, timestamp=2.0)

    points = [MapPoint(np.array([-0.4 + 0.04 * i, -0.1 + 0.01 * i, 2.0 + 0.02 * (i % 3)])) for i in range(25)]

    kf0.points = []
    kf1.points = []
    attach_observations(kf0, points, Tcw0_true)
    attach_observations(kf1, points, Tcw1_true)

    err_before = np.linalg.norm(kf1.Tcw()[:3, 3] - Tcw1_true[:3, 3])

    mse, _ = bundle_adjustment(
        keyframes=[kf0, kf1],
        points=points,
        local_window_size=None,
        fixed_points=True,
        rounds=15,
        use_robust_kernel=True,
    )

    err_after = np.linalg.norm(kf1.Tcw()[:3, 3] - Tcw1_true[:3, 3])

    assert np.isfinite(mse)
    assert err_after < err_before * 0.1
    np.testing.assert_allclose(kf1.Tcw()[:3, 3], Tcw1_true[:3, 3], atol=1e-3)


def test_local_bundle_adjustment_runs_on_covisible_keyframes():
    setup_tracker()

    Tcw0 = make_Tcw(0.0, 0.0, 0.0)
    Tcw1 = make_Tcw(0.10, 0.0, 0.0)
    Tcw1_init = make_Tcw(0.15, -0.03, 0.05)

    kf0 = make_synthetic_keyframe(Tcw0, Tcw0, kid=0, timestamp=1.0)
    kf1 = make_synthetic_keyframe(Tcw1, Tcw1_init, kid=1, timestamp=2.0)

    points = [MapPoint(np.array([-0.4 + 0.04 * i, 0.02 * (i % 5), 2.0 + 0.02 * (i % 3)])) for i in range(25)]

    kf0.points = []
    kf1.points = []
    attach_observations(kf0, points, Tcw0)
    attach_observations(kf1, points, Tcw1)

    kf0.update_connections()
    kf1.update_connections()

    result = local_bundle_adjustment(kf1, rounds=8)

    assert result.success
    assert result.num_edges > 0
    assert result.num_inliers > 0


def test_global_bundle_adjustment_wrapper_returns_mse():
    setup_tracker()

    Tcw0 = make_Tcw(0.0, 0.0, 0.0)
    Tcw1 = make_Tcw(0.10, 0.0, 0.0)

    kf0 = make_synthetic_keyframe(Tcw0, Tcw0, kid=0, timestamp=1.0)
    kf1 = make_synthetic_keyframe(Tcw1, Tcw1, kid=1, timestamp=2.0)

    points = [MapPoint(np.array([-0.2 + 0.04 * i, 0.01 * (i % 5), 2.0])) for i in range(15)]

    kf0.points = []
    kf1.points = []
    attach_observations(kf0, points, Tcw0)
    attach_observations(kf1, points, Tcw1)

    result_dict = {}
    mse, result_dict = global_bundle_adjustment(
        keyframes=[kf0, kf1],
        points=points,
        rounds=5,
        result_dict=result_dict,
    )

    assert np.isfinite(mse)
    assert "keyframes" in result_dict
    assert "points" in result_dict
