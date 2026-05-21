from types import SimpleNamespace

import cv2
import g2o
import numpy as np

from visual_slam.orbslam.local_features import create_orb2_feature_tracker
from visual_slam.orbslam.slam import (
    FeatureTrackerShared,
    Frame,
    KeyFrame,
    LocalMapping,
    LocalMappingCore,
    Map,
    MapPoint,
    PinholeCamera,
    SensorType,
    Tracking,
)
from visual_slam.orbslam.utilities.geom_triangulation import triangulate_normalized_points


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


def make_slam():
    tracker = setup_tracker()
    cam = make_camera()
    slam_map = Map()
    slam = SimpleNamespace(
        camera=cam,
        sensor_type=SensorType.RGBD,
        map=slam_map,
        feature_tracker=tracker,
        loop_closing=None,
        local_mapping=None,
    )
    slam.tracking = Tracking(slam)
    slam.local_mapping = LocalMapping(slam)
    return slam


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


def make_points_and_descriptors(n=35):
    points = []
    descriptors = []

    for i in range(n):
        p = MapPoint(np.array([
            -0.45 + 0.025 * i,
            -0.15 + 0.02 * (i % 8),
            2.0 + 0.03 * (i % 4),
        ], dtype=np.float64))
        d = np.full(32, i % 255, dtype=np.uint8)
        p.set_descriptor(d)
        points.append(p)
        descriptors.append(d)

    return points, np.asarray(descriptors, dtype=np.uint8)


def make_frame(cam, Tcw_obs, Tcw_pose, points, descriptors, timestamp=1.0):
    frame = Frame(
        camera=cam,
        img=None,
        depth_img=None,
        pose=g2o.Isometry3d(Tcw_pose),
        timestamp=timestamp,
    )

    kps = []
    uRs = []
    for p in points:
        u, v, ur = project(Tcw_obs, p.get_position(), cam)
        kps.append(cv2.KeyPoint(float(u), float(v), 20.0, 0.0, 1.0, 0))
        uRs.append(ur)

    frame.kps = kps
    frame.kpsu = kps
    frame.des = np.asarray(descriptors, dtype=np.uint8)
    frame.depths = np.full(len(kps), 2.0, dtype=np.float32)
    frame.uRs = np.asarray(uRs, dtype=np.float32)
    frame.kps_ur = frame.uRs
    frame.points = [None] * len(kps)
    frame.outliers = np.zeros(len(kps), dtype=bool)
    frame.idxs = np.arange(len(kps), dtype=np.int32)
    return frame


def add_keyframe_with_points(slam, Tcw, kid, timestamp, points, descriptors):
    f = make_frame(slam.camera, Tcw, Tcw, points, descriptors, timestamp=timestamp)
    kf = KeyFrame(f, kid=kid)
    slam.map.add_keyframe(kf)

    for i, p in enumerate(points):
        f.points[i] = p
        kf.points[i] = p
        p.add_frame_view(f, i)
        p.add_observation(kf, i)
        p.update_info()
        slam.map.add_point(p)

    kf.update_connections()
    return f, kf


def test_triangulate_normalized_points_valid_geometry():
    cam = make_camera()

    pw = np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.2], [-0.1, 0.1, 2.1]], dtype=np.float64)

    T1 = make_Tcw(0.0)
    T2 = make_Tcw(0.1)

    def norm_project(T, pts):
        out = []
        for p in pts:
            pc = T[:3, :3] @ p + T[:3, 3]
            out.append([pc[0] / pc[2], pc[1] / pc[2]])
        return np.asarray(out, dtype=np.float64)

    kpn1 = norm_project(T1, pw)
    kpn2 = norm_project(T2, pw)

    pts3d, mask = triangulate_normalized_points(T1, T2, kpn1, kpn2)

    assert np.count_nonzero(mask) == 3
    np.testing.assert_allclose(pts3d[mask], pw[mask], atol=1e-5)


def test_local_mapping_core_process_new_keyframe_updates_observations_and_connections():
    slam = make_slam()
    points, descriptors = make_points_and_descriptors(30)

    f0, kf0 = add_keyframe_with_points(slam, make_Tcw(0.0), 0, 1.0, points, descriptors)

    f1 = make_frame(slam.camera, make_Tcw(0.1), make_Tcw(0.1), points, descriptors, timestamp=2.0)
    kf1 = KeyFrame(f1, kid=1)
    slam.map.add_keyframe(kf1)

    for i, p in enumerate(points):
        f1.points[i] = p
        kf1.points[i] = p

    core = LocalMappingCore(slam.map, SensorType.RGBD)
    core.kf_cur = kf1
    core.process_new_keyframe()

    assert all(p.is_in_keyframe(kf1) for p in points)
    assert len(kf1.get_covisible_keyframes()) >= 1


def test_local_mapping_core_culls_bad_recent_points():
    slam = make_slam()
    core = LocalMappingCore(slam.map, SensorType.RGBD)

    points, descriptors = make_points_and_descriptors(5)
    f0, kf0 = add_keyframe_with_points(slam, make_Tcw(0.0), 0, 1.0, points, descriptors)

    core.kf_cur = kf0

    bad_point = MapPoint(np.array([0.0, 0.0, 2.0]))
    bad_point.set_bad()
    slam.map.add_point(bad_point)
    core.add_points([bad_point])

    count = core.cull_map_points()

    assert count == 1
    assert bad_point.is_bad()


def test_local_mapping_core_local_ba_runs():
    slam = make_slam()
    points, descriptors = make_points_and_descriptors(30)

    f0, kf0 = add_keyframe_with_points(slam, make_Tcw(0.0), 0, 1.0, points, descriptors)

    f1 = make_frame(slam.camera, make_Tcw(0.1), make_Tcw(0.12, -0.02, 0.04), points, descriptors, timestamp=2.0)
    kf1 = KeyFrame(f1, kid=1)
    slam.map.add_keyframe(kf1)

    for i, p in enumerate(points):
        f1.points[i] = p
        kf1.points[i] = p
        p.add_observation(kf1, i)

    kf0.update_connections()
    kf1.update_connections()

    core = LocalMappingCore(slam.map, SensorType.RGBD)
    core.kf_cur = kf1

    err, tracked = core.local_BA()

    assert np.isfinite(err)
    assert tracked >= 20


def test_local_mapping_wrapper_do_local_mapping_sequence():
    slam = make_slam()

    points, descriptors = make_points_and_descriptors(35)
    f0, kf0 = add_keyframe_with_points(slam, make_Tcw(0.0), 0, 1.0, points, descriptors)

    f1 = make_frame(slam.camera, make_Tcw(0.1), make_Tcw(0.1), points, descriptors, timestamp=2.0)
    kf1 = KeyFrame(f1, kid=1)
    slam.map.add_keyframe(kf1)

    for i, p in enumerate(points):
        f1.points[i] = p
        kf1.points[i] = p

    lm = slam.local_mapping
    lm.kf_cur = kf1
    lm.do_local_mapping()

    assert lm.last_num_culled_points is not None
    assert lm.last_num_triangulated_points is not None
    assert lm.last_num_fused_points is not None
    assert lm.last_num_culled_keyframes is not None
    assert np.isfinite(lm.mean_ba_chi2_error)
    assert kf1.num_tracked_points(3) >= 20


def test_local_mapping_queue_step_processes_keyframe():
    slam = make_slam()

    points, descriptors = make_points_and_descriptors(35)
    f0, kf0 = add_keyframe_with_points(slam, make_Tcw(0.0), 0, 1.0, points, descriptors)

    f1 = make_frame(slam.camera, make_Tcw(0.1), make_Tcw(0.1), points, descriptors, timestamp=2.0)
    kf1 = KeyFrame(f1, kid=1)
    slam.map.add_keyframe(kf1)

    for i, p in enumerate(points):
        f1.points[i] = p
        kf1.points[i] = p

    lm = slam.local_mapping
    lm.insert_keyframe(kf1)
    lm.step()

    assert lm.is_idle()
    assert lm.last_num_culled_points is not None
    assert lm.last_num_fused_points is not None
