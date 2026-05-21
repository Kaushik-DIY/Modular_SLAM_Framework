from pathlib import Path
import warnings
import inspect

import numpy as np

from visual_slam.orbslam.io import associate_tum_rgbd, make_tum_rgbd_camera
from visual_slam.orbslam.run_tum_rgbd_smoke import run_tum_rgbd_smoke
from visual_slam.orbslam.slam import SensorType


def test_tum_association_nearest_timestamp():
    rgb = [
        (1.00, "rgb/1.png"),
        (1.10, "rgb/2.png"),
        (1.20, "rgb/3.png"),
    ]
    depth = [
        (1.01, "depth/1.png"),
        (1.09, "depth/2.png"),
        (1.50, "depth/3.png"),
    ]

    assoc = associate_tum_rgbd(rgb, depth, max_difference=0.02)

    assert len(assoc) == 2
    assert assoc[0][1] == "rgb/1.png"
    assert assoc[0][3] == "depth/1.png"
    assert assoc[1][1] == "rgb/2.png"
    assert assoc[1][3] == "depth/2.png"


def test_tum_camera_fr1_parameters():
    cam = make_tum_rgbd_camera("rgbd_dataset_freiburg1_desk")

    assert cam.sensor_type == SensorType.RGBD
    assert abs(cam.fx - 517.3) < 1e-9
    assert abs(cam.fy - 516.5) < 1e-9
    assert abs(cam.cx - 318.6) < 1e-9
    assert abs(cam.cy - 255.3) < 1e-9
    # pySLAM stores the reciprocal of TUM DepthMapFactor.
    # TUM raw depth is converted as depth_m = raw_depth * depth_factor.
    assert abs(cam.depth_factor - (1.0 / 5000.0)) < 1e-12


def test_tum_camera_fr2_parameters():
    cam = make_tum_rgbd_camera("rgbd_dataset_freiburg2_desk")

    assert cam.sensor_type == SensorType.RGBD
    assert abs(cam.fx - 520.9) < 1e-9
    assert abs(cam.fy - 521.0) < 1e-9
    assert abs(cam.cx - 325.1) < 1e-9
    assert abs(cam.cy - 249.7) < 1e-9


def test_runner_entrypoint_signature():
    sig = inspect.signature(run_tum_rgbd_smoke)
    params = list(sig.parameters.keys())

    assert params[:5] == [
        "dataset",
        "output_dir",
        "max_frames",
        "start_index",
        "print_every",
    ]


def test_projection_matcher_rejects_nonfinite_projection_candidates():
    import cv2
    import g2o
    import numpy as np

    from visual_slam.orbslam.local_features import create_orb2_feature_tracker
    from visual_slam.orbslam.slam import FeatureTrackerShared, Frame, MapPoint
    from visual_slam.orbslam.slam.geometry_matchers import ProjectionMatcher

    FeatureTrackerShared.reset()
    FeatureTrackerShared.set_feature_tracker(create_orb2_feature_tracker())

    cam = make_tum_rgbd_camera("rgbd_dataset_freiburg1_desk")

    f_ref = Frame(camera=cam, img=None, depth_img=None, pose=g2o.Isometry3d(np.eye(4)), timestamp=1.0)
    f_cur = Frame(camera=cam, img=None, depth_img=None, pose=g2o.Isometry3d(np.eye(4)), timestamp=2.0)

    valid_mp = MapPoint(np.array([0.0, 0.0, 2.0], dtype=np.float64))
    invalid_mp = MapPoint(np.array([np.inf, 0.0, 2.0], dtype=np.float64))

    valid_desc = np.zeros(32, dtype=np.uint8)
    invalid_desc = np.ones(32, dtype=np.uint8)

    valid_mp.set_descriptor(valid_desc)
    invalid_mp.set_descriptor(invalid_desc)

    f_ref.kps = [
        cv2.KeyPoint(float(cam.cx), float(cam.cy), 20.0, 0.0, 1.0, 0),
        cv2.KeyPoint(float(cam.cx + 20.0), float(cam.cy), 20.0, 0.0, 1.0, 0),
    ]
    f_ref.kpsu = f_ref.kps
    f_ref.des = np.vstack([valid_desc, invalid_desc]).astype(np.uint8)
    f_ref.depths = np.array([2.0, 2.0], dtype=np.float32)
    f_ref.uRs = np.array([cam.cx - cam.bf / 2.0, cam.cx + 20.0 - cam.bf / 2.0], dtype=np.float32)
    f_ref.kps_ur = f_ref.uRs
    f_ref.points = [valid_mp, invalid_mp]
    f_ref.outliers = np.zeros(2, dtype=bool)
    f_ref.idxs = np.arange(2, dtype=np.int32)

    valid_mp.add_frame_view(f_ref, 0)
    invalid_mp.add_frame_view(f_ref, 1)

    f_cur.kps = [
        cv2.KeyPoint(float(cam.cx), float(cam.cy), 20.0, 0.0, 1.0, 0),
    ]
    f_cur.kpsu = f_cur.kps
    f_cur.des = np.vstack([valid_desc]).astype(np.uint8)
    f_cur.depths = np.array([2.0], dtype=np.float32)
    f_cur.uRs = np.array([cam.cx - cam.bf / 2.0], dtype=np.float32)
    f_cur.kps_ur = f_cur.uRs
    f_cur.points = [None]
    f_cur.outliers = np.zeros(1, dtype=bool)
    f_cur.idxs = np.arange(1, dtype=np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        idxs_ref, idxs_cur, count = ProjectionMatcher.search_frame_by_projection(
            f_ref,
            f_cur,
            max_reproj_distance=15.0,
            max_descriptor_distance=100,
            ratio_test=0.9,
            is_monocular=False,
        )

    assert count >= 1
    assert 0 in list(idxs_ref)
    assert all(np.isfinite(f_cur.kps[i].pt[0]) and np.isfinite(f_cur.kps[i].pt[1]) for i in idxs_cur)
