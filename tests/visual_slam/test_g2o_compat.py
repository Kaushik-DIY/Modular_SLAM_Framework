import numpy as np

from visual_slam.g2o_compat import (
    G2OCamera,
    add_camera_parameters,
    add_mono_edge,
    add_point_vertex,
    add_pose_vertex,
    add_stereo_edge,
    make_optimizer,
    optimize,
    project_mono_with_g2o_camera,
    project_stereo_with_g2o_camera,
)


def test_g2o_compat_mono_zero_residual():
    optimizer = make_optimizer(verbose=False)

    camera = G2OCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, bf=0.0)
    add_camera_parameters(optimizer, camera, parameter_id=0)

    Tcw = np.eye(4, dtype=np.float64)
    pose_v = add_pose_vertex(optimizer, vertex_id=0, Tcw=Tcw, fixed=False)

    point_w = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    point_v = add_point_vertex(
        optimizer,
        vertex_id=1,
        point_w=point_w,
        fixed=True,
        marginalized=True,
    )

    uv = project_mono_with_g2o_camera(camera, point_w)
    edge = add_mono_edge(
        optimizer,
        edge_id=0,
        point_vertex=point_v,
        pose_vertex=pose_v,
        uv=uv,
        inv_sigma2=1.0,
        parameter_id=0,
        huber_delta=np.sqrt(5.991),
    )

    result = optimize(optimizer, iterations=5)
    assert result >= 0
    assert edge.chi2() < 1e-9


def test_g2o_compat_stereo_zero_residual():
    optimizer = make_optimizer(verbose=False)

    fx = 500.0
    baseline = 0.08
    camera = G2OCamera(fx=fx, fy=fx, cx=320.0, cy=240.0, bf=fx * baseline)
    add_camera_parameters(optimizer, camera, parameter_id=0)

    Tcw = np.eye(4, dtype=np.float64)
    pose_v = add_pose_vertex(optimizer, vertex_id=0, Tcw=Tcw, fixed=False)

    point_w = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    point_v = add_point_vertex(
        optimizer,
        vertex_id=1,
        point_w=point_w,
        fixed=True,
        marginalized=True,
    )

    uvu = project_stereo_with_g2o_camera(camera, point_w)
    edge = add_stereo_edge(
        optimizer,
        edge_id=1,
        point_vertex=point_v,
        pose_vertex=pose_v,
        uvu=uvu,
        inv_sigma2=1.0,
        parameter_id=0,
        huber_delta=np.sqrt(7.815),
    )

    result = optimize(optimizer, iterations=5)
    assert result >= 0
    assert edge.chi2() < 1e-9


def test_g2o_camera_baseline_from_bf():
    camera = G2OCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, bf=40.0)
    assert abs(camera.baseline - 0.08) < 1e-12
    assert abs(camera.focal - 500.0) < 1e-12
