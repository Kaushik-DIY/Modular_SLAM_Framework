"""
=============================================================================
visual_slam/orbslam/slam/optimizer_g2o.py

pySLAM-aligned g2o optimizer subset for ORB/RGB-D SLAM.

Reference:
- pySLAM: pyslam/slam/optimizer_g2o.py

Implemented in this checkpoint:
- pose_optimization(frame)
- bundle_adjustment(keyframes, points, ...)
- local_bundle_adjustment(keyframe, ...)
- global_bundle_adjustment(keyframes, points, ...)

Not implemented yet:
- optimize_sim3
- optimize_essential_graph
- loop-correction specific Sim3 graph
- multiprocessing abort synchronization
- semantic residuals

Important adaptation:
- pySLAM's g2o wrapper exposes projection edges differently.
- This implementation uses visual_slam.g2o_compat to support the installed
  parameter-based g2o API in this workspace.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import g2o
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
)
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.feature_tracker_shared import FeatureTrackerShared
from visual_slam.orbslam.slam.frame import Frame
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.map_point import MapPoint


@dataclass
class OptimizerResult:
    num_edges: int
    num_inliers: int
    num_outliers: int
    mean_squared_error: float
    success: bool


def _as_list(values) -> list:
    if values is None:
        return []
    if hasattr(values, "to_list"):
        return values.to_list()
    return list(values)


def _is_bad_keyframe(kf: KeyFrame) -> bool:
    return hasattr(kf, "is_bad") and kf.is_bad()


def _is_bad_point(point: MapPoint) -> bool:
    return point is None or (hasattr(point, "is_bad") and point.is_bad())


def _camera_to_g2o(camera) -> G2OCamera:
    bf = getattr(camera, "bf", 0.0)
    if bf is None:
        bf = 0.0
    return G2OCamera(
        fx=float(camera.fx),
        fy=float(camera.fy),
        cx=float(camera.cx),
        cy=float(camera.cy),
        bf=float(bf),
    )


def _get_observation_uv(frame_or_kf, idx: int) -> np.ndarray:
    kps = getattr(frame_or_kf, "kps", getattr(frame_or_kf, "keypoints", None))
    if kps is None:
        raise ValueError("Frame/keyframe has no keypoints.")
    kp = kps[int(idx)]
    return np.array([kp.pt[0], kp.pt[1]], dtype=np.float64)


def _get_observation_ur(frame_or_kf, idx: int) -> float:
    uRs = getattr(frame_or_kf, "uRs", getattr(frame_or_kf, "kps_ur", None))
    if uRs is None or idx < 0 or idx >= len(uRs):
        return -1.0
    return float(uRs[int(idx)])


def _get_inv_sigma2(frame_or_kf, idx: int) -> float:
    kps = getattr(frame_or_kf, "kps", getattr(frame_or_kf, "keypoints", None))
    if kps is None or idx < 0 or idx >= len(kps):
        return 1.0

    octave = max(0, int(getattr(kps[int(idx)], "octave", 0)))

    feature_manager = FeatureTrackerShared.feature_manager
    if feature_manager is None:
        return 1.0

    octave = min(octave, len(feature_manager.inv_level_sigmas2) - 1)
    return float(feature_manager.inv_level_sigmas2[octave])


def _extract_Tcw_from_pose_vertex(pose_vertex) -> np.ndarray:
    se3_opt = pose_vertex.estimate()
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = se3_opt.rotation().matrix()
    Tcw[:3, 3] = se3_opt.translation()
    return Tcw


def _set_frame_pose_from_vertex(frame_or_kf, pose_vertex) -> None:
    Tcw = _extract_Tcw_from_pose_vertex(pose_vertex)
    frame_or_kf.update_pose(g2o.Isometry3d(Tcw))


def _add_reprojection_edge(
    optimizer,
    edge_id: int,
    point_vertex,
    pose_vertex,
    frame_or_kf,
    idx: int,
    parameter_id: int = 0,
    use_robust_kernel: bool = True,
):
    uv = _get_observation_uv(frame_or_kf, idx)
    ur = _get_observation_ur(frame_or_kf, idx)
    inv_sigma2 = _get_inv_sigma2(frame_or_kf, idx)

    if ur >= 0.0:
        measurement = np.array([uv[0], uv[1], ur], dtype=np.float64)
        edge = add_stereo_edge(
            optimizer=optimizer,
            edge_id=edge_id,
            point_vertex=point_vertex,
            pose_vertex=pose_vertex,
            uvu=measurement,
            inv_sigma2=inv_sigma2,
            parameter_id=parameter_id,
            huber_delta=Parameters.kHuberStereo if use_robust_kernel else None,
        )
        chi2_threshold = Parameters.kChi2Stereo
        is_stereo = True
    else:
        edge = add_mono_edge(
            optimizer=optimizer,
            edge_id=edge_id,
            point_vertex=point_vertex,
            pose_vertex=pose_vertex,
            uv=uv,
            inv_sigma2=inv_sigma2,
            parameter_id=parameter_id,
            huber_delta=Parameters.kHuberMono if use_robust_kernel else None,
        )
        chi2_threshold = Parameters.kChi2Mono
        is_stereo = False

    return edge, chi2_threshold, is_stereo


def pose_optimization(
    frame: Frame,
    verbose: bool = False,
    rounds: int = 4,
    iterations_per_round: int = 10,
    print=print,
) -> tuple[int, float]:
    """
    pySLAM/ORB-SLAM-style motion-only pose optimization.

    Optimizes only the current frame pose Tcw. Map points are fixed.

    Returns:
        (num_inliers, mean_squared_error)
    """
    if frame is None:
        return 0, float("inf")

    points = list(getattr(frame, "points", []))
    if len(points) == 0:
        return 0, float("inf")

    optimizer = make_optimizer(verbose=verbose)
    add_camera_parameters(optimizer, _camera_to_g2o(frame.camera), parameter_id=0)

    pose_vertex = add_pose_vertex(
        optimizer=optimizer,
        vertex_id=0,
        Tcw=frame.Tcw(),
        fixed=False,
    )

    edges = []
    vertex_id = 1
    edge_id = 0

    for idx, point in enumerate(points):
        if _is_bad_point(point):
            continue

        point_vertex = add_point_vertex(
            optimizer=optimizer,
            vertex_id=vertex_id,
            point_w=point.get_position(),
            fixed=True,
            marginalized=True,
        )

        edge, chi2_threshold, is_stereo = _add_reprojection_edge(
            optimizer=optimizer,
            edge_id=edge_id,
            point_vertex=point_vertex,
            pose_vertex=pose_vertex,
            frame_or_kf=frame,
            idx=idx,
            parameter_id=0,
            use_robust_kernel=True,
        )

        edges.append((edge, idx, chi2_threshold, is_stereo))
        vertex_id += 1
        edge_id += 1

    if len(edges) < Parameters.kRelocalizationPoseOpt1MinMatches:
        return 0, float("inf")

    if not hasattr(frame, "outliers") or len(frame.outliers) != len(points):
        frame.outliers = np.zeros(len(points), dtype=bool)

    num_inliers = 0

    for round_idx in range(int(rounds)):
        try:
            optimize(optimizer, iterations=iterations_per_round, verbose=verbose)
        except Exception as exc:
            print(f"pose_optimization: g2o failed: {exc}")
            return 0, float("inf")

        num_inliers = 0

        for edge, idx, chi2_threshold, _ in edges:
            chi2 = float(edge.chi2())

            if not np.isfinite(chi2) or chi2 > chi2_threshold:
                frame.outliers[idx] = True
                edge.set_level(1)
            else:
                frame.outliers[idx] = False
                edge.set_level(0)
                num_inliers += 1

            # ORB-SLAM removes robust kernels in later rounds.
            if round_idx == 2:
                edge.set_robust_kernel(None)

        optimizer.initialize_optimization(0)

    if num_inliers < Parameters.kRelocalizationPoseOpt1MinMatches:
        return num_inliers, float("inf")

    _set_frame_pose_from_vertex(frame, pose_vertex)

    active_chi2 = [
        float(edge.chi2())
        for edge, idx, _, _ in edges
        if idx < len(frame.outliers) and not frame.outliers[idx]
    ]
    mse = float(np.mean(active_chi2)) if active_chi2 else float("inf")

    return num_inliers, mse


def _bundle_adjustment_core(
    local_keyframes: list[KeyFrame],
    fixed_keyframes: list[KeyFrame],
    points: list[MapPoint],
    fixed_points: bool = False,
    rounds: int = 10,
    use_robust_kernel: bool = False,
    verbose: bool = False,
    result_dict: Optional[dict] = None,
    print=print,
) -> OptimizerResult:
    local_keyframes = [kf for kf in local_keyframes if kf is not None and not _is_bad_keyframe(kf)]
    fixed_keyframes = [kf for kf in fixed_keyframes if kf is not None and not _is_bad_keyframe(kf)]
    points = [p for p in points if not _is_bad_point(p)]

    if len(local_keyframes) == 0 or len(points) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False)

    optimizer = make_optimizer(verbose=verbose)
    add_camera_parameters(optimizer, _camera_to_g2o(local_keyframes[0].camera), parameter_id=0)

    pose_vertices = {}
    point_vertices = {}

    vertex_id = 0

    for kf in local_keyframes:
        v = add_pose_vertex(
            optimizer=optimizer,
            vertex_id=vertex_id,
            Tcw=kf.Tcw(),
            fixed=(kf.kid == 0),
        )
        pose_vertices[kf] = v
        vertex_id += 1

    for kf in fixed_keyframes:
        if kf in pose_vertices:
            continue
        v = add_pose_vertex(
            optimizer=optimizer,
            vertex_id=vertex_id,
            Tcw=kf.Tcw(),
            fixed=True,
        )
        pose_vertices[kf] = v
        vertex_id += 1

    for point in points:
        v = add_point_vertex(
            optimizer=optimizer,
            vertex_id=vertex_id,
            point_w=point.get_position(),
            fixed=bool(fixed_points),
            marginalized=True,
        )
        point_vertices[point] = v
        vertex_id += 1

    edges = []
    edge_id = 0

    for point in points:
        point_vertex = point_vertices.get(point)
        if point_vertex is None:
            continue

        for kf, idx in point.observations():
            pose_vertex = pose_vertices.get(kf)
            if pose_vertex is None:
                continue

            edge, chi2_threshold, is_stereo = _add_reprojection_edge(
                optimizer=optimizer,
                edge_id=edge_id,
                point_vertex=point_vertex,
                pose_vertex=pose_vertex,
                frame_or_kf=kf,
                idx=idx,
                parameter_id=0,
                use_robust_kernel=use_robust_kernel,
            )

            edges.append((edge, point, kf, idx, chi2_threshold, is_stereo))
            edge_id += 1

    if len(edges) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False)

    try:
        optimize(optimizer, iterations=rounds, verbose=verbose)
    except Exception as exc:
        print(f"bundle_adjustment: g2o failed: {exc}")
        return OptimizerResult(len(edges), 0, len(edges), float("inf"), False)

    inliers = []
    outliers = []

    for edge, point, kf, idx, chi2_threshold, _ in edges:
        chi2 = float(edge.chi2())
        if np.isfinite(chi2) and chi2 <= chi2_threshold:
            inliers.append(edge)
        else:
            outliers.append((edge, point, kf, idx))

    for kf, vertex in pose_vertices.items():
        # Fixed vertices can be updated too, but their estimate is unchanged.
        if kf in local_keyframes:
            _set_frame_pose_from_vertex(kf, vertex)

    if not fixed_points:
        for point, vertex in point_vertices.items():
            point.set_position(np.asarray(vertex.estimate(), dtype=np.float64).reshape(3))
            point.update_normal_and_depth()

    if result_dict is not None:
        result_dict["keyframes"] = {kf.kid: kf.Tcw() for kf in pose_vertices.keys()}
        result_dict["points"] = {p.id: p.get_position() for p in point_vertices.keys()}

    mse = float(np.mean([float(edge.chi2()) for edge in inliers])) if inliers else float("inf")

    return OptimizerResult(
        num_edges=len(edges),
        num_inliers=len(inliers),
        num_outliers=len(outliers),
        mean_squared_error=mse,
        success=len(inliers) > 0,
    )


def bundle_adjustment(
    keyframes,
    points,
    local_window_size=None,
    fixed_points: bool = False,
    rounds: int = 10,
    loop_kf_id: int = 0,
    use_robust_kernel: bool = False,
    abort_flag=None,
    mp_abort_flag=None,
    result_dict: Optional[dict] = None,
    verbose: bool = False,
    print=print,
) -> tuple[float, Optional[dict]]:
    """
    pySLAM-compatible bundle_adjustment wrapper.

    Returns:
        (mean_squared_error, result_dict)
    """
    keyframes = _as_list(keyframes)
    points = _as_list(points)

    if local_window_size is None:
        local_keyframes = keyframes
    else:
        local_keyframes = keyframes[-int(local_window_size):]

    # Gauge fixing: keep the first keyframe fixed. This mirrors the role of
    # fixed boundary keyframes in local BA and root fixation in global BA.
    fixed_keyframes = []
    if len(local_keyframes) > 0 and local_keyframes[0].kid != 0:
        fixed_keyframes.append(local_keyframes[0])

    result = _bundle_adjustment_core(
        local_keyframes=local_keyframes,
        fixed_keyframes=fixed_keyframes,
        points=points,
        fixed_points=fixed_points,
        rounds=rounds,
        use_robust_kernel=use_robust_kernel,
        verbose=verbose,
        result_dict=result_dict,
        print=print,
    )

    return result.mean_squared_error, result_dict


def local_bundle_adjustment(
    keyframe: KeyFrame,
    abort_flag=None,
    rounds: int = Parameters.kLocalBAWindowSize,
    verbose: bool = False,
    print=print,
) -> OptimizerResult:
    """
    pySLAM-style local BA from a reference keyframe.

    Local keyframes:
    - reference keyframe
    - best covisible keyframes

    Fixed keyframes:
    - other keyframes observing local map points
    """
    if keyframe is None or keyframe.is_bad():
        return OptimizerResult(0, 0, 0, float("inf"), False)

    local_keyframes = [keyframe]

    for kf in keyframe.get_best_covisible_keyframes(Parameters.kNumBestCovisibilityKeyFrames):
        if kf is not None and not kf.is_bad() and kf not in local_keyframes:
            local_keyframes.append(kf)

    local_points = []
    for kf in local_keyframes:
        for point in kf.get_matched_good_points():
            if point is not None and not point.is_bad() and point not in local_points:
                local_points.append(point)

    fixed_keyframes = []
    for point in local_points:
        for observing_kf, _ in point.observations():
            if observing_kf not in local_keyframes and observing_kf not in fixed_keyframes:
                if not observing_kf.is_bad():
                    fixed_keyframes.append(observing_kf)

    # If there are no boundary fixed keyframes and the root is not in local set,
    # fix the first local keyframe to remove gauge freedom.
    if not fixed_keyframes and local_keyframes:
        if local_keyframes[0].kid != 0:
            fixed_keyframes.append(local_keyframes[0])

    return _bundle_adjustment_core(
        local_keyframes=local_keyframes,
        fixed_keyframes=fixed_keyframes,
        points=local_points,
        fixed_points=False,
        rounds=min(int(rounds), 10),
        use_robust_kernel=True,
        verbose=verbose,
        result_dict=None,
        print=print,
    )


def global_bundle_adjustment(
    keyframes,
    points,
    rounds: int = 10,
    loop_kf_id: int = 0,
    use_robust_kernel: bool = True,
    abort_flag=None,
    result_dict: Optional[dict] = None,
    verbose: bool = False,
    print=print,
) -> tuple[float, Optional[dict]]:
    """pySLAM-compatible global BA wrapper."""
    return bundle_adjustment(
        keyframes=keyframes,
        points=points,
        local_window_size=None,
        fixed_points=False,
        rounds=rounds,
        loop_kf_id=loop_kf_id,
        use_robust_kernel=use_robust_kernel,
        abort_flag=abort_flag,
        result_dict=result_dict,
        verbose=verbose,
        print=print,
    )
