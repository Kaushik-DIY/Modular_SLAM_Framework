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

from contextlib import nullcontext
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
    mean_error_before: float = float("inf")
    mean_error_after: float = float("inf")
    num_keyframes: int = 0
    num_map_points: int = 0
    aborted: bool = False
    reason: str = ""


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


def _abort_requested(abort_flag) -> bool:
    if abort_flag is None:
        return False
    value = getattr(abort_flag, "value", abort_flag)
    if callable(value):
        try:
            value = value()
        except TypeError:
            pass
    return bool(value)


def _is_finite_pose(Tcw: np.ndarray) -> bool:
    Tcw = np.asarray(Tcw, dtype=np.float64)
    return Tcw.shape == (4, 4) and np.all(np.isfinite(Tcw))


def _point_has_positive_depth(frame_or_kf, point_w: np.ndarray) -> bool:
    try:
        Tcw = np.asarray(frame_or_kf.Tcw(), dtype=np.float64).reshape(4, 4)
    except Exception:
        return False
    point_w = np.asarray(point_w, dtype=np.float64).reshape(3)
    if not (_is_finite_pose(Tcw) and np.all(np.isfinite(point_w))):
        return False
    point_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    return bool(np.isfinite(point_c[2]) and point_c[2] > Parameters.kMinDepth)


def _valid_observation(frame_or_kf, idx: int, point_w: np.ndarray) -> bool:
    try:
        uv = _get_observation_uv(frame_or_kf, idx)
        ur = _get_observation_ur(frame_or_kf, idx)
    except Exception:
        return False
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        return False
    if ur >= 0.0 and not np.isfinite(ur):
        return False
    return _point_has_positive_depth(frame_or_kf, point_w)


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
    kps = getattr(frame_or_kf, "kpsu", None)
    if kps is None:
        kps = getattr(frame_or_kf, "kps", getattr(frame_or_kf, "keypoints", None))
    if kps is None:
        raise ValueError("Frame/keyframe has no keypoints.")
    kp = kps[int(idx)]
    if hasattr(kp, "pt"):
        return np.array([kp.pt[0], kp.pt[1]], dtype=np.float64)
    return np.asarray(kp, dtype=np.float64).reshape(2)


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


def _pose_vertex_point_depth(pose_vertex, point_vertex) -> float:
    Tcw = _extract_Tcw_from_pose_vertex(pose_vertex)
    point_w = np.asarray(point_vertex.estimate(), dtype=np.float64).reshape(3)
    point_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]
    return float(point_c[2])


def _is_depth_positive(edge, pose_vertex, point_vertex) -> bool:
    is_depth_positive = getattr(edge, "is_depth_positive", None)
    if callable(is_depth_positive):
        try:
            return bool(is_depth_positive())
        except Exception:
            pass

    # Binding adaptation: this workspace's g2o projection edges do not expose
    # is_depth_positive(), so mirror pySLAM by checking optimized camera depth.
    depth = _pose_vertex_point_depth(pose_vertex, point_vertex)
    return bool(np.isfinite(depth) and depth > 0.0)


def _manual_reprojection_chi2(frame_or_kf, idx: int, pose_vertex, point_vertex, is_stereo: bool) -> float:
    Tcw = _extract_Tcw_from_pose_vertex(pose_vertex)
    point_w = np.asarray(point_vertex.estimate(), dtype=np.float64).reshape(3)
    point_c = Tcw[:3, :3] @ point_w + Tcw[:3, 3]

    if not np.all(np.isfinite(point_c)):
        return float("inf")

    z = float(point_c[2])
    if z <= 0.0:
        return float("inf")

    camera = frame_or_kf.camera
    u = float(camera.fx) * float(point_c[0]) / z + float(camera.cx)
    v = float(camera.fy) * float(point_c[1]) / z + float(camera.cy)

    if not np.isfinite(u) or not np.isfinite(v):
        return float("inf")

    uv = _get_observation_uv(frame_or_kf, idx)
    inv_sigma2 = _get_inv_sigma2(frame_or_kf, idx)
    err2 = (u - float(uv[0])) ** 2 + (v - float(uv[1])) ** 2

    if is_stereo:
        ur_obs = _get_observation_ur(frame_or_kf, idx)
        if ur_obs >= 0.0:
            ur = u - float(getattr(camera, "bf", 0.0)) / z
            err2 += (ur - ur_obs) ** 2

    return float(err2 * inv_sigma2)


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
        point_w = point.get_position()
        if not _valid_observation(frame, idx, point_w):
            continue

        point_vertex = add_point_vertex(
            optimizer=optimizer,
            vertex_id=vertex_id,
            point_w=point_w,
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
    abort_flag=None,
    map_lock=None,
    verbose: bool = False,
    result_dict: Optional[dict] = None,
    write_back: bool = True,
    prune_outliers: bool = False,
    print=print,
) -> OptimizerResult:
    local_keyframes = [kf for kf in local_keyframes if kf is not None and not _is_bad_keyframe(kf)]
    fixed_keyframes = [kf for kf in fixed_keyframes if kf is not None and not _is_bad_keyframe(kf)]
    points = [p for p in points if not _is_bad_point(p)]

    if len(local_keyframes) == 0 or len(points) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False, reason="empty graph input")

    optimizer = make_optimizer(verbose=verbose)
    if abort_flag is not None:
        if hasattr(optimizer, "set_force_stop_flag") and abort_flag.__class__.__module__ == "g2o":
            optimizer.set_force_stop_flag(abort_flag)
    add_camera_parameters(optimizer, _camera_to_g2o(local_keyframes[0].camera), parameter_id=0)

    pose_vertices = {}
    point_vertices = {}
    graph_edges = {}

    # pySLAM uses stable even keyframe vertex ids and odd map-point ids. This
    # makes graph construction traceable across local and global BA windows.
    good_keyframes = []

    for kf in local_keyframes:
        if kf in pose_vertices:
            continue
        v = add_pose_vertex(
            optimizer=optimizer,
            vertex_id=int(kf.kid) * 2,
            Tcw=kf.Tcw(),
            fixed=(kf.kid == 0),
        )
        pose_vertices[kf] = v
        good_keyframes.append(kf)

    for kf in fixed_keyframes:
        if kf in pose_vertices:
            continue
        v = add_pose_vertex(
            optimizer=optimizer,
            vertex_id=int(kf.kid) * 2,
            Tcw=kf.Tcw(),
            fixed=True,
        )
        pose_vertices[kf] = v
        good_keyframes.append(kf)

    for point in points:
        point_w = point.get_position()
        if not np.all(np.isfinite(point_w)):
            continue
        v = add_point_vertex(
            optimizer=optimizer,
            vertex_id=int(point.id) * 2 + 1,
            point_w=point_w,
            fixed=bool(fixed_points),
            marginalized=True,
        )
        point_vertices[point] = v

    edges = []
    edge_id = 0
    points_with_edges = set()

    for point in points:
        point_vertex = point_vertices.get(point)
        if point_vertex is None:
            continue

        for kf, idx in point.observations():
            if _is_bad_keyframe(kf):
                continue
            pose_vertex = pose_vertices.get(kf)
            if pose_vertex is None:
                continue
            if idx < 0 or idx >= len(getattr(kf, "points", [])):
                continue
            if kf.get_point_match(idx) is not point:
                continue
            if not _valid_observation(kf, idx, point.get_position()):
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

            edge_data = (point, kf, idx, chi2_threshold, is_stereo, point_vertex, pose_vertex)
            edges.append((edge, *edge_data))
            graph_edges[edge] = edge_data
            points_with_edges.add(point)
            edge_id += 1

    if len(edges) == 0:
        return OptimizerResult(
            0,
            0,
            0,
            float("inf"),
            False,
            num_keyframes=len(pose_vertices),
            num_map_points=len(point_vertices),
            reason="no valid reprojection edges",
        )

    if verbose:
        optimizer.set_verbose(True)

    if _abort_requested(abort_flag):
        return OptimizerResult(
            len(edges),
            0,
            len(edges),
            float("inf"),
            False,
            num_keyframes=len(pose_vertices),
            num_map_points=len(point_vertices),
            aborted=True,
            reason="aborted before optimization",
        )

    num_bad_edges = 0

    try:
        optimizer.initialize_optimization()
        optimizer.compute_active_errors()
        initial_active_chi2 = float(optimizer.active_chi2())

        if use_robust_kernel:
            optimizer.optimize(5)

            for edge, (
                point,
                kf,
                idx,
                chi2_threshold,
                is_stereo,
                point_vertex,
                pose_vertex,
            ) in graph_edges.items():
                chi2 = _manual_reprojection_chi2(kf, idx, pose_vertex, point_vertex, is_stereo)
                is_bad_edge = (
                    (not np.isfinite(chi2))
                    or chi2 > float(chi2_threshold)
                    or not _is_depth_positive(edge, pose_vertex, point_vertex)
                )
                if is_bad_edge:
                    edge.set_level(1)
                    num_bad_edges += 1
                edge.set_robust_kernel(None)

            if _abort_requested(abort_flag):
                return OptimizerResult(
                    len(edges),
                    0,
                    len(edges),
                    float("inf"),
                    False,
                    mean_error_before=initial_active_chi2 / max(len(edges), 1),
                    num_keyframes=len(pose_vertices),
                    num_map_points=len(point_vertices),
                    aborted=True,
                    reason="aborted after robust optimization",
                )

            optimizer.initialize_optimization()
            optimizer.optimize(int(rounds))
        else:
            optimizer.initialize_optimization()
            optimizer.optimize(int(rounds))
    except Exception as exc:
        print(f"bundle_adjustment: g2o failed: {exc}")
        return OptimizerResult(
            len(edges),
            0,
            len(edges),
            float("inf"),
            False,
            num_keyframes=len(pose_vertices),
            num_map_points=len(point_vertices),
            reason=f"g2o failed: {exc}",
        )

    outlier_observations = []
    inlier_chi2 = []

    for edge, point, kf, idx, chi2_threshold, is_stereo, point_vertex, pose_vertex in edges:
        if _is_bad_point(point) or _is_bad_keyframe(kf):
            continue
        if idx < 0 or idx >= len(getattr(kf, "points", [])):
            continue
        if kf.get_point_match(idx) is not point:
            continue
        chi2 = _manual_reprojection_chi2(kf, idx, pose_vertex, point_vertex, is_stereo)
        is_bad_observation = (
            (not np.isfinite(chi2))
            or chi2 > float(chi2_threshold)
            or not _is_depth_positive(edge, pose_vertex, point_vertex)
        )
        if is_bad_observation:
            outlier_observations.append((point, kf, idx, is_stereo))
        else:
            inlier_chi2.append(chi2)

    pose_updates = {}
    point_updates = {}
    for kf, vertex in pose_vertices.items():
        pose_updates[kf] = _extract_Tcw_from_pose_vertex(vertex)
    if not fixed_points:
        for point, vertex in point_vertices.items():
            if point not in points_with_edges:
                continue
            point_updates[point] = np.array(vertex.estimate(), dtype=np.float64, copy=True).reshape(3)

    if write_back:
        lock_context = map_lock if map_lock is not None else nullcontext()
        with lock_context:
            if prune_outliers:
                for point, kf, idx, _ in outlier_observations:
                    if _is_bad_point(point) or _is_bad_keyframe(kf):
                        continue
                    if idx < 0 or idx >= len(getattr(kf, "points", [])):
                        continue
                    if kf.get_point_match(idx) is point:
                        point.remove_observation(kf, idx, map_no_lock=True)

            for kf, Tcw in pose_updates.items():
                if kf in local_keyframes and not _is_bad_keyframe(kf):
                    kf.update_pose(g2o.Isometry3d(Tcw))
                    if hasattr(kf, "lba_count"):
                        kf.lba_count += 1

            if not fixed_points:
                for point, position in point_updates.items():
                    if _is_bad_point(point):
                        continue
                    point.update_position(position)
                    point.update_normal_and_depth()

    if result_dict is not None:
        result_dict["keyframes"] = {kf.kid: Tcw.copy() for kf, Tcw in pose_updates.items()}
        result_dict["keyframe_updates"] = {
            getattr(kf, "id", kf.kid): Tcw.copy() for kf, Tcw in pose_updates.items()
        }
        result_dict["points"] = {p.id: position.copy() for p, position in point_updates.items()}
        result_dict["point_updates"] = {p.id: position.copy() for p, position in point_updates.items()}
        result_dict["fixed_keyframes"] = [kf.kid for kf, vertex in pose_vertices.items() if vertex.fixed()]
        result_dict["num_edges"] = len(edges)
        result_dict["num_inliers"] = len(inlier_chi2)
        result_dict["num_outliers"] = len(outlier_observations)
        result_dict["mean_error_before"] = initial_active_chi2 / max(len(edges), 1)
        result_dict["mean_error_after"] = float(np.mean(inlier_chi2)) if inlier_chi2 else float("inf")

    mse = float(np.mean(inlier_chi2)) if inlier_chi2 else float("inf")

    return OptimizerResult(
        num_edges=len(edges),
        num_inliers=len(inlier_chi2),
        num_outliers=len(outlier_observations),
        mean_squared_error=mse,
        success=len(inlier_chi2) > 0 and np.isfinite(initial_active_chi2),
        mean_error_before=initial_active_chi2 / max(len(edges), 1),
        mean_error_after=mse,
        num_keyframes=len(pose_vertices),
        num_map_points=len(point_vertices),
        reason="" if len(inlier_chi2) > 0 and np.isfinite(initial_active_chi2) else "no inlier edges",
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
    write_back: bool = True,
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
        abort_flag=abort_flag,
        map_lock=None,
        verbose=verbose,
        result_dict=result_dict,
        write_back=write_back,
        prune_outliers=False,
        print=print,
    )

    return result.mean_squared_error, result_dict


def local_bundle_adjustment(
    keyframes,
    points: Optional[Iterable[MapPoint]] = None,
    keyframes_ref: Optional[Iterable[KeyFrame]] = None,
    fixed_points: bool = False,
    verbose: bool = False,
    rounds: int = 10,
    abort_flag=None,
    mp_abort_flag=None,
    map_lock=None,
    print=print,
) -> OptimizerResult:
    """
    pySLAM-style local BA.

    Preferred pySLAM-compatible call:
        local_bundle_adjustment(keyframes, points, keyframes_ref, ...)

    Compatibility call retained for existing tests:
        local_bundle_adjustment(reference_keyframe, ...)
    """
    if isinstance(keyframes, KeyFrame):
        keyframe = keyframes
        if keyframe is None or keyframe.is_bad():
            return OptimizerResult(0, 0, 0, float("inf"), False)

        local_keyframes = [keyframe]

        for kf in keyframe.get_covisible_keyframes():
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
    else:
        local_keyframes = _as_list(keyframes)
        local_points = _as_list(points)
        fixed_keyframes = _as_list(keyframes_ref)

    if len(local_keyframes) == 0 or len(local_points) == 0:
        return OptimizerResult(0, 0, 0, float("inf"), False)

    # If there are no boundary fixed keyframes and the root is not in local set,
    # fix the first local keyframe to remove gauge freedom.
    if not fixed_keyframes and local_keyframes:
        if local_keyframes[0].kid != 0:
            fixed_keyframes.append(local_keyframes[0])

    return _bundle_adjustment_core(
        local_keyframes=local_keyframes,
        fixed_keyframes=fixed_keyframes,
        points=local_points,
        fixed_points=fixed_points,
        rounds=int(rounds),
        use_robust_kernel=True,
        abort_flag=abort_flag,
        map_lock=map_lock,
        verbose=verbose,
        result_dict=None,
        write_back=True,
        prune_outliers=True,
        print=print,
    )


def global_bundle_adjustment(
    keyframes,
    points,
    rounds: int = 10,
    loop_kf_id: int = 0,
    use_robust_kernel: bool = True,
    abort_flag=None,
    mp_abort_flag=None,
    result_dict: Optional[dict] = None,
    write_back: bool = True,
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
        mp_abort_flag=mp_abort_flag,
        result_dict=result_dict,
        write_back=write_back,
        verbose=verbose,
        print=print,
    )
