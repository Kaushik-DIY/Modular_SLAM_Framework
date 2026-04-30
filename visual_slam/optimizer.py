"""
=============================================================================
visual_slam/optimizer.py

g2o optimization wrappers for Visual SLAM.

IMPORTANT: This module follows pyslam's g2o optimization approach.
---------------------------------------------------------------------------
Ported from pyslam's optimizer_g2o.py.

This module provides four optimization functions:
1. motion_only_ba: Optimize pose with fixed map points (tracking)
2. local_ba: Optimize local keyframes + map points (local mapping)
3. pose_graph_optimization: Optimize keyframe poses (loop closing)
4. global_ba: Full bundle adjustment (after loop closure)

All use g2o for graph optimization with appropriate vertex/edge types.

Functions
---------
motion_only_ba(frame, camera)
    Optimize frame pose with fixed map points.
    
local_ba(local_keyframes, local_map_points, fixed_keyframes)
    Optimize local window of keyframes and map points.
    
pose_graph_optimization(keyframes, loop_edges)
    Optimize keyframe poses with loop closure constraints.
    
global_ba(slam_map)
    Full optimization of all keyframes and map points.

References
----------
pyslam: optimizer_g2o.py
ORB-SLAM2: Optimizer.cc
g2o: Vertex/Edge types for SE(3) + reprojection

=============================================================================
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import g2o
except ImportError:
    g2o = None
    print("WARNING: g2o not available. Optimizer functions will not work.")

from visual_slam.types import Frame, KeyFrame, MapPoint, Map
from slam_core.common.types3d import CameraIntrinsics, Pose3D


# ===========================================================================
# Motion-Only Bundle Adjustment (Pose Optimization)
# ===========================================================================

def motion_only_ba(
    frame: Frame,
    camera: CameraIntrinsics,
    iterations: int = 10,
) -> Optional[Pose3D]:
    """
    Optimize camera pose with fixed 3D map points.
    
    This is motion-only BA: only the pose (6 DOF) is optimized.
    Map points are held fixed. Used by tracking to refine pose estimate.
    
    Parameters
    ----------
    frame : Frame
        Frame with initial pose estimate and matched map points.
    camera : CameraIntrinsics
        Camera parameters for reprojection.
    iterations : int
        Number of optimization iterations.
    
    Returns
    -------
    g2o.Isometry3d or None
        Optimized pose, or None if optimization failed.
    
    Notes
    -----
    Uses EdgeProjectXYZ2UV which takes camera params in constructor.
    """
    if g2o is None:
        print("ERROR: g2o not available for motion_only_ba")
        return None
    
    if frame.pose_world is None:
        print("ERROR: Frame has no initial pose")
        return None
    
    # Create optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    
    # Add camera pose vertex
    v_pose = g2o.VertexSE3Expmap()
    v_pose.set_id(0)
    
    # Convert g2o.Isometry3d to SE3Quat
    T = frame.pose_world.matrix()
    se3 = g2o.SE3Quat(T[:3, :3], T[:3, 3])
    v_pose.set_estimate(se3)
    optimizer.add_vertex(v_pose)
    
    # Create camera parameters object
    cam_params = g2o.CameraParameters(camera.fx, np.array([camera.cx, camera.cy]), 0)
    cam_params.set_id(0)
    optimizer.add_parameter(cam_params)
    
    # Add map point vertices (fixed) and reprojection edges
    edges = []
    vertex_id = 1
    
    for i, mp in enumerate(frame.map_point_matches):
        if mp is None or mp.is_bad:
            continue
        
        if frame.depths[i] <= 0:
            continue  # Invalid depth
        
        # Get observed pixel coordinates
        kp = frame.keypoints[i]
        obs = np.array([kp.pt[0], kp.pt[1]])
        
        # Add map point vertex (FIXED)
        v_point = g2o.VertexPointXYZ()
        v_point.set_id(vertex_id)
        v_point.set_estimate(mp.position_world)
        v_point.set_fixed(True)
        v_point.set_marginalized(True)
        optimizer.add_vertex(v_point)
        
        # Create reprojection edge using EdgeProjectXYZ2UV
        edge = g2o.EdgeProjectXYZ2UV()
        
        # Connect to pose vertex and point vertex
        edge.set_vertex(0, v_point)  # First vertex: 3D point
        edge.set_vertex(1, v_pose)   # Second vertex: camera pose
        edge.set_measurement(obs)
        edge.set_parameter_id(0, 0)  # Use camera parameters
        
        # Information matrix
        info = np.eye(2)
        edge.set_information(info)
        
        # Huber robust kernel
        huber = g2o.RobustKernelHuber()
        huber.set_delta(np.sqrt(5.991))
        edge.set_robust_kernel(huber)
        
        optimizer.add_edge(edge)
        edges.append((edge, i))
        vertex_id += 1
    
    if len(edges) == 0:
        print("WARNING: No valid reprojection edges for motion_only_ba")
        return frame.pose_world
    
    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(iterations)
    
    # Extract optimized pose
    se3_opt = v_pose.estimate()
    T_opt = np.eye(4)
    T_opt[:3, :3] = se3_opt.rotation().matrix()
    T_opt[:3, 3] = se3_opt.translation()
    optimized_pose = g2o.Isometry3d(T_opt)
    
    return optimized_pose




# ===========================================================================
# ORB-SLAM-style Motion-Only Bundle Adjustment
# ===========================================================================

def motion_only_ba_orbslam_style(
    frame: Frame,
    camera: CameraIntrinsics,
    iterations: int = 10,
    min_edges: int = 10,
    outlier_chi2: float = 5.991,
    remove_outliers: bool = True,
) -> Optional[Pose3D]:
    """
    ORB-SLAM/pySLAM-style motion-only BA with the workspace g2o API.

    Important pose convention:
    - frame.pose_world is T_world_from_camera (Twc), used by the current
      tracking code.
    - g2o reprojection edges use T_camera_from_world (Tcw).
    - Therefore this function optimizes Tcw internally and returns Twc.

    This function is intentionally separate from motion_only_ba() so that it
    can be tested and integrated incrementally.
    """
    if g2o is None:
        print("ERROR: g2o not available for motion_only_ba_orbslam_style")
        return None

    if frame.pose_world is None:
        print("ERROR: Frame has no initial pose")
        return None

    if not hasattr(frame, "map_point_matches") or not hasattr(frame, "keypoints"):
        print("ERROR: Frame missing map_point_matches/keypoints")
        return None

    from visual_slam.g2o_compat import (
        G2OCamera,
        add_camera_parameters,
        add_mono_edge,
        add_point_vertex,
        add_pose_vertex,
        make_optimizer,
        optimize,
    )

    # Your current frame pose convention is Twc.
    Twc_init = np.asarray(frame.pose_world.matrix(), dtype=np.float64)
    if Twc_init.shape != (4, 4) or not np.all(np.isfinite(Twc_init)):
        print("ERROR: Invalid initial frame pose")
        return None

    # g2o projection edges expect Tcw.
    Tcw_init = np.linalg.inv(Twc_init)

    bf = float(getattr(camera, "bf", 0.0) or 0.0)
    g2o_cam = G2OCamera(
        fx=float(camera.fx),
        fy=float(camera.fy),
        cx=float(camera.cx),
        cy=float(camera.cy),
        bf=bf,
    )

    optimizer = make_optimizer(verbose=False)
    add_camera_parameters(optimizer, g2o_cam, parameter_id=0)

    pose_vertex = add_pose_vertex(
        optimizer=optimizer,
        vertex_id=0,
        Tcw=Tcw_init,
        fixed=False,
    )

    edges = []
    vertex_id = 1
    edge_id = 0

    n = min(len(frame.map_point_matches), len(frame.keypoints))

    for i in range(n):
        mp = frame.map_point_matches[i]
        if mp is None or getattr(mp, "is_bad", False):
            continue

        point_w = np.asarray(mp.position_world, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point_w)):
            continue

        kp = frame.keypoints[i]
        uv = np.array([kp.pt[0], kp.pt[1]], dtype=np.float64)
        if not np.all(np.isfinite(uv)):
            continue

        octave = int(getattr(kp, "octave", 0))
        octave = max(octave, 0)

        # Temporary ORB-SLAM-like scale weighting.
        # Later this should come from FeatureTrackerShared.feature_manager.
        scale_factor = 1.2
        sigma2 = scale_factor ** (2.0 * octave)
        inv_sigma2 = 1.0 / sigma2

        point_vertex = add_point_vertex(
            optimizer=optimizer,
            vertex_id=vertex_id,
            point_w=point_w,
            fixed=True,
            marginalized=True,
        )

        edge = add_mono_edge(
            optimizer=optimizer,
            edge_id=edge_id,
            point_vertex=point_vertex,
            pose_vertex=pose_vertex,
            uv=uv,
            inv_sigma2=inv_sigma2,
            parameter_id=0,
            huber_delta=np.sqrt(outlier_chi2),
        )

        edges.append((edge, i))
        vertex_id += 1
        edge_id += 1

    if len(edges) < min_edges:
        print(
            f"WARNING: Not enough reprojection edges for "
            f"motion_only_ba_orbslam_style: {len(edges)} < {min_edges}"
        )
        return None

    try:
        optimize(optimizer, iterations=iterations, verbose=False)
    except Exception as exc:
        print(f"ERROR: motion_only_ba_orbslam_style optimization failed: {exc}")
        return None

    # Basic outlier classification after optimization.
    inliers = []
    outliers = []

    for edge, idx in edges:
        chi2 = float(edge.chi2())
        if np.isfinite(chi2) and chi2 <= outlier_chi2:
            inliers.append((edge, idx))
        else:
            outliers.append((edge, idx))

    if len(inliers) < min_edges:
        print(
            f"WARNING: Not enough inliers after "
            f"motion_only_ba_orbslam_style: {len(inliers)} < {min_edges}"
        )
        return None

    if remove_outliers:
        for _, idx in outliers:
            if idx < len(frame.map_point_matches):
                frame.map_point_matches[idx] = None

    se3_opt = pose_vertex.estimate()

    Tcw_opt = np.eye(4, dtype=np.float64)
    Tcw_opt[:3, :3] = se3_opt.rotation().matrix()
    Tcw_opt[:3, 3] = se3_opt.translation()

    if not np.all(np.isfinite(Tcw_opt)):
        print("ERROR: Optimized Tcw has NaN/Inf")
        return None

    # Return Twc, because the rest of your visual_slam tracking uses Twc.
    Twc_opt = np.linalg.inv(Tcw_opt)

    if not np.all(np.isfinite(Twc_opt)):
        print("ERROR: Optimized Twc has NaN/Inf")
        return None

    return g2o.Isometry3d(Twc_opt)


# ===========================================================================
# Local Bundle Adjustment
# ===========================================================================

def local_ba(
    local_keyframes: List[KeyFrame],
    local_map_points: List[MapPoint],
    fixed_keyframes: List[KeyFrame],
    iterations: int = 5,
) -> None:
    """
    Optimize local keyframes and map points.
    
    This is local BA: optimizes both keyframe poses AND map point positions
    within a local window. Fixed keyframes constrain the optimization.
    
    Parameters
    ----------
    local_keyframes : List[KeyFrame]
        Keyframes to optimize (pose is variable).
    local_map_points : List[MapPoint]
        Map points to optimize (position is variable).
    fixed_keyframes : List[KeyFrame]
        Keyframes at the boundary (pose is fixed).
    iterations : int
        Number of optimization iterations.
    
    Notes
    -----
    Updates keyframe poses and map point positions in-place.
    """
    if g2o is None:
        print("ERROR: g2o not available for local_ba")
        return
    
    # Create optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    
    # Add camera parameters
    if len(local_keyframes) > 0:
        cam = local_keyframes[0].frame.camera
        cam_params = g2o.CameraParameters(cam.fx, np.array([cam.cx, cam.cy]), 0)
        cam_params.set_id(0)
        optimizer.add_parameter(cam_params)
    
    # Add keyframe vertices
    kf_vertex_map = {}
    vertex_id = 0
    
    # Local keyframes (optimized)
    for kf in local_keyframes:
        if kf.is_bad or kf.frame.pose_world is None:
            continue
        
        v = g2o.VertexSE3Expmap()
        v.set_id(vertex_id)
        
        # Convert Isometry3d to SE3Quat
        T = kf.frame.pose_world.matrix()
        se3 = g2o.SE3Quat(T[:3, :3], T[:3, 3])
        v.set_estimate(se3)
        v.set_fixed(False)
        optimizer.add_vertex(v)
        kf_vertex_map[kf.keyframe_id] = vertex_id
        vertex_id += 1
    
    # Fixed keyframes (boundary)
    for kf in fixed_keyframes:
        if kf.is_bad or kf.frame.pose_world is None:
            continue
        
        v = g2o.VertexSE3Expmap()
        v.set_id(vertex_id)
        
        # Convert Isometry3d to SE3Quat
        T = kf.frame.pose_world.matrix()
        se3 = g2o.SE3Quat(T[:3, :3], T[:3, 3])
        v.set_estimate(se3)
        v.set_fixed(True)
        optimizer.add_vertex(v)
        kf_vertex_map[kf.keyframe_id] = vertex_id
        vertex_id += 1
    
    # Add map point vertices
    mp_vertex_map = {}
    for mp in local_map_points:
        if mp.is_bad or len(mp.observations) == 0:
            continue
        
        v = g2o.VertexPointXYZ()
        v.set_id(vertex_id)
        v.set_estimate(mp.position_world)
        v.set_marginalized(True)
        optimizer.add_vertex(v)
        mp_vertex_map[mp.point_id] = vertex_id
        vertex_id += 1
    
    # Add observation edges
    for mp in local_map_points:
        if mp.is_bad or mp.point_id not in mp_vertex_map:
            continue
        
        mp_vertex_id = mp_vertex_map[mp.point_id]
        
        for kf, keypoint_idx in mp.observations.items():
            if kf.keyframe_id not in kf_vertex_map:
                continue
            
            kf_vertex_id = kf_vertex_map[kf.keyframe_id]
            
            # Get observation
            kp = kf.frame.keypoints[keypoint_idx]
            obs = np.array([kp.pt[0], kp.pt[1]])
            
            # Create edge using EdgeProjectXYZ2UV
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, optimizer.vertex(mp_vertex_id))  # 3D point
            edge.set_vertex(1, optimizer.vertex(kf_vertex_id))  # Camera pose
            edge.set_measurement(obs)
            edge.set_parameter_id(0, 0)  # Camera parameters
            
            # Information matrix
            info = np.eye(2)
            edge.set_information(info)
            
            # Robust kernel
            huber = g2o.RobustKernelHuber()
            huber.set_delta(np.sqrt(5.991))
            edge.set_robust_kernel(huber)
            
            optimizer.add_edge(edge)
    
    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(iterations)
    
    # Extract optimized values
    for kf in local_keyframes:
        if kf.keyframe_id in kf_vertex_map:
            vertex_id = kf_vertex_map[kf.keyframe_id]
            se3_opt = optimizer.vertex(vertex_id).estimate()
            
            # Convert SE3Quat back to Isometry3d
            T_opt = np.eye(4)
            T_opt[:3, :3] = se3_opt.rotation().matrix()
            T_opt[:3, 3] = se3_opt.translation()
            kf.frame.pose_world = g2o.Isometry3d(T_opt)
    
    for mp in local_map_points:
        if mp.point_id in mp_vertex_map:
            vertex_id = mp_vertex_map[mp.point_id]
            mp.position_world = optimizer.vertex(vertex_id).estimate()


# ===========================================================================
# Pose Graph Optimization (PGO)
# ===========================================================================

def pose_graph_optimization(
    keyframes: List[KeyFrame],
    loop_edges: List[Tuple[int, int, Pose3D]],
    iterations: int = 20,
) -> None:
    """
    Optimize keyframe poses with loop closure constraints.
    
    This is PGO: only poses are optimized, not map points.
    Used after loop closure detection to correct drift.
    
    Parameters
    ----------
    keyframes : List[KeyFrame]
        All keyframes in the map.
    loop_edges : List[Tuple[int, int, g2o.Isometry3d]]
        Loop closure constraints: (from_kf_id, to_kf_id, relative_pose).
    iterations : int
        Number of optimization iterations.
    
    Notes
    -----
    Updates keyframe poses in-place.
    
    Graph structure:
        - VertexSE3 per keyframe
        - EdgeSE3 per loop closure constraint
    """
    if g2o is None:
        print("ERROR: g2o not available for PGO")
        return
    
    # Create optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    
    # Add keyframe vertices
    kf_vertex_map = {}
    for kf in keyframes:
        if kf.is_bad or kf.frame.pose_world is None:
            continue
        
        v = g2o.VertexSE3()
        v.set_id(kf.keyframe_id)
        v.set_estimate(kf.frame.pose_world)  # VertexSE3 accepts Isometry3d directly
        
        # Fix first keyframe
        if kf.keyframe_id == 0:
            v.set_fixed(True)
        
        optimizer.add_vertex(v)
        kf_vertex_map[kf.keyframe_id] = kf.keyframe_id
    
    # Add loop closure edges
    for from_id, to_id, relative_pose in loop_edges:
        if from_id not in kf_vertex_map or to_id not in kf_vertex_map:
            continue
        
        edge = g2o.EdgeSE3()
        edge.set_vertex(0, optimizer.vertex(from_id))
        edge.set_vertex(1, optimizer.vertex(to_id))
        edge.set_measurement(relative_pose)
        
        # Information matrix (higher for loop closures)
        info = np.eye(6) * 100.0
        edge.set_information(info)
        
        optimizer.add_edge(edge)
    
    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(iterations)
    
    # Extract optimized poses
    for kf in keyframes:
        if kf.keyframe_id in kf_vertex_map:
            kf.frame.pose_world = optimizer.vertex(kf.keyframe_id).estimate()


# ===========================================================================
# Global Bundle Adjustment (GBA)
# ===========================================================================

def global_ba(
    slam_map: Map,
    iterations: int = 10,
) -> None:
    """
    Full bundle adjustment over all keyframes and map points.
    
    This is GBA: optimizes both all keyframe poses AND all map point
    positions. Used after loop closure to globally refine the map.
    
    Parameters
    ----------
    slam_map : Map
        The global map with keyframes and map points.
    iterations : int
        Number of optimization iterations.
    
    Notes
    -----
    Updates all keyframe poses and map point positions in-place.
    This is computationally expensive and should be used sparingly.
    """
    # GBA is essentially local_BA with all keyframes as "local"
    # and no fixed keyframes
    local_ba(
        local_keyframes=list(slam_map.keyframes.values()),
        local_map_points=list(slam_map.map_points.values()),
        fixed_keyframes=[],
        iterations=iterations,
    )