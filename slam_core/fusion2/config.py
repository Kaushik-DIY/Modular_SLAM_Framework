"""Fusion v2 run configuration (Python-side knobs; C++ configs built from this)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FusionV2Config:
    # --- run ---
    mode: str = "lidar"                    # orb | lidar | orb_lidar | lidar_orb
    dataset: Path = Path("datasets/lab_hybrid")
    output_dir: Path = Path("fusion2_outputs")
    max_scans: int = 0                     # 0 = all
    print_every: int = 200
    # LiDAR local-mapping variant (modes lidar / lidar_orb):
    #   native_s2s | native_s2m  -> C++ fusion_core front-end (V4.4; ~11-15 ms/scan)
    #   legacy_s2s | legacy_s2m  -> Python hector stack (debug/parity reference)
    lidar_frontend: str = "native_s2s"     # default flipped after V4.4 parity passed
    # visual front-end (modes orb / orb_lidar): native (C++ VO) | legacy
    visual_frontend: str = "native"

    # --- IMU (always-on pose-prior fallback; plan V4 cross-cutting req.) ---
    lidar_use_imu: bool = True             # extrapolator gyro+yaw in LiDAR FEs
    vo_imu_dropout: bool = True            # VO dead-reckoning during dropouts

    # --- memory tiers (RTAB defaults, CLAUDE.md §2.13) ---
    stm_size: int = 30
    wm_cap: int = 200
    rehearsal_similarity: float = 0.20

    # --- keyframe decision (LiDAR front-end) ---
    kf_min_dist_m: float = 0.25
    kf_min_angle_rad: float = math.radians(12.0)
    kf_min_dt_s: float = 2.0

    # --- graph ---
    spine_trans_weight: float = 1e5
    spine_rot_weight: float = 1e5
    # spine weight for REINIT (dead-reckoned) keyframes: these edges are a
    # guess, not a measurement — keep them soft so loop closures can bend the
    # blind segment back into place.
    blind_spine_weight: float = 1e2
    loop_trans_weight: float = 1.1e4
    loop_rot_weight: float = 1e5
    huber_scale: float = 10.0
    optimize_every_n_kf: int = 30

    # --- loop proposing (proximity; bounded per Test-3 lessons) ---
    propose_every_n_kf: int = 5
    min_kf_separation: int = 30
    proposal_radius_m: float = 4.0
    max_candidates_per_query: int = 2

    # --- candidate-local verification (C++ B&B) ---
    retrieval_graph_depth: int = 2
    retrieval_metric_radius: float = 3.0
    grid_resolution: float = 0.05
    # Hector-proven log-odds balance for candidate-local + output grids:
    # hits at +0.85 vs gentle -0.1 free carving (aggressive l_free erodes
    # wall hits — C6 diagnosis). Moved here from runner.py in V4.3.
    grid_l_occ: float = 0.85
    grid_l_free: float = -0.1
    # Window must absorb pre-first-loop odometry drift (the Test-3 lesson: the
    # two-room lab drifts metres before the first revisit). The C++ B&B at ~85x
    # makes the wide window affordable per-candidate.
    bnb_window_xy: float = 6.0
    bnb_window_th: float = math.radians(30.0)
    # orb-led modes: slightly wider angular window than lidar-led, reflecting
    # residual VO yaw drift. (Was ±180° pre-V3.6 when the VO drifted badly —
    # that width let rotationally-ambiguous rooms produce high-scoring WRONG
    # rotations; narrowed in V4.1 now that the VO is drift-bounded.)
    orb_bnb_window_th: float = math.radians(45.0)
    bnb_depth: int = 7
    accept_coarse_min: float = 0.55
    accept_refined_min: float = 0.60
    # scan-side loop verifier: "bnb" (C++ correlative B&B) or "icp" (small_gicp)
    scan_verifier: str = "bnb"
    icp_accept_fitness: float = 0.6
    icp_max_corr_dist: float = 1.0         # GICP correspondence cap (m)
    icp_fitness_dist: float = 0.15         # inlier distance for fitness (m)
    # Relative-pose sanity gate for SCAN verifiers (B&B / ICP), mirroring the
    # PnP gate: a verified loop whose rel pose disagrees wildly with the graph
    # prediction is a rotational-ambiguity false positive (V4.1 — these warped
    # the orb_lidar map at rot_weight 1e5 despite high match scores).
    scan_rel_sanity_m: float = 2.5
    scan_rel_sanity_rad: float = math.radians(45.0)

    # --- visual verification (PnP) ---
    pnp_min_inliers: int = 15
    # Strong PnP (many RANSAC inliers) is self-validating: it bypasses the
    # rel-sanity gate, which otherwise rejects every FIRST loop once pre-loop
    # drift exceeds the bound (observed: 50-79-inlier loops all rejected).
    pnp_strong_inliers: int = 40
    pnp_nndr: float = 0.7
    pnp_rel_sanity_m: float = 2.5
    pnp_rel_sanity_rad: float = math.radians(40.0)
    dbow_min_score: float = 0.05

    # --- native VO front-end ---
    vo_depth_max: float = 4.0              # depth-point cap (m); was hardcoded

    # --- soft sync (RGB-D <-> LiDAR) ---
    sync_tolerance_s: float = 0.05

    # --- misc ---
    seed: int = 0
    extra: dict = field(default_factory=dict)
