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

    # --- IMU priors ---
    lidar_use_imu: bool = True             # extrapolator gyro+yaw in LiDAR FEs
    vo_imu_dropout: bool = True            # VO dead-reckoning during dropouts
    # Replace only the VO spine yaw with IMU relative yaw; VO still owns x/y.
    # The sign is learned online from VO/IMU yaw correlation; alpha blends the swap.
    vo_imu_spine_heading: bool = True
    vo_imu_spine_alpha: float = 1.0

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
    # REINIT keyframes are dead-reckoned, so their spine edge stays soft.
    blind_spine_weight: float = 1e2
    loop_trans_weight: float = 1.1e4
    loop_rot_weight: float = 1e5
    huber_scale: float = 10.0
    optimize_every_n_kf: int = 30

    # --- loop closure on/off ---
    # False keeps only the front-end odometry spine: no propose/verify/loop solve.
    enable_loops: bool = True

    # --- loop proposing ---
    propose_every_n_kf: int = 5
    min_kf_separation: int = 30
    proposal_radius_m: float = 4.0
    max_candidates_per_query: int = 2

    # --- candidate-local verification (C++ B&B) ---
    retrieval_graph_depth: int = 2
    retrieval_metric_radius: float = 3.0
    grid_resolution: float = 0.05
    # Log-odds update used by both candidate-local grids and final map rendering.
    grid_l_occ: float = 0.85
    grid_l_free: float = -0.1
    # B&B window must cover accumulated drift before the first accepted loop.
    bnb_window_xy: float = 6.0
    bnb_window_th: float = math.radians(30.0)
    # Visual-led scan verification gets a wider yaw window for residual VO drift.
    orb_bnb_window_th: float = math.radians(45.0)
    bnb_depth: int = 7
    accept_coarse_min: float = 0.55
    accept_refined_min: float = 0.60
    # scan-side loop verifier: "bnb" (C++ correlative B&B) or "icp" (small_gicp)
    scan_verifier: str = "bnb"
    icp_accept_fitness: float = 0.6        # min correspondence ratio (RTAB-style)
    icp_max_corr_dist: float = 1.0         # GICP correspondence cap (m)
    icp_fitness_dist: float = 0.15         # inlier distance for the corr. ratio (m)
    icp_accept_rmse: float = 0.10          # max inlier RMSE to accept (m)
    # Scan loop must agree with the graph prediction to reject ambiguous matches.
    scan_rel_sanity_m: float = 2.5
    scan_rel_sanity_rad: float = math.radians(45.0)
    # Optional absolute cap on verified cand->query scan-loop translation.
    scan_abs_max_m: float = 1e9

    # --- visual verification (PnP) ---
    pnp_min_inliers: int = 15
    # Strong PnP can bootstrap the first loop even when the prediction has drifted.
    pnp_strong_inliers: int = 40
    pnp_nndr: float = 0.7
    pnp_rel_sanity_m: float = 2.5
    pnp_rel_sanity_rad: float = math.radians(40.0)
    dbow_min_score: float = 0.05

    # --- native VO front-end ---
    vo_depth_max: float = 4.0              # depth-point cap (m)
    # VO tracking/keyframe/BA overrides passed through to fusion_core.VoConfig.
    vo_overrides: dict = field(
        default_factory=lambda: {"match_radius_px": 10.0,
                                 "match_radius_fallback_px": 22.0})

    # --- soft sync (RGB-D <-> LiDAR) ---
    sync_tolerance_s: float = 0.05

    # --- misc ---
    seed: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-dataset tuning (applied in run_realtime after the CLI config is built).
# ---------------------------------------------------------------------------
# Dataset-specific overrides stay narrow because VO/loop behavior differs by scene.
_DATASET_TUNING = {
    # More permissive DBoW threshold for weak but valid indoor revisits.
    "lab_hybrid": dict(dbow_min_score=0.025),
    "lab_hybrid_3_slow": dict(
        # Texture-poor scene: coast longer and insert visual keyframes earlier.
        _vo_extra=dict(reinit_patience=12, kf_ref_ratio=0.60, kf_min_close_points=60),
        # Propose slightly more often and over a wider radius.
        propose_every_n_kf=4, min_kf_separation=20, proposal_radius_m=5.0,
        max_candidates_per_query=3, dbow_min_score=0.018,
        # Tighten verification to reject scan slide-locks.
        accept_coarse_min=0.58, accept_refined_min=0.63,
        icp_accept_fitness=0.75, icp_accept_rmse=0.06,
        scan_rel_sanity_m=2.0, scan_rel_sanity_rad=math.radians(35.0),
        # Drift-independent cap on verified scan-loop translation.
        scan_abs_max_m=2.5,
        pnp_min_inliers=18, pnp_rel_sanity_m=1.8, pnp_rel_sanity_rad=math.radians(28.0),
    ),
}


def apply_dataset_tuning(cfg: "FusionV2Config") -> bool:
    """Apply per-dataset overrides keyed by the dataset folder name. Returns True
    if any override was applied (so the runner can log it)."""
    tuning = _DATASET_TUNING.get(Path(cfg.dataset).name)
    if not tuning:
        return False
    for k, v in tuning.items():
        if k == "_vo_extra":
            cfg.vo_overrides = {**cfg.vo_overrides, **v}
        else:
            setattr(cfg, k, v)
    return True
