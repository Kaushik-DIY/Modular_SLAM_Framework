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
    # Loosely-coupled IMU heading for the VO front-end (modes orb / orb_lidar):
    # replace the DRIFTING VO relative-yaw in each keyframe's spine edge with the
    # drift-free IMU relative-yaw (a scalar ground-plane delta, immune to the ~8 deg
    # camera mount tilt that broke per-frame IMU tracking priors). VO still owns x/y
    # and the visual BA is untouched (no reinit regression). This is the root-cause
    # fix for the orb_lidar heading warp: the ICP/B&B loop seed heading becomes
    # correct, so the EXISTING verifier finds the right yaw. Sign vs the SE(2) frame
    # is learned online by VO/IMU delta correlation. alpha=1.0 fully replaces the
    # spine yaw with IMU; <1 blends. No-op when imu.csv is absent.
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
    # spine weight for REINIT (dead-reckoned) keyframes: these edges are a
    # guess, not a measurement — keep them soft so loop closures can bend the
    # blind segment back into place.
    blind_spine_weight: float = 1e2
    loop_trans_weight: float = 1.1e4
    loop_rot_weight: float = 1e5
    huber_scale: float = 10.0
    optimize_every_n_kf: int = 30

    # --- loop closure on/off ---
    # When False the runner does ONLY front-end local mapping (scan-matching /
    # VO+local-BA odometry chained on the spine graph) — no loop propose/verify/
    # constraint and no global optimize. Lets us measure how far a front-end alone
    # carries (e.g. a small area may map well with no loop closure at all).
    enable_loops: bool = True

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
    icp_accept_fitness: float = 0.6        # min correspondence ratio (RTAB-style)
    icp_max_corr_dist: float = 1.0         # GICP correspondence cap (m)
    icp_fitness_dist: float = 0.15         # inlier distance for the corr. ratio (m)
    icp_accept_rmse: float = 0.10          # max inlier RMSE to accept (m); ICP-native,
    #                                        independent of B&B's accept_refined_min
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
    # VoConfig tracking/KF/BA knobs. Default widens the projection-match radius:
    # tuned 2026-06-14, the 7 px default missed correspondences when the
    # constant-velocity prediction is off on fast turns (the rotation-pathological
    # failure that smeared the small-map orb map). 10 px / 22 px fallback cut
    # end-start drift small 1.22->0.05 m, large 1.83->0.67 m, sharper maps + fewer
    # reinits on BOTH maps. (radius12 / +patience regressed; see git log.)
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
# Only `lab_hybrid_3_slow` deviates. ROOT CAUSE (probed): it is TEXTURE-POOR —
# ~half the ORB features of lab_hybrid (median 1042 vs 1912), so the VO collapses
# to ~4 inliers and loses tracking ~2x as often (285 reinits vs 104), which
# shreds the visual-led maps (orb_lidar ~12 m). The fix that works here — coast
# through the transient texture-poor frames (reinit_patience 12), insert KFs
# earlier, and verify loops strictly (ICP) — CANNOT be global: lab_hybrid's
# collapses are during FAST TURNS where coasting flies the VO off (17 m), so it
# must keep the short default patience. Hence per-dataset, scoped to this one set.
_DATASET_TUNING = {
    "lab_hybrid_3_slow": dict(
        # VO front-end robustness (texture-poor: coast + insert KFs earlier)
        _vo_extra=dict(reinit_patience=12, kf_ref_ratio=0.60, kf_min_close_points=60),
        # loop proposer: relaxed but not extreme
        propose_every_n_kf=4, min_kf_separation=20, proposal_radius_m=5.0,
        max_candidates_per_query=3, dbow_min_score=0.025,
        # verifiers: strict ICP (rejects the warp loops); B&B/PnP moderate
        accept_coarse_min=0.58, accept_refined_min=0.63,
        icp_accept_fitness=0.75, icp_accept_rmse=0.06,
        scan_rel_sanity_m=2.0, scan_rel_sanity_rad=math.radians(35.0),
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
