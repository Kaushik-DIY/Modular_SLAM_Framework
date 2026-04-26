"""
Modular SLAM configuration for the Hector SLAM runner.

How to use
----------
Set DATASET_NAME (and DATASET_SCAN_VARIANT for lab_run_2) at the top of
this file, or override from the command line:

    python -m hector.run_local_slam_new --dataset fr079 --max-scans 1000

All SLAM tuning parameters are loaded automatically from the per-dataset
profile below.  Only the three constants at the top need user attention.

Supported datasets
------------------
  "lab_run_2"  Custom lab recording with JetRacer + YD LiDAR G4 + IMU.
               Full 360° FOV, 16 m range, NO wheel odometry, slow robot.
               Two scan variants:
                 "raw" — 909 beams, maximum detail (default)
                 "360" — 360 beams, lower compute cost

  "fr079"      Freiburg FR079 corridor benchmark (CARMEN log).
               SICK LMS, 180° FOV, 360 beams, 30 m range.
               Wheel odometry embedded in the FLASER records.

  "intel"      Intel Research Lab benchmark (CARMEN log).
               SICK LMS, 180° FOV, 180 beams at 1°, 30 m range.
               Wheel odometry embedded in the FLASER records.
"""

import numpy as np

# ==============================================================
# >>> USER SELECTION — edit these three lines <<<
# ==============================================================
DATASET_NAME         = "lab_run_2"      # "lab_run_2" | "fr079" | "intel"
DATASET_SCAN_VARIANT = "raw"            # lab_run_2 only: "raw" | "360"
MATCHER_TYPE         = "scan_to_map" # "scan_to_submap" | "scan_to_map"
MAX_SCANS            = None             # None → all scans; int → cap for testing
VERBOSE_EVERY        = 10

# ==============================================================
# Per-dataset parameter profiles
# ==============================================================
# Each profile must define every parameter key listed below.
# Keys are injected as module-level attributes so the runner can
# continue to use `cfg.XXX` style access unchanged.

_PROFILES: dict = {

    # ----------------------------------------------------------
    # lab_run_2 — JetRacer AI + YD LiDAR G4 in a lab room.
    #
    # Notable characteristics:
    #   • Full 360° field of view (vs. 180° half-scan on SICK datasets)
    #   • 909 raw beams per scan → voxel filtering is necessary
    #   • NO wheel odometry; IMU available but not fused in Hector mode
    #   • Slow robot (≤ 1 m/s) → dense scan coverage per metre of travel
    #   • Small indoor environment (~8 × 6 m lab room)
    #   • Range limited to 16 m (G4 sensor spec)
    # ----------------------------------------------------------
    "lab_run_2": dict(

        # --- Odometry / initial pose ---
        ODOM_ALPHA         = 0.0,   # no wheel odometry for this dataset
        INITIAL_POSE_X     = 0.0,
        INITIAL_POSE_Y     = 0.0,
        INITIAL_POSE_THETA = 0.0,

        # --- Scan handling ---
        BEAM_STRIDE     = 1,
        LIDAR_MIN_RANGE = 0.10,
        LIDAR_MAX_RANGE = 16.0,

        # --- Voxel pre-processing ---
        # Enabled: 909 raw beams produce more points than the GN solver
        # benefits from.  Voxel filtering also removes motion-blur
        # artefacts from beam clustering near close obstacles.
        VOXEL_FILTER_ENABLED      = True,
        VOXEL_FIXED_SIZE          = 0.03,
        VOXEL_ADAPTIVE_MAX_SIZE   = 0.10,
        VOXEL_ADAPTIVE_MIN_POINTS = 200,
        VOXEL_ADAPTIVE_ITERS      = 6,

        # --- Global map (scan_to_map mode) ---
        MAP_RESOLUTION  = 0.05,
        MAP_SIZE_METERS = 40.0,     # lab room comfortably fits in 40 × 40 m

        # --- Submap (scan_to_submap mode) ---
        SUBMAP_RESOLUTION  = 0.05,
        SUBMAP_SIZE_METERS = 20.0,
        # Slow robot at ~10 Hz → ~0.03 m/scan → 500 scans ≈ 15 m traversal
        # giving 2–3 finished submaps for the full ~34 m lab path.
        SCANS_PER_SUBMAP   = 500,

        # Correlative coarse search — wide window because there is no
        # odometry prior; the extrapolator alone can lag at sharp turns.
        SUBMAP_COARSE_XY_WINDOW = 1.0,
        SUBMAP_COARSE_XY_STEP   = 0.10,
        SUBMAP_COARSE_TH_WINDOW = 0.40,   # ~23 deg
        SUBMAP_COARSE_TH_STEP   = 0.05,

        # Correlative fine search
        SUBMAP_FINE_XY_WINDOW = 0.25,
        SUBMAP_FINE_XY_STEP   = 0.05,
        SUBMAP_FINE_TH_WINDOW = 0.12,
        SUBMAP_FINE_TH_STEP   = 0.02,

        # Match / refine caps
        # After voxel filtering we have ~200 pts/scan; use all of them.
        SUBMAP_MAX_MATCH_POINTS  = 200,
        SUBMAP_MAX_REFINE_POINTS = 200,
        SUBMAP_MIN_VALID         = 30,
        SUBMAP_MIN_SCORE         = 0.50,
        SUBMAP_REFINE_W_TRANS    = 0.1,
        SUBMAP_REFINE_W_ROT      = 1.0,
        SUBMAP_REFINE_MIN_POINTS = 30,

        # scan_to_map GN correlative params
        CORR_MAP_MIN_POINTS     = 60,
        CORR_MAP_MIN_INLIERS    = 60,
        CORR_MAP_MIN_SCORE      = 0.10,
        CORR_MAP_STEP_CLIP_XY   = 0.03,   # original Hector Python: ±0.03 m/iter

        # --- Log-odds occupancy model ---
        P_OCC     = 0.75,
        P_FREE    = 0.40,
        L_OCC     = 1.0,    # stronger update — solid lab walls are reliable
        L_FREE    = -0.1,
        L_MIN     = -5.0,
        L_MAX     = 5.0,
        RAY_STEPS = 20,     # 5 cm resolution → 20 steps adequate

        # --- GN scan-to-map solver ---
        PYRAMID_LEVELS     = 3,
        GN_ITERS_PER_LEVEL = [20, 15, 10],
        GN_DAMPING         = 1e-4,
        N_BOOTSTRAP_SCANS  = 1,

        # --- Pose extrapolator ---
        USE_EXTRAPOLATOR = False,
        EXTRAP_MAX_DT   = 0.5,
        EXTRAP_INIT_VXY = 0.0,
        EXTRAP_INIT_WZ  = 0.0,

        # --- Motion filter — slow robot at 10 Hz ---
        TARGET_INSERT_PERIOD_S = 0.10,
        V_EXPECTED_MPS         = 1.0,
        W_EXPECTED_RPS         = np.deg2rad(45.0),

        # --- Pose graph / PGO ---
        KEYFRAME_STRIDE = 10,
        ODOM_SIGMA_XY   = 0.10,
        ODOM_SIGMA_TH   = np.deg2rad(5.0),
        PGO_ITERS       = 15,
        PGO_DAMPING     = 1e-6,
    ),

    # ----------------------------------------------------------
    # fr079 — Freiburg FR079 corridor dataset (CARMEN log).
    #
    # Notable characteristics:
    #   • SICK LMS laser, 180° FOV (half-scan), 360 beams, 30 m range
    #   • Reliable wheel odometry embedded in the FLASER log records
    #   • Purpose-built mobile robot, faster than JetRacer
    #   • Corridor environment, ~60 m total path
    # ----------------------------------------------------------
    "fr079": dict(

        # --- Odometry / initial pose ---
        # Blend 50% odometry with extrapolator prediction.
        # Initial pose comes from the first FLASER odom record.
        ODOM_ALPHA         = 0.5,
        INITIAL_POSE_X     = 0.0,
        INITIAL_POSE_Y     = 0.0,
        INITIAL_POSE_THETA = 0.0,

        # --- Scan handling ---
        BEAM_STRIDE     = 2,
        LIDAR_MIN_RANGE = 0.10,
        LIDAR_MAX_RANGE = 30.0,
        MAP_UPDATE_EVERY = 5,

        # --- Voxel pre-processing ---
        # Disabled: only 360 beams — no benefit from further thinning.
        VOXEL_FILTER_ENABLED      = False,
        VOXEL_FIXED_SIZE          = 0.05,
        VOXEL_ADAPTIVE_MAX_SIZE   = 0.15,
        VOXEL_ADAPTIVE_MIN_POINTS = 150,
        VOXEL_ADAPTIVE_ITERS      = 6,

        # --- Global map (scan_to_map mode) ---
        MAP_RESOLUTION  = 0.05,
        MAP_SIZE_METERS = 80.0,     # corridor reaches ~31 m from origin; 80 m gives safe margin

        # --- Submap (scan_to_submap mode) ---
        SUBMAP_RESOLUTION  = 0.05,
        SUBMAP_SIZE_METERS = 20.0,
        # Faster robot at ~10 Hz → ~0.05 m/scan → 90 scans ≈ 4.5 m traversal.
        # Matches the carto runner's value for this dataset.
        SCANS_PER_SUBMAP   = 90,

        # Correlative coarse search — tighter than lab because odometry
        # already provides a good initial guess.
        SUBMAP_COARSE_XY_WINDOW = 0.30,
        SUBMAP_COARSE_XY_STEP   = 0.05,
        SUBMAP_COARSE_TH_WINDOW = 0.20,   # ~11 deg
        SUBMAP_COARSE_TH_STEP   = 0.05,

        SUBMAP_FINE_XY_WINDOW = 0.15,
        SUBMAP_FINE_XY_STEP   = 0.03,
        SUBMAP_FINE_TH_WINDOW = 0.08,
        SUBMAP_FINE_TH_STEP   = 0.02,

        SUBMAP_MAX_MATCH_POINTS  = 300,   # use all 360 beams
        SUBMAP_MAX_REFINE_POINTS = 300,
        SUBMAP_MIN_VALID         = 20,
        SUBMAP_MIN_SCORE         = 0.52,
        SUBMAP_REFINE_W_TRANS    = 0.2,
        SUBMAP_REFINE_W_ROT      = 1.0,
        SUBMAP_REFINE_MIN_POINTS = 20,

        # scan_to_map GN correlative params
        CORR_MAP_MIN_POINTS     = 30,
        CORR_MAP_MIN_INLIERS    = 30,
        CORR_MAP_MIN_SCORE      = 0.35,
        CORR_MAP_STEP_CLIP_XY   = 0.03,   # original Hector Python: ±0.03 m/iter

        # --- Log-odds occupancy model ---
        P_OCC     = 0.70,
        P_FREE    = 0.40,
        L_OCC     = 0.85,
        L_FREE    = -0.1,
        L_MIN     = -5.0,
        L_MAX     = 5.0,
        RAY_STEPS = 40,     # more ray steps for longer-range sensor

        # --- GN scan-to-map solver ---
        PYRAMID_LEVELS     = 3,
        GN_ITERS_PER_LEVEL = [2, 2, 1],
        GN_DAMPING         = 2e-2,
        N_BOOTSTRAP_SCANS  = 1,

        # --- Pose extrapolator ---
        USE_EXTRAPOLATOR = False,
        EXTRAP_MAX_DT   = 1.0,   # longer window for variable CARMEN timestamps
        EXTRAP_INIT_VXY = 0.0,
        EXTRAP_INIT_WZ  = 0.0,

        # --- Motion filter — benchmark robot at ~10 Hz ---
        TARGET_INSERT_PERIOD_S = 0.10,
        V_EXPECTED_MPS         = 0.5,
        W_EXPECTED_RPS         = np.deg2rad(30.0),

        # --- Pose graph / PGO ---
        KEYFRAME_STRIDE = 10,
        ODOM_SIGMA_XY   = 0.05,
        ODOM_SIGMA_TH   = np.deg2rad(3.0),
        PGO_ITERS       = 15,
        PGO_DAMPING     = 1e-6,
    ),

    # ----------------------------------------------------------
    # intel — Intel Research Lab dataset (CARMEN log).
    #
    # Notable characteristics:
    #   • SICK LMS, 180° FOV, 180 beams at 1° angular resolution
    #   • Sparser angular coverage than fr079 (1° vs ~0.5°)
    #   • Wheel odometry available but has known drift over long runs
    #   • Large office building environment (~800 m full path)
    # ----------------------------------------------------------
    "intel": dict(

        # --- Odometry / initial pose ---
        # Lower odometry trust than fr079: intel odometry accumulates
        # more drift, so we rely more heavily on the scan matcher.
        ODOM_ALPHA         = 0.35,
        INITIAL_POSE_X     = 0.0,
        INITIAL_POSE_Y     = 0.0,
        INITIAL_POSE_THETA = 0.0,

        # --- Scan handling ---
        BEAM_STRIDE     = 1,
        LIDAR_MIN_RANGE = 0.10,
        LIDAR_MAX_RANGE = 30.0,

        # --- Voxel pre-processing ---
        # Disabled: only 180 beams.
        VOXEL_FILTER_ENABLED      = False,
        VOXEL_FIXED_SIZE          = 0.05,
        VOXEL_ADAPTIVE_MAX_SIZE   = 0.15,
        VOXEL_ADAPTIVE_MIN_POINTS = 100,
        VOXEL_ADAPTIVE_ITERS      = 6,

        # --- Global map (scan_to_map mode) ---
        MAP_RESOLUTION  = 0.05,
        MAP_SIZE_METERS = 80.0,

        # --- Submap (scan_to_submap mode) ---
        SUBMAP_RESOLUTION  = 0.05,
        SUBMAP_SIZE_METERS = 20.0,
        SCANS_PER_SUBMAP   = 90,

        # Correlative coarse search — tight, odometry provides good prior
        SUBMAP_COARSE_XY_WINDOW = 0.30,
        SUBMAP_COARSE_XY_STEP   = 0.05,
        SUBMAP_COARSE_TH_WINDOW = 0.20,
        SUBMAP_COARSE_TH_STEP   = 0.05,

        SUBMAP_FINE_XY_WINDOW = 0.15,
        SUBMAP_FINE_XY_STEP   = 0.03,
        SUBMAP_FINE_TH_WINDOW = 0.08,
        SUBMAP_FINE_TH_STEP   = 0.02,

        # Lower point caps to match the sparser 180-beam scanner
        SUBMAP_MAX_MATCH_POINTS  = 150,
        SUBMAP_MAX_REFINE_POINTS = 150,
        SUBMAP_MIN_VALID         = 15,
        SUBMAP_MIN_SCORE         = 0.60,
        SUBMAP_REFINE_W_TRANS    = 0.2,
        SUBMAP_REFINE_W_ROT      = 1.0,
        SUBMAP_REFINE_MIN_POINTS = 15,

        # scan_to_map GN correlative params
        CORR_MAP_MIN_POINTS     = 20,
        CORR_MAP_MIN_INLIERS    = 20,
        CORR_MAP_MIN_SCORE      = 0.10,
        CORR_MAP_STEP_CLIP_XY   = 0.03,   # original Hector Python: ±0.03 m/iter

        # --- Log-odds occupancy model ---
        P_OCC     = 0.70,
        P_FREE    = 0.40,
        L_OCC     = 0.85,
        L_FREE    = -0.1,
        L_MIN     = -5.0,
        L_MAX     = 5.0,
        RAY_STEPS = 40,

        # --- GN scan-to-map solver ---
        PYRAMID_LEVELS     = 3,
        GN_ITERS_PER_LEVEL = [15, 10, 8],
        GN_DAMPING         = 1e-4,
        N_BOOTSTRAP_SCANS  = 1,

        # --- Pose extrapolator ---
        USE_EXTRAPOLATOR = False,
        EXTRAP_MAX_DT   = 1.0,
        EXTRAP_INIT_VXY = 0.0,
        EXTRAP_INIT_WZ  = 0.0,

        # --- Motion filter ---
        TARGET_INSERT_PERIOD_S = 0.10,
        V_EXPECTED_MPS         = 0.5,
        W_EXPECTED_RPS         = np.deg2rad(30.0),

        # --- Pose graph / PGO ---
        KEYFRAME_STRIDE = 10,
        ODOM_SIGMA_XY   = 0.05,
        ODOM_SIGMA_TH   = np.deg2rad(3.0),
        PGO_ITERS       = 15,
        PGO_DAMPING     = 1e-6,
    ),
}


def _apply_profile(dataset_name: str) -> None:
    """Inject all profile keys for *dataset_name* as module-level attributes."""
    import sys
    mod = sys.modules[__name__]
    if dataset_name not in _PROFILES:
        raise ValueError(
            f"Unknown DATASET_NAME={dataset_name!r}. "
            f"Supported: {', '.join(_PROFILES)}"
        )
    for key, value in _PROFILES[dataset_name].items():
        setattr(mod, key, value)


# Apply the active profile immediately at import time so that every
# cfg.XXX attribute reflects the chosen dataset.
_apply_profile(DATASET_NAME)

# ==============================================================
# Derived constants (dataset-independent)
# ==============================================================
P0 = 0.5
L0 = float(np.log(P0 / (1.0 - P0)))   # always 0.0; kept for downstream compat
