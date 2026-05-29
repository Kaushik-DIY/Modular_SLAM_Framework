"""
Modular SLAM configuration for the Hector SLAM runner.

How to use
----------
Set DATASET_NAME (and DATASET_SCAN_VARIANT for lab_run_2) at the top of
this file, or override from the command line:

    python -m hector.run_local_slam_new --dataset fr079 --max-scans 1000

This file is the single source of truth for tuning — both
run_local_slam_new.py and run_realtime_viz.py read every parameter from here
via `cfg.XXX`, so you should never need to edit the runner files.

Where to change things
----------------------
  • USER SELECTION block  — dataset, scan variant, matcher type, scan cap.
  • _COMMON dict          — knobs shared across datasets: the GN-LM refine
                            solver, motion filter / IMU extrapolator, live
                            switching, the PGO back-end, and realtime-viz
                            runtime defaults.
  • _PROFILES[<dataset>]  — only what differs per dataset (sensor geometry,
                            environment size, per-dataset solver tuning); these
                            override _COMMON.

_apply_profile() applies _COMMON first, then the dataset profile.

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
# Shared tuning defaults (apply to every dataset)
# ==============================================================
# These are the knobs that are normally the same across datasets (matcher
# internals, the GN-LM refine solver, live-switching, the PGO back-end, and the
# realtime-viz runtime defaults).  They are applied FIRST, then a dataset profile
# below can override any of them.  Both run_local_slam_new.py and
# run_realtime_viz.py read every value here via `cfg.XXX`, so this file is the
# single place to tune behaviour — no need to edit the runner files.
_COMMON: dict = dict(

    # --- Matcher selection default (overridable by --matcher) ---
    MAP_UPDATE_EVERY = 1,          # scan_to_map: insert every Nth scan when motion filter is OFF

    # --- IMU-aided extrapolator + motion-filter keyframing (opt-in) ---
    # OFF by default so the baseline path (match every scan, last-pose prior) is
    # preserved exactly.  Enable per run with --use-imu / --use-motion-filter
    # (either flag also forces the extrapolator ON).
    USE_IMU                  = False,   # feed gyro yaw-rate + quaternion yaw into the extrapolator
    IMU_YAW_CORRECTION_ALPHA = 0.02,    # blend toward IMU absolute heading per predict (0=gyro-rate only)
    USE_MOTION_FILTER        = False,   # skip GN on sub-threshold scans (dead-reckon them)
    MF_MAX_DIST_M            = 0.10,    # keyframe if translation since last keyframe exceeds this [m]
    MF_MAX_ANGLE_DEG         = 2.0,     # keyframe if rotation since last keyframe exceeds this [deg]
    MF_MAX_TIME_S            = 0.5,     # keyframe if time since last keyframe exceeds this [s]

    # --- scan_to_map GN-LM solver (extra knobs; per-level iters/damping are per-profile) ---
    GN_STEP_CLIP_TH_DEG = 1.0,     # max GN rotation step per iteration [deg]

    # --- scan_to_submap two-stage front-end (correlative search + native GN-LM refine) ---
    # Correlative search executor: False = scalar Python loop (deterministic, the
    # stable default); True = vectorized NumPy batch (~10x faster, same scores, may
    # break exact ties differently). Opt-in via --vectorized-search.
    SUBMAP_VECTORIZED_SEARCH    = False,
    SUBMAP_PRECOMP_LEVELS       = 3,    # multi-resolution precompute levels for correlative search
    SUBMAP_COARSE_LEVEL         = 2,    # pyramid level used by the coarse correlative pass
    SUBMAP_FINE_LEVEL           = 0,    # pyramid level used by the fine correlative pass
    SUBMAP_REFINE_ITERS         = 12,   # GN-LM iterations in the local refine stage
    SUBMAP_REFINE_DAMPING       = 1e-3, # LM damping for the local refine stage
    SUBMAP_REFINE_STEP_CLIP_XY  = 0.10, # max refine translation step per iteration [m]
    SUBMAP_REFINE_STEP_CLIP_TH_DEG = 5.0,  # max refine rotation step per iteration [deg]

    # --- Matcher manager / live front-end switching (realtime viz) ---
    ROLLING_BUFFER_SIZE  = 30,     # recent matched scans kept for warm-starting a switched matcher
    MIN_BUFFER_FOR_SWITCH = 20,    # min buffered scans before a live matcher switch may take effect
    SWITCH_GRACE_SCANS   = 15,     # scans the old matcher keeps running after a switch is requested

    # --- Online PGO: g2o SE2 solver ---
    PGO_LOCAL_TRANS_WEIGHT = 1e5,  # information weight on the local-SLAM (front-end) pose prior, xy
    PGO_LOCAL_ROT_WEIGHT   = 1e5,  # information weight on the local-SLAM pose prior, rotation
    PGO_HUBER_SCALE        = 10.0, # Huber robust-kernel scale for loop edges
    PGO_MAX_ITERATIONS     = 50,   # Levenberg iterations per global solve

    # --- Online PGO: pose-graph constraints ---
    PGO_INTRA_TRANS_WEIGHT = 5e2,  # intra-submap (node↔submap) constraint weight, xy
    PGO_INTRA_ROT_WEIGHT   = 1.6e3,# intra-submap constraint weight, rotation
    PGO_OPTIMIZE_EVERY_N_NODES = 90,  # run a global solve every N inserted keyframe nodes

    # --- Online PGO: loop-closure detection ---
    PGO_LOOP_MIN_SCORE        = 0.50,  # min branch-and-bound score to accept a loop candidate
    PGO_LOOP_TRANS_WEIGHT     = 1.1e4, # accepted loop-closure constraint weight, xy
    PGO_LOOP_ROT_WEIGHT       = 1e5,   # accepted loop-closure constraint weight, rotation
    PGO_LOOP_SEARCH_XY        = 7.0,   # loop-search translational window half-extent [m]
    PGO_LOOP_SEARCH_TH_DEG    = 30.0,  # loop-search rotational window half-extent [deg]
    PGO_LOOP_PRECOMP_LEVELS   = 7,     # branch-and-bound precompute depth for loop search
    PGO_LOOP_BNB_DEPTH        = 7,     # branch-and-bound recursion depth limit
    PGO_LOOP_BNB_MIN_ROT_STEP = 0.02,  # branch-and-bound minimum rotational step [rad]
    PGO_LOOP_BNB_BRANCHING    = 4,     # branch-and-bound branching factor
    PGO_MIN_NODE_SEPARATION   = 30,    # min node-index gap between the two ends of a loop
    PGO_SPATIAL_SEARCH_RADIUS = 8.0,   # only search submaps within this radius of the new node [m]
    PGO_MAX_CANDIDATE_TARGETS = 3,     # max loop candidates evaluated per new node
    PGO_HISTORICAL_NODE_STRIDE = 3,    # stride when scanning historical nodes for candidates
    PGO_CHECK_EVERY_N_NODES   = 5,     # only run loop search on every Nth node (bounds cost)
    PGO_RECENT_SUBMAP_EXCLUSION = 0,   # exclude this many most-recent finished submaps from loop search
    PGO_CORRECTION_ALPHA      = 0.5,   # blend factor when writing optimized poses back online

    # --- Realtime-viz runtime defaults (overridable by CLI flags) ---
    PLAYBACK_SPEED         = 1.0,   # 1.0=real-time, 2.0=2x, 0=as-fast-as-possible
    DRAW_EVERY             = 5,     # redraw the live plot every N scans
    REALTIME_VERBOSE_EVERY = 20,    # print a live timing line every N scans
    POINTS_MAX             = 80000, # cap displayed --live-points cloud (stride-subsampled above this)
)


# ==============================================================
# Per-dataset parameter profiles
# ==============================================================
# A profile only needs to list keys that DIFFER from _COMMON (sensor geometry,
# environment size, per-dataset solver tuning).  _apply_profile() applies _COMMON
# first, then the profile, so any key here overrides the shared default.
# Keys are injected as module-level attributes so the runner can use `cfg.XXX`.

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
        SCANS_PER_SUBMAP   = 250,

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
        # IMU/motion-filter knobs (USE_IMU, USE_MOTION_FILTER, MF_*, etc.) are
        # shared defaults in _COMMON; override here only if lab needs different.
        USE_EXTRAPOLATOR = False,
        EXTRAP_MAX_DT   = 0.5,
        EXTRAP_INIT_VXY = 0.0,
        EXTRAP_INIT_WZ  = 0.0,

        # --- Motion filter (velocity-derived, scan_to_submap cadence) — slow robot at 10 Hz ---
        TARGET_INSERT_PERIOD_S = 0.10,    # target seconds between submap inserts
        V_EXPECTED_MPS         = 1.0,     # expected linear speed → derives insert distance
        W_EXPECTED_RPS         = np.deg2rad(45.0),  # expected angular speed → derives insert angle

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
    """Inject shared defaults then the dataset profile as module-level attributes.

    _COMMON is applied first; the per-dataset profile is applied second so any
    key it defines overrides the shared default.
    """
    import sys
    mod = sys.modules[__name__]
    if dataset_name not in _PROFILES:
        raise ValueError(
            f"Unknown DATASET_NAME={dataset_name!r}. "
            f"Supported: {', '.join(_PROFILES)}"
        )
    for key, value in _COMMON.items():
        setattr(mod, key, value)
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


# ==============================================================
# Cartographer-style fast local matching + IMU (run_realtime_viz --fast-match / --use-imu)
# ==============================================================
# --fast-match runs the SAME full correlative search + refine as the default path, but
# vectorized (batched NumPy) plus vectorized submap insertion (fast_insert). Same map
# quality, ~10x faster. Does NOT affect the default scalar path or the stable runner.
# --use-imu feeds gyro yaw-rate + quaternion yaw into the extrapolator. Finding on this
# slow indoor robot: scan-derived velocity already predicts well, so the gyro adds
# bias/noise and IMU is OFF by default (kept for fast-motion / feature-poor runs).
# The IMU/motion-filter knobs themselves (USE_IMU, IMU_YAW_CORRECTION_ALPHA,
# USE_MOTION_FILTER, MF_*) live in _COMMON above.
