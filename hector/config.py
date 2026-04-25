import numpy as np

# =========================
# Dataset selection
# =========================
# Supported values:
#   - "fr079"
#   - "intel"
#   - "lab_run_2"
DATASET_NAME = "lab_run_2"

# Used only by the lab dataset.
#   - "raw"  -> 909 beams, maximum scan detail
#   - "360"  -> 360 beams, lower compute cost
DATASET_SCAN_VARIANT = "raw"

# =========================
# Runtime / logging
# =========================
MATCHER_TYPE = "scan_to_submap"   # "scan_to_map" or "scan_to_submap"
MAX_SCANS = None               # None = use all scans (1087 for lab_run_2)
VERBOSE_EVERY = 50

# Odometry blending for datasets that provide odometry.
# Keep this at 0.0 for LiDAR-only lab experiments.
ODOM_ALPHA = 0.0

# Initial pose used when the dataset has no odometry prior.
INITIAL_POSE_X = 0.0
INITIAL_POSE_Y = 0.0
INITIAL_POSE_THETA = 0.0

# =========================
# Shared scan handling
# =========================
# Subsample every k-th beam after dataset geometry is applied.
BEAM_STRIDE = 1

# Conservative LiDAR limits. Dataset-specific geometry is resolved separately.
# Note: for lab_run_2 range_max=16m is used directly from the dataset profile.
LIDAR_MIN_RANGE = 0.10
LIDAR_MAX_RANGE = 30.0  # conservative upper bound; profile.range_max takes precedence

# =========================
# Map parameters
# =========================
MAP_RESOLUTION = 0.05      # meters per cell (5cm — good match for 16m-range lab LiDAR)
MAP_SIZE_METERS = 40.0     # lab room fits within 40x40m; reduces out-of-bounds risk

# =========================
# Submap matcher parameters (scan_to_submap)
# =========================
# Submap physical size — must be large enough to contain all scan endpoints
# from all poses that belong to the submap.  For the lab LiDAR (range_max=16 m)
# the worst-case footprint radius from the submap origin is:
#   max_traverse + range_max  ≈  travel_per_submap + 16 m.
# 20 m is a comfortable fit for traversals up to ~4 m and still keeps the grid
# compact (400×400 cells at 5 cm).
SUBMAP_RESOLUTION   = 0.05
SUBMAP_SIZE_METERS  = 20.0

# === Distance-based submap rotation ===
# Strategy: rotate submaps every ~17 m of path,  so that each submap covers
# one major segment of the environment.  This is expressed as a scan count
# via the observed scan density:
#
#   lab_run_2 stats (scan_to_map reference trajectory):
#     total path ≈  34.3 m, 1087 accepted scans → mean step ≈ 0.032 m/scan
#     ⇒ ~5 m traversal is  ~160 scans
#     ⇒ ~17 m traversal is ~530 scans
#
# With SCANS_PER_SUBMAP = 500 the Cartographer rotation logic gives:
#   submap-0 finished at scan 500, rotated out at scan 750  → 2 finished submaps
#   submap-1 finished at scan 750, still active at scan 1087
#   ⇒ total ≥ 2 finished + 1 active  (3 submaps total for the lab)
#
# For a larger dataset (e.g., intel = 13k scans, ~800 m path) the same setting
# gives ~26 submaps — good coverage without being excessively fine.
SCANS_PER_SUBMAP    = 500

# Correlative coarse search window (in submap frame).
# Wider than default to recover from extrapolator lag at corners/turns.
SUBMAP_COARSE_XY_WINDOW  = 1.0    # metres
SUBMAP_COARSE_XY_STEP    = 0.10   # 2x finer than default 0.20 → better init
SUBMAP_COARSE_TH_WINDOW  = 0.40   # ~23 deg, enough for sharp lab turns
SUBMAP_COARSE_TH_STEP    = 0.05

# Correlative fine search window.
SUBMAP_FINE_XY_WINDOW    = 0.25
SUBMAP_FINE_XY_STEP      = 0.05
SUBMAP_FINE_TH_WINDOW    = 0.12
SUBMAP_FINE_TH_STEP      = 0.02

# Match/refine point caps.
# After voxel filtering we have ~204 pts/scan; use 200 for the correlative
# stage (covers the full preprocessed cloud) and all for refinement.
SUBMAP_MAX_MATCH_POINTS  = 200    # was 60 — too few for 909-beam scanner
SUBMAP_MAX_REFINE_POINTS = 200
SUBMAP_MIN_VALID         = 30     # min in-submap points to score a candidate
SUBMAP_MIN_SCORE         = 0.50   # correlative score threshold

# Refinement priors.  Loose translation prior allows GN to move freely within
# the scan-matched window; tighter rotation prior keeps heading stable.
SUBMAP_REFINE_W_TRANS    = 0.1
SUBMAP_REFINE_W_ROT      = 1.0
SUBMAP_REFINE_MIN_POINTS = 30

# =========================
# Log-odds parameters
# =========================
P0 = 0.5
L0 = np.log(P0 / (1.0 - P0))  # = 0

P_OCC = 0.75   # stronger occupied confidence for lab walls
P_FREE = 0.40

L_FREE = -0.1
L_OCC = 1.0    # stronger log-odds update per hit (was 0.85)

L_MIN = -5.0
L_MAX = 5.0

# Mapping update speed: 20 ray steps adequate for 5cm resolution, 2x faster than 40
RAY_STEPS = 20

# =========================
# Hector scan matcher params
# =========================
PYRAMID_LEVELS = 3
# Per-level GN iterations from coarse to fine.
# Each level: [coarse, mid, fine].  More iters = better convergence, more CPU.
GN_ITERS_PER_LEVEL = [20, 15, 10]  # proper multi-resolution schedule
GN_DAMPING = 1e-4                   # lower damping = sharper, faster convergence

# Number of scans to integrate at dead-reckoned positions before GN matching starts.
# Multi-scan seeding gives the map enough evidence density for GN gradients to be
# meaningful, eliminating the degenerate single-scan bootstrap failure.
N_BOOTSTRAP_SCANS = 5

# -------------------------
# Pose extrapolator
# -------------------------
EXTRAP_MAX_DT = 0.5  # seconds
EXTRAP_INIT_VXY = 0.0
EXTRAP_INIT_WZ = 0.0

# -------------------------
# Motion filter tuning
# -------------------------
TARGET_INSERT_PERIOD_S = 0.10   # update map every scan (10 Hz lab scan rate)
V_EXPECTED_MPS = 1.0            # walking human in lab ~1 m/s
W_EXPECTED_RPS = np.deg2rad(45.0)   # faster turns in tight lab space

# -------------------------
# Pose-graph / Loop closure
# -------------------------
KEYFRAME_STRIDE = 10
ODOM_SIGMA_XY = 0.10
ODOM_SIGMA_TH = np.deg2rad(5.0)
PGO_ITERS = 15
PGO_DAMPING = 1e-6