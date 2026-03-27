
import numpy as np

# =========================
# Map parameters
# =========================
MAP_RESOLUTION = 0.05      # meters per cell
MAP_SIZE_METERS = 80.0    # map width/height (square)

# =========================
# Log-odds parameters
# =========================
P0 = 0.5
L0 = np.log(P0 / (1.0 - P0))  # = 0

P_OCC = 0.70
P_FREE = 0.40

L_FREE = -0.1
L_OCC = 0.85
# L_OCC = np.log(P_OCC / (1.0 - P_OCC))
# L_FREE = np.log(P_FREE / (1.0 - P_FREE))

L_MIN = -5.0
L_MAX =  5.0

# =========================
# Hector scan matcher params
# =========================
PYRAMID_LEVELS = 3
GN_ITERS_PER_LEVEL = [2, 2, 1]  # coarse -> fine
GN_DAMPING = 2e-2

# Use every k-th beam for speed
BEAM_STRIDE = 4

# Lidar limits (set conservative defaults)
LIDAR_MIN_RANGE = 0.10
LIDAR_MAX_RANGE = 30.0

# Mapping update strength
RAY_STEPS = 40 # coarse ray sampling count

# -------------------------
# Pose-graph / Loop closure
# -------------------------
KEYFRAME_STRIDE = 10 # add a graph node every N scans

ODOM_SIGMA_XY = 0.10      # m (edge strength)
ODOM_SIGMA_TH = np.deg2rad(5.0)
PGO_ITERS = 15
PGO_DAMPING = 1e-6
