import numpy as np

# --- Sensor model (for fr079) ---
NUM_BEAMS = 360
ANGLE_MIN = -np.pi / 2.0
ANGLE_MAX = +np.pi / 2.0
ANGLE_INC = (ANGLE_MAX - ANGLE_MIN) / (NUM_BEAMS - 1)

RANGE_MIN = 0.10
RANGE_MAX = 30.0
BEAM_STRIDE = 4

# --- Submaps ---
SUBMAP_RESOLUTION = 0.05
SUBMAP_SIZE_METERS = 20.0
SCANS_PER_SUBMAP = 90

# --- Occupancy update (log odds) ---
L0 = 0.0
L_FREE = -0.1
L_OCC = 0.85
L_MIN = -5.0
L_MAX = 5.0

RAY_STEPS = 40

# --- Motion Filter ---
MOTION_FILTER_TRANSLATION = 0.05    #m
MOTION_FILTER_ROTATION = 0.05       #rad

# --- constant-velocity extrapolation ---
EXTRAP_MAX_DT = 0.5          # seconds
EXTRAP_INIT_VXY = 0.0
EXTRAP_INIT_WZ  = 0.0

# --- correlative scan matching search window ---
CORR_XY_WINDOW = 0.3
CORR_TH_WINDOW = 0.15
CORR_XY_STEP   = 0.10
CORR_TH_STEP   = 0.05
MIN_MATCH_SCORE = 0.52