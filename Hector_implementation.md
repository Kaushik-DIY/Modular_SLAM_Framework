# Hector SLAM — Complete Implementation Review

**Target audience:** Anyone curious about this SLAM system, including readers with little or no robotics background.

---

## What Is SLAM and Why Does It Matter?

Imagine you are dropped blindfolded into an unknown building and told to draw a map of it while figuring out where you are. That is exactly the problem robots face, and it is called **SLAM — Simultaneous Localization and Mapping**.

- **Localization** = "Where am I?"
- **Mapping** = "What does the environment look like?"

The challenge is circular: you need a map to know where you are, but you need to know where you are to build a map. SLAM solves both problems at the same time.

This implementation solves 2D SLAM — the robot moves on a flat floor, and the map is a 2D bird's-eye-view of the environment.

---

## High-Level Pipeline at a Glance

```
[Sensor]
   │  Raw distance measurements (range data)
   ▼
[Data Loader]
   │  Reads the log file and sensor geometry
   ▼
[Range → Point Cloud]
   │  Converts distances into 2D (x, y) points
   ▼
[Voxel Filter]  (optional, dataset-dependent)
   │  Removes redundant points, keeps ~200 clean points
   ▼
[Pose Predictor]
   │  "Where do I think I am right now?"
   ▼
[Scan Matcher]  ← CORE OF THE SYSTEM
   │  Aligns the new scan to the map to get the precise pose
   ▼
[Map Updater]
   │  Paints the new scan into the occupancy grid map
   ▼
[Trajectory Writer]
   │  Appends (timestamp, x, y, θ, score) to output file
   ▼
[Post-Run PGO]  (optional, run separately)
   │  Finds loop closures, globally corrects drift
   ▼
[Map & Trajectory Plots]
   │  Visualizes the trajectory overlaid on the final map
```

---

## Part 1 — Sensors Used

### Three Supported Datasets

The system supports three different real-world sensor setups. All produce **laser range scans** — the sensor spins a laser beam and measures how far away the nearest obstacle is at each angle.

---

### 1.1 Lab Run 2 — Custom JetRacer Robot

| Property | Value |
|---|---|
| Robot | JetRacer AI Kit (small wheeled robot) |
| Sensor | YD LiDAR G4 (spinning laser rangefinder) |
| Field of View | 360° — full circle around the robot |
| Number of beams | 909 raw beams per scan |
| Range | 0.10 m to 16.0 m |
| Odometry | None (no wheel encoders used) |
| Environment | Small indoor lab room, roughly 8 × 6 metres |
| Data format | Custom CARMEN log file (`scans.carmen`) |

The LiDAR G4 fires 909 laser beams in a circle at roughly 10 Hz. Each beam returns one distance measurement. Because 909 points is more than the solver benefits from, the voxel filter is switched on for this dataset.

An IMU (Inertial Measurement Unit) is present in the dataset but is **not used** in this Hector SLAM pipeline — it exists for future use.

---

### 1.2 Freiburg FR079 — Corridor Benchmark

| Property | Value |
|---|---|
| Robot | University of Freiburg research robot |
| Sensor | SICK LMS laser scanner |
| Field of View | 180° (a half-circle in front of the robot) |
| Number of beams | 360 beams per scan |
| Range | 0.10 m to 30.0 m |
| Odometry | Wheel odometry embedded in the log |
| Environment | Long university corridor, ~60 m total path |
| Data format | CARMEN log file (`fr079.clf`) |

---

### 1.3 Intel Research Lab — Large Building Benchmark

| Property | Value |
|---|---|
| Robot | Intel Research Lab robot |
| Sensor | SICK LMS laser scanner |
| Field of View | 180° |
| Number of beams | 180 beams per scan (coarser, 1° per beam) |
| Range | 0.10 m to 30.0 m |
| Odometry | Wheel odometry (but known to drift over long runs) |
| Environment | Large office building, ~800 m total path |
| Data format | CARMEN log file (`intel.clf`) |

---

## Part 2 — Configuration System

**File:** [hector/config.py](hector/config.py)

Before running, every parameter the system needs is loaded from a per-dataset profile defined in `config.py`. You can think of this as the "settings panel" — one set of knobs per dataset, covering everything from sensor geometry to how aggressively the scan matcher should iterate.

### What You Control at the Top

```python
DATASET_NAME         = "lab_run_2"    # which dataset to run
DATASET_SCAN_VARIANT = "raw"          # lab_run_2 only: "raw" (909 beams) or "360" (360 beams)
MATCHER_TYPE         = "scan_to_map"  # "scan_to_map" or "scan_to_submap"
MAX_SCANS            = None           # None = all scans; set an integer to limit for testing
```

Or override from the command line:
```bash
python -m hector.run_local_slam_new --dataset fr079 --matcher scan_to_map --max-scans 500
```

### Key Parameters per Dataset (summary)

| Parameter | Lab Run 2 | FR079 | Intel | What it controls |
|---|---|---|---|---|
| `ODOM_ALPHA` | 0.0 | 0.5 | 0.35 | How much to trust wheel odometry (0 = ignore, 1 = full trust) |
| `VOXEL_FILTER_ENABLED` | True | False | False | Whether to thin out the point cloud |
| `MAP_RESOLUTION` | 0.05 m | 0.05 m | 0.05 m | Grid cell size (5 cm) |
| `PYRAMID_LEVELS` | 3 | 3 | 3 | Number of map resolution levels |
| `GN_ITERS_PER_LEVEL` | [20, 15, 10] | [2, 2, 1] | [15, 10, 8] | GN iterations at each pyramid level |
| `L_OCC` | 1.0 | 0.85 | 0.85 | How strongly a hit updates the map |
| `L_FREE` | -0.1 | -0.1 | -0.1 | How strongly a miss updates the map |
| `RAY_STEPS` | 20 | 40 | 40 | Steps used when casting a ray through the map |

---

## Part 3 — Step-by-Step Pipeline

### Step 1: Startup and Dataset Loading

**File:** [hector/run_local_slam_new.py](hector/run_local_slam_new.py)  
**Supporting file:** [slam_core/dataio/dataset_catalog.py](slam_core/dataio/dataset_catalog.py)

The main script `run_local_slam_new.py` is the entry point. When it starts:

1. It reads the command-line arguments and applies any overrides to the config.
2. It calls `load_dataset_scans(cfg.DATASET_NAME)`, which:
   - Looks up a **DatasetProfile** — a frozen record containing the sensor geometry (number of beams, angle range, range limits, whether odometry is available, the file path of the log).
   - Calls the appropriate reader (`read_carmen_log`, `read_intel_carmen_log`, or `read_lab_carmen_log`) which parses the log file line by line and returns a Python list of scan dictionaries.

Each scan dictionary looks like:
```python
{
    "t":      1234567890.123,       # Unix timestamp in seconds
    "ranges": [0.82, 0.95, ...],   # list of N distance values (one per beam)
    "odom":   (x, y, theta)        # optional wheel odometry pose (world frame)
}
```

The system prints a startup summary:
```
Dataset      : lab_run_2
Scan variant : raw
Geometry     : beams=909  angle=[-180.0°, 180.0°]  range=[0.10, 16.00] m  has_odom=False
Total scans  : 3420
Matcher      : scan_to_map
```

---

### Step 2: Converting Range Data to a 2D Point Cloud

**File:** `carto/local_slam/range_to_points.py` — function `ranges_to_points()`

The raw sensor data is just a list of distances — one number per laser beam. To do any geometry, we first need to convert these distances into 2D (x, y) coordinates **in the robot's local frame** (the robot is at the origin, facing the +x direction).

**How it works (for a single beam):**

The sensor fires beam number `i` at a known angle:
```
angle_i = angle_min + i * angle_increment
```

The measured distance `r_i` converts to Cartesian:
```
x_i = r_i * cos(angle_i)
y_i = r_i * sin(angle_i)
```

**Filters applied:**
- Beams shorter than `LIDAR_MIN_RANGE` (0.10 m) are discarded — these are sensor noise or self-reflections.
- Beams at or beyond `range_max` are discarded — these hit nothing (open space).
- `BEAM_STRIDE` can skip every Nth beam to reduce computation (lab_run_2: stride 1 = use all; fr079: stride 2 = use every other beam).

After this step, each scan is a NumPy array of shape `(N, 2)` — N points with (x, y) coordinates in the robot frame, measured in metres.

---

### Step 3: Voxel Filtering (Point Cloud Thinning)

**File:** [slam_core/matching/preprocessing.py](slam_core/matching/preprocessing.py)  
**Class:** `PointCloudProcessor`

**Why is this needed?**  
The lab_run_2 dataset produces 909 laser beams per scan. Near a wall 0.5 m away, dozens of nearby beams all hit essentially the same spot. Having 50 nearly-identical points from the same wall patch does not help the solver — it just wastes compute. Voxel filtering reduces these to one representative point per region.

**How it works — two stages:**

**Stage 1 — Fixed Voxel Filter** (voxel size = 0.03 m for lab):
- Divide 2D space into a grid of 3 cm × 3 cm cells (voxels).
- All points that fall into the same cell are collapsed into their centroid (average position).
- This removes tightly-clustered duplicates while preserving spatial coverage.

**Stage 2 — Adaptive Voxel Filter:**
- Binary search for the largest voxel size that still keeps at least `VOXEL_ADAPTIVE_MIN_POINTS` (200 for lab_run_2) points.
- This ensures the output always has enough points for the scan matcher regardless of how sparse or dense the scene is.

**Result for lab_run_2:** 909 raw beams → typically ~200 clean, well-distributed points.  
**FR079 and Intel:** Voxel filtering disabled — 360/180 beams is already manageable.

---

### Step 4: Pose Prediction ("Where Am I Before Matching?")

**Files:** `carto/local_slam/pose_extrapolator.py` and [hector/adapter.py](hector/adapter.py)

Before trying to match the new scan against the map, the system needs a starting guess for the robot's current pose. This guess is called the **predicted pose**.

**What is a pose?**  
A 2D pose is three numbers: `(x, y, θ)` — position in metres and heading angle in radians. The robot starts at `(0, 0, 0)`.

**Two prediction strategies:**

**Strategy A — Constant Velocity Extrapolator** (`PoseExtrapolatorCV`):  
Tracks the robot's recent velocity. Predicts: `pose(t) = pose(t_last) + velocity × Δt`. This is like saying "if the robot was moving at 0.3 m/s to the right, it's probably 0.03 m further right 0.1 seconds later."

**Strategy B — Raw Odometry** (when `USE_EXTRAPOLATOR = False`):  
Uses the wheel encoder reading directly as the prediction. Simpler but relies entirely on the quality of the odometry.

**Odometry blending** (fr079 and intel only):  
When odometry is available, the two predictions are blended:
```
predicted_pose = (1 - ODOM_ALPHA) × extrapolator_pose + ODOM_ALPHA × odometry_pose
```
For fr079: `ODOM_ALPHA = 0.5` (50/50 blend because the odometry is reliable).  
For intel: `ODOM_ALPHA = 0.35` (less trust because intel odometry drifts).  
For lab_run_2: `ODOM_ALPHA = 0.0` (no odometry, ignore).

---

### Step 5: Scan Matching — The Heart of the System

This is the most important step. The scan matcher takes the predicted pose and the filtered point cloud, and finds the **best pose** that makes the scan align with the current map.

There are two interchangeable matcher implementations:

---

#### Matcher A — Scan-to-Map (Hector-Style)

**File:** [slam_core/matching/scan_to_map.py](slam_core/matching/scan_to_map.py)

**Core idea:** Match the new laser scan directly against a single large global occupancy grid map using iterative mathematical optimization. This is the classic Hector SLAM approach.

---

##### 5A.1 — The Occupancy Grid Map

Think of the map as a giant sheet of graph paper where each 5 cm × 5 cm square (cell) stores a single number representing what the robot believes about that cell:

- **Positive number** → probably occupied (wall, obstacle)
- **Zero** → completely unknown
- **Negative number** → probably free space (open area)

The stored number is actually in **log-odds** form (a mathematical transformation of probability) to make updates numerically stable. The actual occupancy probability is recovered as:
```
P(occupied) = 1 / (1 + exp(-logodds))
```

**Multi-resolution pyramid:**  
Three versions of this map are maintained simultaneously at different resolutions:

| Level | Cell Size | Purpose |
|---|---|---|
| Level 0 (coarsest) | 0.20 m (20 cm) | Big-picture alignment, tolerates large errors |
| Level 1 (medium) | 0.10 m (10 cm) | Medium refinement |
| Level 2 (finest) | 0.05 m (5 cm) | Sub-centimetre precision |

All three levels cover the same physical area and are kept updated in sync. This is called a **map pyramid** and is the key to efficient coarse-to-fine alignment.

---

##### 5A.2 — Bootstrap Phase

Before the first real scan-to-map alignment can work, the map must contain some information — otherwise the gradient field is flat and the solver has nothing to pull on.

The **bootstrap** integrates the first scan directly at the predicted pose (no optimization), painting it into all three map levels. Only 1 bootstrap scan is needed (configurable). After that, the map has enough evidence for the Gauss-Newton optimizer to work.

---

##### 5A.3 — Coarse-to-Fine Gauss-Newton Optimization

This is the core of the scan matcher. It runs on the map pyramid, from the coarsest level to the finest level, iteratively adjusting the pose to minimize the mismatch between the new scan and the map.

**What "alignment" means in plain language:**

Imagine you are holding a transparent overlay sheet (the new scan) over the graph-paper map. The overlay has dots where the laser hit walls. You slide and rotate the overlay until the dots line up with the dark (occupied) areas on the map as well as possible. Gauss-Newton is the mathematical procedure for finding that best position and angle automatically.

**Gauss-Newton iteration (one step):**

1. Take the current pose estimate `(x, y, θ)`.
2. Transform all scan points from robot frame to world frame using `(x, y, θ)`.
3. For each transformed point, look up the **occupancy probability** in the current map level (using bilinear interpolation for sub-pixel accuracy — this means reading a smoothly interpolated value between the four nearest grid cells rather than snapping to the nearest cell).
4. Compute the **residual** for each point: `r = 1.0 - P(occupied)`. A residual of 0 means the point sits exactly on a known wall; a residual of 1 means the point is in unknown or free space.
5. Compute the **Jacobian** — how much would the residuals change if we moved the pose slightly in x, y, or θ? This uses the map's spatial gradient (computed once per iteration using finite differences across adjacent cells) and the derivative of the rotation matrix with respect to θ.
6. Solve the **normal equations**: `H·δ = -g` where `H = Jᵀ J + λI` and `g = Jᵀ r`. The small damping term `λI` (Levenberg–Marquardt regularization) prevents numerical explosions when H is nearly singular.
7. **Clip the step size** to at most ±0.03 m in x/y and ±1° in θ. This prevents the optimizer from jumping too far in one step and overshooting.
8. Apply the update: `pose += δ`.
9. Repeat until the step magnitude is below 1e-6 (converged) or the maximum iteration count is reached.

**Coarse-to-fine sequence:**

| Map Level | Iterations | Why |
|---|---|---|
| Coarsest (0.20 m) | 20 (lab), 2 (fr079) | Large steps, gets close fast |
| Medium (0.10 m) | 15 (lab), 2 (fr079) | Medium refinement |
| Finest (0.05 m) | 10 (lab), 1 (fr079) | Fine-grained precision |

The output of each level is fed as the starting pose for the next (finer) level.

**Acceptance test:**

After the finest-level GN finishes, the system checks whether the match is trustworthy:
- Compute `score = mean P(occupied)` for all scan points at the matched pose on the finest grid.
- If `score ≥ min_score` (0.10 for lab, 0.35 for fr079), the match **succeeds** and the returned pose is used.
- If the score is too low (points landed in free space — bad alignment), the match **fails** and the predicted pose is returned as a fallback.

The score threshold is intentionally low (0.10) for lab_run_2 because the sparse indoor environment naturally scores lower than a dense corridor.

---

#### Matcher B — Scan-to-Submap (Cartographer-Style)

**File:** [slam_core/matching/scan_to_submap_old.py](slam_core/matching/scan_to_submap_old.py)

**Core idea:** Instead of one giant global map, maintain a rolling set of **submaps** — small local maps that cover only a recent portion of the robot's path. Match each new scan against the currently active submap using a two-stage search.

---

##### 5B.1 — Submap Management

A submap is an independent occupancy grid (20 m × 20 m, 5 cm resolution) anchored at a fixed world position. The system always maintains **two active submaps**:

- **Submap N** — the current (newest) submap, still being filled.
- **Submap N-1** — the previous submap, recently finished but kept for matching.

When a submap accumulates enough scans (`SCANS_PER_SUBMAP`: 500 for lab, 90 for fr079/intel), it is **finished** (frozen) and a new submap is created. A finished submap is available for matching until it is eventually rotated out.

This design means the system never needs a single monolithic global map in memory. For a long 800-metre run like the Intel dataset, many submaps are created and finished in sequence.

---

##### 5B.2 — Correlative Two-Stage Search

Unlike scan-to-map which starts from the predicted pose and only refines, scan-to-submap first performs a **brute-force search** over a grid of candidate poses. This makes it more robust to larger prediction errors.

**Stage 1 — Coarse Search (on the 4× downsampled map level):**
- Systematically try every combination of x, y, θ within a search window around the predicted pose.
  - Lab: ±1.0 m in x/y, ±23° in θ; step = 0.10 m / 2.9°
  - FR079: ±0.30 m in x/y, ±11°; step = 0.05 m / 2.9°
- For each candidate pose, transform the scan points and compute the mean occupancy score on the coarse (4× downsampled) grid.
- Record the candidate with the highest score.

This is slow but thorough — it cannot miss a good match as long as it is within the search window.

**Stage 2 — Fine Search (on the finest map level):**
- Center a tighter search window on the coarse winner.
  - Lab: ±0.25 m / ±6.9°; step = 0.05 m / 1.1°
  - FR079: ±0.15 m / ±4.6°; step = 0.03 m / 1.1°
- Repeat the brute-force grid evaluation but at full resolution.
- Record the best fine candidate.

**Stage 3 — Gauss-Newton-LM Refinement:**

The best fine-search candidate is refined further with a GN-LM solver (Gauss-Newton with Levenberg–Marquardt damping). Unlike the scan-to-map GN above, this version includes a **regularization term** that anchors the refined pose close to the predicted pose:

- Residuals = scan-to-grid alignment residuals (same as before) + weighted penalty for drifting from the prior pose.
- Weights: `w_trans = 0.1–0.2` (gentle translation anchor), `w_rot = 1.0` (stronger rotation anchor).
- Up to 12 iterations; step clipped to 0.10 m / 5° per iteration.

**Acceptance:** Score ≥ `SUBMAP_MIN_SCORE` (0.50–0.60 depending on dataset) for success.

---

##### 5B.3 — Motion Filter (Scan-to-Submap Only)

Not every matched scan needs to be inserted into the submap. Inserting when the robot has barely moved wastes compute and adds noise. The **motion filter** only triggers a submap update when:
- The robot has moved more than `max_distance_meters` (typically 0.05 m), OR
- The robot has rotated more than `max_angle_radians` (typically 0.5°–10°), OR
- More than `max_time_seconds` (0.10 s) has elapsed since the last insertion.

For scan-to-map mode, the motion filter is replaced by a simpler `MAP_UPDATE_EVERY` — insert every Nth scan (default 1, meaning every scan).

---

### Step 6: Map Update (Painting the Scan into the Map)

**File:** [slam_core/matching/scan_to_map.py](slam_core/matching/scan_to_map.py) — `integrate_scan_simple()`

After finding the matched pose, the system updates the map with what the new scan observed. This uses **ray casting**:

For each laser beam endpoint:
1. Draw a line (ray) from the robot's position to the endpoint.
2. All grid cells along the ray (except the endpoint) are marked **more free**: `logodds += L_FREE` (−0.1).
3. The endpoint cell is marked **more occupied**: `logodds += L_OCC` (+0.85 to +1.0).
4. Both updates are clipped to the range `[L_MIN, L_MAX]` = `[−5, +5]` to prevent any single reading from dominating.

The ray is traced using **linspace** (uniform steps along the line from robot to endpoint). The number of steps is `RAY_STEPS` (20 for lab, 40 for fr079/intel — more steps for the longer-range SICK sensor).

This is done for all three pyramid levels simultaneously, keeping the coarse levels consistent with the fine level.

---

### Step 7: Trajectory Recording

**File:** [hector/run_local_slam_new.py](hector/run_local_slam_new.py) — lines 398–407

After every scan, the final pose is written to two output files in the `hector_outputs/` directory:

**Trajectory file** (5 columns, one row per scan):
```
timestamp   x(m)    y(m)    theta(rad)  score
1234567.891  0.123  -0.045   0.012       0.634
```

**Debug log file** (13 columns):
```
k  t   x   y   theta  score  inliers  dx  dy  dtheta  do_insert  did_insert
```
This contains internal diagnostics — how many inlier points were found, how much the pose moved per scan, whether the scan was inserted into the map.

The filename encodes all relevant settings, e.g.:
```
hector_outputs/trajectory_lab_run_2_raw_scan_to_map_3420.txt
```

---

### Step 8: Post-Run Pose Graph Optimization (PGO)

**File:** [hector/eval/pgo_any.py](hector/eval/pgo_any.py)

This step is **run separately** after the main SLAM has finished. It corrects the accumulated drift in the trajectory by finding **loop closures** — places where the robot revisited a previously seen location — and globally adjusting all poses to make the map consistent.

---

#### 8.1 — What Is Drift and Why Does It Matter?

Each individual scan match is very accurate — typically within a few millimetres. But small errors accumulate over hundreds of metres. After travelling 100 metres and returning to the start, the trajectory might show the endpoint displaced by 0.5–2 metres from the actual starting point. This accumulated drift makes the map inconsistent (walls appear doubled or blurred).

---

#### 8.2 — Loop Closure Detection

The system uses three complementary strategies to find revisited places:

**Strategy 1 — Spatial Search:**
- Every 10th pose is a **keyframe**.
- For each keyframe `j`, find all earlier keyframes `i` within `search_radius` (1.2–2.0 m).
- Skip `i` if `|j - i| < min_index_gap` (80–120) to avoid matching adjacent poses.
- The remaining candidates are geometrically verified.

**Strategy 2 — Geometric Verification (Map-Based):**
- Build a small local mini-map around candidate keyframe `i` from ±10 surrounding scans.
- Run the GN scan matcher to align keyframe `j`'s scan against that mini-map.
- If the match score is high enough, a **loop closure constraint** is created: `(i, j, relative_pose_z_ij)`.

**Strategy 3 — Scan-Context Descriptor:**
- For each keyframe, compute a **Scan-Context** descriptor: a polar histogram binned by radial rings and angular sectors, capturing the distribution of points around the robot.
- Find candidates with high descriptor similarity (cosine similarity), then verify geometrically.
- This works for rotational revisits where the robot approaches from a different direction.

**Consistency gate:** A candidate loop closure `(i, j, z_ij)` is accepted only if the measured relative pose `z_ij` agrees with the predicted relative pose from the current trajectory within 0.75 m and 20°. This prevents false positives.

---

#### 8.3 — Pose Graph Structure

The **pose graph** is a network where:
- **Nodes** are robot poses at keyframe locations.
- **Edges** are constraints between poses:
  - **Odometry / sequential edges** — every consecutive pair of keyframes.
  - **Loop closure edges** — the newly found loop closures.

Each edge carries a measurement `z_ij` (the measured relative pose between nodes i and j) and an **information matrix** (the inverse of covariance — how confident the measurement is):

| Edge type | Translation weight | Rotation weight |
|---|---|---|
| Sequential (intra-submap) | 500 | 1600 |
| Loop closure (inter-submap) | 11 000 | 100 000 |

Loop closure edges have much higher weight because they are verified matches, while sequential edges accumulate odometric uncertainty.

---

#### 8.4 — Optimization Solver

**Algorithm: Gauss-Newton on the condensed Hessian (sparse)**

The pose graph optimization minimizes the total weighted squared error across all edge measurements:

1. **Fix the first node** (the robot starts at origin — this removes the gauge freedom that would otherwise make the system underdetermined).
2. Linearize all edge residuals around the current pose estimates: `r_ij = z_ij ⊖ (x_i⁻¹ ⊕ x_j)` (using SE(2) composition operators).
3. Assemble the **sparse Hessian** `H` (a 3N × 3N matrix, mostly zeros) using SciPy's CSR sparse matrix format.
4. Solve the reduced linear system `H·Δx = −g` using SciPy's sparse direct solver (`spsolve`).
5. **Huber robust weighting:** Each residual is down-weighted if it is large (likely an outlier): `w = min(1, δ_Huber / |residual|)`. This prevents bad loop closures from corrupting the solution.
6. Apply the update `x += Δx`, rewrap all angles.
7. Repeat up to 15 iterations or until `‖Δx‖ < 1e-7` (converged).

**Key technical choice — no external library:**  
The entire PGO is implemented in pure Python using only NumPy and SciPy. There is no dependency on g2o, Ceres, or any C++ SLAM library.

---

#### 8.5 — What Changes After PGO?

The optimizer adjusts all keyframe poses globally. The corrected trajectory is then written to a new file. The map is rebuilt from scratch by replaying all scans at their corrected poses (see `hector/eval/rebuild_map_any.py`).

---

## Part 4 — Outputs: Trajectory and Map

### Trajectory

The trajectory file `hector_outputs/trajectory_*.txt` contains the robot's estimated position at every scan:

```
1618560000.100  0.000  0.000  0.000  -1.000   ← bootstrap scan (score=-1)
1618560000.200  0.024  0.001  0.003   0.721   ← first real match
1618560000.300  0.051  0.002  0.005   0.758
...
```

Each row is one moment in time: timestamp, x position, y position, heading angle, match quality score.

### Map

The occupancy grid is held in memory as a NumPy array throughout the run. After the run:

- **Probability image:** Each cell's log-odds value is converted to probability (0–1), then to a greyscale pixel. White = free space, black = occupied wall, grey = unknown.
- **Trajectory overlay:** The (x, y) positions are plotted on top of the map image.

**Plotting scripts:**

| Script | What it produces |
|---|---|
| [hector/plot_single_trajectory.py](hector/plot_single_trajectory.py) | XY trajectory, score vs. scan index, heading vs. scan index |
| [hector/plot_matcher_trajectories.py](hector/plot_matcher_trajectories.py) | Side-by-side comparison of scan_to_map vs. scan_to_submap |
| [hector/viz/live_view.py](hector/viz/live_view.py) | Rendered map frames saved during the run |
| [hector/viz/plot_final.py](hector/viz/plot_final.py) | Final map rendered after run |
| [hector/eval/pgo_any.py](hector/eval/pgo_any.py) — PGO section | Before/after map comparison with trajectory correction arrows |

---

## Part 5 — All Algorithms and Solvers, Summarized

| Component | Algorithm | Implementation |
|---|---|---|
| Scan matching (local) | Hector-style coarse-to-fine Gauss-Newton on occupancy grid pyramid | Pure NumPy in `scan_to_map.py` |
| Search-then-match | Two-stage correlative search + GN-LM refinement | Pure NumPy in `scan_to_submap_old.py` |
| Map representation | Log-odds occupancy grid | NumPy float32 array |
| Map update | Ray casting via linspace | `integrate_scan_simple()` in `scan_to_map.py` |
| Ray casting (PGO maps) | Bresenham's line algorithm | `_bresenham_cells()` in `pgo_any.py` |
| Motion model | Constant-velocity extrapolator | `PoseExtrapolatorCV` in carto module |
| Pose graph optimization | Gauss-Newton with sparse Hessian, Huber loss | SciPy spsolve in `pgo_any.py` |
| Loop closure detection | Spatial search + geometric GN verification + Scan-Context descriptor | `pgo_any.py` |
| Point cloud preprocessing | Fixed voxel centroid filter + adaptive binary-search voxel filter | `preprocessing.py` |
| Linear algebra | Direct dense solve (GN per scan), sparse direct solve (PGO) | `np.linalg.solve` and `scipy.sparse.linalg.spsolve` |

**No PGO during the main run:** Loop closure and global optimization are post-processing steps. The main SLAM loop is purely local (each new scan is matched against the current map without any global graph update).

**No external SLAM libraries:** No g2o, no Ceres, no ROS, no GTSAM. Everything is NumPy + SciPy.

---

## Part 6 — End-to-End Data Flow with File References

```
datasets/lab_run_2/scans.carmen
         │
         │ read_lab_carmen_log()
         ▼
slam_core/dataio/dataset_catalog.py     ← DatasetProfile (geometry, file paths)
         │
         │ load_dataset_scans()
         ▼
hector/run_local_slam_new.py            ← MAIN ENTRY POINT (main loop)
         │
         │ ranges_to_points()
         ▼
carto/local_slam/range_to_points.py     ← laser angles → (x,y) point cloud
         │
         │ PointCloudProcessor.process()
         ▼
slam_core/matching/preprocessing.py    ← voxel filtering
         │
         │ HectorLocalSlamAdapter._predict_world_pose()
         ▼
hector/adapter.py                       ← pose prediction (extrap + odom blend)
carto/local_slam/pose_extrapolator.py  ← PoseExtrapolatorCV
         │
         │ ScanToMapMatcher.match()        OR    OldScanToSubmapMatcher.match()
         ▼                                       ▼
slam_core/matching/scan_to_map.py       slam_core/matching/scan_to_submap_old.py
  ├── MapPyramid (3 GridMaps)             ├── SubmapBuilder2D
  ├── align_pose_gauss_newton()           ├── correlative_match_two_stage()
  └── integrate_scan_simple()            ├── GaussNewtonLM.solve()
                                         └── CartoRefinementProblem
         │
         │ write trajectory row
         ▼
hector_outputs/trajectory_*.txt         ← (timestamp, x, y, θ, score)
hector_outputs/trajectory_*_debug.txt   ← (full diagnostics per scan)

         [OPTIONAL POST-PROCESSING]
         │
         │ python -m hector.eval.pgo_any
         ▼
hector/eval/pgo_any.py
  ├── load_trajectory_full()
  ├── load_aligned_scan_points()
  ├── loop closure detection (spatial + GN + scan-context)
  ├── PGO with scipy.sparse spsolve + Huber weighting
  └── save corrected trajectory + before/after map plots
```

---

## Part 7 — Frequently Asked Questions

**Q: Does this implementation use loop closure during the main SLAM run?**  
A: No. The main run is purely local — each scan is matched against the current map, the map is updated, and the system moves on. Loop closure is a separate post-processing step run via `pgo_any.py` after the trajectory file is saved.

**Q: Is there IMU fusion?**  
A: The lab_run_2 dataset includes an IMU CSV file, but the Hector pipeline does not fuse IMU measurements. The IMU data is present for potential future use.

**Q: What happens if the scan matcher fails?**  
A: The system falls back to the predicted pose (from the extrapolator/odometry). On the next scan, it tries again from that predicted position. A message like `!! FALLBACK k=1234  score=0.021<0.100` is printed for every non-bootstrap failure.

**Q: Can the two matchers be switched at runtime?**  
A: The `MatcherManager` in `slam_core/matching/core.py` supports swapping matchers mid-run using a rolling buffer of recent scans. The new matcher is seeded from those buffered scans. However, in normal operation, one matcher type is selected at startup and used for the entire run.

**Q: What coordinate frame is used?**  
A: The robot starts at world position `(0, 0, 0)` — x pointing forward, y pointing left, angles measured counter-clockwise. All trajectory outputs and map origins are in this world frame.

**Q: How accurate is it?**  
A: For well-structured indoor environments (fr079, intel), the scan-to-map matcher typically achieves sub-5 cm position accuracy between consecutive scans. Accumulated drift over a full run (before PGO) is typically 0.3–2.0 m depending on path length and environment complexity.
