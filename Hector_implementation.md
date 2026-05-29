# Hector SLAM — Complete Implementation & Mathematical Reference

**Target audience.** Researchers and engineers who want the exact mathematics of
*this* implementation, not a textbook survey. Every equation below corresponds to
code that actually runs in the repository; nothing is added "for completeness."
A short plain-language paragraph precedes each mathematical block so that readers
without a SLAM background can still follow the intent.

> **Scope discipline.** This document describes only what the runner
> `hector/run_local_slam_new.py` and the modules it imports actually execute. Dead
> or optional code paths (e.g. the constant-velocity extrapolator and odometry
> blending, the legacy `pyceres` refine backend, the standalone scipy PGO tool) are
> flagged explicitly as *present but inactive in the shipped profiles* rather than
> presented as part of the live pipeline.

---

## 0. What SLAM is (one paragraph)

SLAM — *Simultaneous Localization and Mapping* — is the problem of estimating, at
the same time, **where a sensor is** (localization) and **what the environment
looks like** (a map), using only the sensor's own measurements. It is circular:
you need a map to localize, and a localized pose to extend the map. This codebase
solves the **2-D** version: the platform moves on a plane and the map is a
bird's-eye occupancy grid. The estimated quantity at each step is a planar pose
$\xi=(x,y,\theta)$, and the map is a grid of occupancy probabilities.

---

## 1. Notation and SE(2) algebra

All poses live in the special Euclidean group of the plane, $SE(2)$. A pose is the
triple

$$
\xi = (x,\,y,\,\theta)\in\mathbb{R}^2\times\mathbb{S}^1 ,
$$

interpreted as the rigid transform that maps a point $\mathbf{p}=(p_x,p_y)$ in the
body frame to the world frame:

$$
T(\xi)\,\mathbf{p} \;=\; R(\theta)\,\mathbf{p} + \mathbf{t},
\qquad
R(\theta)=\begin{bmatrix}\cos\theta & -\sin\theta\\[2pt]\sin\theta & \cos\theta\end{bmatrix},
\qquad
\mathbf{t}=\begin{bmatrix}x\\y\end{bmatrix}.
$$

In code this is `transform_points_pose` / `_transform_points` (a right-multiply by
$R^\top$ on row-vector point arrays, which is algebraically identical).

**Composition** ($\oplus$), `pose_compose(a,b)` — apply $b$ then $a$:

$$
\xi_a \oplus \xi_b =
\begin{bmatrix}
x_a + \cos\theta_a\, x_b - \sin\theta_a\, y_b\\[2pt]
y_a + \sin\theta_a\, x_b + \cos\theta_a\, y_b\\[2pt]
\operatorname{wrap}(\theta_a+\theta_b)
\end{bmatrix}.
$$

**Inverse** ($\ominus$ as a unary op), `inverse_pose(p)`:

$$
\xi^{-1}=
\begin{bmatrix}
-(\cos\theta\, x + \sin\theta\, y)\\[2pt]
-(-\sin\theta\, x + \cos\theta\, y)\\[2pt]
\operatorname{wrap}(-\theta)
\end{bmatrix}.
$$

**Relative pose** (used everywhere for constraints), `pose_relative(a,b)`:

$$
{}^{a}\xi_{b} \;=\; \xi_a^{-1}\oplus\xi_b
\qquad(\text{"pose of $b$ expressed in the frame of $a$"}).
$$

**Angle wrap**, `wrap_angle`, keeps angles in $(-\pi,\pi]$:

$$
\operatorname{wrap}(\alpha)=\big((\alpha+\pi)\bmod 2\pi\big)-\pi .
$$

These five operations are the algebraic spine of the whole system. Defined in
[slam_core/common/se2.py](slam_core/common/se2.py).

---

## 2. Pipeline at a glance

**The core pipeline is: sensor → preprocessing → prediction → scan matching → map
update → trajectory.** Scan matching is the heart of the system, and it comes in
two **independent, interchangeable** flavours — `scan_to_map` (§8) and
`scan_to_submap` (§9) — selected at startup. Each is a complete standalone matching
pipeline; you run *one* of them. The pose graph (§10) is an **optional add-on stage
layered on top of `scan_to_submap`** to reduce accumulated drift; it is off unless
`--enable-pgo` is passed and changes nothing about how scans are matched or how the
map is built. Read §8 and §9 as the two main paths; read §10 only as a refinement
that can be bolted onto §9.

```
 raw laser ranges  r_i                                  (Part 3 — sensor)
        │  ranges_to_points()                           (Stage A — §5)
        ▼
 body-frame point cloud  P = { p_i }                    (carto/local_slam/range_to_points.py)
        │  PointCloudProcessor.process()                (Stage B — §6)
        ▼
 thinned point cloud  P'  (lab only; ~200 pts)          (slam_core/matching/preprocessing.py)
        │  HectorLocalSlamAdapter._predict_world_pose() (Stage C — §7)
        ▼
 predicted pose  ξ̂_k                                    (hector/adapter.py)
        │
        ▼
 ┌─ SCAN MATCHING — choose ONE path ─────────────────────────────────────┐
 │  PATH 1: scan_to_map  (§8)    OR   PATH 2: scan_to_submap  (§9)        │
 │  Hector multi-res GN          two-stage correlative + GN-LM refine    │
 │  slam_core/matching/scan_to_map.py    slam_core/matching/scan_to_submap/│
 └───────────────────────────────────────────────────────────────────────┘
        │  matched pose  ξ_k,  score  s_k
        ▼
 map / submap update (ray casting, log-odds)            (map update — §8.4, §9.1)
        │
        ├──────────────► trajectory_*.txt  (t, x, y, θ, s)   (output — §11)
        │
        ▼  ╌╌ OPTIONAL improvement, only with --enable-pgo (path 2 only) ╌╌
 ONLINE POSE GRAPH (g2o SE2)                             (§10, optional)
   nodes + intra-submap + loop-closure + spine constraints
   carto/pose_graph/*  +  carto/pose_graph/backends/g2o_backend_2d.py
        │  final Levenberg solve + write-back
        ▼
 trajectory_*_pgo.txt  (dense, drift-corrected)
```

The two matchers are interchangeable; the back-end pose graph is optional and only
meaningful for `scan_to_submap`.

---

## 3. Sensors and datasets

All three datasets deliver **planar laser range scans**: a spinning beam reports
the distance $r_i$ to the nearest surface at a known angle $\theta_i$.

| Property | lab_run_2 | fr079 | intel |
|---|---|---|---|
| Sensor | YD LiDAR G4 | SICK LMS | SICK LMS |
| FOV | 360° | 180° | 180° |
| Beams / scan | 909 (`raw`) or 360 | 360 | 180 (1°) |
| Range | 0.10–16 m | 0.10–30 m | 0.10–30 m |
| Wheel odometry | none | yes | yes (drifts) |
| Path length | ~34 m loop | ~60 m | ~800 m |
| Log format | `scans.carmen` | `fr079.clf` | `intel.clf` |

The sensor geometry (beam count, `angle_min`, `angle_inc`, range limits, odometry
flag) is stored as a frozen `DatasetProfile` and loaded by
[slam_core/dataio/dataset_catalog.py](slam_core/dataio/dataset_catalog.py). An IMU
CSV exists for lab_run_2 but **is not fused** in the live Hector path (the IMU
hooks in the extrapolator are disabled by default — see §7).

---

## 4. Configuration system

[hector/config.py](hector/config.py) holds one parameter **profile** per dataset.
Selecting a dataset injects every key as a module attribute (`cfg.XXX`), so the
runner code is dataset-agnostic. The three user knobs at the top are
`DATASET_NAME`, `DATASET_SCAN_VARIANT` (lab only), and `MATCHER_TYPE`; CLI flags
(`--dataset`, `--matcher`, `--enable-pgo`, `--scans-per-submap`, `--max-scans`)
override them.

Key constants referenced by the math below, per dataset:

| Symbol (this doc) | config key | lab_run_2 | fr079 | intel |
|---|---|---|---|---|
| occupancy hit log-odds $\ell_{\text{occ}}$ | `L_OCC` | 1.0 | 0.85 | 0.85 |
| miss log-odds $\ell_{\text{free}}$ | `L_FREE` | −0.1 | −0.1 | −0.1 |
| clamp $[\ell_{\min},\ell_{\max}]$ | `L_MIN/L_MAX` | ±5 | ±5 | ±5 |
| base resolution $\rho$ (m/cell) | `MAP_RESOLUTION` | 0.05 | 0.05 | 0.05 |
| pyramid levels $L$ | `PYRAMID_LEVELS` | 3 | 3 | 3 |
| GN iters / level | `GN_ITERS_PER_LEVEL` | [20,15,10] | [2,2,1] | [15,10,8] |
| GN damping $\lambda$ | `GN_DAMPING` | 1e−4 | 2e−2 | 1e−4 |
| accept score $s_{\min}$ (map) | `CORR_MAP_MIN_SCORE` | 0.10 | 0.35 | 0.10 |
| ray steps $N_{\text{ray}}$ | `RAY_STEPS` | 20 | 40 | 40 |
| scans per submap $N_s$ | `SCANS_PER_SUBMAP` | 500 | 90 | 90 |
| submap accept score | `SUBMAP_MIN_SCORE` | 0.50 | 0.52 | 0.60 |
| odom trust $\alpha$ | `ODOM_ALPHA` | 0.0 | 0.5 | 0.35 |
| use CV extrapolator | `USE_EXTRAPOLATOR` | **False** | **False** | **False** |

The last two rows matter for §7: **all three profiles disable the extrapolator**,
so $\alpha$ (which only acts inside the extrapolator branch) is inert in the
shipped runs.

---

## 5. Stage A — laser ranges to a Cartesian point cloud

**Plain idea.** A scan is a list of distances. Convert each distance + its known
beam angle into an $(x,y)$ point in the sensor's own frame.

**Math.** For beam index $i$ the firing angle and Cartesian endpoint are

$$
\theta_i=\theta_{\min}+i\,\Delta\theta,
\qquad
\mathbf{p}_i=\begin{bmatrix}r_i\cos\theta_i\\ r_i\sin\theta_i\end{bmatrix},
$$

where $\theta_{\min}$ is `angle_min` and $\Delta\theta$ is `angle_inc` from the
dataset profile. A beam is kept only if it is finite and within range,

$$
r_{\min}\le r_i\le r_{\max},\qquad r_{\min}=\max(\texttt{LIDAR\_MIN\_RANGE},\,\texttt{range\_min}).
$$

Optional decimation keeps every `BEAM_STRIDE`-th beam (lab: 1, fr079: 2, intel: 1);
indices and ranges are strided together so $\theta_i$ stays correct. Output: an
$(N,2)$ array in the **body frame**. Code: `ranges_to_points` in
[carto/local_slam/range_to_points.py](carto/local_slam/range_to_points.py).

---

## 6. Stage B — voxel preprocessing (thinning)

**Plain idea.** With 909 beams, a nearby wall produces dozens of almost-identical
points; they add cost without adding information. Collapse clusters to one
representative point. Enabled only for lab_run_2 (`VOXEL_FILTER_ENABLED=True`);
fr079/intel pass through unchanged.

**Stage B.1 — fixed voxel centroid filter.** Partition the plane into square cells
of side $v$ (`VOXEL_FIXED_SIZE` = 0.03 m). Each point is assigned the integer cell
index $\lfloor \mathbf{p}/v\rfloor$; all points sharing a cell are replaced by their
centroid

$$
\bar{\mathbf{p}}_c=\frac{1}{|C|}\sum_{\mathbf{p}\in C}\mathbf{p}.
$$

**Stage B.2 — adaptive voxel filter.** Binary-search the largest voxel size in
$[0,\,v_{\max}]$ (`VOXEL_ADAPTIVE_MAX_SIZE` = 0.10 m) whose fixed-filter output
still retains at least $n_{\min}$ points (`VOXEL_ADAPTIVE_MIN_POINTS` = 200):

$$
v^\star=\max\{\,v: |\,\text{fixed}_v(P)\,|\ge n_{\min}\,\},
$$

found by `VOXEL_ADAPTIVE_ITERS` = 6 bisection steps. This guarantees a roughly
constant point budget regardless of scene density. Result for lab: 909 → ~200
well-spread points. Code: `PointCloudProcessor` in
[slam_core/matching/preprocessing.py](slam_core/matching/preprocessing.py).

---

## 7. Stage C — pose prediction (the matcher's initial guess)

**Plain idea.** Before matching, supply a starting pose. A good guess makes the
matcher converge fast and avoids local minima.

**What actually runs (shipped profiles).** Because `USE_EXTRAPOLATOR=False` for all
three datasets, `_predict_world_pose` in [hector/adapter.py](hector/adapter.py)
takes the simple branch:

$$
\hat{\xi}_k=
\begin{cases}
\xi_k^{\text{odom}}, & \text{wheel odometry present (fr079, intel)},\\[4pt]
\xi_{k-1}^{\text{match}}, & \text{otherwise (lab\_run\_2): last accepted pose},\\[4pt]
(0,0,0), & \text{before the first match.}
\end{cases}
$$

So for lab_run_2 the prior is a **constant-position** model (the previous matched
pose); for fr079/intel it is the **raw odometry** pose. Note that with the
extrapolator off, the odometry-blend weight $\alpha=$ `ODOM_ALPHA` is never applied,
and the matched pose does not feed back into the extrapolator.

**Present but inactive — the constant-velocity extrapolator.** When
`USE_EXTRAPOLATOR=True`, `PoseExtrapolatorCV`
([carto/local_slam/pose_extrapolator.py](carto/local_slam/pose_extrapolator.py))
estimates a body velocity from the two ends of a short pose queue,

$$
v_x=\frac{x_1-x_0}{t_1-t_0},\quad
v_y=\frac{y_1-y_0}{t_1-t_0},\quad
\omega=\frac{\operatorname{wrap}(\theta_1-\theta_0)}{t_1-t_0},
$$

and predicts $\hat\xi_k = (x+v_x\,\Delta t,\; y+v_y\,\Delta t,\;
\operatorname{wrap}(\theta+\omega\,\Delta t))$ with $\Delta t$ clamped to
`EXTRAP_MAX_DT`. Odometry blending would then mix the two predictions
componentwise with weight $\alpha$, including a wrapped angular blend
$\theta\leftarrow\operatorname{wrap}\!\big(\hat\theta+\alpha\,
\operatorname{wrap}(\theta^{\text{odom}}-\hat\theta)\big)$. IMU aiding (gyro
yaw-rate substitution + a small absolute-yaw correction $\alpha_{\text{imu}}=0.02$)
is also implemented here but disabled. **None of this executes in the documented
runs** — it is retained for fast-motion / feature-poor scenarios.

---

## 8. Scan matching, Path 1 — `scan_to_map` (classic Hector matcher)

*One of the two interchangeable scan-matching pipelines (the other is §9). Pick one
at startup; this section is self-contained.*

**Core idea.** Maintain one global occupancy grid and align each new scan to it by
iterative nonlinear least squares (Gauss–Newton), coarse-to-fine over a
resolution pyramid. File:
[slam_core/matching/scan_to_map.py](slam_core/matching/scan_to_map.py).

### 8.1 Occupancy grid and the log-odds field

Each cell stores a **log-odds** value $\ell$. The occupancy probability is the
logistic (sigmoid) map

$$
M = \sigma(\ell)=\frac{1}{1+e^{-\ell}}\in(0,1),
$$

so $\ell=0\Rightarrow M=0.5$ (unknown), $\ell>0$ occupied, $\ell<0$ free. Log-odds
is used because Bayesian evidence accumulation becomes simple addition (§8.4).
World→grid mapping (cell units, grid centred on the world origin):

$$
g_x=\frac{x}{\rho}+o_x,\qquad g_y=\frac{y}{\rho}+o_y,\qquad o=\tfrac{1}{2}\,\text{size}.
$$

**Bilinear interpolation.** $M$ is read at continuous grid coordinates so that the
field is differentiable. With $x_0=\lfloor g_x\rfloor$, $y_0=\lfloor g_y\rfloor$,
$d_x=g_x-x_0$, $d_y=g_y-y_0$:

$$
M(g)=(1-d_y)\big[(1-d_x)M_{00}+d_x M_{10}\big]+d_y\big[(1-d_x)M_{01}+d_x M_{11}\big].
$$

**Spatial gradient.** The gradient field used by the optimizer is computed once per
iteration by central differences on the probability grid, then converted to
per-metre units by dividing by $\rho$:

$$
\frac{\partial M}{\partial x}\Big|_{\text{cell}}=\tfrac12\big(M_{i+1,j}-M_{i-1,j}\big),
\qquad
\nabla_{\!w} M=\frac{1}{\rho}\,\big(\partial_x M,\;\partial_y M\big).
$$

Probability and gradient arrays are cached and recomputed only when the grid is
marked dirty (a `_dirty` flag set on every map update), so unchanged scans reuse
them.

### 8.2 Resolution pyramid

$L=3$ grids share the same physical extent but differ in cell size:

$$
\rho_\ell=\rho\cdot 2^{\,L-1-\ell},\qquad
\rho_0=0.20\,\text{m (coarse)},\;\;\rho_1=0.10,\;\;\rho_2=0.05\,\text{m (fine)}.
$$

Coarse levels have wide basins of convergence (tolerate large initial error);
fine levels give precision. The matcher optimizes coarse→fine, feeding each
level's result as the next level's initial pose.

### 8.3 Gauss–Newton scan-to-map alignment

**Plain idea.** Slide and rotate the scan so its endpoints land on cells the map
says are occupied. "Occupied" means $M\to1$, so we drive each endpoint's
**residual** $r_i = 1-M(\cdot)$ toward zero.

For a pose $\xi$ the $i$-th body point maps to the world point
$\mathbf{q}_i(\xi)=R(\theta)\mathbf{p}_i+\mathbf{t}$, and the per-point residual is

$$
r_i(\xi)=1-M\big(\mathbf{q}_i(\xi)\big).
$$

The objective minimized over the in-bounds points is

$$
E(\xi)=\sum_i r_i(\xi)^2 = \sum_i\big[\,1-M(\mathbf{q}_i(\xi))\,\big]^2 .
$$

**Jacobian.** Each residual depends on $\xi=(x,y,\theta)$ through the map value at a
moving point. By the chain rule, with $\partial r/\partial M=-1$:

$$
\frac{\partial r_i}{\partial \xi}
= -\,\nabla_{\!w} M(\mathbf{q}_i)^\top\,\frac{\partial \mathbf{q}_i}{\partial \xi},
\qquad
\frac{\partial \mathbf{q}_i}{\partial(x,y)}=I_2,
\qquad
\frac{\partial \mathbf{q}_i}{\partial\theta}=\frac{dR}{d\theta}\mathbf{p}_i,
$$

with

$$
\frac{dR}{d\theta}=\begin{bmatrix}-\sin\theta & -\cos\theta\\ \cos\theta & -\sin\theta\end{bmatrix}.
$$

So the $1\times3$ Jacobian row is

$$
J_i=\Big[\,-\partial_x M,\;\; -\partial_y M,\;\;
-\big(\nabla_{\!w}M\big)^\top \tfrac{dR}{d\theta}\mathbf{p}_i\,\Big].
$$

This is exactly the Hector SLAM Jacobian (Kohlbrecher et al., 2011) and matches
`align_pose_gauss_newton` lines 277–286.

**Normal equations (Gauss–Newton with Levenberg–Marquardt damping).** Stacking
$J$ (size $m\times3$) and the residual vector $r$ (size $m$):

$$
H=J^\top J+\lambda I_3,\qquad g=J^\top r,\qquad
\boxed{\;\delta=-H^{-1}g\;}
$$

$\lambda=$ `GN_DAMPING` keeps $H$ invertible when the geometry is degenerate
(e.g. a single straight wall constrains only one translation direction). The step
is clamped per component to avoid overshoot,

$$
\delta_x,\delta_y\in[-\,c_{xy},\,c_{xy}]\ (c_{xy}=\texttt{CORR\_MAP\_STEP\_CLIP\_XY}=0.03\,\text{m}),
\qquad
\delta_\theta\in[-1^\circ,1^\circ],
$$

then applied, $\xi\leftarrow(x+\delta_x,\,y+\delta_y,\,\operatorname{wrap}(\theta+\delta_\theta))$.
Iteration stops at `GN_ITERS_PER_LEVEL[ℓ]`, when fewer than `min_points` endpoints
are in bounds, or when $\lVert\delta\rVert<10^{-6}$.

**Coarse-to-fine loop.** Levels are processed from largest $\rho$ to smallest; the
pose carries over between levels.

### 8.3.1 Bootstrap and acceptance

- **Bootstrap.** With an empty map the gradient field is flat and GN cannot move.
  The first `N_BOOTSTRAP_SCANS` (=1) scans are integrated *at the predicted pose
  without optimization* (`match()` returns `success=False`, score $=-1$, so the
  prediction loop is not polluted). After that the map has enough evidence.
- **Score.** After the finest level, the match quality is the mean occupancy at the
  matched endpoints,
  $$
  s=\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} M(\mathbf{q}_i),\qquad
  \mathcal{I}=\{\text{in-bounds finest-grid endpoints}\}.
  $$
- **Acceptance.** Hector itself never rejects on inliers or jump size; only a score
  guard remains: the match is accepted iff $s\ge s_{\min}$ (`CORR_MAP_MIN_SCORE`,
  0.10 lab / 0.35 fr079). On failure the **predicted** pose is returned as a
  fallback and a `!! FALLBACK` line is printed.

### 8.4 Map update — painting the scan into the map (ray casting)

**Plain idea.** Everything between the sensor and a hit is free space; the hit cell
is occupied. Add evidence accordingly.

For each accepted endpoint, the ray from the sensor origin $g_0$ to the endpoint
$g_1$ (both in grid coords) is sampled at $N_{\text{ray}}$ uniform points
(`np.linspace`). All cells **except** the endpoint receive the free update; the
endpoint receives the occupied update, with clamping:

$$
\ell \leftarrow \operatorname{clip}\big(\ell+\ell_{\text{free}},\,\ell_{\min},\ell_{\max}\big)\ \text{(along ray)},
\qquad
\ell \leftarrow \operatorname{clip}\big(\ell+\ell_{\text{occ}},\,\ell_{\min},\ell_{\max}\big)\ \text{(endpoint)}.
$$

This is the additive log-odds form of Bayesian occupancy updating; clamping to
$\pm5$ bounds any single reading's influence so the map stays adaptable. The update
is applied to **all three pyramid levels**. Code: `integrate_scan_simple`. The
adapter throttles insertions via `MAP_UPDATE_EVERY` (1 = every scan); on a matcher
*failure* it still inserts at the predicted pose (dead-reckoning) so the map keeps
growing along the path instead of stalling at the bootstrap region.

---

## 9. Scan matching, Path 2 — `scan_to_submap` (Cartographer-style front-end)

*The second of the two interchangeable scan-matching pipelines (the other is §8).
Self-contained as a local matcher; the optional pose-graph improvement in §10 can be
layered on top of it.*

**Core idea.** Instead of one global grid, keep a small rolling set of **submaps**.
Match each scan to the active submap by a discrete **correlative search** (robust
to larger initial error) followed by a continuous **GN-LM refine**. This is the
single unified front-end shared with the Cartographer pipeline; there is no second
copy (`scan_to_submap_old.py` was removed). Package:
[slam_core/matching/scan_to_submap/](slam_core/matching/scan_to_submap/).

### 9.1 Submap representation and lifecycle

A `ProbabilityGrid` is a $20\,\text{m}\times20\,\text{m}$, 0.05 m log-odds grid
anchored at a world pose $\xi_s$ (its origin), with the same $\sigma(\ell)$
probability and the same clamp as §8.1. Insertion uses **Bresenham** rays (integer
line rasterization) rather than `linspace`, but the log-odds update is identical:
free along the ray, occupied at the endpoint, clamped.

Scans are first transformed into the submap frame before insertion:

$$
\mathbf{q}^{\,s}=\xi_s^{-1}\oplus\big(\xi_k\oplus\mathbf{p}\big),
$$

i.e. world endpoints rebased into the submap's local frame.

**Lifecycle (`SubmapBuilder2D`, Cartographer-faithful).** Exactly two active
submaps are kept: `active[0]` is the mature matching target, `active[1]` is filling.

- When `active[0]` reaches $N_s$ scans (`SCANS_PER_SUBMAP`) it is **marked
  finished** but *not* removed — it remains the match target.
- The pair **rotates** only once `active[1]` has $\ge N_s/2$ scans: `active[0]` is
  dropped, the finished submap moves to the finished list, and a fresh empty submap
  is appended. This guarantees the target always has $\ge N_s/2$ scans of evidence,
  avoiding score $=-1$ fallbacks at handoffs.

Matching targets the **oldest** active submap (richest evidence).

### 9.2 Discrete correlative search (two-stage brute force)

**Plain idea.** Try a grid of candidate poses around the prediction; keep the one
whose scan endpoints land on the most-occupied cells. Two stages: a fast coarse
sweep on a downsampled grid, then a fine sweep around the winner.

A **precomputation stack** is built by repeated $2\times2$ **max-pooling** of the
submap probability image; level $\ell$ has been downsampled by $2^\ell$. The score
of a candidate submap-frame pose $\xi$ is the mean occupancy at its
nearest-cell-rounded endpoints, evaluated at level $\ell$:

$$
\text{score}(\xi,\ell)=
\frac{1}{|\mathcal{V}|}\sum_{i\in\mathcal{V}}
\text{Grid}_\ell\Big[\,\operatorname{round}\!\big(g(\mathbf q_i(\xi))/2^\ell\big)\Big],
$$

requiring at least `min_valid` in-bounds points (else $-\infty$). Using a max-pooled
level gives an **optimistic upper bound**, so a coarse winner cannot hide a better
fine solution nearby.

**Stage 1 (coarse, level 2 = 4× down).** Exhaustively sweep offsets

$$
(\delta x,\delta y)\in[-W_{xy},W_{xy}]^2\ \text{step}\ \Delta_{xy},
\qquad
\delta\theta\in[-W_\theta,W_\theta]\ \text{step}\ \Delta_\theta,
$$

centred on the predicted submap-frame pose. (lab: $W_{xy}=1.0$ m, $W_\theta\approx
23^\circ$, $\Delta_{xy}=0.10$ m, $\Delta_\theta\approx2.9^\circ$.)

**Stage 2 (fine, level 0 = full res).** Repeat in a tight window around the coarse
winner (lab: $\pm0.25$ m / $\pm6.9^\circ$, step 0.05 m / $1.1^\circ$).

Code: `correlative_match_two_stage` in
[slam_core/matching/scan_to_submap/correlative.py](slam_core/matching/scan_to_submap/correlative.py).
The point set is downsampled to `SUBMAP_MAX_MATCH_POINTS` for this stage only.

### 9.3 Continuous GN-LM refinement with a motion prior

**Plain idea.** Polish the discrete winner to sub-cell accuracy, while gently
anchoring it to the prediction so a weak/ambiguous scan can't teleport the pose.

This solves an augmented least-squares problem (`CartoRefinementProblem` in
[slam_core/matching/scan_to_submap/refine.py](slam_core/matching/scan_to_submap/refine.py))
whose residual stacks two blocks:

**(a) Occupancy residuals**, one per in-bounds point, identical in spirit to §8.3
but using the bilinearly-interpolated submap probability $p$ and its analytic
bilinear gradient $(\partial_x p,\partial_y p)$:

$$
r_i^{\text{occ}}=1-p(\mathbf q_i),\qquad
J_i^{\text{occ}}=\Big[-\partial_x p,\;-\partial_y p,\;
-\big(\partial_x p\,\dot q_{x,i}+\partial_y p\,\dot q_{y,i}\big)\Big],
$$

where $\dot q_{x}=-\sin\theta\,p_x-\cos\theta\,p_y$ and
$\dot q_{y}=\cos\theta\,p_x-\sin\theta\,p_y$ are $\partial\mathbf q/\partial\theta$.

**(b) Prior residuals** anchoring to the predicted submap-frame pose
$\hat\xi=(\hat x,\hat y,\hat\theta)$, with separate translation/rotation weights:

$$
r^{\text{prior}}=
\begin{bmatrix}
w_t\,(x-\hat x)\\
w_t\,(y-\hat y)\\
w_r\,\operatorname{wrap}(\theta-\hat\theta)
\end{bmatrix},
\qquad
J^{\text{prior}}=\operatorname{diag}(w_t,w_t,w_r).
$$

Weights (lab): $w_t=$ `SUBMAP_REFINE_W_TRANS` $=0.1$ (gentle translation anchor),
$w_r=$ `SUBMAP_REFINE_W_ROT` $=1.0$ (firmer rotation anchor). The combined system

$$
r=\begin{bmatrix}r^{\text{occ}}\\ r^{\text{prior}}\end{bmatrix},\quad
J=\begin{bmatrix}J^{\text{occ}}\\ J^{\text{prior}}\end{bmatrix}
$$

is minimized by the generic `GaussNewtonLM` solver
([slam_core/optimisers/gn_lm.py](slam_core/optimisers/gn_lm.py)), i.e. the same
$H=J^\top J+\lambda I$, $\delta=-H^{-1}J^\top r$ update with per-iteration step
clipping (0.10 m / 5°), up to 12 iterations. The 3-DOF local refine is **native
NumPy** — g2o is reserved for the pose graph (§10).

### 9.4 Acceptance, score, and the motion filter

- **Reported score** is the *coarse correlative* score $s$ (not the refined score);
  the refined pose is the actual estimate returned.
- **Acceptance.** With `reject_below_min_score=True` (the Hector profile), a coarse
  score below `SUBMAP_MIN_SCORE` yields a **FALLBACK** to the predicted pose
  instead of refining from a weak initializer — this avoids accepting ambiguous
  matches during fast turns.
- **Motion filter** (submap mode only) decides *whether to insert* a scan into the
  submap. Insertion fires if any threshold is exceeded since the last insertion:

$$
\Delta\text{trans} > d_{\max}\ \ \lor\ \ \Delta\text{rot} > a_{\max}\ \ \lor\ \ \Delta t > t_{\max}.
$$

The thresholds derive from expected motion,
$d_{\max}=\operatorname{clip}(v_{\exp}\,t_{\text{period}},\,0.05,\,0.50)\,\text{m}$
and $a_{\max}=\operatorname{clip}(\omega_{\exp}\,t_{\text{period}},\,0.5^\circ,\,10^\circ)$,
with $t_{\max}=$ `TARGET_INSERT_PERIOD_S` $=0.10$ s (`adapter.py`,
`make_motion_filter_from_expected_velocity`). `scan_to_map` does not use this; it
uses the simpler `MAP_UPDATE_EVERY` cadence (§8.4).

---

## 10. Optional improvement stage — online pose-graph optimization (g2o SE(2))

> **This is not part of the core scan-matching pipeline.** Sections 8 and 9 are each
> complete on their own and produce a full trajectory and map without any of the
> machinery below. The pose graph is an **optional refinement layered on top of
> `scan_to_submap`**: it leaves the front-end (prediction, matching, map insertion)
> untouched and only post-corrects the *poses* to reduce accumulated drift. It is
> inactive unless `--enable-pgo` is passed, and it has no effect on `scan_to_map`.

**Enabled by** `--enable-pgo` with `--matcher scan_to_submap`. This is the
Cartographer-style **back-end** that turns the drift-prone front-end estimate into a
globally consistent one. Everything runs *online* during the scan loop, with a final
solve at the end. Modules: [carto/pose_graph/](carto/pose_graph/) +
[carto/pose_graph/backends/g2o_backend_2d.py](carto/pose_graph/backends/g2o_backend_2d.py).

### 10.1 Graph structure

A pose graph is a set of **vertices** (poses to estimate) and **edges**
(relative-pose measurements with confidences).

- **Submap vertices** $\xi_{s}$ — one per submap origin.
- **Node vertices** $\xi_{n}$ — one per *accepted, inserted* scan keyframe (added
  by `add_node_with_intra_constraints`). FALLBACK poses never become nodes.
- **Edges** carry a measurement $z$ (a relative pose) and an **information matrix**
  $\Omega$ (inverse covariance — bigger = more trusted), $\Omega=\operatorname{diag}(w_t,w_t,w_r)$.

`G2oBackend2D` maps the two id spaces into one disjoint g2o vertex space (submaps →
even ids, nodes → odd ids) and uses `VertexSE2` / `EdgeSE2`.

### 10.2 The three constraint families

**(1) Intra-submap constraints** (trusted local matches). For each submap a node
was inserted into, the measurement is the node pose expressed in the submap frame:

$$
z^{\text{intra}}_{s,n}=\xi_s^{-1}\oplus\xi_n,
\qquad (w_t,w_r)=(5\times10^2,\,1.6\times10^3).
$$

**(2) Inter-submap (loop-closure) constraints** (§10.3). Same algebraic form, but
between a *historical* submap $t$ and a current node, with much higher weight and a
robust kernel:

$$
z^{\text{loop}}_{t,n}=\xi_t^{-1}\oplus\xi_n^{\text{matched}},
\qquad (w_t,w_r)=(1.1\times10^4,\,1\times10^5).
$$

**(3) Local-trajectory "spine" constraints** (consecutive nodes). These preserve
the front-end's trajectory shape under loop-closure deformation. Between
consecutive node ids $i,i{+}1$ the measurement uses the online (pre-solve) poses,

$$
z^{\text{spine}}_{i,i+1}=\xi_i^{-1}\oplus\xi_{i+1},
\qquad (w_t,w_r)=(1\times10^5,\,1\times10^5),
$$

added in `_add_local_trajectory_regularization`. (Using the global online pose
rather than the submap-relative pose avoids a spurious large jump at submap
boundaries.)

### 10.3 Online loop-closure detection

A loop closure is the recognition that the robot has returned to an earlier place,
giving a constraint that ties distant parts of the trajectory together. Detection
has three filters (`CartoLoopClosureAdapter`, `LoopClosureManager`):

**(i) Spatial candidate gating.** For a new node, consider only *finished* submaps
whose current estimated origin lies within `spatial_search_radius` $=8.0$ m of the
node's guess, excluding submaps the node already belongs to, and only every
`check_every_n_nodes` $=5$-th node (cost control). A node-index separation of
`min_node_index_separation` $=30$ prevents matching near-neighbours. (For the small
lab loop, `recent_finished_submap_exclusion` $=0$ so the robot can close against the
very first submap when it returns to the origin.)

**(ii) Geometric verification (branch-and-bound).** Each candidate is verified by
re-matching the node's scan against the candidate submap with the **branch-and-bound
correlative matcher**
([branch_and_bound_backend.py](slam_core/matching/scan_to_submap/branch_and_bound_backend.py)).
This uses a stack of **forward-looking max-filter precomputation grids** (level $i$
stores the max probability over a $2^i\times2^i$ forward box) to compute, cheaply,
an **upper bound** on the achievable score for a whole block of candidate offsets.
The recursion (`_branch_and_bound`) explores high-score branches first and prunes
any branch whose upper bound cannot beat the current best — guaranteeing the global
optimum within the search window without scoring every cell. The angular step is
derived from the maximum scan radius $d_{\max}$ and resolution so that one angular
bin corresponds to roughly one cell of arc:

$$
\Delta\theta=\max\!\Big(\theta_{\min},\;\arccos\!\big(1-\tfrac{\rho^2}{2d_{\max}^2}\big)\Big).
$$

The best candidate is then continuously refined (the §9.3 GN-LM problem).

**(iii) Consistency gate.** Even a high-scoring match is rejected unless it agrees
with the current graph estimate. With predicted and matched node-in-target relative
poses $z^{\text{pred}}=\xi_t^{-1}\oplus\xi_n^{\text{guess}}$ and
$z^{\text{match}}=\xi_t^{-1}\oplus\xi_n^{\text{matched}}$, the residual

$$
\Delta=\big(z^{\text{pred}}\big)^{-1}\oplus z^{\text{match}},\qquad
\lVert\Delta_{xy}\rVert\le 1.0\,\text{m},\quad |\Delta_\theta|\le 0.209\,\text{rad}\,(12^\circ)
$$

must hold (defaults `max_loop_translation_residual_m`,
`max_loop_rotation_residual_rad`). The score must also clear `min_score` $=0.50$.
Accepted constraints are deduplicated per (node, target) pair.

### 10.4 The g2o optimization

**Objective.** With $\mathcal{V}=\{\xi_s\}\cup\{\xi_n\}$ the optimizer minimizes the
total Mahalanobis edge error,

$$
\min_{\mathcal V}\;
\sum_{e\in\mathcal E}\rho_e\!\Big(\;e_e^\top\,\Omega_e\,e_e\Big),
\qquad
e_e \;=\; \log_{SE(2)}\!\Big(z_e^{-1}\cdot\big(\xi_a^{-1}\oplus\xi_b\big)\Big),
$$

where edge $e$ connects vertices $a,b$ with measurement $z_e$, $e_e\in\mathbb{R}^3$
is the $SE(2)$ residual in the tangent space (g2o's `EdgeSE2` error), and $\rho_e$
is identity for intra/spine edges and the **Huber** kernel for loop edges. Huber
caps the influence of a residual once $\lVert e\rVert$ exceeds $\delta_H=$
`huber_scale` $=10$,

$$
\rho_H(s)=
\begin{cases}
s, & s\le \delta_H^2,\\
2\delta_H\sqrt{s}-\delta_H^2, & s> \delta_H^2,
\end{cases}
$$

so a single bad loop closure cannot dominate the solution.

**Gauge fix.** Submap 0 is held fixed (`set_fixed("submap",0)`), removing the global
translation/rotation gauge freedom that would otherwise leave the problem rank-3
deficient.

**Solver.** g2o's `SparseOptimizer` with a `BlockSolverSE2` over an Eigen sparse
linear solver and the **Levenberg–Marquardt** algorithm
(`OptimizationAlgorithmLevenberg`), up to 50 iterations. At each LM step it solves
the damped normal equations $(\mathbf H+\mu\,\mathrm{diag}(\mathbf H))\,\Delta=
-\mathbf b$, where $\mathbf H=\sum_e \mathbf J_e^\top\Omega_e\mathbf J_e$ and
$\mathbf b=\sum_e \mathbf J_e^\top\Omega_e e_e$ are assembled from per-edge
Jacobians, accepting the step only if the error decreases (otherwise increasing
$\mu$). This is the same GN/LM machinery as §8.3, now over the full sparse graph in
C++ instead of one dense 3×3 system.

**Scheduling & write-back.** `CartoGlobalSlam2D.on_node_inserted` triggers a solve
every `optimize_every_n_nodes` $=90$ nodes (guarded by having $\ge1$ intra
constraint), and `finalize()` runs a last full solve. After each solve, optimized
**submap poses are written back into the live `SubmapBuilder2D`**
(`_sync_submaps_to_builder`) so subsequent insertions and loop-distance checks use
corrected positions — closing the loop between back-end and front-end.

### 10.5 Exporting the corrected trajectory

The online trajectory file holds one row per scan; nodes are sparse keyframes. To
produce a **dense** corrected trajectory, each node's rigid correction is computed
as the left-multiplied delta between its optimized and online (drifted) poses,

$$
\Delta_n=\xi_n^{\text{opt}}\oplus\big(\xi_n^{\text{online}}\big)^{-1},
$$

and every scan inherits the correction of its **most recent keyframe node** $n(k)$:

$$
\xi_k^{\text{pgo}}=\Delta_{n(k)}\oplus\xi_k^{\text{online}}.
$$

The result is written to `trajectory_*_pgo.txt`. Because this is the post-solve
trajectory, it is the one used for map rebuild and evaluation (the hard rule:
*always export the optimized trajectory*).

> **Standalone offline PGO.** A separate scipy-based tool
> [hector/eval/pgo_any.py](hector/eval/pgo_any.py) implements an *offline*
> post-processing pose graph (spatial + scan-context loop detection, sparse
> `spsolve`, Huber weighting) that reads a finished `trajectory_*.txt`. It is **not**
> part of the live runner and is not the g2o path documented above; it remains as an
> alternative analysis tool.

---

## 11. Stage G — outputs

**Trajectory** `hector_outputs/trajectory_<tag>_<matcher>_<N>.txt`, one row per
scan: `t  x  y  θ  score` (bootstrap rows carry score $=-1$). A 13-column
`_debug.txt` adds per-scan diagnostics (`inliers`, per-scan delta
$(\delta x,\delta y,\delta\theta)$, `do_insert`, `did_insert`). With PGO, the dense
corrected `_pgo.txt` is also written (§10.5).

**Map.** The occupancy grid is converted cell-by-cell to probability
$M=\sigma(\ell)$ and rendered as greyscale (white = free, black = occupied, grey =
unknown), with the $(x,y)$ trajectory overlaid. Rebuild/plot helpers:
`hector/eval/rebuild_map_any.py`, `hector/plot_single_trajectory.py`,
`hector/viz/`.

---

## 12. Algorithm summary

| Component | Method | Code |
|---|---|---|
| Range → points | polar→Cartesian + range/finite gating | `range_to_points.py` |
| Preprocess | fixed-voxel centroid + adaptive (bisection) voxel | `preprocessing.py` |
| Prediction (active) | raw odom / last matched pose (`USE_EXTRAPOLATOR=False`) | `adapter.py` |
| Prediction (dormant) | constant-velocity + odom/IMU blend | `pose_extrapolator.py` |
| scan_to_map | multi-res log-odds grid, coarse-to-fine GN+LM | `scan_to_map.py` |
| scan_to_submap search | two-stage correlative (max-pool stack) | `correlative.py` |
| scan_to_submap refine | GN-LM, occupancy + translation/rotation prior | `refine.py`, `gn_lm.py` |
| Submap mgmt | 2-active rotation, finish at $N_s$, rotate at $N_s/2$ | `submaps.py` |
| Map update | log-odds add (linspace ray / Bresenham), clamp ±5 | `scan_to_map.py`, `submaps.py` |
| Loop detection | spatial gate + branch-and-bound verify + consistency gate | `loop_closure*.py`, `branch_and_bound_backend.py` |
| Pose graph | $SE(2)$ vertices/edges, intra + loop + spine, Huber on loops | `pose_graph_2d.py` |
| PGO solver | g2o `BlockSolverSE2` + Levenberg, submap 0 fixed | `g2o_backend_2d.py` |
| Trajectory export | per-node rigid correction propagated to all scans | `run_local_slam_new.py` |

---

## 13. What this implementation deliberately does *not* do

- **No IMU fusion** in the live path (hooks exist, disabled).
- **No constant-velocity prediction or odom blending** in the shipped profiles
  (`USE_EXTRAPOLATOR=False`); the prediction is raw odom or the last matched pose.
- **No loop closure inside `scan_to_map`** — that matcher is purely local; global
  consistency is only available via the `scan_to_submap` + g2o back-end.
- **g2o is used only for the pose graph.** All per-scan 3-DOF matching is native
  NumPy Gauss–Newton; there is no `pyceres` in the default path.
- **No ROS / GTSAM / Ceres** in the documented pipeline.

---

## 14. File-by-file data flow (current)

```
datasets/<name>/<log>                                  raw log
  └► slam_core/dataio/dataset_catalog.py               DatasetProfile + reader → scan dicts
       └► hector/run_local_slam_new.py                 MAIN LOOP (CLI, scheduling, IO)
            ├► carto/local_slam/range_to_points.py      ranges → body-frame points  (§5)
            ├► slam_core/matching/preprocessing.py      voxel thinning              (§6)
            ├► hector/adapter.py                         predict + motion filter + node insert (§7,§9.4)
            │    └► carto/local_slam/pose_extrapolator.py  (dormant CV extrapolator)
            ├► slam_core/matching/core.py                MatcherManager (swappable)
            ├─[scan_to_map] slam_core/matching/scan_to_map.py          (§8)
            │     MapPyramid · align_pose_gauss_newton · integrate_scan_simple
            └─[scan_to_submap] slam_core/matching/scan_to_submap/      (§9)
                  matcher.py · correlative.py · two_stage_backend.py
                  refine.py (CartoRefinementProblem) · submaps.py (SubmapBuilder2D)
                  └─[--enable-pgo] carto/pose_graph/  +  loop_closure*  (§10)
                        pose_graph_2d.py · global_slam_2d.py
                        branch_and_bound_backend.py (loop verify)
                        backends/g2o_backend_2d.py  (VertexSE2/EdgeSE2 + Levenberg)
       outputs ► hector_outputs/trajectory_*.txt              (online, per scan)
                 hector_outputs/trajectory_*_debug.txt         (diagnostics)
                 hector_outputs/trajectory_*_pgo.txt           (dense, optimized — PGO only)
```

---

## 15. Reproducing the validated result (lab_run_2)

```bash
# scan_to_map (Hector global grid)
.venv/bin/python -m hector.run_local_slam_new --dataset lab_run_2 --matcher scan_to_map
# scan_to_submap, no back-end
.venv/bin/python -m hector.run_local_slam_new --dataset lab_run_2 --matcher scan_to_submap
# scan_to_submap + online g2o PGO (recommended: smaller submaps → more loop closures)
.venv/bin/python -m hector.run_local_slam_new --dataset lab_run_2 --matcher scan_to_submap \
    --enable-pgo --scans-per-submap 250
```

Start↔end drift gap (lower is better): `scan_to_map` 0.186 m · `scan_to_submap`
no-PGO 0.525 m · `scan_to_submap` + PGO (sps 250) 0.306 m. PGO removes the no-PGO
ghosting and cuts drift ~42%; the residual gap to `scan_to_map` is front-end
per-scan jitter that a keyframe pose graph does not correct.
