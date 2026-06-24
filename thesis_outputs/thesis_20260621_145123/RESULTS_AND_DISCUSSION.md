# Results & Discussion — modular fusion SLAM across two lab environments

All nine method combinations were run through the **fusion-layer real-time runner**
(`run_realtime.py --source dataset --speed 1.0`) on two hybrid RGB-D + 2D-LiDAR + IMU
datasets: **`lab_hybrid_small`** (a single cluttered workshop room, 164 s) and
**`lab_hybrid`** (a two-room lab, 431 s). Because the runs are played at real sensor
cadence, the lag / late-frame figures are literal. Each run produced a fused occupancy
map, a trajectory, per-keyframe timing/memory, and a per-candidate loop log.

The two visual-VO front-ends (`orb`, `orb_lidar`) apply a **loosely-coupled IMU heading**
correction: each keyframe's spine edge takes its relative *yaw* from the drift-free IMU
(absolute yaw drifts ≈ 0.001° over a run), keeping VO for translation and leaving the visual
bundle-adjustment untouched. This is applied at the SE(2) graph boundary — a scalar
ground-plane yaw, immune to the camera's ~8° mount tilt that makes per-frame IMU *tracking*
priors fail — and it bounds the VO heading drift that otherwise feeds the loop verifiers.

The nine combinations span the modular axes — LiDAR front-end (`scan-to-submap` /
`scan-to-map`), loop verifier (B&B / ICP / ORB-PnP), and proposer (proximity / DBoW):

| id | front-end | loop verification |
|---|---|---|
| lidar_s2s_bnb / _icp | LiDAR scan-to-submap | scan B&B / scan ICP |
| lidar_s2m_bnb / _icp | LiDAR scan-to-map | scan B&B / scan ICP |
| lidar_orb_s2s / _s2m | LiDAR (s2s / s2m) | ORB + PnP (visual) |
| orb_lidar_bnb / _icp | visual VO | scan B&B / scan ICP |
| orb | visual VO | ORB + PnP (visual) |

### Loop-closure ground truth (no GT poses available)

There is no motion-capture ground truth, so each proposed loop candidate was labelled
**true/false by a *physical-revisit* test**: TRUE if the two keyframes are genuinely the
same place, evidenced by **camera overlap** (ORB + RANSAC fundamental-matrix inliers ≥ 18)
**OR** **rotation-invariant LiDAR scan overlap** (B&B-aligned raw scans, nearest-neighbour
overlap ≥ 0.40). The dual criterion is *sensor-fair*: the 360° LiDAR registers a revisit
even when the narrow-FOV camera faced a different wall (≈ 70 % of valid LiDAR loops), which
a camera-only test would wrongly reject. The auto-labels were eyeball-validated on the RGB
contact sheets — e.g. it correctly separates visually-aliased under-furniture views
(scan ≈ 0 → different spot → false) from true revisits (scan ≈ 0.8). **Precision** = true /
accepted; **recall** = true-accepted / (true-accepted + true-loops-rejected).

---

## Single-room lab (`lab_hybrid_small`)

| combo | kf | proposed | accepted | true-acc | false-acc | true-rej | precision | recall | lag s | late % | ms/kf | RSS GB | sharp | drift m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 159 | 41 | 32 | 19 | 13 | 1 | 0.59 | 0.95 | 0.01 | 6 | 4.1 | 0.13 | 0.55 | 0.50 |
| lidar_s2s_icp | 159 | 41 | 34 | 17 | 17 | 2 | 0.50 | 0.90 | 0.01 | 6 | 3.8 | 0.13 | 0.55 | 0.48 |
| lidar_s2m_bnb | 155 | 38 | 32 | 17 | 13 | 3 | 0.57 | 0.85 | 0.02 | 10 | 6.4 | 0.12 | **0.63** | **0.03** |
| lidar_s2m_icp | 155 | 38 | 36 | 20 | 14 | 0 | 0.59 | 1.00 | 0.02 | 9 | 6.2 | 0.13 | **0.64** | **0.02** |
| lidar_orb_s2s | 159 | 41 | 9 | 9 | 0 | 15 | **1.00** | 0.38 | 0.01 | 7 | 4.7 | 0.26 | 0.56 | 0.44 |
| lidar_orb_s2m | 155 | 38 | 8 | 6 | 0 | 14 | **1.00** | 0.30 | 0.02 | 10 | 7.1 | 0.26 | **0.64** | 0.03 |
| orb_lidar_bnb | 249 | 9 | 1 | 1 | 0 | 8 | **1.00** | 0.11 | 0.00 | 18 | 18.1 | 0.66 | 0.49 | 0.02 |
| orb_lidar_icp | 249 | 9 | 7 | 7 | 0 | 2 | **1.00** | 0.78 | 0.00 | 18 | 18.2 | 0.67 | 0.49 | 0.39 |
| orb | 249 | 9 | 9 | 9 | 0 | 0 | **1.00** | 1.00 | 0.00 | 18 | 17.9 | 0.67 | 0.51 | 0.05 |

Maps: `figures/lab_hybrid_small__<combo>.png`.

**Per-run notes** (one notable point each):
- **lidar_s2s_bnb** — High recall (0.95) but moderate precision (0.59): in one cluttered room the proximity proposer + B&B accept many marginal same-room overlaps, several of which are perceptually-aliased under-furniture views the 360° scan rejects.
- **lidar_s2s_icp** — Lowest precision (0.50): standalone ICP's looser fitness admits a few more aliased loops than B&B on the same s2s front-end; map quality is identical to s2s_bnb.
- **lidar_s2m_bnb** — Crispest scan map (sharpness 0.63, drift 0.03 m): the growing global scan-to-map reference tightens geometry, giving a near-drift-free single room at the lowest memory (0.12 GB).
- **lidar_s2m_icp** — Recall 1.00 with the same crisp s2m map (drift 0.02 m): ICP accepts every real revisit here; the standout when both recall and map quality matter.
- **lidar_orb_s2s** — Precision 1.00 but recall only 0.38: visual PnP confirms only loops the camera also saw, missing the heading-divergent revisits; the map is its s2s front-end's (driftier, 0.44 m).
- **lidar_orb_s2m** — Precision 1.00 on the crisp s2m map (sharpness 0.64, drift 0.03 m) at the lowest recall (0.30): the precision-first, clean-map option.
- **orb_lidar_bnb** — Only 1 loop accepted (recall 0.11): VO drifts little over the short single room, so almost no revisit registers a graph offset for B&B to confirm.
- **orb_lidar_icp** — Recall 0.78 (vs B&B's 0.11) on the same VO front-end: with the IMU-corrected heading seeding it, standalone ICP recovers most revisits at precision 1.00 even over the short room — the same map that, two-room, was its catastrophic case before the heading fix.
- **orb** — Recall 1.00: DBoW + PnP catches every appearance revisit in the compact room; but the camera-only map is the least sharp of the LiDAR-fused options (0.50).

**Single-room discussion.** Map quality clearly favours the **scan-to-map** variants
(`lidar_s2m_*`, `lidar_orb_s2m`: sharpness ≈ 0.63–0.64, drift ≈ 0.02–0.03 m) over
scan-to-submap and the visual-led maps. On loops, two regimes emerge: **scan verifiers**
(B&B/ICP) give *high recall / moderate precision* — they find nearly every revisit but
admit aliased ones in the repetitive room — whereas **PnP verifiers** (`lidar_orb*`, `orb`)
give *precision 1.0 / low recall* — only confident, camera-confirmed loops. Cost separates
the front-ends sharply: LiDAR-led runs use 0.12–0.26 GB at 4–10 % late frames, while the
VO-led runs use ≈ 0.66 GB at ≈ 18 % late (and re-initialise 70× in the rotation-heavy room).
No single winner: **`lidar_s2m_bnb`** is the crisp-map-per-watt choice; **`lidar_orb_s2m`**
adds false-loop immunity (precision 1.0) on the same clean map; **scan verifiers** maximise
revisit detection; the **visual VO** modes are the fallback when wheel/LiDAR odometry is
unavailable, at a clear compute and stability cost.

---

## Two-room lab (`lab_hybrid`)

| combo | kf | proposed | accepted | true-acc | false-acc | true-rej | precision | recall | lag s | late % | ms/kf | RSS GB | sharp | drift m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 382 | 110 | 69 | 48 | 20 | 16 | 0.71 | 0.75 | 0.0 | 6 | 4.2 | 0.21 | 0.53 | 0.70 |
| lidar_s2s_icp | 382 | 110 | 69 | 42 | 26 | 14 | 0.62 | 0.75 | 0.0 | 6 | 3.8 | 0.21 | 0.53 | 0.41 |
| lidar_s2m_bnb | 368 | 104 | 76 | 48 | 25 | 9 | 0.66 | 0.84 | 0.0 | 10 | 6.4 | 0.18 | **0.63** | **0.14** |
| lidar_s2m_icp | 368 | 104 | 85 | 49 | 33 | 6 | 0.60 | 0.89 | 0.0 | 11 | 7.1 | 0.19 | **0.63** | 0.28 |
| lidar_orb_s2s | 382 | 110 | 29 | 29 | 0 | 29 | **1.00** | 0.50 | 0.0 | 8 | 5.1 | 0.35 | 0.54 | 0.47 |
| lidar_orb_s2m | 368 | 103 | 24 | 22 | 0 | 29 | **1.00** | 0.43 | 0.0 | 11 | 7.4 | 0.32 | **0.63** | **0.13** |
| orb_lidar_bnb | 647 | 47 | 40 | 40 | 0 | 7 | **1.00** | 0.85 | 0.0 | 18 | 17.8 | 0.75 | 0.58 | 0.57 |
| orb_lidar_icp | 647 | 47 | 40 | 40 | 0 | 7 | **1.00** | 0.85 | 0.0 | 18 | 17.7 | 0.75 | 0.58 | 1.11 |
| orb | 647 | 47 | 31 | 31 | 0 | 16 | **1.00** | 0.66 | 0.0 | 18 | 17.7 | 0.75 | 0.58 | 0.63 |

Maps: `figures/lab_hybrid__<combo>.png`.

**Per-run notes** (same mode, vs the single-room run):
- **lidar_s2s_bnb** — Precision rises 0.59 → 0.71 vs the small map: the second room makes cross-room candidates unambiguously rejectable, so fewer aliased accepts; recall eases to 0.75 over the longer route.
- **lidar_s2s_icp** — Tracks s2s_bnb but ~0.1 lower precision again (looser ICP fitness); end-start drift actually better here (0.41 m) than s2s_bnb (0.70 m), a run-to-run effect of which loops landed.
- **lidar_s2m_bnb** — Holds the crispest map across both maps (sharpness 0.63, drift 0.14 m) at the lowest RSS (0.18 GB): scan-to-map scales to two rooms without losing geometry — the most consistent performer.
- **lidar_s2m_icp** — Highest recall on this map (0.89) on the crisp s2m map, but drift (0.28 m) is double s2m_bnb's — ICP's marginal edges loosen the global solution slightly.
- **lidar_orb_s2s** — Precision 1.0 again, recall up 0.38 → 0.50 vs small: the longer route offers more camera-confirmable revisits, though half are still missed (heading-divergent).
- **lidar_orb_s2m** — Precision 1.0 + the crisp s2m map (drift 0.13 m) again at low recall (0.43): the same precision-first / clean-map profile as on the small map, now over two rooms.
- **orb_lidar_bnb** — Recall jumps 0.11 → 0.85 vs the small map: the long two-room route offers many camera-confirmable revisits, and with the IMU-bounded heading the B&B seed is reliable enough to confirm them (drift 0.57 m) — vision-led loop closure needs distance to "earn" loops.
- **orb_lidar_icp** — **Drift 1.11 m (was 4.25 m before the IMU heading fix)**: with the VO spine yaw now IMU-locked, the standalone ICP seed lands in the right basin instead of a corridor slide-lock, so it matches B&B's loop set exactly (recall 0.85); the residual gap to B&B (0.57 m) is ICP's local-refinement precision, not a warp — unlike the small room, the two-room route is where this fix mattered.
- **orb** — Recall falls 1.00 → 0.66 vs the small map: DBoW proposes fewer of the long route's revisits (appearance is viewpoint-dependent over distance); the camera-only map remains coarser than the LiDAR-fused ones.

**Two-room discussion.** The larger map sharpens every trend. **`lidar_s2m_bnb`** is again the
best crisp-map-per-cost option (sharpness 0.63, drift 0.14 m, 0.18 GB, 6 % late) and the most
consistent across both environments. The precision/recall split is now unmistakable:
**scan verifiers** sit at recall 0.75–0.89 / precision 0.60–0.71 (high coverage, some aliased
accepts), while **PnP verifiers** sit at precision 1.0 / recall 0.43–0.85 (no false loops, but
they discard heading-divergent revisits). The **visual-VO** modes reach competitive recall
here (`orb_lidar_bnb` and `orb_lidar_icp` both 0.85) but at ≈ 0.75 GB and ≈ 18 % late frames;
once the VO heading is IMU-bounded, `orb_lidar_icp` tracks B&B's loop set exactly and its map
deforms only by ICP's local-refinement precision (drift 1.11 m vs B&B's 0.57 m) rather than the
4.25 m slide-lock warp it suffered when the seed heading was left to drift. The framework's value is exactly
this spread: a deployment that needs a **crisp metric map cheaply** picks `lidar_s2m_bnb`; one
that **cannot tolerate a false loop** picks a PnP-verified mode; one **without LiDAR odometry**
falls back to `orb_lidar`, accepting higher cost and the need for route length before loops
close. No mode dominates on all of {map quality, precision, recall, memory, latency} — which
is the case for keeping all of them selectable.
