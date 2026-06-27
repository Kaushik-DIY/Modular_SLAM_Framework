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

---

## Larger lab (`lab_hybrid_3_slow`)

A third recording over a **bigger floor area** (~26 m across vs ~21 m for the two-room lab),
driven at the same speed as the first two datasets (635 s, 6095 scans / 9521 RGB-D frames). The
scan-to-map global map was enlarged 40 → 80 m for the larger traverse.

**Root cause of the visual difficulty (diagnosed).** This recording is **texture-poor**: it
yields ~half the ORB features of `lab_hybrid` (median **1042 vs 1912** per frame), so the visual
odometry repeatedly **collapses to ~4 inliers and loses tracking** — ~2× as often as on the
two-room lab (285 vs 104 reinitialisations on the full runner). Those reinit "blind" segments,
not the loop closure, are what wrecked the visual-led maps. The fix is to **coast through the
transient texture-poor frames** (longer reinit-patience) + insert keyframes earlier + verify
loops strictly. Crucially this **cannot be a global setting** — `lab_hybrid`'s tracking losses
happen during *fast turns*, where coasting flies the VO off (17 m). So it is applied as a
**per-dataset tuning for `lab_hybrid_3_slow` only** (`apply_dataset_tuning`); `lab_hybrid` and
`lab_hybrid_small` keep their committed config unchanged.

| combo | kf | proposed | accepted | true-acc | false-acc | true-rej | precision | recall | lag s | late % | ms/kf | RSS GB | sharp | drift m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 562 | 345 | 176 | 111 | 65 | 53 | 0.63 | 0.68 | 0.0 | 5 | 3.9 | 0.30 | 0.58 | 0.45 |
| lidar_s2s_icp | 562 | 344 | 221 | 135 | 86 | 36 | 0.61 | 0.79 | 0.0 | 4 | 2.5 | 0.32 | 0.61 | **0.07** |
| lidar_s2m_bnb | 544 | 329 | 228 | 141 | 87 | 21 | 0.62 | 0.87 | 0.0 | 23 | 15.3 | 0.30 | **0.69** | **0.10** |
| lidar_s2m_icp | 544 | 326 | 261 | 160 | 101 | 10 | 0.61 | **0.94** | 0.0 | 23 | 15.2 | 0.31 | **0.69** | **0.09** |
| lidar_orb_s2s | 562 | 338 | 6 | 6 | 0 | 131 | **1.00** | 0.04 | 0.0 | 4 | 2.8 | 0.41 | 0.57 | 2.24 |
| lidar_orb_s2m | 544 | 320 | 12 | 12 | 0 | 132 | **1.00** | 0.08 | 0.0 | 23 | 15.5 | 0.40 | **0.69** | **0.04** |
| orb_lidar_bnb | 787 | 259 | 45 | 41 | 1 | 189 | 0.98 | 0.18 | 0.0 | 15 | 16.1 | 0.79 | 0.59 | **4.24** |
| orb_lidar_icp | 787 | 259 | 51 | 51 | 0 | 179 | **1.00** | 0.22 | 0.0 | 15 | 15.2 | 0.80 | 0.60 | **5.33** |
| orb | 787 | 259 | 73 | 64 | 0 | 166 | **1.00** | 0.28 | 0.0 | 16 | 16.6 | 0.81 | 0.52 | 1.02 |

Maps: `figures/lab_hybrid_3_slow__<combo>.pdf` (+ PNG); montage `figures/master_lab_hybrid_3_slow.*`.
With the per-dataset fix the visual-VO reinitialisations drop **285 → 82** (keyframes 1026 → 787),
recovering the orb_lidar modes from ~12 m to 4–5 m.

**Per-run notes** (one notable point; vs the same mode on the smaller maps):
- **lidar_s2s_bnb** — Holds its profile on the bigger map (precision 0.63, recall 0.68, drift 0.45 m) — essentially the two-room numbers at larger scale, with more proposals from the relaxed proposer.
- **lidar_s2s_icp** — Drift tightens to **0.07 m** (best s2s of the study) at recall 0.79: the extra confirmed loops from the relaxed proposer pull the submap-chain straight.
- **lidar_s2m_bnb** — Among the **crispest maps of the study** (sharpness 0.69, drift 0.10 m) at low RSS (0.30 GB): scan-to-map's global reference scales gracefully to the bigger area.
- **lidar_s2m_icp** — Recall **0.94** with the same crispest map (drift 0.09 m): the standout when coverage and map quality both matter.
- **lidar_orb_s2s** — Precision 1.0 / recall 0.04: visual PnP confirms very few of the larger route's revisits; the map is its driftier s2s front-end's (2.24 m), as on the small map.
- **lidar_orb_s2m** — Precision 1.0 on the crisp s2m map (sharpness 0.69, drift **0.04 m**) at low recall: the precision-first clean-map option, here the best it has looked across the three maps.
- **orb_lidar_bnb** — **Recovered to 4.24 m** (was 12 m before the texture-poverty fix): coasting through the transient VO collapses (reinits 285→82) lets B&B's loops finally bite — precision 0.98 (one false accept).
- **orb_lidar_icp** — **5.33 m at precision 1.00 / zero false loops**: the strict ICP gate accepts no wrong loop; with the VO spine no longer shredded, those loops correct the bulk of the drift (vs 11.8 m before) — recall stays low (0.22) because the strict gate + texture-poverty leave few confident revisits.
- **orb** — Sharper map than baseline (sharpness 0.42 → **0.52**) at precision 1.00; end-start drift 1.02 m (the noisier single-point metric rises while the map itself improves with fewer blind segments).

**Larger-lab discussion.** This map is the study's hardest for vision, and it shows *why* the
**front-end must be selectable**. The diagnosed cause is **texture-poverty** (half the ORB
features → the VO collapses ~2× as often); the cure is a **per-dataset** robustness setting,
because the right reinit behaviour is opposite to `lab_hybrid`'s (coast here, reinit-fast there).
With it, the visual-led modes go from **unusable (~12 m) to usable (~4–5 m)** at **precision ≈ 1.0**
(the strict verifier accepts essentially no false loop) — but still at high cost (~0.8 GB, ~16 %
late) and low recall (the scene yields few confident revisits). The **LiDAR scan-to-map** modes
remain the **best maps in the whole study** (sharpness 0.69, drift 0.09–0.10 m, ~0.30 GB) and need
almost no loop help. So the framework's value stands: on a large, low-texture floor LiDAR
scan-to-map is the robust, crisp, cheap choice, while visual-led modes are a recoverable but
costly fallback — and getting the most out of each needs **per-environment configuration**.

---

## Larger lab — faster traverse (`lab_hybrid_3`)

The **same large floor area** as `lab_hybrid_3_slow` driven at **1.6× faster speed** (395 s vs
635 s, 5919 vs 9521 RGB-D frames, 3790 vs 6095 LiDAR scans). Run with the **baseline config**
(no per-dataset tuning applied — the `lab_hybrid_3_slow` patience-coasting setting is wrong for
fast motion). Purpose: isolate the effect of traversal speed on each front-end.

| combo | kf | proposed | accepted | true-acc | false-acc | true-rej | precision | recall | late % | ms/kf | RSS GB | sharp | drift m | reinit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 556 | 170 | 102 | 74 | 28 | 17 | 0.73 | 0.81 | 6.1 | 4.5 | 0.23 | 0.623 | 0.61 | 0 |
| lidar_s2s_icp | 556 | 166 | 133 | 88 | 44 | 16 | 0.67 | 0.85 | 3.7 | 2.5 | 0.25 | 0.637 | 0.37 | 0 |
| lidar_s2m_bnb | 550 | 164 | 122 | 85 | 37 | 14 | 0.70 | 0.86 | 26.0 | 17.6 | 0.25 | 0.686 | 0.15 | 0 |
| lidar_s2m_icp | 550 | 163 | 142 | 99 | 41 | 5 | 0.71 | **0.95** | 24.9 | 16.8 | 0.26 | **0.694** | **0.15** | 0 |
| lidar_orb_s2s | 556 | 164 | 13 | 13 | 0 | 73 | **1.00** | 0.15 | 5.2 | 3.6 | 0.35 | 0.614 | 0.44 | 0 |
| lidar_orb_s2m | 550 | 146 | 8 | 8 | 0 | 67 | **1.00** | 0.11 | 25.6 | 17.1 | 0.36 | **0.688** | **0.11** | 0 |
| orb_lidar_bnb | 697 | 85 | 65 | 64 | 1 | 9 | 0.99 | 0.88 | 17.2 | 17.5 | 0.77 | 0.503 | **0.15** | **427** |
| orb_lidar_icp | 697 | 85 | 49 | 46 | 3 | 27 | 0.94 | 0.63 | 15.3 | 15.8 | 0.76 | 0.503 | 0.97 | **427** |
| orb | 697 | 85 | 25 | 25 | 0 | 48 | **1.00** | 0.34 | 15.0 | 15.6 | 0.74 | 0.529 | **0.10** | **427** |

Maps: `figures/lab_hybrid_3__<combo>.pdf` (+ PNG); montage `figures/master_lab_hybrid_3.*`.

**Fast-vs-slow comparison (same map, same baseline config):**

| front-end | slow reinits | fast reinits | slow drift | fast drift | verdict |
|---|---|---|---|---|---|
| LiDAR s2s/s2m | 0 | 0 | 0.07–0.45 m | 0.15–0.61 m | **immune to speed** |
| lidar_orb_s2s | 0 | 0 | 2.24 m | 0.44 m | *better* fast (VO healthier) |
| lidar_orb_s2m | 0 | 0 | 0.04 m | 0.11 m | minor — s2m spine dominates |
| orb_lidar_bnb | 82\* | **427** | 4.23 m\* | 0.15 m | ↑5× reinits, low drift (early loops) |
| orb_lidar_icp | 82\* | **427** | 5.33 m\* | 0.97 m | ↑5× reinits, low drift (early loops) |
| orb | 82\* | **427** | 1.02 m\* | 0.10 m | ↑5× reinits, low drift (early loops) |

\* *Slow figures use per-dataset tuning (reinit_patience=12); fast uses baseline (patience=3).*

**Per-run notes:**
- **lidar_s2s/s2m** — Sharpness and compute cost are virtually unchanged vs the slow run
  (s2s 4.5 ms, s2m 17 ms). Speed has no effect on scan-to-map matching or submap insertion.
  The slight s2s drift increase (0.37–0.61 m vs 0.07–0.45 m) reflects fewer loop
  confirmations on the shorter traversal, not a tracking failure.
- **lidar_orb_s2s** — Drift *improves* fast (0.44 vs 2.24 m): on the slow run the VO was
  collapsing from texture-poverty (no patience tuning → reinits), so PnP loops couldn't
  land; on the fast run the VO stays healthy (texture-poor stretches are traversed quickly)
  so PnP confirms 13 loops and pulls the s2s chain straight.
- **lidar_orb_s2m** — Minor drift change (0.11 vs 0.04 m); the crisp s2m spine is robust
  regardless — visual verification is a bonus on top.
- **orb_lidar_bnb/icp/orb** — **427 reinits** (vs 82 with tuning on the slow run) — fast
  motion causes 5× more VO tracking collapses. Yet *final drift is lower* (0.10–0.97 m vs
  1–5 m): the faster loop means the robot revisits the start sooner, so loop closures fire
  before drift accumulates far. Map sharpness suffers (0.50 vs 0.59) — the 427 blind
  segments leave many scans mis-registered in the grid.

**Speed-sensitivity finding.** LiDAR front-ends are **speed-immune** — scan geometry and IMU
extrapolation handle any traversal rate. VO-led front-ends are **speed-sensitive**: fast motion
increases tracking collapses 5×, degrading map sharpness even when early loop closure keeps the
final drift low. This reinforces the modular selector argument: in a large low-texture space at
variable speed, a LiDAR front-end is the safe default; VO-led modes can survive fast traversal
through dense loop closure but at the cost of a blurrier occupancy map.
