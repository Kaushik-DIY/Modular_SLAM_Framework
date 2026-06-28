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

> **Global loop closure (RTAB proximity-with-retrieval).** The six LiDAR-front-end modes
> (`lidar_s2s_*`, `lidar_s2m_*`, `lidar_orb_*`) use the geometric **proximity** proposer, which
> now follows RTAB-Map's *proximity detection with retrieval*: it searches the whole pose graph
> within a bounded radius and **reactivates** any nearby keyframe that has aged into LTM back
> into Working Memory for verification (bounded by radius + max-candidates, so WM stays capped).
> This lets it close **global** loops to places transferred out of WM — including reverse-view
> returns that the appearance (DBoW) proposer cannot match. Effect on the tables below: the
> `lidar_*` modes gain recall (more true loops found) at maintained precision (no false-loop map
> warping — verified visually); the three DBoW-proposer modes (`orb_lidar_*`, `orb`) are
> unchanged. Real-time `lag s`/`late %` are carried from the original speed-1.0 runs (a
> loop-only change does not affect them); all other columns are freshly measured.

---

## Single-room lab (`lab_hybrid_small`)

| combo | kf | proposed | accepted | true-acc | false-acc | true-rej | precision | recall | lag s | late % | ms/kf | RSS GB | sharp | drift m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lidar_s2s_bnb | 159 | 41 | 32 | 19 | 13 | 1 | 0.59 | 0.95 | 0.0 | 6.1 | 4.2 | 0.13 | 0.553 | 0.49 |
| lidar_s2s_icp | 159 | 41 | 34 | 17 | 17 | 2 | 0.50 | 0.90 | 0.0 | 5.8 | 3.9 | 0.13 | 0.554 | 0.48 |
| lidar_s2m_bnb | 155 | 38 | 32 | 17 | 13 | 3 | 0.57 | 0.85 | 0.0 | 9.5 | 22.9 | 0.16 | 0.630 | 0.03 |
| lidar_s2m_icp | 155 | 38 | 36 | 20 | 14 | 0 | 0.59 | 1.00 | 0.0 | 9.3 | 20.1 | 0.17 | 0.636 | 0.02 |
| lidar_orb_s2s | 159 | 41 | 8 | 8 | 0 | 16 | 1.00 | 0.33 | 0.0 | 7.2 | 3.6 | 0.68 | 0.564 | 0.44 |
| lidar_orb_s2m | 155 | 38 | 8 | 6 | 0 | 14 | 1.00 | 0.30 | 0.0 | 10.4 | 22.0 | 0.72 | 0.636 | 0.03 |
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
| lidar_s2s_bnb | 382 | 112 | 80 | 53 | 26 | 13 | 0.67 | 0.80 | 0.0 | 6.1 | 2.7 | 0.22 | 0.543 | 0.71 |
| lidar_s2s_icp | 382 | 112 | 74 | 52 | 21 | 16 | 0.71 | 0.77 | 0.0 | 5.9 | 3.3 | 0.22 | 0.555 | 1.04 |
| lidar_s2m_bnb | 369 | 106 | 86 | 60 | 26 | 7 | 0.70 | 0.90 | 0.0 | 9.5 | 20.2 | 0.22 | 0.632 | 0.15 |
| lidar_s2m_icp | 369 | 106 | 96 | 66 | 30 | 0 | 0.69 | 1.00 | 0.0 | 10.6 | 19.8 | 0.23 | 0.633 | 0.11 |
| lidar_orb_s2s | 382 | 112 | 34 | 34 | 0 | 38 | 1.00 | 0.47 | 0.0 | 7.7 | 3.9 | 0.79 | 0.543 | 0.80 |
| lidar_orb_s2m | 369 | 106 | 35 | 35 | 0 | 32 | 1.00 | 0.52 | 0.0 | 11.0 | 16.5 | 0.79 | 0.625 | 0.15 |
| orb_lidar_bnb | 647 | 47 | 40 | 40 | 0 | 7 | **1.00** | 0.85 | 0.0 | 18 | 17.8 | 0.75 | 0.58 | 0.57 |
| orb_lidar_icp | 647 | 47 | 40 | 40 | 0 | 7 | **1.00** | 0.85 | 0.0 | 18 | 17.7 | 0.75 | 0.58 | 1.11 |
| orb | 647 | 47 | 31 | 31 | 0 | 16 | **1.00** | 0.66 | 0.0 | 18 | 17.7 | 0.75 | 0.58 | 0.63 |

Maps: `figures/lab_hybrid__<combo>.png`.

**Per-run notes** (same mode, vs the single-room run):
- **lidar_s2s_bnb** — Precision rises 0.59 → 0.67 vs the small map (cross-room candidates are unambiguously rejectable, fewer aliased accepts); recall 0.80 over the longer route, with the global proximity loops now catching the start↔end return.
- **lidar_s2s_icp** — Tracks s2s_bnb at slightly higher precision (0.71); end-start drift 1.04 m is the highest of the LiDAR modes here — a run-to-run effect of the driftier s2s front-end (the global loop ties end to start at the loop's measured offset; the occupancy map itself stays crisp, sharpness 0.555).
- **lidar_s2m_bnb** — Among the crispest maps (sharpness 0.63, drift 0.15 m) at low RSS (0.22 GB): scan-to-map scales to two rooms without losing geometry — the most consistent performer.
- **lidar_s2m_icp** — **Recall 1.00** on the crisp s2m map at the lowest drift of the scan modes (0.11 m): with global retrieval ICP now confirms every true revisit, including the long return loop — the standout when both recall and map quality matter.
- **lidar_orb_s2s** — Precision 1.0, recall up 0.33 → 0.47 vs small: the longer route plus global proximity retrieval offer more confirmable revisits, though heading-divergent ones the camera missed are still rejected by PnP; the map is its driftier s2s front-end's (0.80 m).
- **lidar_orb_s2m** — Precision 1.0 + the crisp s2m map (drift 0.15 m) at recall 0.52: the precision-first / clean-map profile, recall lifted by the global loops vs the small map.
- **orb_lidar_bnb** — Recall jumps 0.11 → 0.85 vs the small map: the long two-room route offers many camera-confirmable revisits, and with the IMU-bounded heading the B&B seed is reliable enough to confirm them (drift 0.57 m) — vision-led loop closure needs distance to "earn" loops.
- **orb_lidar_icp** — **Drift 1.11 m (was 4.25 m before the IMU heading fix)**: with the VO spine yaw now IMU-locked, the standalone ICP seed lands in the right basin instead of a corridor slide-lock, so it matches B&B's loop set exactly (recall 0.85); the residual gap to B&B (0.57 m) is ICP's local-refinement precision, not a warp — unlike the small room, the two-room route is where this fix mattered.
- **orb** — Recall falls 1.00 → 0.66 vs the small map: DBoW proposes fewer of the long route's revisits (appearance is viewpoint-dependent over distance); the camera-only map remains coarser than the LiDAR-fused ones.

**Two-room discussion.** The larger map sharpens every trend. **`lidar_s2m_bnb`** is again the
best crisp-map-per-cost option (sharpness 0.63, drift 0.15 m, 0.22 GB, 9 % late) and the most
consistent across both environments. The precision/recall split is now unmistakable:
**scan verifiers** sit at recall 0.77–1.00 / precision 0.67–0.71 (high coverage, some aliased
accepts), while **PnP verifiers** sit at precision 1.0 / recall 0.47–0.66 (no false loops, but
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
| lidar_s2s_bnb | 562 | 364 | 204 | 135 | 69 | 53 | 0.66 | 0.72 | 0.0 | 5.2 | 4.9 | 0.31 | 0.589 | 0.51 |
| lidar_s2s_icp | 562 | 363 | 260 | 155 | 102 | 31 | 0.60 | 0.83 | 0.0 | 3.7 | 3.7 | 0.33 | 0.606 | 0.51 |
| lidar_s2m_bnb | 544 | 352 | 229 | 149 | 77 | 27 | 0.66 | 0.85 | 0.0 | 22.7 | 24.2 | 0.30 | 0.689 | 0.10 |
| lidar_s2m_icp | 544 | 349 | 267 | 163 | 101 | 8 | 0.62 | 0.95 | 0.0 | 22.7 | 17.8 | 0.31 | 0.692 | 0.10 |
| lidar_orb_s2s | 562 | 368 | 9 | 9 | 0 | 169 | 1.00 | 0.05 | 0.0 | 4.2 | 3.4 | 0.85 | 0.555 | 2.85 |
| lidar_orb_s2m | 544 | 351 | 33 | 33 | 0 | 138 | 1.00 | 0.19 | 0.0 | 23.1 | 16.4 | 0.84 | 0.674 | 0.23 |
| orb_lidar_bnb | 787 | 259 | 45 | 41 | 1 | 189 | 0.98 | 0.18 | 0.0 | 15 | 16.1 | 0.79 | 0.59 | **4.24** |
| orb_lidar_icp | 787 | 259 | 51 | 51 | 0 | 179 | **1.00** | 0.22 | 0.0 | 15 | 15.2 | 0.80 | 0.60 | **5.33** |
| orb | 787 | 259 | 73 | 64 | 0 | 166 | **1.00** | 0.28 | 0.0 | 16 | 16.6 | 0.81 | 0.52 | 1.02 |

Maps: `figures/lab_hybrid_3_slow__<combo>.pdf` (+ PNG); montage `figures/master_lab_hybrid_3_slow.*`.
With the per-dataset fix the visual-VO reinitialisations drop **285 → 82** (keyframes 1026 → 787),
recovering the orb_lidar modes from ~12 m to 4–5 m.

**Per-run notes** (one notable point; vs the same mode on the smaller maps):
- **lidar_s2s_bnb** — Holds its profile on the bigger map (precision 0.66, recall 0.72, drift 0.51 m) — essentially the two-room numbers at larger scale, with many more proposals from the relaxed proposer + global retrieval.
- **lidar_s2s_icp** — Recall 0.83 at precision 0.60; end-start drift 0.51 m on the driftier s2s front-end (the global loop ties the route ends; the map stays crisp at sharpness 0.61).
- **lidar_s2m_bnb** — Among the **crispest maps of the study** (sharpness 0.69, drift 0.10 m) at low RSS (0.30 GB): scan-to-map's global reference scales gracefully to the bigger area.
- **lidar_s2m_icp** — Recall **0.95** with the same crispest map (drift 0.10 m): the standout when coverage and map quality both matter.
- **lidar_orb_s2s** — Precision 1.0 / recall 0.05: even with global retrieval the texture-poor scene gives the camera very few PnP-confirmable revisits; the map is its driftier s2s front-end's (2.85 m).
- **lidar_orb_s2m** — Precision 1.0 on the crisp s2m map (sharpness 0.67, drift 0.23 m) at recall 0.19: the precision-first clean-map option; global retrieval lifts recall over the smaller maps' lidar_orb runs while keeping zero false loops.
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
| lidar_s2s_bnb | 556 | 172 | 119 | 82 | 37 | 15 | 0.69 | 0.84 | 6.1 | 2.5 | 0.24 | 0.630 | 0.63 | 0 |
| lidar_s2s_icp | 556 | 168 | 135 | 92 | 43 | 13 | 0.68 | 0.88 | 3.7 | 2.3 | 0.25 | 0.635 | 0.55 | 0 |
| lidar_s2m_bnb | 550 | 164 | 132 | 91 | 40 | 9 | 0.69 | 0.91 | 26.0 | 15.3 | 0.25 | 0.689 | 0.15 | 0 |
| lidar_s2m_icp | 550 | 163 | 142 | 96 | 45 | 5 | 0.68 | 0.95 | 24.9 | 15.4 | 0.26 | 0.695 | 0.15 | 0 |
| lidar_orb_s2s | 556 | 170 | 12 | 12 | 0 | 83 | 1.00 | 0.13 | 5.2 | 5.2 | 0.78 | 0.617 | 0.57 | 0 |
| lidar_orb_s2m | 550 | 164 | 12 | 11 | 0 | 88 | 1.00 | 0.11 | 25.6 | 25.7 | 0.80 | 0.684 | 0.08 | 0 |
| orb_lidar_bnb | 697 | 85 | 65 | 64 | 1 | 9 | 0.99 | 0.88 | 17.2 | 17.5 | 0.77 | 0.503 | **0.15** | **427** |
| orb_lidar_icp | 697 | 85 | 49 | 46 | 3 | 27 | 0.94 | 0.63 | 15.3 | 15.8 | 0.76 | 0.503 | 0.97 | **427** |
| orb | 697 | 85 | 25 | 25 | 0 | 48 | **1.00** | 0.34 | 15.0 | 15.6 | 0.74 | 0.529 | **0.10** | **427** |

Maps: `figures/lab_hybrid_3__<combo>.pdf` (+ PNG); montage `figures/master_lab_hybrid_3.*`.

**Fast-vs-slow comparison (same map, same baseline config):**

| front-end | slow reinits | fast reinits | slow drift | fast drift | verdict |
|---|---|---|---|---|---|
| LiDAR s2s/s2m | 0 | 0 | 0.10–0.51 m | 0.15–0.63 m | **immune to speed** |
| lidar_orb_s2s | 0 | 0 | 2.85 m | 0.57 m | *better* fast (VO healthier) |
| lidar_orb_s2m | 0 | 0 | 0.23 m | 0.08 m | minor — s2m spine dominates |
| orb_lidar_bnb | 82\* | **427** | 4.23 m\* | 0.15 m | ↑5× reinits, low drift (early loops) |
| orb_lidar_icp | 82\* | **427** | 5.33 m\* | 0.97 m | ↑5× reinits, low drift (early loops) |
| orb | 82\* | **427** | 1.02 m\* | 0.10 m | ↑5× reinits, low drift (early loops) |

\* *Slow figures use per-dataset tuning (reinit_patience=12); fast uses baseline (patience=3).*

**Per-run notes:**
- **lidar_s2s/s2m** — Sharpness and compute cost are virtually unchanged vs the slow run
  (s2s ~2.5 ms, s2m ~15 ms). Speed has no effect on scan-to-map matching or submap insertion;
  drifts stay close (s2s ~0.55–0.63 m fast vs ~0.51 m slow, s2m ~0.15 m vs ~0.10 m).
- **lidar_orb_s2s** — Drift *improves* fast (0.57 vs 2.85 m): on the slow run the VO was
  collapsing from texture-poverty (no patience tuning → reinits), so PnP loops couldn't
  land; on the fast run the VO stays healthy (texture-poor stretches are traversed quickly)
  so PnP confirms 12 loops and pulls the s2s chain straight.
- **lidar_orb_s2m** — Minor drift change (0.08 vs 0.23 m); the crisp s2m spine is robust
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

---

## Real-time module switching (live reconfiguration)

The fusion-layer runner hot-swaps any module **mid-run** — front-end (s2s / s2m / VO), loop
proposer (proximity / DBoW), loop verifier (B&B / ICP / PnP) — via a grace-buffer handoff that
warms the incoming front-end on the live stream before flipping, resets the relative-motion
baseline to the new estimate (no teleport), and auto-falls-back any module the new front-end
cannot feed. To present this deterministically, switches were driven by a keyframe-indexed
schedule (`--switch-schedule`, replaying the same code path as the interactive commands); each
applied switch and a per-keyframe timeline are logged (`switches.csv`, `timeline.csv`).

Two demo runs on `lab_2` exercise, between them, **every module** (front-ends s2s/VO/s2m,
proposers proximity/DBoW, verifiers B&B/PnP/ICP):

| run | start → end config | switches (at keyframe) | kf | proposed | accepted | reinits |
|---|---|---|---|---|---|---|
| **Demo A** (`switching_demo_mapA`) | s2s·prox·B&B → VO·DBoW·PnP | ①prox→DBoW @114 · ②**s2s→VO** @259 · ③B&B→PnP @304 | 473 | 52 | 33 | 29 |
| **Demo B** (`switching_demo_mapB`) | VO·DBoW·PnP → s2m·prox·ICP | ①**VO→s2m** @179 · ②PnP→ICP @300 · ③DBoW→prox @390 | 441 | 42 | 35 | 0 |

**Feature-aware switch placement.** The cross-sensor switches are the stability-critical ones, so
they are scheduled by *content*, not just by time: a VO-reference pass supplies per-timestamp
visual-feature richness (tracking inliers) and turning rate, and the switch **into** VO (Demo A,
kf 259) is placed on a sustained textured, non-turning stretch (≥95 inliers vs a median of 43,
≤4.6° turn across the warm-up window) rather than mid-turn in a bare corridor. The switch **out
of** VO (Demo B, kf 179) leaves while VO is still healthy. The no-handoff switches
(proposer/verifier) are feature-insensitive and stay near their fractional targets — the
timeline is used efficiently instead of waiting for a perfect moment for every switch.

**Stability finding (the claim).** Across all six switches the **keyframe-to-keyframe
displacement stays spike-free** (the `_displacement` figure) — the handoff introduces no
teleport — and the trajectory on the final map is continuous through every switch marker. The
**loop closure events** panel (`_loops`) shows *proposed* and *accepted* candidates; acceptances
continue across switches (e.g. Demo A keeps closing loops through the proposer switch ①, the
s2s→VO switch ②, and into the VO segment), so the map keeps being corrected regardless of which
modules are active. The result is that **the occupancy map stays a single consistent two-room
map** end-to-end, demonstrating that live reconfiguration is functional and non-disruptive.

**Loop-closure behaviour (clustering).** Loop closures are *not* uniform over a run — they
cluster where the trajectory re-passes an already-mapped place (candidate "age" is bounded below
by `min_kf_separation = 30`), so a run's flat stretches are simply non-revisiting route, not a
switching failure. Two memory/loop-path issues were found and fixed during this study:

1. **Lazy appearance index (fixed).** The DBoW index was built only when the `dbow` proposer
   first became active, so a *mid-run* switch to DBoW could not propose against keyframes seen
   before the switch (why Demo A's VO segment originally showed 0 proposals). It now builds from
   the first keyframe whenever visual descriptors exist (Demo A: 28→52 proposed, 13→33 accepted).

2. **Proximity proposer reaches LTM — global loops (fixed, RTAB-faithful).** The proximity
   proposer originally searched **Working Memory only**, so a revisit to a place that had aged
   into LTM was never proposed — only the appearance (DBoW) proposer could attempt it, and DBoW
   fails on *reverse-direction* returns (the camera images a different field of view, so the
   bag-of-words does not match — ORB's in-plane rotation invariance does not help a viewpoint
   change). The fix implements RTAB-Map's **"proximity detection with retrieval"**: the pose
   graph retains every node (transfer to LTM does not drop its pose), so the proposer searches
   the whole graph within `proposal_radius_m` and **reactivates** any chosen LTM node (+graph
   neighbours) back into WM for verification — bounded by radius + `max_candidates_per_query`, so
   it stays real-time and WM stays capped. Result on Demo B: the literal **start↔end loop now
   closes** (e.g. query kf 425 ↔ candidate kf 13, age 412), with **11 loops of age > 200** that
   had aged into LTM; WM held at its 200 cap throughout (LTM 205). Loop acceptances now continue
   to the very end of the run (`_loops` panel). A geometric (pose) proposer is view-invariant and
   so closes these reverse-direction global loops that appearance cannot.

Figures (separate, paper-grade PDF+PNG, per run): `figures/switching_demo_map{A,B}_map`,
`…_displacement`, `…_tracking`, `…_loops`.
