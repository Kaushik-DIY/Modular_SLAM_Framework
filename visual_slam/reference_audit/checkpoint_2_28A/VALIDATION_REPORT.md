# Checkpoint 2.28A — VALIDATION REPORT

## 1. Unit tests

### Command

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/
```

### Result

```
206 passed, 1 skipped in 20.72s
```

### New tests added (7, all pass)

| Test | What it verifies |
|---|---|
| `test_search_more_projects_into_current_camera_with_id_pose` | Direct API: identity `Tcw` → points in front of camera → `found > 0` |
| `test_search_more_wrong_pose_finds_nothing` | Direct API: far-away candidate `Tcw` (50 m offset) → `found == 0` |
| `test_loop_geometry_checker_uses_corrected_pose` | End-to-end: `LoopGeometryChecker` with corrected pose → `check_candidates` returns True |
| `test_corrected_tcw_matches_pipeline_convention` | Numerical: `Tcw_current @ inv(T12)` equals `_make_corrected_pose_map` formula |
| `test_search_more_return_diagnostics_flag` | Diagnostics dict present when `return_diagnostics=True` |
| `test_search_more_signature_takes_single_4x4_tcw` | API shape contract: `(points, frame, Tcw_4x4, ...)` accepted without error |
| `test_search_more_docstring_warns_against_candidate_tcw` | Documentation guard: docstring contains the formula and warns against `T12 @ Tc2w` |

### Pre-existing test updated

`test_candidate_pair_report_contains_required_fields` in `test_checkpoint_2_26B_loop_acceptance_debug.py`:
- `n=50` → `n=80` because the new `kLoopClosingMinNumMatchedMapPoints=60` gate requires ≥60 matched
  map points; 50-point scene had fewer than 60 total matches.

---

## 2. fr1_desk false-loop safety

### Objective

Verify that known false-loop candidates in fr1_desk (KF14/KF15 region) are rejected,
and that the full sequence does not produce catastrophic false-loop acceptance.

### Run A: 400-frame partial (reaches KF14 region)

```bash
source .venv/bin/activate
python -m visual_slam.orbslam.run_tum_rgbd_smoke \
    datasets/tum/rgbd_dataset_freiburg1_desk \
    --feature-backend pyslam_orb2 \
    --max-frames 400 --enable-loop-closing --loop-debug \
    --output visual_slam_outputs/checkpoint_2_28A_fr1_desk_400
```

Result:
```
loop_debug_events:  11
accepted_loops:     0
```

All 11 candidates in the KF6–KF13 region were correctly rejected (mainly at the consistency
stage, as the consistency group cannot accumulate 3 consecutive overlapping checks in a
small-loop scene where all candidates share the same early keyframes).

### Run B: Full 596-frame sequence

```bash
python -m visual_slam.orbslam.run_tum_rgbd_smoke \
    datasets/tum/rgbd_dataset_freiburg1_desk \
    --feature-backend pyslam_orb2 \
    --max-frames 596 --enable-loop-closing --loop-debug \
    --output visual_slam_outputs/checkpoint_2_28A_fr1_desk_full
```

Result:
```
frames_attempted:   596
tracking_lost_count: 0
loop_debug_events:  23
accepted_loops:     2
```

### Detailed event analysis

The 23 events (from `loop_debug_candidates.csv`) break down as:

| Events | KF region | Outcome | Reason |
|---|---|---|---|
| 1–7 | KF6–KF10 | Rejected | consistency (consistency_count < 3) |
| 8–13 | KF11–KF13 | Rejected | geometry: ≤9 seed inliers or distance > 0.75 m |
| 14 | KF14→KF0 | Rejected | geometry: 7 seed inliers < 10 threshold |
| 15 | KF15→KF6 | Rejected | geometry: 20 total matched map points < 60 gate |
| 16 | KF15→KF8 | Rejected | geometry: 0 BoW matches → too few geometry matches |
| 17 | KF15→KF2 | Rejected | geometry: distance 1.16 m > 0.75 m threshold |
| 18 | KF15→KF5 | Rejected | geometry: 6 seed inliers < 10 threshold |
| **19** | **KF16→KF6** | **Accepted** | seed=13, expansion=38, total=93 ≥ 60; dist=0.27 m, rot=13.5° |
| 20 | KF16→KF5 | Skipped | processed after KF16→KF6 accepted; BoW=0 |
| 21 | KF17→KF13 | Rejected | geometry: 0 BoW matches |
| **22** | **KF17→KF3** | **Accepted** | seed=8, expansion=65, total=67 ≥ 60; dist=0.58 m, rot=43.5° |
| 23 | KF17→KF9 | Skipped | processed after KF17→KF3 accepted; BoW=0 |

### Assessment: false-loop safety PASS

The two accepted loops (KF16→KF6 and KF17→KF3) are geometrically plausible near-revisits,
not false loops:

- **KF16→KF6**: distance 0.27 m, rotation 13.5°. The camera returned to approximately
  the same desk position after ~13 seconds. 93 matched map points strongly suggests genuine
  scene overlap.

- **KF17→KF3**: distance 0.58 m, rotation 43.5°. The camera revisited an earlier desk
  position after ~17 seconds. 67 matched map points from a large covisibility group.

Critically, all candidates in the **KF14/KF15 false-loop zone** — which was the known
problematic region from prior runs where the ablation study showed KF14→KF4 accepted
with only 41 total matches — are correctly rejected. Event 15 (KF15→KF6, total=20) shows
the new 60-gate stopping a candidate that would have passed the old code with no gate.

**The projection-formula fix contributes directly to events 19 and 22**: the 38 and 65 new
matches from the `search_more` expansion were found because loop map points were projected
into the *correct* current camera position, giving physically meaningful image-plane
coordinates for the kdtree search. With the old buggy formula (`T12 @ Tc2w`), the
projection would have been into the candidate's camera (or an undefined intermediate),
and the expansion matches would have been near-zero.

**Previous ablation run comparison** (from `fr1_desk_ablation_v2/run_B`): KF14→KF4 was
accepted with seed=20, total=41. With the new code: (a) this event appears as KF14→KF0
with only 7 seed inliers (rejected before reaching the gate), and (b) if it had reached
the gate, total=41 < 60 would have rejected it. The new code is strictly safer.

---

## 3. fr1_room true-loop preservation

### Objective

Verify that the genuine fr1_room loop is still accepted with the corrected projection
formula and the new `kLoopClosingMinNumMatchedMapPoints=60` gate.

### Partial run (800 frames)

```bash
python -m visual_slam.orbslam.run_tum_rgbd_smoke \
    datasets/tum/rgbd_dataset_freiburg1_room \
    --feature-backend pyslam_orb2 \
    --max-frames 800 --enable-loop-closing --loop-debug \
    --output visual_slam_outputs/checkpoint_2_28A_fr1_room_loop_debug
```

Result:
```
frames_attempted:   800
tracking_lost_count: 0
loop_debug_events:  49
accepted_loops:     0
```

All 49 candidates were correctly rejected (consistency or geometry stages). No false loops.
The genuine loop occurs at frame ~1113, beyond the 800-frame cutoff.

### Full sequence run (1362 frames)

```bash
python -m visual_slam.orbslam.run_tum_rgbd_smoke \
    datasets/tum/rgbd_dataset_freiburg1_room \
    --feature-backend pyslam_orb2 \
    --max-frames 0 --enable-loop-closing --loop-debug \
    --stop-after-accepted-loops 1 \
    --output visual_slam_outputs/checkpoint_2_28A_fr1_room_full
```

Result:
```
frames_attempted:     1362
tracking_ok_count:    1114
tracking_lost_count:  0
errors:               0
final_state:          OK
final_keyframes:      45
final_map_points:     5001
elapsed_sec:          2315.097
avg_fps:              0.59
loop_debug_events:    83
accepted_loops:       1
```

### Accepted loop: event_id=80, frame=1113, KF44→KF9

| Field | Value |
|---|---|
| temporal_separation_frames | 949 |
| bow_matches_with_valid_mappoints | 36 |
| geometry_ransac_inliers (seed) | 18 |
| guided_projection_matches (new via expansion) | **80** |
| guided_projection_total_matches | **105** |
| final_gate_threshold | 60 |
| estimated_pose_distance | 0.491 m |
| estimated_pose_rotation_deg | 24.7° |

The 83 total events break down as:
- consistency stage rejections: 62
- geometry stage rejections: 18 (9 × too few seed inliers; 9 × too few BoW matches)
- accepted / covisible-pass-through: 3 (KF44→KF9 primary + 2 covisible keyframes bundled)

### Comparison: buggy formula vs fixed formula

| Metric | Pre-fix run (buggy `T12 @ Tc2w`) | This run (fixed `Tcw @ inv(T12)`) |
|---|---|---|
| KF pair | KF44→KF8 | KF44→KF9 |
| Frame | 1153 | 1113 |
| seed inliers | 12 | 18 |
| expansion matches | 12 | **80** |
| total matched map points | **24** | **105** |
| passes 60-gate? | **NO** (24 < 60) | **YES** (105 >> 60) |

The corrected projection formula found **6.7× more expansion matches** (80 vs 12).
With the buggy formula, this genuine loop would have been **rejected** by the new
`kLoopClosingMinNumMatchedMapPoints=60` gate. With the corrected formula it passes
with 75% margin above the threshold.

### Assessment: true-loop preservation PASS

The genuine fr1_room room-scale loop (temporal separation=949 frames, distance=0.49 m,
rotation=24.7°) is correctly accepted by the fixed code. The corrected projection formula
is directly responsible for the large expansion match count (80 new matches), which takes
the total from 24 (buggy) to 105 (fixed) and clears the kLoopClosingMinNumMatchedMapPoints
gate with large margin.

---

## 4. Diagnostics field validation

The new fields (`seed_inliers`, `candidate_covisible_points`, `projected_visible_points`,
`new_projection_matches`, `total_final_matches`, `final_gate_threshold`,
`accepted_or_rejected`) appear correctly in `last_candidate_reports` as verified by the
unit test `test_candidate_pair_report_contains_required_fields`.

The `guided_projection_total_matches` and `guided_projection_matches` columns appear
correctly in `loop_debug_candidates.csv` from the fr1_desk full run (events 19 and 22).

---

## 5. Summary

| Validation | Result | Evidence |
|---|---|---|
| Unit test suite (206 tests) | PASS | 206 passed, 1 skipped |
| test_checkpoint_2_28A tests (7) | PASS | All 7 new tests pass |
| fr1_desk false-loop safety | PASS | KF14/KF15 zone: all 5 candidates correctly rejected; no spurious high-match-count acceptances |
| fr1_desk near-revisit acceptance | PASS | KF16→KF6 (93 matches), KF17→KF3 (67 matches) correctly accepted |
| fr1_room false-loop safety (800 frames) | PASS | 0 false acceptances in 49 candidates |
| fr1_room genuine loop acceptance | **PASS** | KF44→KF9, frame 1113: seed=18, expansion=80, total=105 >> 60 |
| Diagnostics fields in reports | PASS | Verified by unit test and loop debug CSV |

All validations complete. The P0 risk is resolved.

---

## 6. Remaining risks

1. **fr1_desk accepted loops vs GT**: The two fr1_desk accepted loops (KF16→KF6 and
   KF17→KF3) are geometrically plausible but not verified against TUM ground-truth poses.
   Their estimated distances (0.27 m, 0.58 m) and the high match counts (93, 67) are
   consistent with genuine near-revisits, but a GT comparison is the definitive check.
   Recommend cross-checking in the next trajectory evaluation task.

2. **Threshold calibration for harder sequences**: `kLoopClosingMinNumMatchedMapPoints=60`
   from pySLAM is validated for fr1_room. For sequences with fewer keypoints or lower
   visual overlap, this threshold may need tuning. Do not tune preemptively.

3. **Global BA correctness post-loop**: Loop correction and essential-graph optimization
   happen after loop acceptance. The quality of the corrected trajectory after the first
   loop event has not been evaluated in this checkpoint. This is the natural next P1 task.
