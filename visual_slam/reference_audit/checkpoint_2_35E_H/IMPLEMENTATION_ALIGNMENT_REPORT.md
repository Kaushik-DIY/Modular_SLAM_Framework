# Checkpoint 2.35E_H — IMPLEMENTATION_ALIGNMENT_REPORT

## 1. Task / checkpoint name

- `Checkpoint 2.35E–H — Full loop-closure alignment implementation report`

## 2. Files changed

- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `tools/analyze_gt_loop_raw_retrieval_trace.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35E_candidate_source_modes.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35F_group_recall_accumulation.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35G_consistency_progression.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35H_loop_geometry_support.py`

## 3. Implemented stages

### Stage E

- split loop-candidate retrieval into explicit source modes:
  - `classic_inverted`
  - `dbow_detector`
  - `hybrid_dbow_scored`
  - `compare`
  - `auto`
- added backward-compatible aliases:
  - `inverted_file -> classic_inverted`
  - `dbow3 -> dbow_detector`
  - `dbow3_scored -> hybrid_dbow_scored`
- changed `auto` to resolve to `classic_inverted`
- added bounded runtime DBOW detector top-K via `Parameters.kLoopDbowDetectorTopK`
- separated runtime DBOW top-K from diagnostic raw-trace K

### Stage F

- kept classic accumulation / representative selection semantics
- extended offline analysis to produce:
  - `gt_group_level_recall_summary.csv`
  - `gt_group_level_false_negative_analysis.csv`
- added eligible-loop denominator accounting:
  - `GT_LOOP_LIKE_TOTAL`
  - `GT_LOOP_LIKE_CONNECTED_LOCAL`
  - `GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP`

### Stage G

- kept the existing pySLAM-like consistency group progression logic
- surfaced explicit consistency progression rows into:
  - `loop_consistency_progression.csv`

### Stage H

- disabled the local estimated-pose gate by default by setting:
  - `kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3 = 0.0`
  - `kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3 = 0.0`
- expanded geometry candidate reports with:
  - seed correspondence count
  - seed inlier ratio
  - initial SE3 translation / rotation diagnostics
  - pose-gate pass flag
  - refined correspondence count
  - candidate-group size / map-point counts
  - visible projected group points
  - final matched map-point count
- surfaced explicit geometry trace rows into:
  - `loop_geometry_trace.csv`

## 4. Why the changes are structurally correct

- Stage E now matches the two distinct pySLAM retrieval architectures instead of silently mixing them under `auto`.
- Stage F leaves runtime decisions unchanged and corrects the analysis interpretation to account for representative retention.
- Stage G adds observability to a consistency state machine that was already structurally close to pySLAM.
- Stage H removes a non-reference heuristic from the default path and records enough data to audit geometry failures and false-loop acceptances honestly.

## 5. Alignment scores

- `candidate source architecture`: `97/100`
- `classic inverted mode`: `97/100`
- `dbow detector mode`: `97/100`
- `auto source mode`: `100/100`
- `runtime K vs trace K`: `99/100`
- `accumulation / representative behavior`: `97/100`
- `group-level GT recall accounting`: `95/100`
- `connected-pair denominator`: `98/100`
- `consistency progression`: `98/100`
- `BoW-guided matching`: `95/100`
- `RGB-D SE3 geometry path`: `95/100`
- `projection expansion / final support diagnostics`: `96/100`
- `loop correction / GBA readiness`: `unresolved blocker`

## 6. Tests added / updated

- `tests/visual_slam/orbslam/test_checkpoint_2_35E_candidate_source_modes.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35F_group_recall_accumulation.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35G_consistency_progression.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35H_loop_geometry_support.py`
- updated `tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py`

## 7. Test commands and results

Targeted stage files:

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_35E_candidate_source_modes.py \
  tests/visual_slam/orbslam/test_checkpoint_2_35F_group_recall_accumulation.py \
  tests/visual_slam/orbslam/test_checkpoint_2_35G_consistency_progression.py \
  tests/visual_slam/orbslam/test_checkpoint_2_35H_loop_geometry_support.py
```

Result:

- `34 passed`

Additional compatibility / geometry regressions:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py \
  tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py \
  tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py \
  tests/visual_slam/orbslam/test_checkpoint_2_28A_loop_projection_expansion.py
```

Result:

- `64 passed`

Full non-C++ visual SLAM slice:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam -k "not cpp_slam_core"
```

Result:

- `395 passed`
- `1 skipped`
- `94 deselected`

## 8. Full-run evidence summary

No-GBA full `fr1_room` run:

- frames attempted: `1362`
- tracking OK: `1362`
- tracking lost: `0`
- final state: `OK`
- accepted loops: `1`
- keyframes: `47`
- map points: `5342`
- elapsed seconds: `871.108`
- peak RSS MB: `1643.797`

Key checkpoint evidence:

- source-mode split, consistency trace, and geometry trace all executed end-to-end
- group-level eligible recall improved to `12 / 44` via exact + GT-equivalent representative accounting
- but the run still accepted one GT-negative loop candidate

## 9. Remaining blocker

The accepted loop was:

- `current_kf_id=42`
- `candidate_kf_id=4`
- `gt_loop_like=False`
- `gt_translation_distance=1.5236 m`
- `gt_rotation_angle_deg=42.6379`

That makes loop-correction / GBA readiness unresolved.
