# Checkpoint 2.35D — VALIDATION_REPORT

## 1. Task / checkpoint name

- `Checkpoint 2.35D — Final raw retrieval trace`

## 2. Tests run and results

### Targeted checkpoint tests

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py
```

Result:

- `15 passed`

### Additional loop retrieval regression coverage

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py \
  tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py
```

Result:

- `31 passed`

### Full non-C++ visual SLAM slice

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam \
  -k "not cpp_slam_core"
```

Result:

- `361 passed`
- `1 skipped`
- `94 deselected`

## 3. Full diagnostic run result

Run directory:

- `visual_slam_outputs/checkpoint_2_35D/fr1_room_full_raw_retrieval_trace`

Run summary:

- frames attempted: `1362`
- tracking OK count: `1362`
- tracking lost count: `0`
- final state: `OK`
- keyframes: `48`
- map points: `4810`
- accepted loops: `1`
- elapsed seconds: `894.960`

## 4. Output files generated

Runtime trace files:

- `loop_raw_dbow_trace.csv`
- `loop_inverted_word_trace.csv`
- `loop_score_filter_trace.csv`
- `loop_accumulation_trace.csv`
- `loop_retained_candidate_trace.csv`
- `loop_gt_positive_trace.csv`

Existing loop-debug outputs retained:

- `loop_candidate_oracle.csv`
- `loop_retrieval_profile.csv`
- `loop_candidate_source_comparison.csv`
- `loop_keyframe_density_profile.csv`

Offline analysis outputs:

- `visual_slam_outputs/checkpoint_2_35D/gt_retrieval_stage_analysis/gt_retrieval_stage_funnel.csv`
- `visual_slam_outputs/checkpoint_2_35D/gt_retrieval_stage_analysis/gt_retrieval_stage_summary.json`
- `visual_slam_outputs/checkpoint_2_35D/gt_retrieval_stage_analysis/gt_retrieval_false_negatives_detailed.csv`
- `visual_slam_outputs/checkpoint_2_35D/gt_retrieval_stage_analysis/gt_retrieval_stage_report.md`

## 5. Retrieval-stage funnel summary

Current trace analysis used meaningful keyframe-gap filter `>10`.

Funnel:

- GT-loop-like total: `44`
- raw DBOW present: `41`
- inverted/shared-word present: `41`
- passed connected/temporal: `36`
- passed common-word: `15`
- passed minScore: `15`
- passed accumulation: `10`
- retained candidate: `4`
- passed consistency: `2`
- passed geometry: `1`
- accepted: `1`

## 6. Dominant failure stage

- `FAILED_COMMON_WORD_FILTER`
- count: `21`

Secondary stages:

- `NOT_RETAINED_AFTER_ACCUMULATION`: `6`
- `FAILED_ACCUMULATION_FILTER`: `5`
- `FAILED_CONNECTED_FILTER`: `5`
- `MISSING_FROM_RAW_DBOW`: `3`
- `FAILED_CONSISTENCY`: `3`

## 7. Next recommended implementation checkpoint

- `Checkpoint 2.35E — common-word filter parity audit/correction against pySLAM`

Recommended scope:

- inspect candidate pool entering common-word evaluation
- verify `max_common_words` / `min_common_words` computation
- verify `common_words > min_common_words` gating parity
- keep thresholds unchanged until parity is confirmed

## 8. No-threshold / no-decision-change confirmation

Confirmed for 2.35D:

- no loop thresholds changed
- no tracking logic changed
- no keyframe scheduling changed
- no LocalMapping logic changed
- no Global BA logic changed
- no C++ code changed
- GT remained diagnostic-only

## 9. Remaining risks

- The current run’s GT-positive population after meaningful-gap filtering is `44`, while the historical 2.35B reference count is `41`.
- This indicates minor current-vs-historical run population drift, so the 2.35D current trace was treated as authoritative and 2.35B was used as historical reference only.
- Common-word filtering now stands out clearly, but accumulation/retention is still a meaningful secondary loss stage and should be revisited after the common-word audit.

## 10. Generated audit files

- `visual_slam/reference_audit/checkpoint_2_35D/PRE_TRACE_PYSLAM_RETRIEVAL_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_35D/INPUT_DISCOVERY_REPORT.md`
- `visual_slam/reference_audit/checkpoint_2_35D/TRACE_IMPLEMENTATION_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_35D/GT_RAW_RETRIEVAL_TRACE_ANALYSIS_REPORT.md`
- `visual_slam/reference_audit/checkpoint_2_35D/VALIDATION_REPORT.md`

## 11. Git status

Observed `git status --short` at end of task:

```text
 m third_party/g2opy
 M visual_slam/orbslam/run_rgbd_slam.py
 M visual_slam/orbslam/run_tum_rgbd_smoke.py
 M visual_slam/orbslam/slam/config_parameters.py
 M visual_slam/orbslam/slam/essential_graph.py
 M visual_slam/orbslam/slam/frame.py
 M visual_slam/orbslam/slam/geometry_matchers.py
 M visual_slam/orbslam/slam/keyframe.py
 M visual_slam/orbslam/slam/keyframe_database.py
 M visual_slam/orbslam/slam/local_mapping.py
 M visual_slam/orbslam/slam/local_mapping_core.py
 M visual_slam/orbslam/slam/loop_closing.py
 M visual_slam/orbslam/slam/loop_detector.py
 M visual_slam/orbslam/slam/map.py
 M visual_slam/orbslam/slam/map_point.py
 M visual_slam/orbslam/slam/slam.py
 M visual_slam/orbslam/slam/tracking.py
 ?? tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py
 ?? tools/analyze_gt_loop_raw_retrieval_trace.py
 ?? visual_slam/reference_audit/checkpoint_2_35D/
 ...plus many pre-existing unrelated workspace files already outside 2.35D scope.
```

## 12. Git diff stat

Observed `git diff --stat` at end of task:

```text
 third_party/g2opy                              |    0
 visual_slam/orbslam/run_rgbd_slam.py           | 1617 +++++++++++++++++++-----
 visual_slam/orbslam/run_tum_rgbd_smoke.py      |  340 +----
 visual_slam/orbslam/slam/config_parameters.py  |   30 +
 visual_slam/orbslam/slam/essential_graph.py    |    4 +-
 visual_slam/orbslam/slam/frame.py              |   48 +
 visual_slam/orbslam/slam/geometry_matchers.py  |   70 +-
 visual_slam/orbslam/slam/keyframe.py           |   20 +
 visual_slam/orbslam/slam/keyframe_database.py  |  641 +++++++++-
 visual_slam/orbslam/slam/local_mapping.py      |  241 +++-
 visual_slam/orbslam/slam/local_mapping_core.py |    4 +-
 visual_slam/orbslam/slam/loop_closing.py       |  756 +++++++++--
 visual_slam/orbslam/slam/loop_detector.py      |   63 +-
 visual_slam/orbslam/slam/map.py                |  165 ++-
 visual_slam/orbslam/slam/map_point.py          |   19 +
 visual_slam/orbslam/slam/slam.py               |    1 +
 visual_slam/orbslam/slam/tracking.py           | 1061 +++++++++++-----
 17 files changed, 3974 insertions(+), 1106 deletions(-)
```

Important note:

- This diff stat is for the whole current worktree, not just 2.35D.
- The repository already contained substantial unrelated modifications before this checkpoint.
