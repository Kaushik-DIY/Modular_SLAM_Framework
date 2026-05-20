# Checkpoint 2.35C-DIAG Pre-Change Audit

## 1. Task / checkpoint name
- `Checkpoint 2.35C-DIAG — Diagnostic-only GT-positive loop retrieval tracing and pySLAM comparison`

## 2. Objective
- Understand the current loop-closure retrieval / verification pipeline.
- Compare the current control flow with pySLAM / ORB-SLAM-style logic.
- Split GT-positive losses into proven downstream stages where possible.
- Explicitly state what cannot be proven from current logs.

## 3. Files inspected before changes
- `AGENTS.md`
- `CODEX_CHECKPOINT_2_35C_DIAGNOSTIC_ONLY_GT_LOOP_TRACE.md`
- `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle/*`
- `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/*`
- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_base.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/loop_detector.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/sim3_solver.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`

## 4. Root cause / hypothesis before changes
- The 2.35B diagnostic already showed:
  - `41` GT-loop-like pairs
  - only `5` reaching the retained/oracle path
  - only `1` accepted
- The main unresolved question was whether the `36` broad misses were:
  - absent from raw DBOW
  - present in raw DBOW but removed before retention
  - removed later by consistency / geometry / final support
- Initial hypothesis:
  - existing logs are likely enough to prove downstream failures
  - existing logs are likely insufficient to prove pair-level raw retrieval loss

## 5. Planned changes
- Add an offline trace-only analyzer:
  - `tools/analyze_gt_loop_retrieval_trace.py`
- Add synthetic trace tests:
  - `tests/visual_slam/orbslam/test_checkpoint_2_35C_diag_gt_loop_trace.py`
- Add checkpoint reports:
  - `visual_slam/reference_audit/checkpoint_2_35C_diag/*.md`

## 6. Non-negotiable constraints kept
- No core SLAM behavior changes
- No loop threshold changes
- No candidate-retrieval logic changes
- No SLAM rerun
- GT used only for offline diagnostics
