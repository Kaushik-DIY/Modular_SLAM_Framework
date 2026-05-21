# Checkpoint 2.35E_H — SOURCE_CONSISTENCY_CHECK

## 1. Task / checkpoint name

- `Checkpoint 2.35E–H — Full loop-closure alignment source consistency check`

## 2. Files checked

- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/loop_detector.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `tools/analyze_gt_loop_raw_retrieval_trace.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py`
- `visual_slam/reference_audit/checkpoint_2_35D/TRACE_IMPLEMENTATION_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_35D/VALIDATION_REPORT.md`

## 3. Required 2.35D trace infrastructure presence

Required support from the 2.35E–H task file was checked against current source.

Present:

- CLI flag `--loop-retrieval-trace` in `run_rgbd_slam.py`
- CLI flag `--loop-retrieval-trace-raw-k` in `run_rgbd_slam.py`
- runtime output `loop_raw_dbow_trace.csv`
- runtime output `loop_inverted_word_trace.csv`
- runtime output `loop_score_filter_trace.csv`
- runtime output `loop_accumulation_trace.csv`
- runtime output `loop_retained_candidate_trace.csv`
- runtime output `loop_gt_positive_trace.csv`
- analysis tool `tools/analyze_gt_loop_raw_retrieval_trace.py`
- trace aggregation fields in `LoopDiagnostics`
- trace plumbing from `KeyFrameDatabase.detect_loop_candidates()` through `LoopDetector` into `LoopClosing` and `run_rgbd_slam.py`

## 4. Whether current source matches the reported 2.35D state

Result: `Yes, with one important interpretation note`.

Confirmed matches to the 2.35D reported state:

- The local source still contains the deep retrieval tracing infrastructure reported in checkpoint 2.35D.
- `loop_raw_dbow_trace`, `loop_inverted_word_trace`, `loop_score_filter_trace`, `loop_accumulation_trace`, `loop_retained_candidate_trace`, and `loop_gt_positive_trace` are still produced by the current code path.
- The current retrieval architecture still explains the 2.35D funnel:
  - raw DBOW query first
  - hybrid rescoring / common-word filtering / minScore / accumulation
  - retained candidates handed to consistency
- `LoopDetector.detect()` still computes `min_score` from connected keyframes and delegates candidate retrieval into `KeyFrameDatabase.detect_loop_candidates()`.
- `KeyFrameDatabase.detect_loop_candidates()` still selects `dbow3_scored` when a raw DBOW database is available, even when source mode is `auto`.

Interpretation note:

- The 2.35D task treated the runtime behavior as diagnostic-only and explicitly forbade behavior changes.
- The current code still reflects that diagnostic checkpoint rather than the required 2.35E–H pySLAM/ORB-SLAM-aligned production behavior.
- That is not a stale-source blocker; it is the expected starting point for this checkpoint.

## 5. Current-source findings relevant to 2.35E–H

- `auto` currently resolves to the hybrid DBOW-scored path whenever the DBOW database is available.
- `hybrid_dbow_scored` is not an explicit public source mode yet; it is effectively the default runtime path.
- classic inverted retrieval exists, but it is not the default.
- runtime DBOW query size is currently `size(database)` rather than a bounded runtime top-K parameter.
- diagnostic `loop_retrieval_trace_raw_k` is stored in metadata but does not currently control a separate raw-trace query path.
- stage-F/H outputs required by the new checkpoint are not present yet:
  - `gt_group_level_recall_summary.csv`
  - `gt_group_level_false_negative_analysis.csv`
  - `loop_consistency_progression.csv`
  - `loop_geometry_trace.csv`

## 6. Blockers

No stale-source blocker was found.

There is no evidence that the workspace is missing checkpoint 2.35D tracing support, so implementation may proceed.

## 7. Root cause / starting hypothesis

The source consistency check supports the 2.35D conclusion:

- the dominant early loss is still structurally tied to the hybrid candidate-source path and common-word/retention logic
- the current runtime still mixes:
  - DBOW native retrieval as the initial pool
  - ORB-SLAM classic common-word + accumulation filtering afterward

That hybrid architecture is the first confirmed parity gap to correct in stage E.

## 8. Next recommended action

- Write the full pre-change parity audit against pySLAM/ORB-SLAM.
- Then implement stage E first:
  - explicit source modes
  - `auto` no longer selecting the hybrid path
  - bounded DBOW detector mode separated from classic inverted mode
  - runtime-K and trace-K split
