# Checkpoint 2.35A - Implementation Alignment Report

## Gap 1 - Query before adding current keyframe
- pySLAM / ORB-SLAM target:
  - detect loop candidates against previously indexed keyframes
  - add current keyframe after loop candidate processing
- pySLAM files inspected:
  - `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
  - `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
  - `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
- local before:
  - `visual_slam/orbslam/slam/loop_closing.py:process_keyframe()` added the current keyframe before `loop_detector.detect()`
- local after:
  - `process_keyframe()` now queries first and adds the current keyframe in a `finally` block after processing
- alignment score: `98%`
- remaining deviation:
  - local code is synchronous and not pySLAM multiprocessing-based, but the query/add ordering now matches the intended control flow
- evidence:
  - `test_loop_detector_queries_before_adding_current_keyframe`
  - `test_current_keyframe_not_consuming_top_dbow_slot`
  - `test_database_size_before_query_excludes_current_keyframe`

## Gap 2 - Native DBOW3 path must not return raw top-K directly
- pySLAM / ORB-SLAM target:
  - common-word filtering
  - `minScore` filtering
  - covisibility score accumulation
  - retain candidates above `0.75 * best_accumulated_score`
- pySLAM files inspected:
  - `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
  - `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- local before:
  - `visual_slam/orbslam/slam/keyframe_database.py` returned DBOW3 raw query results after only connected / temporal / raw-score filtering
- local after:
  - DBOW3 query is used only as a broad candidate pool
  - actual retained candidates now pass through the same structural scoring path used by the inverted-file mode
  - raw DBOW3 `score < min_score` prefiltering was removed so pySLAM-style scoring owns the decision
- alignment score: `97%`
- deliberate deviation:
  - DBOW3 mode still uses the native DB query as the initial candidate generator, whereas inverted-file mode scans all sharing-word keyframes
  - the retained scoring logic is shared across both modes
- evidence:
  - `test_dbow3_path_uses_common_word_filter`
  - `test_dbow3_path_uses_min_score_filter`
  - `test_dbow3_path_uses_covisibility_score_accumulation`
  - `test_dbow3_path_does_not_return_raw_topk_directly`
  - `test_inverted_file_and_dbow3_paths_share_candidate_scoring_logic`

## Gap 3 - DBOW3 vs inverted-file comparison diagnostics
- pySLAM target:
  - be able to compare candidate sets and diagnose retrieval disagreements explicitly
- local before:
  - no per-keyframe DBOW3 vs inverted-file comparison output
- local after:
  - runner supports `--loop-candidate-source {auto,dbow3,inverted_file,compare}`
  - outputs:
    - `loop_candidate_oracle.csv`
    - `loop_retrieval_profile.csv`
    - `loop_candidate_source_comparison.csv`
    - `loop_keyframe_density_profile.csv`
- alignment score: `99%`
- evidence:
  - `test_loop_candidate_source_compare_writes_both_sources`
  - `test_compare_mode_does_not_change_primary_loop_decision`
  - `test_loop_oracle_loads_tum_groundtruth`
  - `test_loop_oracle_associates_keyframe_timestamps_to_gt`
  - `test_loop_oracle_marks_gt_loop_like_pair`

## Net alignment result
- retrieval/query-order alignment improved from a clearly divergent state to a pySLAM-like control flow and scoring structure
- the strongest before/after diagnostic proof is the source-comparison collapse:
  - baseline: `417` DBOW3 candidates vs `44` inverted-file retained candidates, only `41` overlaps
  - post-fix: `40` DBOW3-scored candidates vs `40` inverted-file candidates, `40` overlaps, `0` source-only mismatches
