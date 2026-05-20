# pySLAM vs Local Loop Pipeline Comparison

## 1. pySLAM files / functions inspected
- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
  - `KeyFrameDatabaseDBow.detect_loop_candidates`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_base.py`
  - `compute_reference_similarity_score`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
  - `LoopGroupConsistencyChecker.check_candidates`
  - `LoopGeometryChecker.check_candidates`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`

## 2. Local files / functions inspected
- `visual_slam/orbslam/slam/keyframe_database.py`
  - `detect_loop_candidates`
  - `_detect_loop_candidates_dbow3_raw`
  - `_score_candidate_pool`
- `visual_slam/orbslam/slam/loop_detector.py`
  - `compute_reference_similarity_score`
  - `detect`
- `visual_slam/orbslam/slam/loop_closing.py`
  - `process_keyframe`
  - `LoopGroupConsistencyChecker.check_candidates`
  - diagnostic row builders
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/sim3_solver.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`

## 3. pySLAM loop retrieval pipeline
- DB query / inverted file candidate collection:
  - pySLAM supports DBOW-backed detector flows in `loop_detector_dbow2.py` / `loop_detector_dbow3.py`.
  - In the classic inverted-file path, `KeyFrameDatabaseDBow.detect_loop_candidates` starts from shared-word keyframes.
- Connected / temporal filtering:
  - pySLAM DBOW detector rejects temporally-close and connected keyframes before candidate output.
  - The inverted-file keyframe database rejects connected keyframes; temporal filtering is mainly handled in detector-level query logic.
- Common-word filter:
  - pySLAM keeps candidates with `num_loop_words > int(0.8 * max_common_words)`.
- minScore filter:
  - pySLAM computes `min_score` from the lowest similarity to a connected covisible keyframe.
  - Candidates must satisfy `score >= min_score`.
- Score accumulation:
  - pySLAM accumulates score over the candidate and up to 10 best covisible neighbors that also share enough words.
  - It retains candidates with `acc_score > 0.75 * best_acc_score`.
- Consistency check:
  - pySLAM expands each candidate to its covisibility group and requires consistency across multiple detections.
- Geometry verification:
  - pySLAM uses Sim3-based verification with keypoint matches and RANSAC.
- Projection expansion:
  - pySLAM performs `ProjectionMatcher.search_by_sim3`, then Sim3 optimization, then `search_more_map_points_by_projection`.
- Acceptance:
  - pySLAM accepts only if post-expansion matched map points exceed the minimum support threshold.

## 4. Local loop retrieval pipeline
- DB query / inverted file candidate collection:
  - Local code supports:
    - raw DBOW3 query via `_detect_loop_candidates_dbow3_raw`
    - inverted-file scoring via `_detect_loop_candidates_inverted_scored`
    - source comparison between the two retained candidate sets
- Connected / temporal filtering:
  - Local code applies temporal and connected filtering inside `_score_candidate_pool`.
  - For raw DBOW3, the raw query is also pre-filtered by connected / temporal conditions.
- Common-word filter:
  - Same gating concept as pySLAM: `common_words > int(0.8 * max_common_words)`.
- minScore filter:
  - Same reference idea as pySLAM: local `LoopDetector.compute_reference_similarity_score` uses the minimum score against connected covisible keyframes.
- Score accumulation:
  - Same broad structure: candidate score plus score of up to 10 best covisible neighbors sharing enough words.
  - Same `0.75 * best_acc_score` retention rule.
- Consistency check:
  - Same ORB-SLAM-style covisibility-group persistence concept.
- Geometry verification:
  - Local RGB-D path uses fixed-scale SE3 / rigid alignment style verification instead of full monocular Sim3 parity.
- Projection expansion:
  - Local code logs guided projection matches and final matched-map-point support.
- Acceptance:
  - Local code applies a final matched-map-point threshold and only then marks the candidate accepted.

## 5. Observed implementation differences
- pySLAM detector path:
  - detector returns candidate IDs directly from the detector backend.
- local path:
  - local keyframe database reconstructs two retained candidate sets:
    - `dbow3_scored`
    - `inverted_file`
  - local diagnostics also compare those retained sets.
- pySLAM geometry:
  - Sim3-centric loop verification and `search_by_sim3`.
- local RGB-D geometry:
  - scale-fixed / SE3-oriented verification for RGB-D benchmark scope.
- local diagnostics:
  - local code logs retrieval counts, retained candidate lists, oracle rows, and density rows.
  - pySLAM reference code does not provide the same thesis-specific CSV trace structure.

## 6. Differences that are likely harmless
- Extra local diagnostics and source-comparison bookkeeping
- Structural Python refactors around result containers
- RGB-D-specific scale-fixed verification instead of full monocular Sim3 parity, for this benchmark scope

## 7. Differences that may cause missed GT loops
- Local source selection behavior prefers `dbow3_scored` whenever raw DBOW3 results exist.
- Current local logs expose only:
  - raw candidate counts
  - retained candidate identities
  - downstream oracle rows
- They do not expose:
  - raw DBOW candidate identities and ranks for each current keyframe
  - raw shared-word candidate identities before common-word / minScore / accumulation filtering
- Because of that, true GT misses cannot be localized to:
  - absent from raw DBOW
  - common-word removal
  - minScore removal
  - accumulated-score removal

## 8. Diagnostic fields needed to prove each hypothesis
- To prove raw DBOW absence:
  - raw DBOW candidate ID list per current keyframe
  - raw DBOW rank per candidate
  - raw DBOW score per candidate
- To prove common-word filtering loss:
  - per-candidate shared-word counts before retention
  - threshold used for `min_common_words`
- To prove minScore filtering loss:
  - per-candidate score versus current `min_score`
- To prove accumulated-score retention loss:
  - per-candidate accumulated score
  - best accumulated score
  - retained / rejected marker before consistency
- To prove source-selection loss:
  - if both source modes are computed, per-pair retained / dropped markers for each source

## Bottom line
- The local retrieval pipeline is structurally close to pySLAM at the stage-concept level.
- The main blocker is not an obviously missing stage in the local pipeline.
- The main blocker is missing per-pair raw retrieval trace data, which prevents proving where most GT-positive misses disappear before the retained candidate list.
