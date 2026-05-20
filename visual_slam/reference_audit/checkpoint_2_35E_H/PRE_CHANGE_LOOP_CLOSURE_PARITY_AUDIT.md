# Checkpoint 2.35E_H — PRE_CHANGE_LOOP_CLOSURE_PARITY_AUDIT

## 1. Task / checkpoint name

- `Checkpoint 2.35E–H — Pre-change loop-closure parity audit`

## 2. Files inspected

Local files:

- `visual_slam/orbslam/slam/keyframe_database.py`
- `visual_slam/orbslam/slam/loop_detector.py`
- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/bow_matcher.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/sim3_solver.py`
- `visual_slam/orbslam/slam/essential_graph.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `tools/analyze_gt_loop_raw_retrieval_trace.py`
- `tools/analyze_gt_loop_recall.py`

pySLAM reference files:

- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
- `third_party/pyslam_reference/pyslam/slam/optimizer_g2o.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`

## 3. pySLAM / ORB-SLAM retrieval pipeline

Classic inverted-file retrieval from `pyslam/loop_closing/keyframe_database.py::detect_loop_candidates()`:

1. iterate query keyframe BoW words
2. gather keyframes sharing at least one word through the inverted file
3. exclude connected keyframes while marking per-query state with `loop_query_id`
4. count `num_loop_words` per candidate
5. compute `max_common_words`
6. require `num_loop_words > int(0.8 * max_common_words)`
7. compute vocabulary score `voc.score(query, candidate)`
8. require `score >= min_score`
9. accumulate over `candidate.get_best_covisible_keyframes(10)`
10. only accumulate neighbors from the same query with `neighbor.loop_query_id == query.id`
11. also require `neighbor.num_loop_words > min_common_words`
12. select the best-scoring keyframe in each covisibility group as the representative
13. retain representatives with `acc_score > 0.75 * best_acc_score`
14. deduplicate representatives
15. hand retained representatives to consistency

DBOW detector behavior from `loop_detector_dbow2.py` and `loop_detector_dbow3.py`:

1. compute `min_score` from connected keyframes
2. query the native DBOW database with bounded `kMaxResultsForLoopClosure`
3. exclude temporal-near keyframes with `abs(other_frame_id - frame_id) > kMinDeltaFrameForMeaningfulLoopClosure`
4. exclude connected keyframes
5. require `score >= min_score`
6. return candidates directly
7. do not run classic common-word filtering
8. do not run classic covisibility-score accumulation

Conclusion:

- pySLAM exposes two distinct architectures:
  - classic inverted-file ORB-SLAM retrieval
  - native DBOW detector retrieval
- it does not silently combine them into one default hybrid retrieval stack

## 4. pySLAM / ORB-SLAM DBOW detector pipeline

Reference behavior from:

- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`

Observed control flow:

- bounded top-K query
- connected and temporal filtering
- minScore gating
- direct handoff to loop consistency

Important parity implication:

- if the runtime mode is intended to represent DBOW detector behavior, then applying classic common-word and accumulation stages afterward is a structural deviation

## 5. pySLAM / ORB-SLAM consistency-group progression

Reference behavior from `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py::LoopGroupConsistencyChecker.check_candidates()`:

1. each candidate expands to `candidate.get_connected_keyframes() + candidate`
2. compare that group against previous consistency groups
3. any overlap marks the candidate consistent with that previous group
4. new consistency count is `previous_group.consistency + 1`
5. candidate passes if the count reaches `consistency_threshold`
6. if no overlap exists, create a new group with initial consistency `0`
7. replace `self.consistent_groups` with the current query’s groups after the query

Local pre-change behavior in `visual_slam/orbslam/slam/loop_closing.py::LoopGroupConsistencyChecker.check_candidates()`:

- same connected-group expansion
- same overlap semantics
- same increment rule
- same threshold semantics
- same initial new-group consistency `0`
- same end-of-query replacement model

Pre-change alignment assessment:

- this stage is already structurally close to pySLAM
- likely remaining work is mainly diagnostic surfacing and evidence reporting, not a large algorithm rewrite

## 6. pySLAM / ORB-SLAM Sim3 / RGB-D geometry verification logic

Reference behavior from `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py::LoopGeometryChecker.check_candidates()`:

1. descriptor matching between current keyframe and consistent candidates
2. optional orientation filtering
3. reject if too few initial matches
4. prepare 3D correspondences from matched map points
5. run Sim3 RANSAC solver
6. take inlier correspondences
7. run guided projection matching through `ProjectionMatcher.search_by_sim3`
8. optimize Sim3 with all found correspondences
9. accept a candidate if optimized inliers exceed the geometry minimum and optimization improves
10. build the candidate covisible group with `success_loop_kf.get_covisible_keyframes() + success_loop_kf`
11. gather loop-side map points from that group
12. expand support using `ProjectionMatcher.search_more_map_points_by_projection(...)`
13. require final matched map points >= `kLoopClosingMinNumMatchedMapPoints`

Local pre-change behavior in `visual_slam/orbslam/slam/loop_closing.py::LoopGeometryChecker.check_candidates()`:

1. BoW-guided matcher is used when available, otherwise descriptor fallback
2. orientation filtering is applied
3. require minimum loop geometry matches
4. convert matched map-point correspondences to 3D
5. run scale-fixed SE3 RANSAC via `estimate_scale_fixed_sim3(...)`
6. require seed inliers >= `kLoopClosingSE3GuidedMinSeedInliers`
7. apply a local estimated-pose distance / rotation gate before guided refinement when seed support is weak
8. perform custom guided projection refinement in candidate image space
9. rerun scale-fixed SE3 estimation on refined correspondences
10. build loop-side expansion set from `candidate.get_covisible_keyframes() + candidate`
11. expand support with `ProjectionMatcher.search_more_map_points_by_projection(...)`
12. require final matched map points >= `kLoopClosingMinNumMatchedMapPoints`

Key confirmed deviations:

- local path is RGB-D SE3 / fixed-scale only, not full pySLAM Sim3
- local path adds a non-pySLAM estimated-pose gate before guided refinement
- local guided refinement is custom and does not directly reuse pySLAM `search_by_sim3`
- local path performs a second fixed-scale estimation rather than pySLAM `optimize_sim3`

Because monocular Sim3 parity is out of scope for this RGB-D checkpoint, the acceptable target is:

- pySLAM-aligned control flow and support-building semantics
- RGB-D scale-fixed SE3 implementation where Sim3 scale remains fixed to 1

## 7. pySLAM / ORB-SLAM projection expansion / fusion logic

Reference behavior:

- support expansion uses the loop candidate covisible group
- final gate checks total matched map-point support after projection expansion
- loop correction then fuses loop-side map points into corrected current-side keyframes

Local pre-change behavior:

- geometry expansion already uses `candidate.get_covisible_keyframes() + candidate`
- correction-side fusion uses bounded best-covisible keyframe neighborhoods
- loop map-point collection during correction uses `get_best_covisible_keyframes(Parameters.kNumBestCovisibilityKeyFrames)`

Potential parity gap:

- geometry-stage support expansion is close
- correction/fusion neighborhood sizing may still be more bounded than pySLAM’s broader covisible group behavior
- this matters after loop acceptance, but it is not the first retrieval bottleneck

## 8. Local current implementation summary by stage

### Stage E area: candidate-source architecture

Current local behavior:

- `LoopDetector` always computes a classic ORB-SLAM-style `min_score`
- `KeyFrameDatabase.detect_loop_candidates()` evaluates:
  - raw DBOW query
  - hybrid DBOW-scored rescoring path
  - classic inverted-file path
- `auto` currently resolves to `dbow3_scored` whenever the DBOW database exists
- source modes currently exposed are:
  - `auto`
  - `dbow3`
  - `inverted_file`
  - `compare`

Confirmed gap:

- `auto -> hybrid DBOW-scored` is explicitly forbidden by the checkpoint
- DBOW detector semantics and classic inverted semantics are not cleanly separated

### Stage F area: accumulation and group-level recall

Current local behavior:

- accumulation logic is already close to pySLAM in `_score_candidate_pool()`
- retained trace is pair-centric
- GT analysis currently emphasizes exact pair retention
- there is no group-level GT-equivalent representative classification yet
- there is no explicit denominator split between:
  - all GT-loop-like pairs
  - connected-local GT pairs
  - eligible-for-loop GT pairs

Confirmed gap:

- retention analysis can overstate failure when the retained representative is GT-equivalent but not the exact pair

### Stage G area: consistency progression

Current local behavior:

- consistency logic is near pySLAM
- candidate debug rows already store overlap and before/after counts
- there is no standalone `loop_consistency_progression.csv`

Confirmed gap:

- diagnostics and checkpoint-required reporting are incomplete even if the core algorithm is close

### Stage H area: geometry and final support

Current local behavior:

- BoW-guided matching exists
- fixed-scale SE3 RANSAC exists
- guided projection refinement exists
- covisible-group projection expansion exists
- final support threshold is `60`
- additional estimated-pose gate exists before guided refinement

Confirmed gaps:

- extra estimated-pose gate is not pySLAM/ORB-SLAM classic logic
- refinement path is locally customized, so trace-level evidence is needed before deciding which parts to keep
- no standalone `loop_geometry_trace.csv` is produced yet

## 9. Confirmed deviations from source and 2.35D reports

1. `auto` source mode currently chooses the hybrid path, not a primary pySLAM/ORB-SLAM mode.
2. DBOW detector runtime query K is not bounded by an explicit runtime parameter; it currently queries `size(database)`.
3. diagnostic raw-K is not separated from runtime-K in the actual query implementation.
4. classic inverted mode exists but uses duplicated raw candidate collection from the inverted file rather than using that mode as the primary `auto` path.
5. group-level GT recall accounting is missing.
6. connected-pair denominator separation is missing from the checkpoint 2.35B/2.35D-style recall analysis.
7. consistency progression CSV output is missing.
8. geometry trace CSV output is missing.
9. an estimated-pose gate exists in local geometry verification without pySLAM parity evidence.

## 10. Root cause / current hypothesis

Primary root cause for the 2.35D funnel:

- the local runtime defaults to a hybrid pipeline that starts from a DBOW-native candidate pool, then applies classic common-word and accumulation logic meant for the inverted-file candidate architecture

Secondary likely causes:

- pair-level recall analysis is overstating some accumulation losses
- geometry verification includes a local pose-prior gate that may reject true loops before projection expansion can recover support

## 11. Planned staged corrections E / F / G / H

### E — Candidate-source parity

- add explicit modes:
  - `classic_inverted`
  - `dbow_detector`
  - `hybrid_dbow_scored`
  - `compare`
  - `auto`
- keep backward-compatible aliases:
  - `inverted_file -> classic_inverted`
  - `dbow3 -> dbow_detector`
  - `dbow3_scored -> hybrid_dbow_scored`
- change `auto` to a documented primary mode, expected target:
  - `auto -> classic_inverted`
- separate:
  - runtime DBOW detector top-K
  - diagnostic raw-trace K

### F — Group-level recall and accumulation

- keep pySLAM accumulation semantics
- add GT-equivalent representative diagnostics
- add eligible-loop denominator reporting

### G — Consistency progression

- keep current near-pySLAM consistency logic unless a real control-flow deviation is found during testing
- add explicit progression trace outputs and tests

### H — Geometry and final support

- audit BoW-guided matching and support growth
- add geometry trace CSV
- remove or justify non-pySLAM pose gating only if evidence shows it is the rejecting deviation
- keep scale-fixed RGB-D SE3 design, but align control-flow semantics as closely as possible

## 12. Expected validation outputs

Unit / targeted outputs expected after implementation:

- `tests/visual_slam/orbslam/test_checkpoint_2_35E_candidate_source_modes.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35F_group_recall_accumulation.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35G_consistency_progression.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_35H_loop_geometry_support.py`

Runtime / analysis outputs expected:

- `loop_consistency_progression.csv`
- `loop_geometry_trace.csv`
- `gt_group_level_recall_summary.csv`
- `gt_group_level_false_negative_analysis.csv`

Full-run validation target:

- `fr1_room` full loop-closing run without GBA first
- GT funnel regenerated from the new runtime traces
- automatic second full run with GBA only if the checkpoint satisfaction criteria pass

## 13. Pre-change readiness decision

Behavior changes should proceed.

Reason:

- the source is not stale
- the pre-change parity gap is concrete and documented
- stage ordering is clear
- the first required fix is architectural rather than threshold tuning
