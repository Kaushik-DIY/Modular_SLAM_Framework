# Checkpoint 2.35I — VALIDATION_REPORT

## 1. Commands run

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m visual_slam.orbslam.run_rgbd_slam --help > visual_slam/reference_audit/checkpoint_2_35I/RUNNER_HELP.txt
python tools/analyze_gt_loop_raw_retrieval_trace.py --help
```
Full no-GBA comparison runs:
```bash
/usr/bin/time -v python -m visual_slam.orbslam.run_rgbd_slam "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" --dataset-type tum_rgbd --camera-profile auto --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_classic_inverted_loop_no_gba" --feature-backend pyslam_orb2 --enable-loop-closing --disable-global-ba --no-map-export --profile-runtime --profile-memory --profile-keyframes --profile-local-map --loop-debug --loop-candidate-source classic_inverted --loop-retrieval-trace --loop-retrieval-trace-raw-k 100 --memory-profile-every 100 --runtime-profile-every 100 --memory-limit-gb 12 --print-every 100 2>&1 | tee "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_classic_inverted_loop_no_gba/full_run_console.log"
/usr/bin/time -v python -m visual_slam.orbslam.run_rgbd_slam "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" --dataset-type tum_rgbd --camera-profile auto --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_dbow_detector_loop_no_gba" --feature-backend pyslam_orb2 --enable-loop-closing --disable-global-ba --no-map-export --profile-runtime --profile-memory --profile-keyframes --profile-local-map --loop-debug --loop-candidate-source dbow_detector --loop-retrieval-trace --loop-retrieval-trace-raw-k 100 --memory-profile-every 100 --runtime-profile-every 100 --memory-limit-gb 12 --print-every 100 2>&1 | tee "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_dbow_detector_loop_no_gba/full_run_console.log"
/usr/bin/time -v python -m visual_slam.orbslam.run_rgbd_slam "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" --dataset-type tum_rgbd --camera-profile auto --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_hybrid_dbow_scored_loop_no_gba" --feature-backend pyslam_orb2 --enable-loop-closing --disable-global-ba --no-map-export --profile-runtime --profile-memory --profile-keyframes --profile-local-map --loop-debug --loop-candidate-source hybrid_dbow_scored --loop-retrieval-trace --loop-retrieval-trace-raw-k 100 --memory-profile-every 100 --runtime-profile-every 100 --memory-limit-gb 12 --print-every 100 2>&1 | tee "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_hybrid_dbow_scored_loop_no_gba/full_run_console.log"
```
Shared GT analyzer command form:
```bash
python tools/analyze_gt_loop_raw_retrieval_trace.py --trace-dir "$TRACE_DIR" --gt-loop-classified "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv" --output "$OUTPUT_DIR"
```
False-loop search / audit commands used:
```bash
find "$HOME/slam_ws/visual_slam_outputs" -type f \( -name "loop_candidate_oracle.csv" -o -name "loop_geometry_trace.csv" -o -name "loop_gt_positive_trace.csv" -o -name "run_summary.json" \) | sort
```

## 2. Output folders

- `classic_inverted` run: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_classic_inverted_loop_no_gba`
- `classic_inverted` analyzer: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35I/gt_retrieval_stage_analysis_classic_inverted`
- `dbow_detector` run: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_dbow_detector_loop_no_gba`
- `dbow_detector` analyzer: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35I/gt_retrieval_stage_analysis_dbow_detector`
- `hybrid_dbow_scored` run: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35I/fr1_room_full_hybrid_dbow_scored_loop_no_gba`
- `hybrid_dbow_scored` analyzer: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35I/gt_retrieval_stage_analysis_hybrid_dbow_scored`
- false-loop audit source run: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35E_H/fr1_room_full_loop_no_gba_best_mode`

## 3. Candidate-source comparison summary

- `classic_inverted`: full tracking survived, `1` accepted loop, but the accepted pair `KF16 <-> KF1` was GT-negative. Dominant failure stage stayed `FAILED_COMMON_WORD_FILTER`.
- `dbow_detector`: full tracking survived, `0` accepted loops, and no accepted false loops. Dominant analyzer failure shifted to `MISSING_FROM_INVERTED_WORD_SET`, reflecting that the classic inverted-word funnel no longer applies cleanly.
- `hybrid_dbow_scored`: full tracking survived, `1` accepted loop, but the accepted pair `KF42 <-> KF2` was GT-negative. Dominant analyzer failure was also `MISSING_FROM_INVERTED_WORD_SET`, and the mode remained architecturally mixed.

## 4. Best mode by GT recall
- `classic_inverted`. No mode produced a GT-valid accepted loop, but `classic_inverted` had the strongest GT-like funnel survival: `GROUP_RECALLED_TOTAL=12`, `PASSED_CONSISTENCY=5`, `PASSED_GEOMETRY=2`.

## 5. Best mode by false-loop safety
- `dbow_detector`. It accepted `0` loops, so this comparison run contained no accepted false loops in that mode.

## 6. Whether any mode is clearly unacceptable
- `hybrid_dbow_scored` is clearly unacceptable as the next primary implementation target because it still accepted a GT-negative loop while mixing classic and DBOW semantics.
- `classic_inverted` is unacceptable as benchmark-ready behavior today because it also accepted a GT-negative loop, even though it remains the strongest parity baseline.
- `dbow_detector` is unacceptable as the sole benchmark mode today because it produced no GT-valid accepted loops and no retained GT-equivalent group recall.

## 7. False loop KF42-KF4 audit summary

- The false loop came from `checkpoint_2_35E_H/fr1_room_full_loop_no_gba_best_mode` and was accepted with `gt_translation_distance=1.5236 m` and `gt_rotation_angle_deg=42.6379`.
- It passed every runtime gate: raw DBOW visibility, shared-word filtering, minScore, accumulation, consistency, seed SE3, guided projection expansion, and final support.
- The strongest single counterfactual is the pre-H pose-distance gate: using the old thresholds `0.75 m` and `45 deg`, the logged estimated pose distance `1.6757 m` suggests the loop would likely have been rejected by distance even though its later support exploded to `107` final matched map points.
- The audit also shows the current diagnostics are missing residual distributions, spatial coverage, and per-keyframe support attribution, which prevents a principled pySLAM-aligned false-loop rejection decision today.

## 8. Exact next implementation recommendation
- Use `classic_inverted` as the primary next-checkpoint path, keep `dbow_detector` as the safety comparison mode, and make the next implementation checkpoint a `classic_inverted` common-word parity plus false-loop geometry/support audit/correction checkpoint. Do not continue on `hybrid_dbow_scored` as the default path.

## 9. Files generated

- `visual_slam/reference_audit/checkpoint_2_35I/RUNNER_HELP.txt`
- `visual_slam/reference_audit/checkpoint_2_35I/CANDIDATE_SOURCE_MODE_COMPARISON_REPORT.md`
- `visual_slam/reference_audit/checkpoint_2_35I/FALSE_LOOP_KF42_KF4_GEOMETRY_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_35I/FALSE_LOOP_KF42_KF4_GEOMETRY_AUDIT.json`
- `visual_slam/reference_audit/checkpoint_2_35I/VALIDATION_REPORT.md`
- `visual_slam_outputs/checkpoint_2_35I/fr1_room_full_classic_inverted_loop_no_gba`
- `visual_slam_outputs/checkpoint_2_35I/fr1_room_full_dbow_detector_loop_no_gba`
- `visual_slam_outputs/checkpoint_2_35I/fr1_room_full_hybrid_dbow_scored_loop_no_gba`
- `visual_slam_outputs/checkpoint_2_35I/gt_retrieval_stage_analysis_classic_inverted`
- `visual_slam_outputs/checkpoint_2_35I/gt_retrieval_stage_analysis_dbow_detector`
- `visual_slam_outputs/checkpoint_2_35I/gt_retrieval_stage_analysis_hybrid_dbow_scored`

## 10. git status --short
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
?? AGENTS.md
?? CODEX_CHECKPOINT_2_31A_FIX_RUNTIME_MEMORY_PROFILE.md
?? CODEX_CHECKPOINT_2_31A_PYSLAM_MEMORY_POLICY.md
?? CODEX_CHECKPOINT_2_32A_PYSLAM_LOCAL_MAP_ALIGNMENT.md
?? CODEX_CHECKPOINT_2_33A_KF_LOCAL_MAPPING_SCHEDULING.md
?? CODEX_CHECKPOINT_2_35A_LOOP_CANDIDATE_RETRIEVAL_ORACLE.md
?? CODEX_CHECKPOINT_2_35B_GT_LOOP_RECALL_ANALYSIS.md
?? CODEX_CHECKPOINT_2_35C_DIAGNOSTIC_ONLY_GT_LOOP_TRACE.md
?? CODEX_CHECKPOINT_2_35D_FINAL_RAW_RETRIEVAL_TRACE.md
?? CODEX_CHECKPOINT_2_35E_H_FULL_LOOP_CLOSURE_ALIGNMENT.md
?? effective_run_config.json
?? tests/visual_slam/orbslam/test_checkpoint_2_31A_fix_memory_runtime.py
?? tests/visual_slam/orbslam/test_checkpoint_2_31A_memory_policy.py
?? tests/visual_slam/orbslam/test_checkpoint_2_32A_local_map_pyslam_alignment.py
?? tests/visual_slam/orbslam/test_checkpoint_2_33A_kf_local_mapping_schedule.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35B_gt_loop_oracle.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35C_diag_gt_loop_trace.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35D_raw_retrieval_trace.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35E_candidate_source_modes.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35F_group_recall_accumulation.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35G_consistency_progression.py
?? tests/visual_slam/orbslam/test_checkpoint_2_35H_loop_geometry_support.py
?? third_party/cpp_slam_core.zip
?? third_party/cpp_slam_core/build_asan/
?? tools/analyze_gt_loop_raw_retrieval_trace.py
?? tools/analyze_gt_loop_recall.py
?? tools/analyze_gt_loop_retrieval_trace.py
?? visual_slam/orbslam/slam/loop_oracle.py
?? visual_slam/orbslam/slam/runtime_profiler.py
?? visual_slam/reference_audit/checkpoint_2_31A/
?? visual_slam/reference_audit/checkpoint_2_31A_fix/
?? visual_slam/reference_audit/checkpoint_2_32A.zip
?? visual_slam/reference_audit/checkpoint_2_32A/
?? visual_slam/reference_audit/checkpoint_2_33A.zip
?? visual_slam/reference_audit/checkpoint_2_33A/
?? visual_slam/reference_audit/checkpoint_2_34A.zip
?? visual_slam/reference_audit/checkpoint_2_34A/
?? visual_slam/reference_audit/checkpoint_2_35A.zip
?? visual_slam/reference_audit/checkpoint_2_35A/
?? visual_slam/reference_audit/checkpoint_2_35B.zip
?? visual_slam/reference_audit/checkpoint_2_35B/
?? visual_slam/reference_audit/checkpoint_2_35C_diag.zip
?? visual_slam/reference_audit/checkpoint_2_35C_diag/
?? visual_slam/reference_audit/checkpoint_2_35D.zip
?? visual_slam/reference_audit/checkpoint_2_35D/
?? visual_slam/reference_audit/checkpoint_2_35E_H.zip
?? visual_slam/reference_audit/checkpoint_2_35E_H/
?? visual_slam/reference_audit/checkpoint_2_35I/
```

## 11. git diff --stat
```text
third_party/g2opy                              |    0
 visual_slam/orbslam/run_rgbd_slam.py           | 1703 +++++++++++++++++++-----
 visual_slam/orbslam/run_tum_rgbd_smoke.py      |  340 +----
 visual_slam/orbslam/slam/config_parameters.py  |   35 +-
 visual_slam/orbslam/slam/essential_graph.py    |    4 +-
 visual_slam/orbslam/slam/frame.py              |   48 +
 visual_slam/orbslam/slam/geometry_matchers.py  |   70 +-
 visual_slam/orbslam/slam/keyframe.py           |   20 +
 visual_slam/orbslam/slam/keyframe_database.py  |  818 +++++++++++-
 visual_slam/orbslam/slam/local_mapping.py      |  241 +++-
 visual_slam/orbslam/slam/local_mapping_core.py |    4 +-
 visual_slam/orbslam/slam/loop_closing.py       |  849 ++++++++++--
 visual_slam/orbslam/slam/loop_detector.py      |   63 +-
 visual_slam/orbslam/slam/map.py                |  165 ++-
 visual_slam/orbslam/slam/map_point.py          |   19 +
 visual_slam/orbslam/slam/slam.py               |    1 +
 visual_slam/orbslam/slam/tracking.py           | 1061 ++++++++++-----
 17 files changed, 4337 insertions(+), 1104 deletions(-)
```
