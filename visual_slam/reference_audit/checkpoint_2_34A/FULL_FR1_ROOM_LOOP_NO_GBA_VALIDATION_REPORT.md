# Checkpoint 2.34A - Full `fr1_room` loop-closure validation, Global BA disabled

## 1. Task/Checkpoint name

Checkpoint 2.34A - full TUM `fr1_room` validation after Checkpoints 2.32A and 2.33A, with loop closing enabled and Global BA disabled.

## 2. Files inspected

- `AGENTS.md`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/reference_audit/checkpoint_2_33A/VALIDATION_REPORT.md`
- `visual_slam_outputs/CODEX_CHECKPOINT_2_33A_KF_LOCAL_MAPPING_SCHEDULING.md`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/effective_run_config.json`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/run_summary.json`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/runtime_profile.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/runtime_profile.json`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/frame_timing.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/frame_log_rgbd_dataset_freiburg1_room.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/keyframe_decision_log.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/local_mapping_schedule_log.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/local_map_profile.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/loop_debug_candidates.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/loop_candidate_pair_reports/*.json`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/memory_profile.csv`
- `visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/full_run_console.log`
- `visual_slam/reference_audit/checkpoint_2_34A/RUNNER_HELP.txt`
- `visual_slam/reference_audit/checkpoint_2_34A/OUTPUT_FILE_LIST.txt`

## 3. pySLAM files inspected

None for this checkpoint. This was a validation/reporting task with no visual-SLAM logic change.

## 4. Root cause found or ruled out

No new tracking/runtime regression was found. Full-sequence tracking completed cleanly. The remaining loop-closure issue is not "no candidates"; it is "candidates exist but are rejected", dominated first by loop-consistency gating and then by geometry/final-match gates.

## 5. Run configuration

Runner flag audit:

- `RUNNER_HELP.txt` confirmed all requested flags existed.
- No equivalent substitutions were needed.

Environment:

- Verified interpreter: `/home/kaushik/slam_ws/.venv/bin/python`
- RAM available at start from `free -h`: about `9.3 GiB`
- Chosen safety limit: `--memory-limit-gb 8`

Run settings:

- Dataset path: `/home/kaushik/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room`
- Output path: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba`
- Feature backend: `pyslam_orb2`
- Loop closing: enabled
- Global BA: disabled
- Map export: disabled
- Profiling: runtime, memory, keyframes, local map, loop debug, loop candidate pair reports
- LocalMapping threading: disabled (sequential)

Exact command used:

```bash
source .venv/bin/activate && /usr/bin/time -v python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_room" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba" \
  --feature-backend pyslam_orb2 \
  --enable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-memory \
  --profile-keyframes \
  --profile-local-map \
  --loop-debug \
  --dump-loop-candidate-reports \
  --memory-profile-every 50 \
  --runtime-profile-every 50 \
  --memory-limit-gb 8 \
  --print-every 100 \
  2>&1 | tee "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_34A/fr1_room_full_loop_no_gba/full_run_console.log"
```

## 6. Run summary from `run_summary.json`

- `frames_attempted`: `1362`
- `tracking_ok_count`: `1362`
- `tracking_lost_count`: `0`
- `final_state`: `OK`
- `final_keyframes`: `46`
- `final_map_points`: `4649`
- `trajectory_poses`: `1362`
- `elapsed_sec`: `775.229`
- `avg_fps`: `1.757`
- `peak_rss_mb`: `1621.051`
- `final_rss_mb`: `1621.176`
- `accepted_loops`: `0`
- `loop_debug_events`: `105`
- `essential_graph_runs`: not exposed by the runner; inferred `0` because `accepted_loops=0`
- `global_ba_started`: `0` from `frame_log_rgbd_dataset_freiburg1_room.csv`
- `global_ba_success`: `0` from `frame_log_rgbd_dataset_freiburg1_room.csv`

`/usr/bin/time -v` summary:

- Wall time: `12:58.12`
- CPU utilization: `164%`
- Max RSS: `1660084 kB` (`1621.18 MiB`)

## 7. Runtime breakdown

Total runtime:

- `frame.total`: total `775.175 s`, mean `0.569 s`, max `12.077 s`

Requested sections:

- `slam.track`: total `499.428 s`, mean `0.367 s`, max `3.860 s`
- `tracking.track_local_map`: total `180.925 s`, mean `0.133 s`, max `0.260 s`
- `tracking.create_new_keyframe`: total `109.527 s`, mean `2.434 s`, max `3.647 s`
- `local_mapping.step`: total `245.037 s`, mean `5.445 s`, max `9.163 s`
- `local_mapping.local_BA`: total `86.241 s`, mean `1.916 s`, max `5.131 s`
- `local_mapping.cull_map_points`: total `152.058 s`, mean `3.379 s`, max `6.128 s`
- `local_mapping.fuse_map_points`: total `4.501 s`, mean `0.100 s`, max `0.418 s`
- `loop_closing.step`: total `3.894 s`, mean `0.087 s`, max `0.497 s`
- `loop.detect_candidates`: total `0.096 s`, mean `0.0021 s`, max `0.0045 s`
- `loop.compute_geometry`: total `3.721 s`, mean `0.207 s`, max `0.492 s`
- `loop.search_more_projection`: total `0.064 s`, mean `0.032 s`, max `0.039 s`

Memory/image overhead:

- `memory.profile_snapshot`: total `0.0081 s`, mean `0.000278 s`, max `0.000516 s`
- `frame_timing.csv` `memory_profile_sec`: mean `0.000006 s`, max `0.000531 s`
- `frame.load_rgb`: total `17.616 s`, mean `0.0129 s`, max `0.0229 s`
- `frame.load_depth`: total `7.906 s`, mean `0.0058 s`, max `0.0173 s`

Observed runtime bottleneck after this full run:

- Aggregate non-tracking sidecar cost is still led by `local_mapping.step` (`245.0 s`).
- Inside LocalMapping, the dominant hot path is `cull_map_points` (`152.1 s`).
- Inside tracking, the dominant per-frame hot path remains `track_local_map` (`180.9 s` total).

## 8. Keyframe scheduling analysis

From `keyframe_decision_log.csv`:

- Total keyframes: `46`
- Inserted new keyframes during run: `45`
- Average frames between keyframes: `28.49`
- Min frames between keyframes: `4`
- Max frames between keyframes: `101`
- Insertion reasons/counts: `local_mapping_accepting=45`
- Rejection reasons/counts: `conditions_not_met=1316`

Trigger statistics on inserted keyframes:

- `c1a`: `18/45`
- `c1b`: `42/45`
- `c1c`: `39/45`
- `c2`: `45/45`

Mapper/queue behavior:

- `local_mapping_accepting=True` for all `1361` decision rows
- `local_mapping_queue_size`: min `0`, mean `0`, max `0`
- Sequential LocalMapping therefore showed no queue backpressure during this run

Assessment after 2.33A:

- Scheduling remained reasonable across the full sequence.
- The system stayed far away from the old dense-keyframe failure mode.
- There were short dense bursts at initialization and around frames `841-849`, but the full-run average stayed controlled at about one keyframe every `28.5` frames.

## 9. Local mapping schedule analysis

From `local_mapping_schedule_log.csv`:

- Local BA started/completed/skipped/aborted/forced:
  - started `45`
  - completed `45`
  - skipped `0`
  - aborted `0`
  - forced due starvation `0`
- Local BA starvation:
  - `keyframes_since_last_successful_ba` max `0`
  - `last_successful_local_ba_kid=45`
  - Local BA was not starved

Timing:

- Local BA total time: `86.242 s`
- Local BA mean/max per run: `1.916 s` / `5.131 s`
- LocalMapping step total time: `245.034 s`
- `cull_map_points` total: `152.058 s`
- `fuse_map_points` total: `4.501 s`

Queue behavior:

- queue before: min `0`, mean `0`, max `0`
- queue after: min `0`, mean `0`, max `0`

Skipped fusion/BA reasons:

- none recorded

## 10. Local map projection analysis

From `local_map_profile.csv`:

- Average local keyframes/frame: `26.39`
- Average local points/frame: `4143.02`
- Average visible projected points/frame: `1373.10`
- Average already-seen rejections/frame: `213.41`
- Average descriptor comparisons/frame: `449.19`
- Average projection matches/frame: `201.03`
- `track_local_map`: mean `0.1329 s`, max `0.2595 s`
- `search_map_by_projection_sec`: mean `0.1079 s`
- `local_map_build_sec`: mean `0.0129 s`
- `pose_optimization_sec`: mean `0.0099 s`

Assessment after 2.32A:

- The local-map construction/projection workflow stayed stable over the full sequence.
- Local points did not collapse, projection matching stayed productive, and `track_local_map` remained bounded.
- The current full-run hotspot is no longer uncontrolled local-map growth; it is mostly the cumulative cost of map-point culling and keyframe creation work.

## 11. Loop-closure analysis

From `loop_debug_candidates.csv` and `loop_candidate_pair_reports/`:

- Loop candidate events: `105`
- Accepted loops: `0`
- Rejected loops: `105`
- Accepted loop pairs: none
- Pair-report files written: `40`

Rejection reason counts:

- `rejected_by_consistency`: `65`
- `not enough SE3 RANSAC seed inliers`: `28`
- `too few loop geometry matches`: `6`
- `estimated pose distance too large for guided SE3 loop seed`: `3`
- `too few matched map points after covisibility expansion (38 < 60)`: `1`
- `too few matched map points after covisibility expansion (37 < 60)`: `1`
- `not enough final guided loop matches`: `1`

Rejection buckets requested by task:

- BoW score: `0`
- Consistency: `65`
- Geometry / seed match count: `28`
- Geometry / pose-distance gate: `3`
- Projection-expansion count / loop geometry matches: `6`
- Final matched map-point gate after covisibility expansion: `2`
- Final guided loop matches: `1`
- Duplicate/near-duplicate check: `0`
- Other: `0`

Rejected-candidate statistics:

- Candidate score: mean `0.01682`, median `0.01593`, max `0.03779`
- Consistency count: mean `2.49`, median `1`, max `10`
- BoW matches with valid map points: mean `9.77`, median `0`, max `44`
- Geometry refined inliers: mean `2.51`, median `0`, max `38`
- Guided projection matches: mean `0.50`, median `0`, max `26`
- Final inliers: mean `0.59`, median `0`, max `38`

Strongest rejected examples:

- `KF 40 -> KF 6`: score `0.01332`, consistency `3`, refined inliers `38`, guided projection matches `26/38`, rejected at final matched map-point gate `38 < 60`
- `KF 43 -> KF 11`: score `0.02886`, consistency `6`, refined inliers `24`, guided projection matches `23/37`, rejected at final matched map-point gate `37 < 60`
- `KF 44 -> KF 12`: score `0.01681`, consistency `7`, refined inliers `13`, guided projection matches `3/16`, rejected for not enough final guided loop matches
- `KF 43 -> KF 8`: score `0.03407`, consistency `6`, refined inliers `18`, rejected because estimated pose distance was too large for guided SE3 loop seeding

Conclusion:

- Loops were not absent; candidates were found.
- The dominant failure mode was consistency rejection, followed by geometry rejection.
- A small number of late-sequence candidates came close enough to justify pair-level inspection before any threshold tuning.

## 12. Memory behavior

From `memory_profile.csv`:

- Peak RSS: `1621.051 MB`
- Final RSS: `1621.176 MB`
- RSS trend: `734.637 MB -> 1621.051 MB`, then flat at the end
- `recent_frames`: min `1`, max `20` -> bounded
- `old_frame_views_total`: min `0`, max `0` -> bounded and fully pruned
- `num_frame_views_total`: start `0`, peak `10646`, end `4698`
- `keyframe_depth_images`: start `0`, peak `0`, end `0`
- Map points: start `1689`, peak `5193`, end `4649`
- Keyframes: start `1`, peak/end `46`

Assessment:

- Memory policy remained stable.
- There was no runaway growth in retained recent frames.
- Old frame views remained at zero throughout sampled memory snapshots.
- Depth images were not being retained on keyframes in this run.

## 13. Comparison against previous known results

Qualitative comparison targets:

- Old 700-frame memory run: `233` keyframes, about `35k` map points, about `6.1 hours`
- Checkpoint 2.33A 300-frame run: `16` keyframes, `3953` map points, `206.8 s`, `300/300 OK`

Current full run:

- `1362/1362` OK
- `46` keyframes
- `4649` map points
- `775.2 s` runner elapsed, `12:58.12` wall time with `/usr/bin/time -v`

Assessment:

- Runtime is not just within the expected `20-40 minute` range; it is materially better at about `13 minutes`.
- Keyframe growth stayed controlled relative to both the old 700-frame regression and the early post-2.32A dense-keyframe behavior.
- Final map-point count stayed close to the 2.33A 300-frame scale rather than exploding toward the old `~35k` regime.

## 14. Tests added/updated

None. This checkpoint did not change source code.

## 15. Test commands run

Environment check:

```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
```

Result:

- `/home/kaushik/slam_ws/.venv/bin/python`

Runner help audit:

```bash
source .venv/bin/activate
python -m visual_slam.orbslam.run_rgbd_slam --help \
  | tee visual_slam/reference_audit/checkpoint_2_34A/RUNNER_HELP.txt
```

Result:

- All requested flags were present.

Dataset validation command:

- See exact full-run command in Section 5.

Dataset validation result:

- Full sequence completed successfully with `1362/1362` tracking OK and `0` tracking losses.

## 16. Correctness / benchmarkability / memory / runtime status

- Correctness: improved confidence; full tracking completed and trajectory/keyframe consistency was exact in the runner summary (`max diff 0.0000 m`)
- Benchmarkability: improved; this is now a reproducible full-sequence artifact set with runtime/memory/keyframe/local-map/loop diagnostics
- Memory: stable
- Runtime: acceptable and substantially improved

## 17. Remaining gaps

- No loop was accepted, so loop correction, essential graph correction, and post-loop behavior remain unvalidated on this sequence.
- The runner does not expose an explicit `essential_graph_runs` counter; current report infers `0` because no loop was accepted.
- Sequential LocalMapping kept queue statistics trivial (`0` throughout), so queue-pressure behavior was not stress-tested here.

## 18. Next recommended action

Recommended next action: inspect near-accept rejected loop pairs before changing thresholds.

Best candidates to inspect first:

- `candidate_84_kf_40_6.json`
- `candidate_96_kf_43_11.json`
- `candidate_99_kf_44_12.json`

Reasoning:

- Loop candidates clearly appeared.
- Two candidates reached `37-38` matched map points and failed only at the final `60`-point matched-map-point gate.
- That makes this checkpoint better aligned with option `d`: inspect rejected candidates that were close to acceptance, and only then consider evidence-backed threshold tuning.
- Do not run loop+GBA validation yet; accepted loops did not occur.

## 19. Files changed

- `visual_slam/reference_audit/checkpoint_2_34A/PRE_CHANGE_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_34A/IMPLEMENTATION_AUDIT.md`
- `visual_slam/reference_audit/checkpoint_2_34A/VALIDATION_REPORT.md`
- `visual_slam/reference_audit/checkpoint_2_34A/FULL_FR1_ROOM_LOOP_NO_GBA_VALIDATION_REPORT.md`
- `visual_slam/reference_audit/checkpoint_2_34A/FULL_FR1_ROOM_LOOP_NO_GBA_SUMMARY.json`
- `visual_slam/reference_audit/checkpoint_2_34A/RUNNER_HELP.txt`
- `visual_slam/reference_audit/checkpoint_2_34A/OUTPUT_FILE_LIST.txt`

## 20. Intentionally excluded files

- No source files under `visual_slam/orbslam/`
- No tests under `tests/`
- No dataset contents
- No generated heavy outputs committed or staged
