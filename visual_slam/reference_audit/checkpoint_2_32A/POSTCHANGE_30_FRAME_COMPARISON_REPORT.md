# Checkpoint 2.32A - Post-Change 30-Frame Comparison Report

## 1. Task/checkpoint name

Checkpoint 2.32A - pySLAM-aligned local map construction and projection-search workflow.

## 2. Files inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`

## 3. pySLAM files inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`

Reference commit: `a95db3a0e95764b8c68b81fade544bdd6ecb912e`.

## 4. Validation commands

Baseline command was run before behavior changes:

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_32A/baseline_fr1_desk_30_localmap_profile" \
  --feature-backend pyslam_orb2 \
  --max-frames 30 \
  --disable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-local-map \
  --print-every 10
```

Post-change command:

```bash
python -m visual_slam.orbslam.run_rgbd_slam \
  "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" \
  --dataset-type tum_rgbd \
  --camera-profile auto \
  --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_32A/postchange_fr1_desk_30_localmap_profile" \
  --feature-backend pyslam_orb2 \
  --max-frames 30 \
  --disable-loop-closing \
  --disable-global-ba \
  --no-map-export \
  --profile-runtime \
  --profile-local-map \
  --print-every 10
```

## 5. Output paths

- Baseline: `visual_slam_outputs/checkpoint_2_32A/baseline_fr1_desk_30_localmap_profile`
- Post-change: `visual_slam_outputs/checkpoint_2_32A/postchange_fr1_desk_30_localmap_profile`

## 6. Comparison table

| Metric | Baseline value | Post-change value | Improved/same/worse | Comment |
| --- | ---: | ---: | --- | --- |
| frames_attempted | 30 | 30 | same | Same run length. |
| tracking_ok_count | 30 | 30 | same | Tracking quality preserved. |
| tracking_lost_count | 0 | 0 | same | No regression. |
| final_state | OK | OK | same | No state regression. |
| final_keyframes | 10 | 10 | same | Keyframe count preserved. |
| final_map_points | 4288 | 4239 | same | 1.1% lower, no collapse. |
| trajectory_poses | 30 | 30 | same | Trajectory remains valid. |
| elapsed_sec | 290.892 | 100.814 | improved | 65.3% lower elapsed time. |
| avg_fps | 0.103 | 0.298 | improved | 2.9x higher FPS. |
| average local_keyframes/frame | 4.724 | 4.724 | same | Same average KF neighborhood size. |
| average local_points/frame | 3122.379 | 3106.724 | improved | Slight reduction without blind cap. |
| average visible_projected_points/frame | 2833.724 | 2273.690 | improved | Already-matched points are skipped before projection. |
| average already_seen_rejections/frame | 0.000 | 546.862 | improved | Gap 7 marker now active. |
| average descriptor_comparisons/frame | 1132.897 | 821.793 | improved | 27.5% fewer descriptor comparisons. |
| average projection_matches/frame | 490.655 | 490.759 | same | Match yield preserved. |
| average track_local_map_sec | 6.638 | 0.132 | improved | 98.0% lower. |
| average search_map_by_projection_sec | 0.105 | 0.089 | improved | 14.8% lower. |
| peak_rss_mb | 1013.973 | 1012.770 | same | No memory regression in this short run. |

## 7. Acceptance criteria

| Criterion | Result |
| --- | --- |
| tracking_lost_count must not increase | Passed: 0 -> 0 |
| final_state must remain OK or not worse | Passed: OK -> OK |
| map point count must not collapse | Passed: 4288 -> 4239 |
| trajectory poses must remain valid | Passed: 30 -> 30 |
| track_local_map_sec should improve or not worsen significantly | Passed: 6.638s -> 0.132s |
| local_points/frame should reduce if baseline was inflated | Passed: 3122.379 -> 3106.724, with much fewer projected candidates |
| no blind cap should be required | Passed: no local-map-point cap added |

## 8. Root cause and fix summary

The baseline bottleneck was local-map construction from a stale/reference keyframe plus unbounded reference covisibles. The fix follows pySLAM's tracking-local-map control flow: current-frame observation voting, max-vote reference selection before local map construction, voted keyframe seed set, bounded covisibility/parent/child expansion, local point uniqueness markers, and already-matched seen markers before projection search.

## 9. Test context

Targeted local-map parity tests passed:

```text
17 passed in 0.79s
```

Python/non-C++ visual SLAM slice passed:

```text
278 passed, 1 skipped, 94 deselected in 27.39s
```

Full visual SLAM slice has a user-waived out-of-scope native blocker:

```text
tests/visual_slam/orbslam/test_cpp_slam_core_phase2_frame.py::TestFramePointMatch::test_reset_points
Fatal Python error: Segmentation fault
```

The standalone C++ segfault reproduces independently and was explicitly waived for this checkpoint because C++ extension changes are out of scope.

## 10. Remaining risks

- This is a 30-frame desk validation; the recommended next run is the checkpoint's 100-frame fr1_room command, not an automatic full sequence.
- Child expansion remains bounded but child identity can depend on local set iteration when multiple valid children exist.
- The C++ extension segfault remains unresolved outside this checkpoint.

## 11. Next recommended action

Run the 100-frame fr1_room local-map profile only after user confirmation.
