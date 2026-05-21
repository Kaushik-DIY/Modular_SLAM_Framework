# Checkpoint 2.32A - Baseline Local Map Diagnostic Report

## 1. Task/checkpoint name

Checkpoint 2.32A - pySLAM-aligned local map construction and projection-search workflow.

## 2. Files inspected

- `visual_slam/orbslam/slam/tracking.py`
- `visual_slam/orbslam/slam/map.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/local_mapping.py`
- `visual_slam/orbslam/slam/local_mapping_core.py`
- `visual_slam/orbslam/slam/config_parameters.py`
- `visual_slam/orbslam/run_rgbd_slam.py`

## 3. pySLAM files inspected

- `third_party/pyslam_reference/pyslam/slam/tracking.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`

Reference commit: `a95db3a0e95764b8c68b81fade544bdd6ecb912e`.

## 4. Baseline command

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

## 5. Output artifacts

- Output directory: `visual_slam_outputs/checkpoint_2_32A/baseline_fr1_desk_30_localmap_profile`
- `run_summary.json`
- `runtime_profile.csv`
- `local_map_profile.csv`
- `frame_timing.csv`
- `frame_log_rgbd_dataset_freiburg1_desk.csv`

## 6. run_summary.json values

| Metric | Baseline value |
| --- | ---: |
| frames_attempted | 30 |
| tracking_ok_count | 30 |
| tracking_lost_count | 0 |
| errors | 0 |
| final_state | OK |
| keyframes | 10 |
| map_points | 4288 |
| trajectory_poses | 30 |
| elapsed_sec | 290.892 |
| avg_fps | 0.103 |
| peak_rss_mb | 1013.973 |

## 7. runtime_profile.csv summary

| Section | Calls | Total sec | Mean sec | Max sec |
| --- | ---: | ---: | ---: | ---: |
| frame.total | 30 | 290.855 | 9.695 | 27.429 |
| slam.track | 30 | 216.451 | 7.215 | 13.961 |
| tracking.track_local_map | 29 | 192.493 | 6.638 | 12.108 |
| local_mapping.step | 9 | 73.873 | 8.208 | 13.452 |

`tracking.track_local_map` accounts for about 66.2% of total frame runtime and about 88.9% of `slam.track` runtime in this baseline.

## 8. local_map_profile.csv summary

| Metric | Baseline average | Baseline max |
| --- | ---: | ---: |
| local keyframes/frame | 4.724 | 9 |
| voted local keyframes/frame | 4.724 | 9 |
| expanded local keyframes/frame | 0.000 | 0 |
| local points/frame | 3122.379 | 4019 |
| visible projected points/frame | 2833.724 | 3359 |
| already-seen rejections/frame | 0.000 | 0 |
| descriptor comparisons/frame | 1132.897 | 1394 |
| projection matches/frame | 490.655 | 624 |
| track_local_map_sec | 6.638 | 12.108 |
| search_map_by_projection_sec | 0.105 | 0.136 |
| local_map_build_sec | 6.504 | 11.960 |
| pose_optimization_sec | 0.025 | 0.044 |

## 9. Root cause or current hypothesis

The baseline confirms that the largest cost inside `track_local_map` is local-map construction, not descriptor projection search. The average local-map build time is 6.504 seconds out of 6.638 seconds per `track_local_map` call.

The pre-change audit identified the structural cause: local code builds the local map from `kf_ref` and all of its covisibles, then elects the best reference keyframe afterward. pySLAM instead starts from the current frame's matched map points, votes observing keyframes, selects the max-vote reference first, and expands a bounded graph neighborhood.

## 10. Evidence of local map inflation

The baseline local-map size is substantial for a 30-frame run: average 3122 local points and max 4019 local points per frame. Projection search itself is not the dominant bottleneck yet, but the local point set is built by the non-pySLAM reference-covisibility workflow and costs most of the tracking time.

The baseline diagnostics show zero already-seen rejections because local code does not mark current-frame matched points as seen before projection search.

## 11. Exact changes already made for diagnostics

- Added `--profile-local-map`.
- Added `local_map_profile.csv` and standardized local-map profile artifact output.
- Added optional projection diagnostics around `ProjectionMatcher.search_map_by_projection()`.
- Added per-frame local-map profile row collection in `Tracking.track_local_map()`.

These changes are diagnostic-only and do not alter local-map construction behavior.

## 12. Tests added/updated

No targeted parity tests have been added yet. They will be added after the required baseline, during the seven-gap implementation phase.

## 13. Test commands run

```bash
python -m py_compile \
  visual_slam/orbslam/slam/geometry_matchers.py \
  visual_slam/orbslam/slam/tracking.py \
  visual_slam/orbslam/run_rgbd_slam.py
```

Result: passed.

```bash
python -m visual_slam.orbslam.run_rgbd_slam --help | rg "profile-local-map|profile-runtime"
```

Result: passed; `--profile-local-map` is exposed.

## 14. Dataset validation commands and results

The required 30-frame baseline completed successfully:

- `frames_attempted`: 30
- `tracking_ok_count`: 30
- `tracking_lost_count`: 0
- `final_state`: OK
- `trajectory_poses`: 30

## 15. Remaining risks

- The current diagnostics count "expanded" local keyframes by comparing local keyframes to vote keys. Since behavior has not yet changed, this is a baseline diagnostic approximation, not pySLAM marker semantics.
- Projection diagnostics currently preserve the baseline filtering order. Gap 7 will intentionally change the order so already-seen points are skipped before expensive projection/KD work.
- Parent/child expansion order may require careful testing because children are stored in sets.

## 16. Next recommended action

Implement the seven pySLAM local-map/projection-search gaps in order, then add targeted parity tests before running the full visual SLAM test slice.
