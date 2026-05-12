# Checkpoint 2.22 Loop Projection Search and Map-Point Fusion Audit

## pySLAM files inspected

- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py`
- `third_party/pyslam_reference/pyslam/slam/geometry_matchers.py`
- `third_party/pyslam_reference/pyslam/slam/keyframe.py`
- `third_party/pyslam_reference/pyslam/slam/map_point.py`
- `third_party/pyslam_reference/pyslam/slam/map.py`
- `third_party/pyslam_reference/pyslam/config_parameters.py`

## Current files inspected

- `visual_slam/orbslam/slam/loop_closing.py`
- `visual_slam/orbslam/slam/geometry_matchers.py`
- `visual_slam/orbslam/slam/keyframe.py`
- `visual_slam/orbslam/slam/map_point.py`
- `visual_slam/orbslam/slam/map.py`
- `tests/visual_slam/orbslam/test_checkpoint_2_21_loop_closing.py`

## Previous fusion limitation

- Loop correction only fused direct matches from the verified current keyframe to the loop keyframe.
- `ProjectionMatcher.search_and_fuse_for_loop_correction()` was not implemented.
- Loop-side covisible map points were not projected into current-side corrected covisible keyframes.
- `MapPoint.replace_with()` could transfer observations and then clear transferred keyframe slots during `set_bad()` cleanup.

## New wider projection/fusion implementation

- Added `ProjectionMatcher.search_and_fuse_for_loop_correction()` with pySLAM-style loop projection search.
- Added `LoopCorrector.search_and_fuse_corrected_keyframes()`.
- Loop-side points are collected from:
  - matched loop keyframe,
  - best covisible neighbors of the loop keyframe.
- Current-side corrected keyframes are collected from:
  - current keyframe,
  - best covisible neighbors of the current keyframe.
- Loop-side map points are projected using the corrected/current-side pose map.
- Projection search checks:
  - finite world position and projection,
  - positive depth,
  - image bounds,
  - predicted scale level,
  - reprojection chi-square,
  - descriptor distance,
  - bad/replaced/duplicate points.

## MapPoint replacement behavior

- `MapPoint.replace_with()` now mirrors pySLAM's safer replacement semantics:
  - refuses self/None/bad replacement targets,
  - records the replacement,
  - clears the old point's observation containers under lock,
  - transfers observations to the replacement,
  - removes duplicate keyframe slots when the replacement already owns that keyframe,
  - carries found/visible counters forward,
  - recomputes descriptor, normal, and depth range on the replacement,
  - removes the replaced point from the map without undoing transferred observations.

## Diagnostics added

`ProjectionFuseDiagnostics` reports:

- `projected_points`
- `visible_projected_points`
- `candidate_matches`
- `added_observations`
- `fused_points`
- `replaced_points`
- `rejected_bad_point`
- `rejected_not_visible`
- `rejected_descriptor`
- `rejected_scale`
- `rejected_duplicate`

Loop diagnostics now retain the aggregate fusion diagnostics from direct matched fusion plus the wider projection fusion pass.

## Tests and results

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_22_loop_fusion.py`
  - Result: `10 passed in 2.47s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam`
  - Result: `154 passed, 1 skipped in 5.83s`.
- Checkpoint 2.22 validation:
  - `python tools/validate_orbslam_pyslam_port.py --dataset "$HOME/slam_ws/datasets/tum/rgbd_dataset_freiburg1_desk" --output "$HOME/slam_ws/visual_slam_outputs/checkpoint_2_22_validation"`
  - Result: `VALIDATION PASSED`.

## TUM smoke results

- `fr1_desk`, loop enabled, 100 frames:
  - Output: `visual_slam_outputs/checkpoint_2_22_loop_enabled_fr1_desk_100`
  - Result: 100/100 tracking OK, 0 lost, 0 errors, final state OK.
- `fr1_room`, loop enabled, 100 frames:
  - Output: `visual_slam_outputs/checkpoint_2_22_loop_enabled_fr1_room_100`
  - Result: 100/100 tracking OK, 0 lost, 0 errors, final state OK.

## Deviations from pySLAM

- pySLAM's loop correction carries Sim3 poses. This checkpoint keeps the project RGB-D-only and uses scale-fixed SE3 poses for projection and correction.
- Orientation consistency for loop projection fusion remains limited to descriptor/geometric gates because the projection-fusion input is a map point rather than a direct keypoint-pair match. Direct BoW loop matching still uses orientation filtering.
- The local implementation does not trigger Global BA after loop correction. Global BA is deferred to Checkpoint 2.24.

## Remaining fusion gaps

- Monocular Sim3 projection/fusion parity is not claimed.
- Projection fusion uses Python-level KD-tree and descriptor loops rather than pySLAM's optional C++ acceleration.
- More detailed per-keyframe fusion timing could be added in a later durability checkpoint.
