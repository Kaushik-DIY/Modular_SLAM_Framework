# Checkpoint 2.28A — IMPLEMENTATION AUDIT

## 1. What was changed

### `visual_slam/orbslam/slam/loop_closing.py`

**`LoopGeometryChecker.check_candidates` (search_more call site, ~line 380)**

*Before* (bug):

```python
Tc2w = np.asarray(candidate.Tcw(), dtype=np.float64).reshape(4, 4)
T12 = np.eye(4, dtype=np.float64)
T12[:3, :3] = estimate.R
T12[:3, 3] = estimate.t
Tc1w = T12 @ Tc2w   # ← WRONG: T12 is world-drift, not camera-to-camera
Rc1w = Tc1w[:3, :3]
tc1w = Tc1w[:3, 3]
_, matches = ProjectionMatcher.search_more_map_points_by_projection(
    loop_map_points, current_keyframe, Rc1w, tc1w, matches, match_idxs, …)
num_matched_map_points = sum(m is not None for m in matches)
```

*After* (fixed):

```python
Tcw_current = np.asarray(current_keyframe.Tcw(), dtype=np.float64).reshape(4, 4)
T12 = np.asarray(estimate.T, dtype=np.float64).reshape(4, 4)
T12_inv = np.linalg.inv(T12)
Tcw_current_corrected = Tcw_current @ T12_inv   # ← CORRECT: world-drift convention
new_projection_matches, matches, search_more_diag = (
    ProjectionMatcher.search_more_map_points_by_projection(
        loop_map_points, current_keyframe,
        Tcw_current_corrected,               # explicit 4×4 SE3
        matches, match_idxs,
        max_reproj_distance=…,
        return_diagnostics=True))
```

Additional diagnostics fields now written into each candidate report:
`seed_inliers`, `candidate_covisible_points`, `projected_visible_points`,
`new_projection_matches`, `total_final_matches`, `final_gate_threshold`,
`accepted_or_rejected`.

### `visual_slam/orbslam/slam/geometry_matchers.py`

**`_search_more_map_points_by_projection` signature and docstring**

- Changed first positional projection-pose argument from `(Rcw, tcw)` (two
  separate arrays) to `current_keyframe_Tcw_corrected: np.ndarray (4×4)`.
  The internal split `Rcw = Tcw[:3,:3]; tcw = Tcw[:3,3]` is done inside.
- Added `return_diagnostics: bool = False` flag returning a third element
  (dict) with `candidate_input_points`, `candidate_unique_points`,
  `projected_visible_points`, `new_projection_matches`.
- Replaced the terse old docstring with a fully-explicit one spelling out
  the `Tcw_current_corrected = current_keyframe.Tcw() @ inv(T12)` formula
  and warning against passing the candidate keyframe's `Tcw`.

### `tests/visual_slam/orbslam/test_checkpoint_2_26B_loop_acceptance_debug.py`

- `test_candidate_pair_report_contains_required_fields`: updated `n=50` to
  `n=80` because the new `kLoopClosingMinNumMatchedMapPoints = 60` gate
  (added as the missing pySLAM-aligned final acceptance gate) requires at
  least 60 matched map points. The old scene with 50 keypoints was
  below threshold.

---

## 2. Why the fix matches pySLAM

pySLAM (`loop_closing.py:363–367`) computes:

```python
self.success_loop_kf_sim3_pose = (
    Sim3Pose(R12, t12, scale12)          # Sc1c2 — camera-to-camera
    @ Sim3Pose().from_se3_matrix(kf.Tcw())  # Sc2w
)  # Sc1w = Sc1c2 * Tc2w (corrected current Tcw)
```

pySLAM's `R12, t12` come from the C++ `sim3solver` which operates on
camera-frame coordinates: it computes `Sc1c2` (a camera-to-camera Sim3).
So `Sc1w = Sc1c2 * Sc2w` is a geometrically valid chain of camera-frame
transforms giving the corrected current-Tcw.

Our solver operates on *world-frame* coordinates (`points_loop_world ≈ R ×
points_current_world + t`), producing the world-frame drift
`T_drift = estimate.T` such that `p_w_loop = T_drift · p_w_current`.

The corrected current Tcw consistent with this world-drift convention is
(derivation in `PRE_CHANGE_LOOP_PROJECTION_EXPANSION_AUDIT.md` §8):

```text
Tcw_current_corrected = Tcw_current @ inv(T_drift)
```

This is exactly what `LoopCorrector._make_corrected_pose_map` and
`optimize_essential_graph_se3` already compute for their keyframe pose
corrections — so our fix brings the `search_more` call into line with the
rest of the pipeline.

Both formulas evaluate to the same *numerical* corrected current Tcw (given
the relationship between pySLAM's camera-frame Sim3 and our world-frame
SE3): pySLAM's `Sc1w` and our `Tcw_current @ inv(T12)` both represent the
pose of the current camera in the loop's world frame.

---

## 3. Transform convention before and after

| | Before | After |
|---|---|---|
| Pose passed to `search_more` | `T12 @ Tc2w` where T12 is world-drift | `Tcw_current @ inv(T12)` where T12 is world-drift |
| Zero-drift limit | reduces to `Tc2w` (candidate's pose — wrong, projects into candidate camera) | reduces to `Tcw_current` (current's pose — correct, projects into current camera) |
| Non-trivial drift | composes a world-frame transform with a camera-from-world transform: geometrically undefined | corrects the current camera's world estimate by removing the drift: geometrically well-defined |
| Consistency with rest of pipeline | inconsistent | consistent with `_make_corrected_pose_map`, `apply_correction_to_map_points`, `essential_graph` |

---

## 4. Why the fix prevents false inflation of projection matches

With the buggy formula, in the small-drift / zero-drift limit the projection
pose = `Tc2w` (candidate's pose). The current camera is at a different
location. Therefore loop map points project to image locations that do NOT
correspond to the current keyframe's actual keypoints. The kdtree radius
search (typically 10 pixel radius) will find no overlap unless the two
cameras are very close, so `found_pts_count ≈ 0`.

While this might seem "safe" (no spurious matches), it actually fails both
ways:

1. True loops: real additional matches are never found. The gate is applied
   only on seed inliers. If seed inliers ≥ 60 a loop may still pass; if < 60
   it silently fails. This led to inconsistent behavior.

2. False loops: with non-trivial drift, the "projection" into a fictitious
   frame randomly scatters points across the image. Any keypoint that
   happens to have matching descriptors and land within 10px of a scattered
   projection can produce a spurious match. The gate counts these, so a
   false loop with many scattered projections could get inflated
   `num_matched_map_points`.

The corrected formula projects loop map points into the current camera at
their *physically expected* locations. Only keypoints in the current image
that actually correspond to the same scene regions will produce low
descriptor distances. False loop candidates whose scenes do not overlap
will have near-zero projections into the visible image area, so fewer
matches are found and the gate is more discriminative.

---

## 5. Why the fix preserves true loop expansion

For a true loop, the drift-corrected current camera pose correctly
back-projects the candidate covisibility group's map points to the image
area in the current frame where those scene points were actually observed.
This means:
- `projected_visible_points` will be high (the points are in front of the
  current camera at plausible distances).
- kdtree hits will be concentrated in the correct image regions.
- Descriptor distances will be low because both keypoints observe the same
  physical scene.
- `new_projection_matches` will be non-zero, potentially recovering
  additional matches the seed RANSAC missed due to octave or radius limits.

This makes the final gate a fairer test: for a true loop the total is at
least the seed inliers, often more; for a false loop the total is the seed
inliers only (random-descriptor matches the only source).

---

## 6. Deliberate deviations from pySLAM

| Topic | pySLAM | Local | Reason |
|---|---|---|---|
| Solver type | C++ Sim3Solver, camera-frame | Python Kabsch, world-frame | pySLAM's C++ sim3solver module unavailable locally; RGB-D fixes scale so world-frame SE3 suffices |
| Pose representation | Sim3Pose object | np.ndarray 4×4 SE3 | No Sim3 scale needed for RGB-D |
| search_more signature | `(points, f_cur, Scw, …)` Scw a Sim3Pose | `(points, f_cur, current_keyframe_Tcw_corrected, …)` a 4×4 SE3 | Explicit naming avoids the ambiguity that caused the original bug |
| Diagnostics | print statements in pySLAM | structured dict returned to candidate report | Exposes per-stage counts in the loop debug CSV |
