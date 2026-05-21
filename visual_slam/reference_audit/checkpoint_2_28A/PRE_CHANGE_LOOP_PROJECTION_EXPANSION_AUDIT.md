# Checkpoint 2.28A — PRE-CHANGE Loop Final Projection Expansion Audit

Date: 2026-05-04
Branch: `feature/orbslam-pyslam-alignment`
Scope: Verify whether `LoopGeometryChecker.check_candidates` composes the
*corrected current-keyframe `Tcw`* used for the final
`search_more_map_points_by_projection` call in the same world-frame
convention that the rest of the loop-closure pipeline (essential graph
correction and map-point correction) already assumes.

---

## 1. Local files inspected

```
visual_slam/orbslam/slam/loop_closing.py
visual_slam/orbslam/slam/geometry_matchers.py
visual_slam/orbslam/slam/sim3_solver.py
visual_slam/orbslam/slam/essential_graph.py
visual_slam/orbslam/slam/keyframe.py
visual_slam/orbslam/slam/frame.py
visual_slam/orbslam/slam/config_parameters.py
```

Key code under audit (loop_closing.py):

- `LoopGeometryChecker._matched_3d_points` (lines 672–703): builds
  `points_current` and `points_loop` from
  `point.get_position()` of the *map points* — i.e. these are **world-frame
  3-D positions**, not camera-frame ones.
- `estimate_scale_fixed_sim3(points_current, points_loop, …)` (sim3_solver.py
  lines 41–109): solves
  `points_loop ≈ R · points_current + t` — so the returned
  `(R, t)` is a **world-to-world** rigid transform that moves a 3-D map-point
  position from the current keyframe’s world frame to the candidate keyframe’s
  world frame. The `Sim3Estimate.T` is `(R | t; 0 1)` with scale fixed to 1
  for RGB-D.
- `LoopGeometryChecker.check_candidates`, lines 380–400 (final search-more):

```python
Tc2w = candidate.Tcw()
T12  = estimate.T                # world-drift current → candidate
Tc1w = T12 @ Tc2w
Rc1w = Tc1w[:3, :3]
tc1w = Tc1w[:3, 3]
ProjectionMatcher.search_more_map_points_by_projection(
    loop_map_points, current_keyframe, Rc1w, tc1w, matches, match_idxs, …)
```

  This is the line under suspicion. It composes the *world-drift* `T12`
  with `Tc2w` using the same algebraic formula pySLAM uses for a *camera-frame*
  Sim3 (`Sc1w = Sc1c2 · Sc2w`). The two compositions are NOT the same
  geometrically.
- `LoopCorrector._make_corrected_pose_map` (lines 918–926): builds the
  corrected `Tcw` for every keyframe in `current_group` using:
  `Tcw_corrected = Tcw @ inv(correction_T)`.
- `apply_correction_to_map_points` (essential_graph.py 387–407):
  `position_world_corrected = correction_T @ position_world`.

So the rest of the pipeline already treats `correction_T == estimate.T` as a
**world-frame drift** (`p_loop_world = correction_T · p_current_world`), and
the matching corrected-camera convention is `Tcw_corrected = Tcw_current ·
correction_T^{-1}`. The line at 388 violates that convention.

---

## 2. pySLAM files inspected

```
third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py
third_party/pyslam_reference/pyslam/slam/geometry_matchers.py
third_party/pyslam_reference/pyslam/slam/frame.py
```

Key references:

- `LoopGeometryChecker.check_candidates`, loop_closing.py lines 260–367:
  pySLAM feeds the C++ `sim3solver.Sim3Solver` with
  `K1, Rcw1, tcw1, K2, Rcw2, tcw2, points_3d_w1, points_3d_w2`. The pySLAM
  sim3solver C++ code projects world points into both camera frames and solves
  the **camera-to-camera** Sim3 (`Sc1c2`) such that
  `P_cam1 ≈ s12 · R12 · P_cam2 + t12`.
- After `optimize_sim3`, pySLAM stores (loop_closing.py 363–367):

```python
self.success_loop_kf_sim3_pose = (
    Sim3Pose(R12, t12, scale12)
    @ Sim3Pose().from_se3_matrix(kf.Tcw())
)  # Sc1w = Sc1c2 * Tc2w
```

  Then this `Sc1w` is the **corrected current Tcw in Sim3** and is passed
  into `search_more_map_points_by_projection(points, f_cur, Scw, …)` at
  loop_closing.py 427–437.
- `_search_more_map_points_by_projection`, geometry_matchers.py 542–658:
  internally extracts SE3 `Rcw, tcw/s` from `Scw` and projects each candidate
  world map-point into the current camera frame using
  `P_cam = Rcw · P_world + tcw`.

So pySLAM’s composition `Sc1w = Sc1c2 · Sc2w` is correct *because* `Sc1c2`
is a **camera-frame** transform. The composition is not interchangeable with
a world-frame drift transform.

---

## 3. pySLAM final loop projection expansion workflow

```
1. seed BoW + orientation matches between current KF and candidate KF
2. prepare 3-D world points (P_w1, P_w2) and feed sim3solver
3. sim3solver returns R12, t12, s12 == Sc1c2 (camera-to-camera)
4. ProjectionMatcher.search_by_sim3 augments matches via Sim3-guided projection
5. optimizer_g2o.optimize_sim3 refines Sc1c2 with all matches
6. accept candidate if num_inliers > kLoopClosingGeometryCheckerMinKpsMatches
7. compute Sc1w = Sc1c2 · Tc2w (corrected current Tcw)
8. collect loop map points from candidate covisibility group
9. search_more_map_points_by_projection(points, f_cur, Scw=Sc1w, …)
10. final gate: num_matched_map_points >= kLoopClosingMinNumMatchedMapPoints
```

---

## 4. Local final loop projection expansion workflow (current state)

```
1. seed BoW + orientation matches between current KF and candidate KF
2. _matched_3d_points returns world-frame P_w1, P_w2
3. estimate_scale_fixed_sim3 returns R, t encoding T_world_drift such that
   P_w_loop ≈ R · P_w_current + t          (world-frame drift)
4. RANSAC seed gate kLoopClosingSE3GuidedMinSeedInliers
5. _guided_projection_refinement runs an SE3-guided projection match using
   correctly composed corrected current Tcw = Tcw_current · T_drift^{-1}
   (see _guided_projection_refinement, OK)
6. refined RANSAC; refined inliers gate (kLoopClosingGeometryCheckerMinKpsMatches)
7. collect loop map points from candidate covisibility group
8. ❌ compute Tc1w := T12 · Tc2w using world-drift T12   (BUG)
9. search_more_map_points_by_projection(points, f_cur, Rc1w, tc1w, …)
10. final gate: num_matched_map_points >= kLoopClosingMinNumMatchedMapPoints
```

---

## 5. Exact transform definitions currently in use

| Symbol | Meaning | Stored as | Frame |
|--------|---------|-----------|-------|
| `Tcw_current` | current KF camera-from-world | `current_keyframe.Tcw()` | SE3 4×4 |
| `Tcw_candidate` | candidate (loop) KF camera-from-world | `candidate.Tcw()` | SE3 4×4 |
| `T12 = estimate.T` | **world-frame drift** mapping current-world → candidate-world | `np.eye(4); R, t from sim3_solver` | SE3 4×4 (scale=1) |
| `correction_T` (`LoopCorrector.correct_loop`) | same object as `T12`; world-frame drift | Sim3Estimate.T | SE3 4×4 |
| `corrected_poses[kf]` | `Tcw_kf · inv(correction_T)` (world-drift convention) | dict | SE3 4×4 |
| `apply_correction_to_map_points` | `p_world_corrected = correction_T · p_world` | n/a | SE3 4×4 |
| **Local: corrected current Tcw passed into search_more_map_points_by_projection** | `T12 · Tc2w` | composed in `LoopGeometryChecker.check_candidates` | SE3 4×4 |
| **pySLAM: corrected current Tcw passed into the same function** | `Sc1c2 · Tc2w` where `Sc1c2` is **camera-frame** Sim3 | composed in pySLAM `LoopGeometryChecker` | Sim3 4×4 |

`search_more_map_points_by_projection` itself (geometry_matchers.py 1115–1232)
correctly expects the *current-camera-from-world* `Rcw, tcw` and projects
world-frame map points using `P_cam = Rcw · P_world + tcw` — i.e. the
function’s contract matches pySLAM. The bug is **what we pass in**, not what
the function does internally.

---

## 6. Which pose is currently passed in

`Rc1w, tc1w` extracted from `Tc1w = T12 @ Tc2w` where:

- `T12` is the world-frame drift (NOT a camera-frame transform).
- `Tc2w` is the candidate KF’s `Tcw`.

So the current code projects loop map points using a 4×4 that is *neither*
the current KF Tcw, *nor* the candidate KF Tcw, *nor* a geometrically
meaningful corrected current Tcw. In the no-drift limit (`T12 = I`) it
reduces to `Tc2w`, i.e. it would project the loop map points into the
**candidate** camera, not the current one. That is unambiguously wrong.

---

## 7. Does the local convention match pySLAM?

**No.** pySLAM composes a camera-to-camera Sim3 with the candidate Tcw to get
the corrected current Tcw. Locally, the same algebraic composition is applied
to a world-to-world drift, which is not the same operation.

The mismatch is a real geometric error, not a notational one.

---

## 8. Expected corrected formula

We want a `Tcw_current_corrected` such that, given a loop map point with
world position `p_w` (anchored in the candidate keyframe’s world frame), it
projects into the current camera at the **same** camera coordinates as the
real physical observation.

Define:
- `T_drift = T12 = estimate.T`, world-frame: `p_w_loop = T_drift · p_w_current`.
- `Tcw_current` = current KF’s drifted `Tcw`.

For a current-side world point `p_w_current` we want
`p_cam_current = Tcw_current · p_w_current` to remain invariant after the
drift correction is propagated to *both* the keyframe pose and the map point.
Substituting `p_w_corrected = T_drift · p_w_current`:

```
p_cam_current  =  Tcw_current_corrected · p_w_corrected
              =  Tcw_current_corrected · T_drift · p_w_current
              =  Tcw_current · p_w_current
=>  Tcw_current_corrected  =  Tcw_current · T_drift^{-1}
```

This is exactly what `_make_corrected_pose_map` already does for the rest of
the connected keyframes. The single inconsistent place is the call site at
lines 380–400 of `loop_closing.py`.

Sanity check (no drift): `T_drift = I` ⇒ `Tcw_corrected = Tcw_current`, and
`search_more` projects loop map points with the current KF’s actual pose —
correct. Whereas the buggy formula gives `Tcw_corrected = Tc2w` (candidate’s
pose), which is geometrically meaningless for projecting *into* the current
keyframe’s camera.

---

## 9. Initial risk assessment

- Severity: **P0** (correctness).
- Symptom 1: in the small-drift regime the call essentially projects loop map
  points into the *candidate* camera, so very few new matches will be found
  in the *current* keyframe → *true* loops can be rejected at the final
  `kLoopClosingMinNumMatchedMapPoints` gate (current threshold = 60).
- Symptom 2: with non-trivial drift the projection is in a fictitious frame
  that has no geometric relation to the current camera; any matches it does
  produce are spurious and fed into the loop fusion / essential graph step,
  potentially introducing bad correspondences that distort the map.
- Symptom 3: the final gate counts these spurious matches, so it can both
  incorrectly accept *false* loops (when noise produces extra matches by
  chance) and incorrectly reject *true* loops (when the misaligned projection
  finds no real overlap).
- Existing diagnostics expose `guided_projection_total_matches` only, and
  do not separately record seed inliers / projection-added / final-after-search-more
  / final-gate threshold. We should add those so future debugging does not
  repeat this audit.

The fix is to compose `Tcw_corrected = Tcw_current · inv(estimate.T)` and
pass that into `search_more_map_points_by_projection`. The signature of
`search_more_map_points_by_projection` will be tightened to take a single
4×4 `Tcw` named explicitly so the convention is unambiguous.

---

## 10. Planned changes (no code yet)

1. `loop_closing.py:check_candidates` — replace
   `Tc1w = T12 @ Tc2w` with
   `Tc1w_corrected = current_keyframe.Tcw() @ inv(estimate.T)` and pass it
   in. Add docstring/inline comment naming the convention.
2. `geometry_matchers.py:_search_more_map_points_by_projection` — keep the
   body (it already projects with `P_cam = Rcw·P_w + tcw`) but tighten the
   signature so callers must pass a single explicitly named
   `current_keyframe_Tcw_corrected: np.ndarray (4×4)` (or document Rcw/tcw
   as `*_current_camera_from_world_corrected`). Add the convention to the
   docstring.
3. Add diagnostics: `seed_inliers`, `candidate_covisible_points`,
   `projected_visible_points` (returned via the matcher), `new_projection_matches`,
   `total_final_matches`, `final_gate_threshold`, `accepted_or_rejected` to
   each candidate report and to `LoopDiagnostics.candidate_pair_reports` so
   the loop-debug CSV exposes them.
4. Tests in `tests/visual_slam/orbslam/test_checkpoint_2_28A_loop_projection_expansion.py`
   covering the seven required cases.

No threshold changes (`kLoopClosingMinNumMatchedMapPoints`,
`kLoopClosingGeometryCheckerMinKpsMatches`,
`kLoopClosingSE3GuidedMinSeedInliers`, `consistency_threshold`) are part of
this audit.
