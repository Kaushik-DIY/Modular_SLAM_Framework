# Checkpoint 2.35I — FALSE_LOOP_KF42_KF4_GEOMETRY_AUDIT

## 1. Pair identity

- current_kf_id: `42`
- candidate_kf_id: `4`
- current_timestamp: `1305031949.768985`
- candidate_timestamp: `1305031913.665421`
- GT translation distance: `1.5235715966110683 m`
- GT rotation angle: `42.63793767760183 deg`
- GT-loop-like: `False`
- GT-near-loop: `False`
- accepted status: `True`
- source run folder: `/home/kaushik/slam_ws/visual_slam_outputs/checkpoint_2_35E_H/fr1_room_full_loop_no_gba_best_mode`

## 2. Candidate retrieval path

- candidate source mode: `classic_inverted`
- raw DBOW present: `True`
- raw DBOW rank: `3`
- raw DBOW score: `0.01816754765507136`
- inverted/shared-word present: `True`
- common words: `42`
- max common words: `44`
- common word ratio: `0.9545454545454546`
- BoW score: `0.01816754765507136`
- minScore: `0.009589279401483528`
- score/minScore: `1.894568600458207`
- accumulated score: `0.03445191178910284`
- best accumulated score: `0.03445191178910284`
- accumulation ratio: `1.0`
- retained rank: `1`
- consistency score: `3`

## 3. BoW-guided matching

- raw BoW matches: `76`
- valid map-point matches: `27`
- ratio-test rejects: `0`
- orientation rejects: `49`
- duplicate / invalid-map-point rejects after orientation: `0`
- duplicate-only rejects: not logged separately
- number of 3D correspondences entering SE3: `27`

## 4. SE3 / geometry verification

- seed correspondences: `27`
- seed inliers: `8`
- seed inlier ratio: `0.2963`
- seed SE3 mean / median / max error: not available in current logs
- refined correspondences: `108`
- refined inliers: `106`
- refined inlier ratio: `0.9815`
- refined SE3 mean / median / max error: not available in current logs
- initial SE3 translation norm: `2.6955654363695523`
- initial SE3 rotation angle: `67.86087255265781`
- available reprojection RMSE diagnostic: `0.08604695746068795`
- runtime pose-distance gate thresholds: distance=`0.0`, rotation=`0.0`
- passed runtime pose-distance gate: `True`
- old pre-H gate thresholds from `config_parameters.py` diff: distance=`0.75`, rotation=`45.0`
- old gate counterfactual using debug estimated pose: distance=`1.6757055817661657`, rotation=`37.43783096299437`
- would the old pose-distance gate have rejected it: `True`

## 5. Guided projection / expansion

- guided projection matches: `100`
- candidate group size: `8`
- candidate group keyframe IDs: `[0, 1, 2, 3, 4, 5, 6, 8]`
- candidate group map points: `1583`
- visible projected group points: `225`
- final matched map points: `107`
- final support threshold: `60`
- final support ratio vs visible projected points: `0.4756`
- final support ratio vs threshold: `1.7833`
- number of contributing covisible keyframes: not logged explicitly
- whether support comes from one keyframe or multiple keyframes: multi-keyframe support is strongly implied by the 8-keyframe candidate group, but per-keyframe contribution counts are not logged

## 6. Spatial distribution

- image grid coverage of matches: not available
- x/y spread: not available
- number of occupied grid cells: not available
- whether matches are concentrated in one image region: not measurable from current logs
- future diagnostic recommendation: log match pixel coordinates or pre-binned grid occupancy for seed and refined correspondences

## 7. Why it passed

- retrieval: it was raw-visible at DBOW rank 3, passed shared-word filtering with ratio 0.9545, passed minScore with score/minScore 1.8946, and won its accumulation group with ratio 1.0.
- consistency: the overlap against the previous consistency group was 8, advancing consistency from 2 to 3 and meeting the threshold of 3.
- seed SE3: 27 correspondences entered geometry and 8 survived seed RANSAC, which met the seed gate.
- refined SE3: guided projection expanded support to 108 correspondences with 106 refined inliers.
- projection expansion: the 8-keyframe candidate group exposed 225 visible projected group points and yielded 100 guided projection matches.
- final support: final matched map points reached 107, comfortably above the gate of 60.

## 8. Candidate non-GT rejection gates for future implementation

- refined SE3 residual quality gate using mean / median / max residuals
- refined inlier-ratio gate, not only refined count
- bidirectional projection agreement check
- spatial coverage / grid-spread gate so one-region matches cannot dominate
- candidate-group diversity requirement that support come from multiple distinct covisible keyframes
- final support ratio gate using matched/visible, not only final matched count
- reintroducing estimated-pose prior as a diagnostic or soft gate only if later evidence justifies it, not as a blind hard filter

## 9. Missing diagnostics

- seed-stage residual distribution
- refined-stage residual distribution
- per-match pixel coordinates or grid occupancy
- per-keyframe support contribution counts inside the candidate group
- bidirectional projection diagnostics
- explicit support-ratio metrics emitted by runtime instead of count-only acceptance
