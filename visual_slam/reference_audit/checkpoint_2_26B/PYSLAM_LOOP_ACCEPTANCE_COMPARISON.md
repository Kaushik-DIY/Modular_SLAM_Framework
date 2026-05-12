# pySLAM Loop Acceptance Comparison

## Candidate Retrieval

pySLAM DBOW3 loop detection queries the native `pydbow3.Database` and then filters recent/current/connected candidates. The local implementation previously used the ORB-SLAM inverted-file/common-word path first, which produced sparse and unstable candidates. The repair adds the native DBOW3 database query path in `visual_slam/orbslam/slam/keyframe_database.py`.

## Consistency Groups

pySLAM accepts candidates after repeated covisibility-group consistency. The local consistency logic already followed that structure, but 2.26B added candidate-level diagnostics for group ids, previous groups, overlap count, consistency count, required count, and pass/fail state.

## BoW-Guided Matching

pySLAM uses BoW-guided ORB matching before geometry. The local matcher already supported BoW-guided matching; 2.26B added diagnostic counts after raw matching, ratio filtering, and orientation filtering.

## Geometry Verification

pySLAM uses Sim3 RANSAC, guided projection, then Sim3 optimization with a final inlier threshold. The local RGB-D path had a one-shot SE3/Kabsch verifier that rejected candidates before guided projection. The repair changed this to:

- robust scale-fixed SE3 RANSAC,
- lower seed gate only for starting guided projection,
- odometry pose-distance/rotation guard for low-seed guided candidates,
- guided projection refinement,
- final SE3 refinement over guided matches,
- unchanged final 20-inlier acceptance gate.

## Loop Correction and GBA Trigger

The downstream loop correction and Global BA path was already present. 2.26B also resets the tracking motion model after successful loop correction/GBA so the next frame does not use stale pre-correction velocity.

## Final Evidence

Final full Run C accepted a real loop pair, added a loop edge, ran essential graph correction, and triggered successful Global BA.
