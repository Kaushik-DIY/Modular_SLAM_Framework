# Checkpoint 2.35A - Loop Geometry Gate Audit

## Accepted post-fix example
- `KF45 -> KF15`
- GT distance / rotation: `0.393 m / 10.17 deg`
- final matched map points: `73`
- outcome: accepted

This is a healthy control example: with structurally aligned retrieval, the current geometry stack can accept a plausible true loop without any threshold reduction.

## Main post-fix rejection classes

### 1. Consistency
- count: `28`
- interpretation:
  - still the most common post-fix rejection
  - no threshold change was made here by design
  - now that retrieval is aligned, consistency failures are more meaningful than they were in the raw-DBOW3 baseline

### 2. SE3 seed inliers
- count: `8`
- representative examples:
  - `KF17 -> KF4`, GT `2.134 m / 21.13 deg`, seed inliers `7`
  - `KF16 -> KF3`, GT `2.250 m / 43.21 deg`, seed inliers `6`
- interpretation:
  - most of these look like plausible rejections rather than suspicious false negatives
  - they are not extremely close GT loops

### 3. Final matched-map-point gate
- count: `1`
- key example:
  - `KF43 -> KF8`, GT `0.540 m / 10.41 deg`
  - seed inliers: `15`
  - final matched map points: `46`
  - rejection: `46 < 60`
- interpretation:
  - this is the most suspicious remaining false-negative style rejection
  - do not lower the `60` gate yet
  - first ask whether denser support or better projection expansion would naturally lift this pair

### 4. Estimated pose-distance gate
- count: `1`
- example:
  - `KF15 -> KF3`, GT `2.280 m / 66.43 deg`
- interpretation:
  - likely a correct conservative rejection

## Geometry conclusion
After the retrieval fix, the remaining geometry failures are much more focused. The geometry stack is no longer failing across a large noisy candidate set. The main follow-up target is not broad threshold tuning; it is careful inspection of marginal GT-loop-like pairs that now fail only at the final map-point support stage.
