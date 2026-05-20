# Checkpoint 2.35A - Validation Report

## Task
pySLAM-aligned loop candidate retrieval, oracle diagnostics, and loop-closure failure root-cause analysis

## Tests run
- targeted checkpoint tests:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py`
  - result: `16 passed`
- broader non-C++ visual SLAM slice:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/visual_slam/orbslam -k "not cpp_slam_core"`
  - result: `325 passed, 1 skipped, 94 deselected`

## Known C++ segfault status
- no C++ code was modified
- known C++-side segfault issues remain out of scope for this checkpoint

## Baseline full-loop diagnostic result
- output: `visual_slam_outputs/checkpoint_2_35A/baseline_fr1_room_full_loop_oracle`
- tracking: `1362/1362` OK
- accepted loops: `2`
- key structural issue found:
  - DBOW3 raw retained-candidate behavior diverged heavily from the pySLAM-style inverted-file path
  - compare summary: `417` DBOW3 candidates vs `44` inverted-file retained candidates, only `41` overlaps

## Post-retrieval-fix diagnostic result
- output: `visual_slam_outputs/checkpoint_2_35A/post_retrieval_fix_fr1_room_full_loop_oracle`
- tracking: `1362/1362` OK
- accepted loops: `1`
- compare summary:
  - DBOW3-scored candidates: `40`
  - inverted-file candidates: `40`
  - overlap: `40`
  - DBOW3-only: `0`
  - inverted-only: `0`

## Root-cause conclusion
- retrieval/query-order gap: confirmed and fixed
- candidate retrieval is no longer the dominant uncertainty
- remaining misses now cluster in:
  - consistency filtering
  - SE3 seed inlier scarcity on some pairs
  - final matched-map-point support on a small number of GT-loop-like pairs

## Benchmarkability / runtime / memory
- correctness:
  - full tracking completes cleanly before and after the retrieval fix
  - at least one post-fix true loop is accepted without threshold reduction
- runtime:
  - improved from `1058.49 s` baseline to `895.73 s` post-fix
- memory:
  - stable in both runs
  - peak RSS dropped from `1656.76 MB` to `1622.77 MB`

## Remaining risks
- exact accepted-loop count is still run-sensitive; the old `0 accepted loops` outcome from 2.34A did not reproduce during this checkpoint
- ORB feature extraction is not currently forced deterministic, so some loop-event variation between full runs is still plausible
- one GT-loop-like pair still dies at `46 < 60`, which is the clearest remaining near-false-negative

## Next recommended action
`2.35B - loop-aware geometry / support audit`
- inspect marginal GT-loop-like pairs that now fail only at consistency or final support
- keep thresholds unchanged until that audit proves a specific gate is overly conservative
- if more evidence points to sparse support, validate a loop-aware keyframe-density experiment separately from retrieval logic
