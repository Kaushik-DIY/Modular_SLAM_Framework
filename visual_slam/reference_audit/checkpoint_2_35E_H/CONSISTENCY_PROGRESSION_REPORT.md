# Checkpoint 2.35E_H — CONSISTENCY_PROGRESSION_REPORT

## 1. Stage / checkpoint name

- `Stage G — Consistency progression audit / correction`

## 2. pySLAM reference used

- `third_party/pyslam_reference/pyslam/loop_closing/loop_closing.py::LoopGroupConsistencyChecker.check_candidates`

## 3. Local behavior audit

The local consistency checker remains structurally close to pySLAM:

- candidate group = `candidate.get_connected_keyframes() + candidate`
- overlap with previous groups increments consistency
- new groups start at consistency `0`
- passing candidates require `current_consistency >= threshold`
- previous groups are replaced after each query

No algorithmic threshold reduction was applied.

## 4. New output produced

- `loop_consistency_progression.csv`

Recorded fields:

- current/candidate keyframe IDs
- candidate group IDs
- previous group IDs
- overlap count
- previous and new consistency counts
- threshold
- pass/fail
- GT loop labels and GT distances when available

## 5. Alignment score

- `consistency progression`: `98/100`

Remaining deviation:

- the local implementation keeps extra debug metadata and uses the local `LoopDiagnostics` / CSV plumbing

## 6. Test evidence

- `tests/visual_slam/orbslam/test_checkpoint_2_35G_consistency_progression.py`: `6 passed`

## 7. Full-run evidence

From the no-GBA run:

- `PASSED_CONSISTENCY = 5`
- first-failure `FAILED_CONSISTENCY = 3`

Interpretation:

- consistency is not the dominant failure stage
- it is now explicitly observable and no longer a black box

## 8. Honest outcome

Stage G is functionally complete for this checkpoint.  
The main remaining blocker is not consistency drift; it is the upstream retrieval / common-word behavior plus downstream false-loop safety.
