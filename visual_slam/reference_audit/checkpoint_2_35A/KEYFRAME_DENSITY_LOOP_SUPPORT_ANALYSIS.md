# Checkpoint 2.35A - Keyframe Density Loop Support Analysis

## Question
Does the lower keyframe density after 2.33A materially limit loop support on `fr1_room`, independent of retrieval correctness?

## Evidence
- baseline pre-fix full run:
  - final keyframes: `48`
  - average frames between inserted keyframes: `26.06`
  - max keyframe gap: `128`
  - GT-loop-like candidate rows: `30`
  - near-final-gate examples:
    - `KF45 -> KF9`: `59 < 60`
    - `KF47 -> KF13`: `21 < 60`
- post-fix full run:
  - final keyframes: `46`
  - average frames between inserted keyframes: `28.47`
  - max keyframe gap: `131`
  - GT-loop-like candidate rows: `5`
  - strongest near-accept:
    - `KF43 -> KF8`: `46 < 60`

## What the data says
1. Retrieval was the first-order problem.
   - Before the fix, DBOW3 raw retrieval produced a huge candidate set that was structurally inconsistent with the inverted-file path.
   - After the fix, both retained-candidate sets matched exactly.

2. Sparse support still matters on some true-loop pairs.
   - The post-fix near-accept pair `KF43 -> KF8` is GT-loop-like (`0.540 m / 10.41 deg`) but still stops at `46` final matched map points against a `60` gate.
   - That is compatible with reduced loop support from a sparse keyframe graph and thinner map-point overlap.

3. The current density is not so sparse that loop closure is impossible.
   - The post-fix run still accepted a true loop (`KF45 -> KF15`, `73` final matched map points).
   - That means density is a contributing factor on marginal cases, not a hard blocker.

## Conclusion
Keyframe density is not the primary root cause anymore. Candidate retrieval is now structurally aligned. The remaining evidence suggests that sparse loop support can hurt marginal GT-near pairs at the final projection-expansion / matched-map-point stage, but the system already has enough density to accept at least some true loops. If density is investigated next, it should be a separate checkpoint (`2.35B`) rather than folded into retrieval tuning.
