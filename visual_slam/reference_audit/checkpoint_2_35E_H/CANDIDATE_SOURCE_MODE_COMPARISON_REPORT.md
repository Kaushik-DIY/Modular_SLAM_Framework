# Checkpoint 2.35E_H — CANDIDATE_SOURCE_MODE_COMPARISON_REPORT

## 1. Stage / checkpoint name

- `Stage E — Candidate-source parity and source-mode comparison`

## 2. pySLAM reference used

- `third_party/pyslam_reference/pyslam/loop_closing/keyframe_database.py::detect_loop_candidates`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow2.py`
- `third_party/pyslam_reference/pyslam/loop_closing/loop_detector_dbow3.py`

## 3. Local behavior before change

- `auto` silently chose the hybrid DBOW-rescored path when a native DBOW database was available.
- DBOW detector behavior and classic inverted-file behavior were not separated cleanly.
- runtime DBOW query size was effectively unbounded (`size(database)`).
- diagnostic raw-K was not separated from runtime decision-K.

## 4. Local behavior after change

- `classic_inverted` follows classic inverted-file retrieval logic.
- `dbow_detector` follows bounded top-K native DBOW detector logic.
- `hybrid_dbow_scored` remains available only as an explicit diagnostic / legacy mode.
- `auto` resolves to `classic_inverted`.
- runtime DBOW detector top-K and diagnostic raw-trace K are now separate.

## 5. Alignment scores

- `candidate source architecture`: `97/100`
  - remaining deviation: local RGB-D pipeline still keeps a temporal frame-gap filter in the classic mode
- `classic inverted mode`: `97/100`
  - remaining deviation: local implementation records richer diagnostics than pySLAM and keeps the local temporal filter
- `dbow detector mode`: `97/100`
  - remaining deviation: local result packaging is richer than pySLAM and reuses the same `LoopDetectorOutput` wrapper
- `auto source mode`: `100/100`
- `runtime K vs trace K`: `99/100`
  - remaining deviation: diagnostic raw trace still uses the same DBOW backend query API, just with a larger K

## 6. Test evidence

- `tests/visual_slam/orbslam/test_checkpoint_2_35E_candidate_source_modes.py`: `14 passed`
- updated `tests/visual_slam/orbslam/test_checkpoint_2_35A_loop_candidate_retrieval.py`: still green

## 7. Full-run evidence

No-GBA `fr1_room` run used:

- `--loop-candidate-source auto`
- resolved runtime source: `classic_inverted`

Observed funnel from `gt_retrieval_stage_funnel.csv`:

- `GT_LOOP_LIKE_TOTAL = 47`
- `RAW_DBOW_PRESENT = 44`
- `INVERTED_WORD_PRESENT = 44`
- `PASSED_CONNECTED_TEMPORAL = 44`
- `PASSED_COMMON_WORD = 14`
- `PASSED_MIN_SCORE = 13`
- `PASSED_ACCUMULATION = 12`
- `RETAINED_CANDIDATE = 8`

## 8. Honest outcome

Stage E architectural parity improved, but it did not solve the dominant retrieval loss.  
`FAILED_COMMON_WORD_FILTER` remains the dominant first-failure stage and got worse on the new run:

- previous 2.35D: `21 / 44`
- current no-GBA run: `30 / 47`

That means the source-mode correction was necessary, but not sufficient.
