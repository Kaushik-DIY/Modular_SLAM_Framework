# Checkpoint 2.35E_H — FINAL_LOOP_CLOSURE_ALIGNMENT_REVIEW

## 1. Completion status

This checkpoint is **not fully complete**.

Completed:

- source consistency check
- pre-change pySLAM audit
- stage E implementation and tests
- stage F implementation and tests
- stage G implementation and tests
- stage H implementation and tests
- full no-GBA `fr1_room` validation
- GT retrieval funnel regeneration

Not completed:

- GBA-enabled validation

Reason:

- no-GBA criteria failed because a GT-negative loop was accepted

## 2. Alignment review

| Component | Score | Status | Remaining deviation | Test evidence | Full-run evidence |
| --- | --- | --- | --- | --- | --- |
| candidate source architecture | 97 | completed | local temporal filter remains around classic retrieval | `test_checkpoint_2_35E_candidate_source_modes.py` | `auto` resolved to `classic_inverted`; hybrid no longer selected implicitly |
| classic inverted mode | 97 | completed | local diagnostics richer than pySLAM reference | `test_checkpoint_2_35E_candidate_source_modes.py`, updated `test_checkpoint_2_35A_loop_candidate_retrieval.py` | retrieval funnel now reflects the classic path, but common-word remains dominant loss |
| dbow detector mode | 97 | completed | local wrapper / result packaging differs structurally | `test_checkpoint_2_35E_candidate_source_modes.py` | bounded DBOW mode available for direct comparison without common-word / accumulation |
| auto source mode | 100 | completed | none | `test_checkpoint_2_35E_candidate_source_modes.py` | full run used `--loop-candidate-source auto` and executed `classic_inverted` |
| runtime K vs trace K | 99 | completed | same DBOW query backend reused for runtime and trace capture | `test_checkpoint_2_35E_candidate_source_modes.py` | raw trace used `K=100` without changing runtime decisions |
| accumulation / representative behavior | 97 | completed | richer local trace payloads | `test_checkpoint_2_35F_group_recall_accumulation.py` | `PASSED_ACCUMULATION = 12`; 4 extra GT pairs were represented by GT-equivalent representatives |
| group-level GT recall | 95 | completed | representative GT-equivalence inferred from GT-positive trace rows | `test_checkpoint_2_35F_group_recall_accumulation.py` | `GROUP_RECALLED_TOTAL = 12 / 44 eligible` |
| connected-pair denominator | 98 | completed | denominator reporting added offline | `test_checkpoint_2_35F_group_recall_accumulation.py` | `GT_LOOP_LIKE_CONNECTED_LOCAL = 3`, `GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP = 44` |
| consistency progression | 98 | completed | local debug/CSV plumbing differs from pySLAM structure | `test_checkpoint_2_35G_consistency_progression.py` | `PASSED_CONSISTENCY = 5`, `FAILED_CONSISTENCY = 3`, with progression trace recorded |
| BoW-guided matching | 95 | completed | local matcher still not literal pySLAM `search_by_sim3` seed path | `test_checkpoint_2_35H_loop_geometry_support.py`, `test_checkpoint_2_21_loop_closing.py` | one candidate reached geometry; false-loop safety still insufficient overall |
| SE3 / RGB-D geometry | 95 | completed | fixed-scale RGB-D SE3 remains a deliberate scope deviation from monocular Sim3 | `test_checkpoint_2_35H_loop_geometry_support.py` | local pose-prior gate disabled by default; accepted loop was still GT-negative |
| projection expansion | 96 | completed | local corrected-pose and matcher plumbing differ | `test_checkpoint_2_35H_loop_geometry_support.py`, `test_checkpoint_2_28A_loop_projection_expansion.py` | accepted false loop reached `guided_projection_matches=100`, `final_matched_map_points=107` |
| final support gate | 96 | completed | gate unchanged, but false-loop safety still insufficient overall | `test_checkpoint_2_35H_loop_geometry_support.py` | `FAILED_FINAL_SUPPORT = 1`, but one GT-negative candidate still passed final support |
| loop correction / GBA readiness | 90 | unresolved | false GT-negative loop accepted in no-GBA run | no failing unit test in scope; blocked by full-run evidence | accepted loop `KF42 <-> KF4` is GT-negative, so GBA was not run |

## 3. No-GBA validation verdict

Run summary:

- tracking OK / attempted: `1362 / 1362`
- tracking lost: `0`
- accepted runtime loops: `1`
- GT-loop-like accepted pairs from analyzer: `0`
- dominant first failure: `FAILED_COMMON_WORD_FILTER`

Critical blocker:

- accepted loop `KF42 <-> KF4` is GT-negative

## 4. Honest overall assessment

The code is more source-aligned and much better instrumented than the 2.35D starting point:

- `auto` no longer hides the hybrid path
- group-level recall is now measured correctly
- consistency progression is observable
- geometry / support traces are explicit
- the non-reference pose gate is disabled by default

But the pipeline is **not benchmark-ready yet** because false-loop safety is still insufficient.

## 5. Recommended next action

`Checkpoint follow-up: classic common-word parity + false-loop safety audit`

Priority:

1. inspect the common-word stage for GT-like pairs that remain raw-visible but are rejected before scoring
2. inspect why the accepted false loop `KF42 <-> KF4` gathered enough support to survive geometry / final support
3. rerun no-GBA full validation only after those two issues are corrected
