# Checkpoint 2.35E_H — GT_GROUP_LEVEL_RECALL_REPORT

## 1. Stage / checkpoint name

- `Stage F — Group-level GT recall and accumulation / representative audit`

## 2. Inputs

- runtime trace directory:
  - `visual_slam_outputs/checkpoint_2_35E_H/fr1_room_full_loop_no_gba_best_mode`
- GT classified input:
  - `visual_slam_outputs/checkpoint_2_35B/gt_loop_recall_post_2_35A/gt_loop_pairs_classified.csv`

## 3. Runtime logic audit result

The local classic accumulation path remains structurally aligned with pySLAM:

- uses `get_best_covisible_keyframes(10)`
- accumulates only neighbors marked for the same query
- requires `neighbor.num_loop_words > min_common_words`
- selects the best scoring representative keyframe in the group
- retains representatives above `0.75 * best_acc_score`
- deduplicates retained representatives

Alignment score:

- `accumulation / representative behavior`: `97/100`

Remaining deviation:

- local implementation adds richer trace payloads and uses frame-ID temporal filtering in the surrounding retrieval flow

## 4. Group-level recall outputs produced

- `gt_group_level_recall_summary.csv`
- `gt_group_level_false_negative_analysis.csv`

## 5. Group-level recall summary

From `gt_group_level_recall_summary.csv`:

- `GT_LOOP_LIKE_TOTAL = 47`
- `GT_LOOP_LIKE_CONNECTED_LOCAL = 3`
- `GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP = 44`
- `GROUP_RECALLED_EXACT = 8`
- `NOT_RETAINED_BUT_GT_EQUIVALENT_REPRESENTATIVE = 4`
- `GROUP_RECALLED_TOTAL = 12`
- `NOT_RETAINED_AND_LOST = 0`

Primary denominator:

- `GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP = 44`

Eligible group-level recall:

- exact retained recall: `8 / 44 = 18.18%`
- GT-equivalent representative rescue: `4 / 44 = 9.09%`
- total group-level recall: `12 / 44 = 27.27%`

## 6. Interpretation

The pair-level view understated retention loss severity in one direction and overstated it in another:

- exact retained GT pairs are still weak
- but 4 eligible GT pairs were not retained exactly and were actually represented by a GT-equivalent representative keyframe
- no eligible accumulated GT pair in this run fell into `NOT_RETAINED_AND_LOST`

## 7. Test evidence

- `tests/visual_slam/orbslam/test_checkpoint_2_35F_group_recall_accumulation.py`: `7 passed`

## 8. Honest outcome

Stage F improved the correctness of the recall interpretation and denominator accounting.  
It does not remove the dominant retrieval bottleneck, which is still the classic common-word gate.
