# PRE_CHANGE_AUDIT

1. task/checkpoint name
Checkpoint `2_36` - common-word filter tuning pipeline

2. files inspected
- `/home/kaushik/slam_ws/CODEX_CHECKPOINT_2_36_CW_TUNING_PLAN.md`
- `/home/kaushik/slam_ws/CODEX_CHECKPOINT_2_35V_DENSE_KF_C1A_OVERRIDE.md`
- `/home/kaushik/slam_ws/visual_slam/orbslam/slam/config_parameters.py`
- `/home/kaushik/slam_ws/tests/visual_slam/orbslam/test_checkpoint_2_1_common.py`

3. pySLAM files inspected, if relevant
- None. This checkpoint is parameter-only tuning with implementation frozen.

4. root cause or current hypothesis
- The dominant loop-recall loss in `2_35V` is `FAILED_COMMON_WORD_FILTER` (`100 / 271` GT pairs).
- The current baseline values already match the `2_35V` plan starting point:
  - `kLoopClosingCommonWordRatioThreshold = 0.67`
  - `kLoopDbowDetectorTopK = 5`
  - `kMaxResultsForLoopClosure = 5`
- KF density guardrail parameters also match the expected baseline and will remain unchanged.

5. exact changes made
- None yet. Documentation scaffold only before tuning.

6. why the changes are structurally correct
- The plan explicitly allows parameter-only edits in `config_parameters.py` and assertion-only edits in the common test file.
- No implementation logic changes are allowed for this checkpoint.

7. tests added/updated
- None yet.

8. test commands run
```bash
cd /home/kaushik/slam_ws
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
```

9. test results
- Python executable confirmed: `/home/kaushik/slam_ws/.venv/bin/python`

10. dataset validation commands and results
- None yet.

11. remaining risks
- The worktree is already dirty, including tracked changes to `config_parameters.py` and the common test file, so parameter edits must be minimal and carefully reverted between runs.
- Each run is long (`~70 min`), so progress depends on disciplined background execution and monitoring.

12. next recommended action
- Run the checkpoint `2_36W` validation and full sequence first, then record results and revert to the `2_35V` parameter baseline before `2_36X`.

