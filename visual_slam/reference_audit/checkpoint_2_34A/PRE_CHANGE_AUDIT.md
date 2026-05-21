# Checkpoint 2.34A Pre-Change Audit

## Task

Checkpoint 2.34A - full `fr1_room` loop-closure validation with Global BA disabled.

## Scope

This checkpoint is validation-only. No source changes were planned unless the runner was blocked by a missing already-implemented flag or a trivial output/path bug.

## Files inspected before run

- `AGENTS.md`
- `visual_slam/orbslam/run_rgbd_slam.py`
- `visual_slam/reference_audit/checkpoint_2_33A/VALIDATION_REPORT.md`
- `visual_slam_outputs/CODEX_CHECKPOINT_2_33A_KF_LOCAL_MAPPING_SCHEDULING.md`

## Pre-run findings

- The required `.venv` Python was available at `/home/kaushik/slam_ws/.venv/bin/python`.
- Runner help confirmed the requested diagnostic/runtime flags already existed.
- Because `free -h` showed about `9.3 GiB` available RAM at run start, the requested `--memory-limit-gb 12` was reduced to `8` to keep headroom on a machine with less than 16 GB usable memory.

## Planned action

Run the full `fr1_room` sequence with loop closing enabled, Global BA disabled, map export disabled, runtime/memory/keyframe/local-map profiling enabled, and detailed loop-candidate reporting enabled.
