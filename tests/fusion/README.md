# tests/fusion/

Per-phase validation tests for the RTAB-inspired multi-modal SLAM build. Each file is a checkpoint that must pass before the next implementation phase begins.

## Naming convention

`test_checkpoint_p<N>_<short_name>.py` where `<N>` is the phase number from `CLAUDE.md` §4 and `RTAB_inspired_implementation_plan.md` §13.

Mirrors the existing `tests/visual_slam/orbslam/test_checkpoint_*` pattern (which uses `test_checkpoint_<X>_<Y>_<name>.py`); the `p` prefix distinguishes fusion phase numbers from the visual_slam checkpoint numbering.

## Phase → test file mapping

| Phase | Test file (to be added during that phase) |
|---|---|
| 1 — Foundation                          | `test_checkpoint_p1_foundation.py` |
| 2 — Memory tier                         | `test_checkpoint_p2_memory.py` |
| 3 — Fusion graph                        | `test_checkpoint_p3_graph.py` |
| 4 — ICP verifier                        | `test_checkpoint_p4_icp.py` |
| 5 — Visual verifier                     | `test_checkpoint_p5_visual.py` |
| 6 — Adapters                            | `test_checkpoint_p6_adapters.py` |
| 7 — Mode A and Mode B (pass-through)    | `test_checkpoint_p7_passthrough.py` |
| 8 — Mode C end-to-end                   | `test_checkpoint_p8_mode_c.py` |
| 9 — Mode D end-to-end                   | `test_checkpoint_p9_mode_d.py` |
| 10 — Map output                         | `test_checkpoint_p10_output.py` |
| 11 — Hardening & docs                   | (no dedicated test; profile + docs review) |

## Running

```bash
# One phase
.venv/bin/pytest tests/fusion/test_checkpoint_p1_foundation.py -x -q

# All fusion tests
.venv/bin/pytest tests/fusion/ -x -q

# Discovery sanity check
.venv/bin/pytest tests/fusion/ --collect-only -q
```

All Python execution MUST go through `.venv/bin/...` per the project venv rule (see `CLAUDE.md` §0).
