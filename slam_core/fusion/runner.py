"""
Unified fusion runner — single entry point for all SLAM modes (CLAUDE.md §2.1).

    .venv/bin/python -m slam_core.fusion.runner --mode {orb,lidar,vlmain,lvmain} ...

Phase 7 implements the two pass-through modes:

- ``--mode orb``   -> delegate to ``visual_slam/orbslam/run_rgbd_slam.py``
- ``--mode lidar`` -> delegate to ``hector/run_local_slam_new.py``

Pass-through is an *identical subprocess invocation* of the existing runner with
all trailing arguments forwarded verbatim, so trajectory output is byte-equal to
running that runner directly (ATE delta = 0 by construction, plan §7.1/§7.2).

The fusion modes ``vlmain`` (Mode C) and ``lvmain`` (Mode D) are wired in
Phases 8 and 9; here they raise a clear NotImplementedError.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Sequence

from slam_core.fusion.config import Mode

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Pass-through targets, relative to the repo root.
PASSTHROUGH_SCRIPTS = {
    Mode.ORB: "visual_slam/orbslam/run_rgbd_slam.py",
    Mode.LIDAR: "hector/run_local_slam_new.py",
}


def build_passthrough_command(
    mode: Mode,
    passthrough_args: Sequence[str],
    python: str = sys.executable,
) -> List[str]:
    """Build the argv that delegates a pass-through mode to its existing runner."""
    if mode not in PASSTHROUGH_SCRIPTS:
        raise ValueError(f"{mode} is not a pass-through mode")
    script = str(_REPO_ROOT / PASSTHROUGH_SCRIPTS[mode])
    return [python, script, *passthrough_args]


def dispatch(
    mode: Mode,
    passthrough_args: Sequence[str],
    executor: Callable[[List[str]], "subprocess.CompletedProcess"] = None,
) -> int:
    """Route a mode to its handler. Returns a process-style exit code."""
    if mode in PASSTHROUGH_SCRIPTS:
        cmd = build_passthrough_command(mode, passthrough_args)
        run = executor if executor is not None else (lambda c: subprocess.run(c, cwd=str(_REPO_ROOT)))
        result = run(cmd)
        return int(getattr(result, "returncode", 0) or 0)

    if mode == Mode.VLMAIN:
        raise NotImplementedError("Mode C (vlmain) is wired in Phase 8")
    if mode == Mode.LVMAIN:
        raise NotImplementedError("Mode D (lvmain) is wired in Phase 9")
    raise ValueError(f"unknown mode {mode}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slam_core.fusion.runner",
        description="Unified multi-modal SLAM runner (orb | lidar | vlmain | lvmain).",
    )
    parser.add_argument("--mode", required=True, choices=[m.value for m in Mode],
                        help="orb/lidar pass through to the existing runners; "
                             "vlmain/lvmain run the fusion pipeline.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER,
                        help="arguments forwarded verbatim to the underlying runner "
                             "(pass-through modes).")
    args = parser.parse_args(argv)

    mode = Mode(args.mode)
    # argparse.REMAINDER keeps a leading '--' if present; drop it.
    forwarded = list(args.passthrough)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return dispatch(mode, forwarded)


if __name__ == "__main__":
    sys.exit(main())
