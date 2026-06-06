#!/usr/bin/env python3
"""Persistent launcher: run the RGB-D SLAM pipeline with the C++ core active
(Parameters.USE_CPP_CORE=True), then delegate to run_rgbd_slam.main().

USE_CPP_CORE has no CLI flag (it is a module-level Parameter the matchers read at
dispatch), so it must be flipped before main() runs. All other args pass straight
through to run_rgbd_slam. Replaces the previous ephemeral /tmp launcher.

Example (F0 threaded re-baseline, loops-off, profiled):
  .venv/bin/python tools/run_lab_cpp.py datasets/lab_rgbd_run_2 \
      --output visual_slam_outputs/f0_loopsoff_cpp_threaded \
      --start-local-mapping-thread --disable-loop-closing \
      --profile-runtime --profile-local-map
"""
import faulthandler
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam import run_rgbd_slam

if __name__ == "__main__":
    # Diagnostics: `kill -USR1 <pid>` dumps every thread's Python stack to stderr
    # (stdlib, no deps; no-op unless the signal is sent). Used to pin threaded deadlocks.
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    Parameters.USE_CPP_CORE = True
    print(f"[run_lab_cpp] USE_CPP_CORE = {Parameters.USE_CPP_CORE}  (SIGUSR1 -> thread dump armed)",
          flush=True)
    raise SystemExit(run_rgbd_slam.main(sys.argv[1:]))
