#!/usr/bin/env python3
"""Run RGB-D ORB-SLAM with the C++ core and IMU fallback trajectory enabled.

This mirrors tools/run_lab_cpp.py, but opt-ins the runner's IMU fallback path so
hybrid datasets with imu.csv produce an additional trajectory during visual
tracking loss. Loop closing and all other run policy stay controlled by CLI
flags passed through to run_rgbd_slam.
"""
import faulthandler
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam import run_rgbd_slam


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    Parameters.USE_CPP_CORE = True
    argv = list(sys.argv[1:])
    if "--use-imu-fallback" not in argv:
        argv.append("--use-imu-fallback")

    print(
        "[run_lab_cpp_imu] USE_CPP_CORE = True, IMU fallback enabled "
        "(SIGUSR1 -> thread dump armed)",
        flush=True,
    )
    raise SystemExit(run_rgbd_slam.main(argv))
