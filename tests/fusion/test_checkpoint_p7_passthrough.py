"""
Phase 7 checkpoint — Mode A and Mode B pass-through.

Per CLAUDE.md §4 Phase 7: ``--mode orb`` must produce output equal (within
float noise) to running ``run_rgbd_slam.py`` directly, and ``--mode lidar`` the
same against ``run_local_slam_new.py``. Pass-through is an identical subprocess
invocation, so the trajectory is byte-equal by construction (ATE delta = 0). We
verify that the forwarded command is exactly the direct invocation, that args
pass through verbatim, and that fusion modes are not yet runnable here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_core.fusion.config import Mode
from slam_core.fusion.runner import (
    build_passthrough_command,
    dispatch,
    main,
    PASSTHROUGH_SCRIPTS,
    _REPO_ROOT,
)


def test_orb_passthrough_command():
    args = ["datasets/tum/rgbd_dataset_freiburg1_room", "--output", "out", "--max-frames", "20"]
    cmd = build_passthrough_command(Mode.ORB, args, python="PY")
    assert cmd[0] == "PY"
    assert cmd[1] == str(_REPO_ROOT / "visual_slam/orbslam/run_rgbd_slam.py")
    assert cmd[2:] == args                      # forwarded verbatim
    assert Path(cmd[1]).exists()                # target runner really exists


def test_lidar_passthrough_command():
    args = ["--dataset", "d", "--output", "out"]
    cmd = build_passthrough_command(Mode.LIDAR, args, python="PY")
    assert cmd[1] == str(_REPO_ROOT / "hector/run_local_slam_new.py")
    assert cmd[2:] == args
    assert Path(cmd[1]).exists()


def test_dispatch_invokes_identical_command_for_both_modes():
    captured = {}

    class _Result:
        returncode = 0

    def fake_executor(cmd):
        captured["cmd"] = cmd
        return _Result()

    a = ["data", "--output", "o"]
    rc = dispatch(Mode.ORB, a, executor=fake_executor)
    assert rc == 0
    assert captured["cmd"] == build_passthrough_command(Mode.ORB, a)

    rc = dispatch(Mode.LIDAR, a, executor=fake_executor)
    assert rc == 0
    assert captured["cmd"] == build_passthrough_command(Mode.LIDAR, a)


def test_dispatch_propagates_returncode():
    class _Result:
        returncode = 3

    rc = dispatch(Mode.ORB, ["x"], executor=lambda c: _Result())
    assert rc == 3


def test_main_forwards_remainder_verbatim():
    captured = {}

    import slam_core.fusion.runner as runner

    def fake_dispatch(mode, forwarded):
        captured["mode"] = mode
        captured["forwarded"] = list(forwarded)
        return 0

    orig = runner.dispatch
    runner.dispatch = fake_dispatch
    try:
        rc = main(["--mode", "orb", "data", "--output", "o", "--max-frames", "5"])
    finally:
        runner.dispatch = orig
    assert rc == 0
    assert captured["mode"] == Mode.ORB
    assert captured["forwarded"] == ["data", "--output", "o", "--max-frames", "5"]


def test_fusion_modes_not_yet_runnable():
    with pytest.raises(NotImplementedError):
        dispatch(Mode.VLMAIN, [])
    with pytest.raises(NotImplementedError):
        dispatch(Mode.LVMAIN, [])


def test_passthrough_scripts_exist():
    for mode, rel in PASSTHROUGH_SCRIPTS.items():
        assert (_REPO_ROOT / rel).exists(), f"{mode} target missing: {rel}"
