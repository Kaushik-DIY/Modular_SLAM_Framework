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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from slam_core.fusion.config import FusionConfig, Mode

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
        raise NotImplementedError(
            "Mode C from the CLI needs --dataset/--output; call run_mode_c() "
            "directly or use _dispatch_fusion()."
        )
    if mode == Mode.LVMAIN:
        raise NotImplementedError("Mode D (lvmain) is wired in Phase 9")
    raise ValueError(f"unknown mode {mode}")


# ==========================================================================
# Mode C — Visual-main + LiDAR (ICP) verifier  (plan §7.3)
# ==========================================================================

@dataclass
class FusionRunResult:
    keyframe_ids: List[int] = field(default_factory=list)
    trajectory: List[Tuple[float, "object"]] = field(default_factory=list)  # (t, Pose2)
    frontend_poses: Dict[int, "object"] = field(default_factory=dict)        # id -> Pose2 (pre-opt)
    optimized_poses: Dict[int, "object"] = field(default_factory=dict)       # id -> Pose2
    accepted_loops: List[Tuple[int, int]] = field(default_factory=list)      # (source, target)
    loop_count: int = 0
    memory_stats: Dict[str, int] = field(default_factory=dict)


def run_mode_c(
    config: FusionConfig,
    frames: Iterable,
    visual_backend,
    *,
    base_T_cam=None,
    loop_detector=None,
    verifier=None,
    optimize_every: Optional[int] = None,
    min_index_separation: int = 10,
) -> FusionRunResult:
    """Wire the visual front-end + memory + graph + ORB proposer + ICP verifier.

    ``frames`` yields objects with ``rgb``, ``depth``, ``rgb_t``, ``scan``
    (e.g. ``FusionFrame``). ``visual_backend`` exposes ``track(rgb, depth, t)``.
    The LiDAR pipeline's PGO is never invoked — the FusionGraph owns optimization.
    """
    from slam_core.common.se2 import pose_compose, pose_inverse
    from slam_core.fusion.signature import Signature, project_pose3d_to_pose2
    from slam_core.fusion.memory import MemoryManager
    from slam_core.fusion.graph import FusionGraph, keyframe_target_id
    from slam_core.fusion.icp_verifier import ICPLoopVerifier
    from slam_core.fusion.frontends import BruteForceOrbDetector
    from slam_core.fusion.adapters import OrbLoopProposer, VisualFrontendService
    from slam_core.loop_closure import ClosureTarget, LoopNode

    def pose_relative(a, b):
        return pose_compose(pose_inverse(a), b)

    mem = MemoryManager(stm_size=config.stm_size, wm_cap=config.wm_cap,
                        ltm_cap=config.ltm_cap, rehearsal_sim=config.rehearsal_similarity)
    graph = FusionGraph(memory=mem)
    detector = loop_detector if loop_detector is not None else BruteForceOrbDetector()
    proposer = OrbLoopProposer(detector=detector,
                               min_index_separation=min_index_separation)
    verifier = verifier or ICPLoopVerifier(
        max_correspondence_distance=config.icp_max_corr,
        fitness_threshold=config.icp_fitness,
        inlier_rmse_threshold=config.icp_inlier_rmse,
    )
    service = VisualFrontendService(visual_backend)
    opt_every = config.optimize_every_n_keyframes if optimize_every is None else optimize_every

    result = FusionRunResult()
    signatures: Dict[int, Signature] = {}
    prev_sig: Optional[Signature] = None
    kf_count = 0

    for frame in frames:
        rec = service.step(frame.rgb, frame.depth, float(frame.rgb_t))
        if rec is None:
            continue

        pose2 = project_pose3d_to_pose2(rec.pose, base_T_cam)
        sig = Signature(id=rec.id, timestamp=rec.timestamp, pose=pose2,
                        keypoints=rec.keypoints, descriptors=rec.descriptors,
                        points3d=rec.points3d, scan=getattr(frame, "scan", None))
        signatures[sig.id] = sig
        result.frontend_poses[sig.id] = pose2

        mem.insert(sig)
        graph.add_node(sig)
        if prev_sig is not None:
            graph.add_neighbor_link(prev_sig, sig)

        # propose candidates against PAST keyframes, then verify with ICP
        for proposal in proposer.poll_candidates(sig):
            cand = signatures.get(int(proposal.candidate_id))
            if cand is None or not sig.has_scan or cand.scan is None:
                continue
            node = LoopNode(node_id=sig.id, scan_points=sig.scan,
                            pose_guess_global=sig.pose, timestamp=sig.timestamp)
            target = ClosureTarget(target_id=keyframe_target_id(cand.id),
                                   target_type="keyframe", pose_global=cand.pose,
                                   is_finished=True, is_fixed=False, map_view=cand.scan)
            res = verifier.verify(node, target)
            if res.success and res.matched_node_pose_global is not None:
                rel = pose_relative(cand.pose, res.matched_node_pose_global)
                graph.add_loop_edge(target_id=cand.id, source_id=sig.id, rel_pose=rel)
                mem.confirm_loop(sig.id, cand.id)
                result.accepted_loops.append((sig.id, cand.id))

        proposer.register(sig)
        mem.tick()
        kf_count += 1
        if opt_every > 0 and kf_count % opt_every == 0 and graph.loop_count > 0:
            graph.solve()
        prev_sig = sig

    if graph.loop_count > 0:
        graph.solve()

    result.keyframe_ids = list(signatures.keys())
    result.optimized_poses = graph.get_all_poses()
    result.trajectory = [(signatures[i].timestamp, graph.get_pose(i))
                         for i in result.keyframe_ids]
    result.loop_count = graph.loop_count
    result.memory_stats = {
        "stm": mem.stm_count, "wm": mem.wm_count, "ltm": mem.ltm_count,
        "live": mem.live_count,
    }
    return result


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
