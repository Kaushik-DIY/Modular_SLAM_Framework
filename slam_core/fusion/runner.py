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
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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


def _passthrough_executor(cmd: List[str]) -> "subprocess.CompletedProcess":
    """Run the underlying runner with the repo root on PYTHONPATH.

    ``python /abs/path/script.py`` puts the *script's* directory on sys.path[0],
    not the repo root, so the existing runners' ``slam_core`` / ``tools`` imports
    fail. Inject the repo root (and run from it) so the delegated invocation
    behaves exactly like the user's own from the repo root.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    return subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)


def dispatch(
    mode: Mode,
    passthrough_args: Sequence[str],
    executor: Callable[[List[str]], "subprocess.CompletedProcess"] = None,
) -> int:
    """Route a mode to its handler. Returns a process-style exit code."""
    if mode in PASSTHROUGH_SCRIPTS:
        cmd = build_passthrough_command(mode, passthrough_args)
        run = executor if executor is not None else _passthrough_executor
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
    keyframe_scans: Dict[int, "object"] = field(default_factory=dict)        # id -> (M,2) scan
    accepted_loops: List[Tuple[int, int]] = field(default_factory=list)      # (source, target)
    loop_count: int = 0
    memory_stats: Dict[str, int] = field(default_factory=dict)
    per_keyframe_ms: List[float] = field(default_factory=list)               # wall time / keyframe
    frontend_poses_3d: Dict[int, "object"] = field(default_factory=dict)     # id -> 4x4 cam pose
    module_times_ms: Dict[str, float] = field(default_factory=dict)          # stage -> total ms
    loop_stats: Dict[str, int] = field(default_factory=dict)                 # proposed/verified/...
    loop_reject_reasons: Dict[str, int] = field(default_factory=dict)        # status -> count

    def runtime_summary(self) -> Dict[str, float]:
        """Per-keyframe runtime stats (ms): count/mean/median/p95/max."""
        ms = np.asarray(self.per_keyframe_ms, dtype=float)
        if ms.size == 0:
            return {"keyframes": 0, "mean_ms": 0.0, "median_ms": 0.0,
                    "p95_ms": 0.0, "max_ms": 0.0}
        return {
            "keyframes": int(ms.size),
            "mean_ms": float(ms.mean()),
            "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)),
            "max_ms": float(ms.max()),
        }


def run_mode_c(
    config: FusionConfig,
    frames: Iterable,
    visual_backend,
    *,
    base_T_cam=None,
    world_transform=None,
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
    import time

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
    mt = {k: 0.0 for k in ("frontend", "signature", "memory", "graph",
                           "propose", "verify", "solve")}
    n_proposed = n_verified = 0
    reject: Dict[str, int] = {}

    for frame in frames:
        _tf = time.perf_counter()
        rec = service.step(frame.rgb, frame.depth, float(frame.rgb_t))
        mt["frontend"] += (time.perf_counter() - _tf) * 1000.0
        if rec is None:
            continue
        _t0 = time.perf_counter()

        _t = time.perf_counter()
        pose2 = project_pose3d_to_pose2(rec.pose, base_T_cam, world_transform)
        sig = Signature(id=rec.id, timestamp=rec.timestamp, pose=pose2,
                        keypoints=rec.keypoints, descriptors=rec.descriptors,
                        points3d=rec.points3d, scan=getattr(frame, "scan", None))
        signatures[sig.id] = sig
        result.frontend_poses[sig.id] = pose2
        result.frontend_poses_3d[sig.id] = np.asarray(rec.pose, dtype=float)
        mt["signature"] += (time.perf_counter() - _t) * 1000.0

        _t = time.perf_counter()
        mem.insert(sig)
        mt["memory"] += (time.perf_counter() - _t) * 1000.0

        _t = time.perf_counter()
        graph.add_node(sig)
        if prev_sig is not None:
            graph.add_neighbor_link(prev_sig, sig)
        mt["graph"] += (time.perf_counter() - _t) * 1000.0

        # propose candidates against PAST keyframes, then verify with ICP
        _t = time.perf_counter()
        proposals = proposer.poll_candidates(sig)
        mt["propose"] += (time.perf_counter() - _t) * 1000.0
        for proposal in proposals:
            cand = signatures.get(int(proposal.candidate_id))
            if cand is None or not sig.has_scan or cand.scan is None:
                continue
            n_proposed += 1
            node = LoopNode(node_id=sig.id, scan_points=sig.scan,
                            pose_guess_global=sig.pose, timestamp=sig.timestamp)
            target = ClosureTarget(target_id=keyframe_target_id(cand.id),
                                   target_type="keyframe", pose_global=cand.pose,
                                   is_finished=True, is_fixed=False, map_view=cand.scan)
            _t = time.perf_counter()
            res = verifier.verify(node, target)
            mt["verify"] += (time.perf_counter() - _t) * 1000.0
            if res.success and res.matched_node_pose_global is not None:
                rel = pose_relative(cand.pose, res.matched_node_pose_global)
                graph.add_loop_edge(target_id=cand.id, source_id=sig.id, rel_pose=rel)
                mem.confirm_loop(sig.id, cand.id)
                result.accepted_loops.append((sig.id, cand.id))
                n_verified += 1
            else:
                reject[res.status] = reject.get(res.status, 0) + 1

        proposer.register(sig)
        mem.tick()
        kf_count += 1
        if opt_every > 0 and kf_count % opt_every == 0 and graph.loop_count > 0:
            _t = time.perf_counter()
            graph.solve()
            mt["solve"] += (time.perf_counter() - _t) * 1000.0
        result.per_keyframe_ms.append((time.perf_counter() - _t0) * 1000.0)
        prev_sig = sig

    if graph.loop_count > 0:
        _t = time.perf_counter()
        graph.solve()
        mt["solve"] += (time.perf_counter() - _t) * 1000.0

    result.module_times_ms = mt
    result.loop_stats = {"proposed": n_proposed, "verified": n_verified,
                         "rejected": n_proposed - n_verified}
    result.loop_reject_reasons = reject

    result.keyframe_ids = list(signatures.keys())
    result.optimized_poses = graph.get_all_poses()
    result.keyframe_scans = {i: signatures[i].scan for i in result.keyframe_ids}
    result.trajectory = [(signatures[i].timestamp, graph.get_pose(i))
                         for i in result.keyframe_ids]
    result.loop_count = graph.loop_count
    result.memory_stats = {
        "stm": mem.stm_count, "wm": mem.wm_count, "ltm": mem.ltm_count,
        "live": mem.live_count,
    }
    return result


# ==========================================================================
# Mode D — LiDAR-main + Visual (ORB+PnP) verifier  (plan §7.4)
# ==========================================================================

def run_mode_d(
    config: FusionConfig,
    frames: Iterable,
    lidar_backend,
    *,
    camera_K,
    base_T_cam=None,
    target_provider=None,
    verifier=None,
    optimize_every: Optional[int] = None,
    min_index_separation: int = 10,
    proximity_radius: float = 1.0,
    orb_features: int = 1000,
    max_depth: float = 8.0,
) -> FusionRunResult:
    """Wire the LiDAR front-end + memory + graph + LiDAR proposer + visual verifier.

    ``frames`` yields objects with ``rgb``, ``depth``, ``rgb_t``, ``scan``.
    ``lidar_backend`` exposes ``process_scan(scan, t)`` (or ``step``) returning
    a world ``pose`` (Pose2) and ``is_keyframe``. Each LiDAR keyframe also
    carries ORB features from the synced RGB frame so the visual verifier can
    confirm proximity proposals. The LiDAR pipeline's own g2o PGO is not driven.
    """
    import cv2
    import time

    from slam_core.common.se2 import pose_compose, pose_inverse
    from slam_core.fusion.signature import Signature
    from slam_core.fusion.memory import MemoryManager
    from slam_core.fusion.graph import FusionGraph, keyframe_target_id
    from slam_core.fusion.visual_verifier import VisualLoopVerifier
    from slam_core.fusion.frontends import ProximityTargetProvider, _backproject
    from slam_core.fusion.adapters import LidarLoopProposer, LidarFrontendService
    from slam_core.loop_closure import ClosureTarget, LoopClosureConfig, LoopNode

    def pose_relative(a, b):
        return pose_compose(pose_inverse(a), b)

    K = np.asarray(camera_K, dtype=np.float64)
    orb = cv2.ORB_create(orb_features)

    mem = MemoryManager(stm_size=config.stm_size, wm_cap=config.wm_cap,
                        ltm_cap=config.ltm_cap, rehearsal_sim=config.rehearsal_similarity)
    graph = FusionGraph(memory=mem)
    service = LidarFrontendService(lidar_backend)

    signatures: Dict[int, Signature] = {}
    provider = target_provider or ProximityTargetProvider(
        signatures, radius=proximity_radius, min_index_separation=min_index_separation)
    proposer = LidarLoopProposer(provider=provider, config=LoopClosureConfig())
    verifier = verifier or VisualLoopVerifier(
        K=K, get_signature=signatures.get,
        nndr=config.visual_nndr, min_inliers=config.visual_min_inliers,
        base_T_cam=base_T_cam)
    opt_every = config.optimize_every_n_keyframes if optimize_every is None else optimize_every

    result = FusionRunResult()
    loop_nodes: Dict[int, LoopNode] = {}
    prev_sig: Optional[Signature] = None
    next_id = 0
    kf_count = 0

    for frame in frames:
        step = service.step(getattr(frame, "scan", None), float(frame.rgb_t))
        if not step.is_keyframe:
            continue
        _t0 = time.perf_counter()

        # ORB payload from the synced RGB frame (for visual verification).
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY) if frame.rgb.ndim == 3 else frame.rgb
        kp, desc = orb.detectAndCompute(gray, None)
        if desc is None:
            continue
        kpts = np.array([k.pt for k in kp], dtype=np.float64)
        pts3d, _ = _backproject(kpts, np.asarray(frame.depth, float), K, max_depth)

        sig = Signature(id=next_id, timestamp=float(frame.rgb_t), pose=step.pose,
                        keypoints=kpts, descriptors=desc, points3d=pts3d,
                        scan=getattr(frame, "scan", None))
        next_id += 1
        signatures[sig.id] = sig
        result.frontend_poses[sig.id] = step.pose

        mem.insert(sig)
        graph.add_node(sig)
        if prev_sig is not None:
            graph.add_neighbor_link(prev_sig, sig, rel_pose=step.rel_pose)

        node = LoopNode(node_id=sig.id, scan_points=sig.scan,
                        pose_guess_global=sig.pose, timestamp=sig.timestamp)
        loop_nodes[sig.id] = node

        for proposal in proposer.poll_candidates(node, all_nodes=loop_nodes):
            cand = signatures.get(int(proposal.candidate_id))
            if cand is None:
                continue
            target = proposal.target or ClosureTarget(
                target_id=keyframe_target_id(cand.id), target_type="keyframe",
                pose_global=cand.pose, is_finished=True, is_fixed=False, map_view=cand)
            res = verifier.verify(node, target)
            if res.success and res.matched_node_pose_global is not None:
                rel = pose_relative(cand.pose, res.matched_node_pose_global)
                graph.add_loop_edge(target_id=cand.id, source_id=sig.id, rel_pose=rel)
                mem.confirm_loop(sig.id, cand.id)
                result.accepted_loops.append((sig.id, cand.id))

        mem.tick()
        kf_count += 1
        if opt_every > 0 and kf_count % opt_every == 0 and graph.loop_count > 0:
            graph.solve()
        result.per_keyframe_ms.append((time.perf_counter() - _t0) * 1000.0)
        prev_sig = sig

    if graph.loop_count > 0:
        graph.solve()

    result.keyframe_ids = list(signatures.keys())
    result.optimized_poses = graph.get_all_poses()
    result.keyframe_scans = {i: signatures[i].scan for i in result.keyframe_ids}
    result.trajectory = [(signatures[i].timestamp, graph.get_pose(i))
                         for i in result.keyframe_ids]
    result.loop_count = graph.loop_count
    result.memory_stats = {
        "stm": mem.stm_count, "wm": mem.wm_count, "ltm": mem.ltm_count,
        "live": mem.live_count,
    }
    return result


def run_fusion_cli(mode: Mode, args: Sequence[str]) -> int:
    """CLI entry for the fusion modes: run on a FusionDataset and emit outputs."""
    import numpy as np

    from slam_core.fusion.dataset import FusionDataset
    from slam_core.fusion.frontends import OrbRgbdVoBackend
    from slam_core.fusion.map_output import emit_run_outputs

    fp = argparse.ArgumentParser(prog=f"fusion {mode.value}")
    fp.add_argument("--dataset", required=True)
    fp.add_argument("--output", default="fusion_outputs")
    fp.add_argument("--max-frames", type=int, default=0)
    fp.add_argument("--stm-size", type=int, default=30)
    fp.add_argument("--wm-cap", type=int, default=200)
    fp.add_argument("--ltm-cap", type=int, default=1000)
    fp.add_argument("--optimize-every", type=int, default=30)
    fa = fp.parse_args(list(args))

    dataset = FusionDataset(fa.dataset)
    config = FusionConfig(mode=mode, dataset_path=fa.dataset, output_dir=fa.output,
                          stm_size=fa.stm_size, wm_cap=fa.wm_cap, ltm_cap=fa.ltm_cap,
                          optimize_every_n_keyframes=fa.optimize_every)
    max_frames = None if fa.max_frames in (0, -1) else fa.max_frames
    frames = list(dataset.iter_frames(max_frames=max_frames))

    if mode == Mode.VLMAIN:
        from slam_core.fusion.signature import CAMERA_GROUND_TRANSFORM
        # 500 features keeps the brute-force stand-in detector near real-time;
        # the production DBoW KeyFrameDatabase is O(1)-ish and faster still.
        backend = OrbRgbdVoBackend(dataset.K, n_features=500)
        result = run_mode_c(config, frames, backend,
                            world_transform=CAMERA_GROUND_TRANSFORM)
    else:  # LVMAIN — needs the real LiDAR front-end backend
        raise NotImplementedError(
            "Mode D CLI needs a LiDAR front-end backend; call run_mode_d() with "
            "an injected backend (scan_to_submap wiring is future work)."
        )

    paths = emit_run_outputs(result, config.output_dir)
    rt = result.runtime_summary()
    print("==== fusion run diagnostics ====")
    print(f"mode               : {mode.value}")
    print(f"keyframes          : {len(result.keyframe_ids)}")
    print(f"accepted loops     : {result.loop_count}")
    print(f"memory (stm/wm/ltm): {result.memory_stats}")
    print(f"runtime/keyframe   : mean {rt['mean_ms']:.1f} ms | median "
          f"{rt['median_ms']:.1f} ms | p95 {rt['p95_ms']:.1f} ms | max {rt['max_ms']:.1f} ms")
    print(f"  -> per-keyframe target (<=100 ms): "
          f"{'PASS' if rt['p95_ms'] <= 100.0 else 'OVER'}")
    print(f"trajectory         : {paths['trajectory']}")
    print(f"occupancy          : {paths['occupancy']}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slam_core.fusion.runner",
        description="Unified multi-modal SLAM runner (orb | lidar | vlmain | lvmain).",
    )
    parser.add_argument("--mode", required=True, choices=[m.value for m in Mode],
                        help="orb/lidar pass through to the existing runners; "
                             "vlmain/lvmain run the fusion pipeline.")
    args, rest = parser.parse_known_args(argv)

    mode = Mode(args.mode)
    if rest and rest[0] == "--":
        rest = rest[1:]

    if mode in (Mode.VLMAIN, Mode.LVMAIN):
        return run_fusion_cli(mode, rest)
    return dispatch(mode, rest)


if __name__ == "__main__":
    sys.exit(main())
