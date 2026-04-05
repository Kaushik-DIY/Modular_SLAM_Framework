from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import inverse_pose, pose_compose, wrap_angle
from slam_core.loop_closure import (
    ClosureTarget,
    ConstraintSink,
    LoopClosureConfig,
    LoopClosureManager,
    LoopConstraint,
    LoopMatchResult,
    LoopNode,
    LoopVerifier,
    TargetProvider,
    pose_translation_distance,
)
from slam_core.matching.scan_to_submap import (
    CartoRefinementProblem,
    ScanToSubmapMatcher,
    Submap2D,
    correlative_match_two_stage,
)
from carto.pose_graph.pose_graph_2d import PoseGraph2D


class CartoTargetProvider(TargetProvider):
    def __init__(self, matcher: ScanToSubmapMatcher, pose_graph: PoseGraph2D) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph

    def get_candidate_targets_for_node(
        self,
        node: LoopNode,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[ClosureTarget]:
        if len(all_nodes) <= config.min_node_index_separation:
            return []

        finished_submaps = self.matcher.submap_builder.get_finished_submaps()
        if not finished_submaps:
            return []

        newest_finished_id = max(int(sm.id) for sm in finished_submaps)

        ranked = []
        for submap in finished_submaps:
            submap_id_int = int(submap.id)
            submap_id = str(submap_id_int)

            if submap_id in node.local_target_ids:
                continue

            if (newest_finished_id - submap_id_int) < int(config.recent_finished_submap_exclusion):
                continue

            target_pose = self.pose_graph.get_submap_pose(submap_id_int)
            dist = pose_translation_distance(node.pose_guess_global, target_pose)
            if dist > config.spatial_search_radius:
                continue

            ranked.append((dist, submap, target_pose))

        ranked.sort(key=lambda x: float(x[0]))
        ranked = ranked[: int(config.max_candidate_targets_per_new_node)]

        targets: List[ClosureTarget] = []
        for _, submap, target_pose in ranked:
            targets.append(
                ClosureTarget(
                    target_id=str(int(submap.id)),
                    target_type="submap",
                    pose_global=target_pose,
                    is_finished=True,
                    is_fixed=False,
                    map_view=submap,
                )
            )

        return targets

    def get_finished_target(self, target_id: str) -> ClosureTarget:
        submap = self.matcher.submap_builder.get_submap_by_id(int(target_id))
        target_pose = self.pose_graph.get_submap_pose(int(target_id))
        return ClosureTarget(
            target_id=str(target_id),
            target_type="submap",
            pose_global=target_pose,
            is_finished=True,
            is_fixed=False,
            map_view=submap,
        )

    def get_candidate_nodes_for_finished_target(
        self,
        target: ClosureTarget,
        all_nodes: Dict[int, LoopNode],
        config: LoopClosureConfig,
    ) -> List[LoopNode]:
        if not all_nodes:
            return []

        latest_node_id = max(all_nodes.keys())
        filtered: List[LoopNode] = []

        for node_id in sorted(all_nodes.keys()):
            node = all_nodes[node_id]

            if target.target_id in node.local_target_ids:
                continue
            if (latest_node_id - node.node_id) < config.min_node_index_separation:
                continue
            if pose_translation_distance(node.pose_guess_global, target.pose_global) > config.spatial_search_radius:
                continue

            filtered.append(node)

        stride = max(1, int(config.historical_node_stride))
        return filtered[::stride]


class CartoLoopVerifier(LoopVerifier):
    def __init__(self, matcher: ScanToSubmapMatcher, min_score: float) -> None:
        self.matcher = matcher
        self.min_score = float(min_score)

    def verify(self, node: LoopNode, target: ClosureTarget) -> LoopMatchResult:
        submap: Submap2D = target.map_view
        pred_sub = pose_compose(inverse_pose(target.pose_global), node.pose_guess_global)

        pts_match = node.scan_points
        max_match_pts = int(self.matcher.corr_params.get("max_match_points", 60))
        if pts_match.shape[0] > max_match_pts:
            stride = max(1, pts_match.shape[0] // max_match_pts)
            pts_match = pts_match[::stride]

        prob_img = submap.grid.probability()

        best_sub, best_score = correlative_match_two_stage(
            prob_img=prob_img,
            grid_origin_xy=submap.grid.origin_world,
            res=submap.grid.res,
            points_local=pts_match,
            initial_submap_pose=pred_sub,
            min_valid=int(self.matcher.corr_params.get("min_valid", 20)),
            precomp_levels=int(self.matcher.corr_params.get("precomp_levels", 3)),
            coarse_level=int(self.matcher.corr_params.get("coarse_level", 2)),
            coarse_xy_window=float(self.matcher.corr_params.get("coarse_xy_window", 0.8)),
            coarse_th_window=float(self.matcher.corr_params.get("coarse_th_window", 0.3)),
            coarse_xy_step=float(self.matcher.corr_params.get("coarse_xy_step", 0.20)),
            coarse_th_step=float(self.matcher.corr_params.get("coarse_th_step", 0.08)),
            fine_level=int(self.matcher.corr_params.get("fine_level", 0)),
            fine_xy_window=float(self.matcher.corr_params.get("fine_xy_window", 0.25)),
            fine_th_window=float(self.matcher.corr_params.get("fine_th_window", 0.12)),
            fine_xy_step=float(self.matcher.corr_params.get("fine_xy_step", 0.05)),
            fine_th_step=float(self.matcher.corr_params.get("fine_th_step", 0.02)),
        )

        if best_score < self.min_score:
            return LoopMatchResult(
                success=False,
                score=float(best_score),
                matched_node_pose_global=None,
            )

        refined_sub = best_sub

        do_refine = self.matcher.corr_params.get("do_refine", True)
        if isinstance(do_refine, str):
            do_refine = do_refine.lower() in ("1", "true", "yes", "y")
        do_refine = bool(do_refine)

        if do_refine:
            x0 = np.array([best_sub.x, best_sub.y, best_sub.theta], dtype=float)
            xpred = np.array([pred_sub.x, pred_sub.y, pred_sub.theta], dtype=float)

            refine_pts = node.scan_points
            max_refine_pts = int(self.matcher.corr_params.get("max_refine_points", 180))
            if refine_pts.shape[0] > max_refine_pts:
                stride = max(1, refine_pts.shape[0] // max_refine_pts)
                refine_pts = refine_pts[::stride]

            problem = CartoRefinementProblem(
                grid=submap.grid,
                pts_local=refine_pts,
                pred_pose_sub=xpred,
                min_points=int(self.matcher.corr_params.get("refine_min_points", 20)),
                w_trans=float(self.matcher.corr_params.get("refine_w_trans", 1.0)),
                w_rot=float(self.matcher.corr_params.get("refine_w_rot", 1.0)),
            )

            x_opt = self.matcher.refine_solver.solve(x0, problem.compute_r_J).reshape(3)
            x_opt[2] = wrap_angle(x_opt[2])
            refined_sub = Pose2(float(x_opt[0]), float(x_opt[1]), float(x_opt[2]))

        matched_world = pose_compose(target.pose_global, refined_sub)

        return LoopMatchResult(
            success=True,
            score=float(best_score),
            matched_node_pose_global=matched_world,
        )


class CartoConstraintSink(ConstraintSink):
    def __init__(self, pose_graph: PoseGraph2D) -> None:
        self.pose_graph = pose_graph

    def add_loop_constraint(self, constraint: LoopConstraint) -> None:
        self.pose_graph.add_loop_submap_node_constraint(
            submap_id=int(constraint.target_id),
            node_id=int(constraint.node_id),
            relative_pose=constraint.relative_pose,
            translation_weight=constraint.translation_weight,
            rotation_weight=constraint.rotation_weight,
        )

    def maybe_optimize(self, node_count: int, config: LoopClosureConfig) -> None:
        if node_count <= 0:
            return
        if (node_count % int(config.optimize_every_n_nodes)) == 0:
            self.pose_graph.solve()


class CartoLoopClosureAdapter:
    def __init__(
        self,
        matcher: ScanToSubmapMatcher,
        pose_graph: PoseGraph2D,
        config: Optional[LoopClosureConfig] = None,
    ) -> None:
        self.matcher = matcher
        self.pose_graph = pose_graph
        self.config = config or LoopClosureConfig()

        provider = CartoTargetProvider(matcher=self.matcher, pose_graph=self.pose_graph)
        verifier = CartoLoopVerifier(matcher=self.matcher, min_score=self.config.min_score)
        sink = CartoConstraintSink(pose_graph=self.pose_graph)

        self.manager = LoopClosureManager(
            config=self.config,
            provider=provider,
            verifier=verifier,
            sink=sink,
        )

    def on_new_node(
        self,
        node_id: int,
        scan_points,
        pose_guess_global: Pose2,
        timestamp: float,
        local_submap_ids: Optional[List[str]] = None,
    ) -> None:
        node = LoopNode(
            node_id=int(node_id),
            scan_points=scan_points,
            pose_guess_global=pose_guess_global,
            timestamp=float(timestamp),
            local_target_ids=list(local_submap_ids or []),
        )
        self.manager.on_new_node(node)

    def on_submap_finished(self, submap_id: int | str) -> None:
        self.manager.on_target_finished(str(submap_id))

    def process_finished_submaps(self) -> None:
        finished_ids = self.matcher.submap_builder.consume_newly_finished_ids()
        for submap_id in finished_ids:
            self.on_submap_finished(submap_id)

    def get_stats(self):
        return self.manager.get_stats()

    def get_recent_events(self, n: int = 20):
        return self.manager.get_recent_events(n)