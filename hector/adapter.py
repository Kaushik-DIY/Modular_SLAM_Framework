from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import hector.config as cfg

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle
from slam_core.matching.core import MatcherManager, MatchResult


@dataclass
class MotionFilterParams:
    """
    Generic local-SLAM motion filter thresholds in tracking/world frame.
    In Hector orchestration, scan_to_map can override sparse insertion
    with its own throttled occupancy-map update cadence.
    """
    max_time_seconds: float
    max_distance_meters: float
    max_angle_radians: float

    min_distance_meters: float = 0.05
    min_angle_radians: float = np.deg2rad(0.5)
    max_distance_cap_meters: float = 0.50
    max_angle_cap_radians: float = np.deg2rad(10.0)


def make_motion_filter_from_expected_velocity(
    target_insert_period_s: float,
    v_expected_mps: float,
    w_expected_rps: float,
    *,
    min_dist: float = 0.05,
    min_ang: float = np.deg2rad(0.5),
    max_dist: float = 0.50,
    max_ang: float = np.deg2rad(10.0),
) -> MotionFilterParams:
    d = float(v_expected_mps) * float(target_insert_period_s)
    a = float(w_expected_rps) * float(target_insert_period_s)

    d = float(np.clip(d, min_dist, max_dist))
    a = float(np.clip(a, min_ang, max_ang))

    return MotionFilterParams(
        max_time_seconds=float(target_insert_period_s),
        max_distance_meters=d,
        max_angle_radians=a,
        min_distance_meters=min_dist,
        min_angle_radians=min_ang,
        max_distance_cap_meters=max_dist,
        max_angle_cap_radians=max_ang,
    )


def motion_filter_decision(
    pred_world: Pose2,
    t: float,
    last_pose: Optional[Pose2],
    last_time: Optional[float],
    p: MotionFilterParams,
) -> Tuple[bool, float, float, float]:
    if last_pose is None or last_time is None:
        return True, 0.0, 0.0, 0.0

    dx = pred_world.x - last_pose.x
    dy = pred_world.y - last_pose.y
    dtrans = float(np.hypot(dx, dy))
    drot = float(abs(wrap_angle(pred_world.theta - last_pose.theta)))
    dtime = float(t - last_time)

    dist_th = float(np.clip(p.max_distance_meters, p.min_distance_meters, p.max_distance_cap_meters))
    ang_th = float(np.clip(p.max_angle_radians, p.min_angle_radians, p.max_angle_cap_radians))
    time_th = float(p.max_time_seconds)

    do_insert = (dtrans > dist_th) or (drot > ang_th) or (dtime > time_th)
    return do_insert, dtrans, drot, dtime


class HectorLocalSlamAdapter:
    """
    Hector-side orchestration layer using the same swappable matcher manager.

    Responsibilities:
      - prediction via extrapolator
      - matcher invocation
      - matcher-specific target update policy
      - rolling matched buffer update

    This local stage does NOT own:
      - global pose graph
      - loop closure
      - scan matching internals
    """

    def __init__(
        self,
        matcher_manager: MatcherManager,
        extrapolator,
        motion_params: Optional[MotionFilterParams] = None,
        use_extrapolator: bool = True,
        pose_graph=None,
        global_slam=None,
        solve_every_n_nodes: int = 20,
    ):
        self.matcher_manager = matcher_manager
        self.extrap = extrapolator
        self.motion_params = motion_params
        self.use_extrapolator = use_extrapolator

        # Optional online pose-graph back-end (Cartographer-style global SLAM).
        # When set, every accepted scan-to-submap node is added to the graph and
        # the global_slam orchestrator runs loop search + periodic optimization
        # with optimized submap poses written back into the live submap builder.
        self.pose_graph = pose_graph
        self.global_slam = global_slam
        self.solve_every_n_nodes = int(solve_every_n_nodes)
        self.node_count: int = 0
        # Id of the most recently created pose-graph node (-1 before any node).
        # Used by the runner to reconstruct a dense optimized trajectory: each
        # scan inherits the correction of its most recent keyframe node.
        self.last_node_id: int = -1

        self.last_insert_pose: Optional[Pose2] = None
        self.last_insert_time: Optional[float] = None
        self._last_match_pose: Optional[Pose2] = None

        self._last_match_result: Optional[MatchResult] = None
        self._last_do_insert: bool = False
        self._last_did_insert: bool = False
        self._last_motion_debug: Optional[dict] = None

    def initialize_extrapolator(self, t0: float, pose0: Pose2) -> None:
        self.extrap.update(float(t0), pose0)

    def _predict_world_pose(self, t: float, odom_pose_world: Optional[Pose2], odom_alpha: float) -> Pose2:
        if not self.use_extrapolator:
            # Original Hector behaviour: raw odom when available, else last matched pose.
            if odom_pose_world is not None:
                return odom_pose_world
            if self._last_match_pose is not None:
                return self._last_match_pose
            return Pose2(0.0, 0.0, 0.0)

        pred_world = self.extrap.predict(t)

        if odom_pose_world is not None and odom_alpha > 0.0:
            pred_world = Pose2(
                x=(1.0 - odom_alpha) * pred_world.x + odom_alpha * odom_pose_world.x,
                y=(1.0 - odom_alpha) * pred_world.y + odom_alpha * odom_pose_world.y,
                theta=wrap_angle(
                    pred_world.theta + odom_alpha * wrap_angle(odom_pose_world.theta - pred_world.theta)
                ),
            )
        return pred_world

    def process_scan(
        self,
        k: int,
        t: float,
        scan_points_local: np.ndarray,
        odom_pose_world: Optional[Pose2] = None,
        *,
        odom_alpha: float = 0.0,
    ) -> Tuple[Pose2, MatchResult, bool, bool]:
        k = int(k)
        t = float(t)

        self.matcher_manager.maybe_activate_pending()

        active_matcher = self.matcher_manager.active_matcher
        matcher_name = getattr(active_matcher, "name", "")

        pred_world = self._predict_world_pose(t, odom_pose_world, odom_alpha)

        if self.motion_params is None:
            motion_insert, dtrans, drot, dtime = True, 0.0, 0.0, 0.0
        else:
            motion_insert, dtrans, drot, dtime = motion_filter_decision(
                pred_world=pred_world,
                t=t,
                last_pose=self.last_insert_pose,
                last_time=self.last_insert_time,
                p=self.motion_params,
            )

        # Hector-style policy:
        # - scan_to_map can throttle occupancy-map insertions
        # - scan_to_submap still uses sparse insertion/update
        if matcher_name == "scan_to_map":
            map_update_every = max(1, int(getattr(cfg, "MAP_UPDATE_EVERY", 1)))
            do_insert = ((k % map_update_every) == 0)
        else:
            do_insert = bool(motion_insert)

        result = self.matcher_manager.match(
            t=t,
            scan_points_local=scan_points_local,
            predicted_pose_world=pred_world,
            odom_pose_world=odom_pose_world,
        )

        self._last_match_result = result
        self._last_do_insert = bool(do_insert)

        final_pose = result.pose_world

        if not np.all(np.isfinite([final_pose.x, final_pose.y, final_pose.theta])):
            raise ValueError(f"Non-finite final pose from matcher: {final_pose}")

        # Track last matched pose for use as GN prior when extrapolator is off.
        if result.success:
            self._last_match_pose = final_pose
            if self.use_extrapolator:
                self.extrap.update(t, final_pose)

        did_insert = False

        if matcher_name == "scan_to_map":
            # The scan_to_map matcher handles bootstrap insertions internally
            # inside its own match() call.  The adapter must NOT call
            # update_active_target for bootstrap scans or they are double-inserted.
            is_bootstrap = (result.debug or {}).get("reason") == "bootstrap_seeding"

            if is_bootstrap:
                did_insert = False

            elif do_insert:
                # Insert at matched pose on success; at predicted pose (dead-reckoning)
                # on failure.
                #
                # WHY dead-reckoning on failure: without it, the map stagnates at the
                # bootstrap region.  The GN gradient then always pulls pose estimates
                # back to that stale region (often 1+ m away in long corridors),
                # causing every subsequent scan to fail — cascade permanent failure.
                # Inserting at the predicted pose (extrapolator + odom blend) keeps
                # the map growing along the robot's actual path so GN has local
                # context to converge against.
                did_insert = self.matcher_manager.update_active_target(
                    pose_world=final_pose,
                    scan_points_local=scan_points_local,
                    t=t,
                )
                if did_insert:
                    self.last_insert_pose = final_pose
                    self.last_insert_time = t

                if not result.success and did_insert:
                    if self.use_extrapolator:
                        self.extrap.update(t, final_pose)

        elif do_insert:
            did_insert = self.matcher_manager.update_active_target(
                pose_world=final_pose,
                scan_points_local=scan_points_local,
                t=t,
            )
            if did_insert:
                self.last_insert_pose = final_pose
                self.last_insert_time = t
                # Online pose-graph back-end (only for the submap matcher and only
                # on a genuine match — FALLBACK poses must not become nodes).
                self._maybe_add_pose_graph_node(
                    t=t,
                    final_pose=final_pose,
                    scan_points_local=scan_points_local,
                    active_matcher=active_matcher,
                    matcher_name=matcher_name,
                    scan_matched=bool(result.success),
                )

        self._last_did_insert = bool(did_insert)

        # Always push matched scan into rolling switch buffer
        self.matcher_manager.push_buffered_scan(
            t=t,
            scan_points_local=scan_points_local,
            pose_world=final_pose,
            score=float(result.score),
        )

        self._last_motion_debug = {
            "matcher_name": matcher_name,
            "dtrans": float(dtrans),
            "drot_deg": float(np.rad2deg(drot)),
            "dtime": float(dtime),
            "motion_filter_insert": bool(motion_insert),
            "do_insert": bool(do_insert),
            "did_insert": bool(did_insert),
        }

        return final_pose, result, bool(do_insert), bool(did_insert)

    def _maybe_add_pose_graph_node(
        self,
        t: float,
        final_pose: Pose2,
        scan_points_local: np.ndarray,
        active_matcher,
        matcher_name: str,
        scan_matched: bool,
    ) -> None:
        """Add a pose-graph node + intra constraints for an accepted submap scan,
        then forward to the global-SLAM orchestrator (loop search + optimize).

        Mirrors CartoLocalSlamAdapter's node-insertion logic so the Hector
        front-end gains the Cartographer back-end without changing its
        prediction/matching/insertion behaviour.
        """
        if self.pose_graph is None:
            return
        if matcher_name == "scan_to_map":
            return
        if not scan_matched:
            return

        if hasattr(active_matcher, "get_last_inserted_submaps"):
            insertion_submaps = active_matcher.get_last_inserted_submaps()
        elif hasattr(active_matcher, "get_active_submaps"):
            insertion_submaps = active_matcher.get_active_submaps()
        else:
            insertion_submaps = []

        if not insertion_submaps:
            return

        node_id = self.pose_graph.add_node_with_intra_constraints(
            t=t,
            node_pose_world=final_pose,
            active_submaps=insertion_submaps,
        )
        self.node_count += 1
        self.last_node_id = int(node_id)

        if self.global_slam is not None:
            self.global_slam.on_node_inserted(
                node_id=node_id,
                timestamp=t,
                scan_points=scan_points_local,
                pose_global=final_pose,
                insertion_submaps=insertion_submaps,
            )
        elif self.node_count % self.solve_every_n_nodes == 0:
            self.pose_graph.solve()

    def finalize(self) -> None:
        """Run the final global optimization pass at end of mapping."""
        if self.global_slam is not None:
            self.global_slam.finalize()
        elif self.pose_graph is not None and self.node_count > 0:
            self.pose_graph.solve()

    def last_motion_debug(self) -> Optional[dict]:
        return self._last_motion_debug

    def last_match_result(self) -> Optional[MatchResult]:
        return self._last_match_result
