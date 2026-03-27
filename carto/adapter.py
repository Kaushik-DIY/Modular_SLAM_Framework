from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle
from slam_core.matching.core import MatcherManager, MatchResult
from carto.pose_graph.pose_graph_2d import PoseGraph2D


@dataclass
class MotionFilterParams:
    """
    Cartographer-style motion filter thresholds in tracking/world frame.
    """
    max_time_seconds: float
    max_distance_meters: float
    max_angle_radians: float

    # defensive clamps
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
    """
    Derive dataset-rate-independent motion filter thresholds
    from expected robot velocities.
    """
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
    """
    Returns:
        do_insert, dtrans, drot, dtime
    All computed in tracking/world frame.
    """
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


class CartoLocalSlamAdapter:
    """
    Cartographer-style orchestration layer using a swappable matcher manager.

    Responsibilities:
      - pose prediction via extrapolator
      - motion filter
      - matcher invocation
      - target update policy
      - pose graph insertion policy
      - rolling matched buffer update

    Does NOT own:
      - matcher internals
      - submap building internals
      - scan matching internals
    """

    def __init__(
        self,
        matcher_manager: MatcherManager,
        extrapolator,
        pose_graph: Optional[PoseGraph2D] = None,
        motion_params: Optional[MotionFilterParams] = None,
        solve_every_n_nodes: int = 30,
    ):
        self.matcher_manager = matcher_manager
        self.extrap = extrapolator
        self.pose_graph = pose_graph

        self.motion_params = motion_params
        self.solve_every_n_nodes = int(solve_every_n_nodes)

        # Reference pose/time for motion statistics or insert gating
        self.last_insert_pose: Optional[Pose2] = None
        self.last_insert_time: Optional[float] = None
        self.node_count: int = 0

        self._last_match_result: Optional[MatchResult] = None
        self._last_do_insert: bool = False
        self._last_did_insert: bool = False
        self._last_motion_debug: Optional[dict] = None

    def initialize_extrapolator(self, t0: float, pose0: Pose2) -> None:
        self.extrap.update(float(t0), pose0)

    def _predict_world_pose(self, t: float, odom_pose_world: Optional[Pose2], odom_alpha: float) -> Pose2:
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
        t: float,
        scan_points_local: np.ndarray,
        odom_pose_world: Optional[Pose2] = None,
        *,
        odom_alpha: float = 0.0,
    ) -> Tuple[Pose2, MatchResult, bool, bool]:
        """
        Process one scan through:
          predict -> motion stats/filter -> matcher -> target update -> graph update -> buffer update

        Returns:
          final_pose, match_result, do_insert, did_insert
        """
        t = float(t)

        # Activate pending matcher, if a switch has been requested and buffer is ready.
        self.matcher_manager.maybe_activate_pending()

        active_matcher = self.matcher_manager.active_matcher
        matcher_name = getattr(active_matcher, "name", "")

        pred_world = self._predict_world_pose(t, odom_pose_world, odom_alpha)

        # Compute motion statistics always, even if some matcher ignores motion filtering.
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

        # Real Hector-style scan_to_map should update map every scan.
        if matcher_name == "scan_to_map":
            do_insert = True
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

        # Extrapolator always continues on final chosen pose
        self.extrap.update(t, final_pose)

        did_insert = False

        # Hector-style scan_to_map: update target every scan
        if matcher_name == "scan_to_map":
            did_insert = self.matcher_manager.update_active_target(
                pose_world=final_pose,
                scan_points_local=scan_points_local,
                t=t,
            )

        # Cartographer-style scan_to_submap: update only on motion-filter insert
        elif do_insert:
            did_insert = self.matcher_manager.update_active_target(
                pose_world=final_pose,
                scan_points_local=scan_points_local,
                t=t,
            )

        self._last_did_insert = bool(did_insert)

        # Update motion reference:
        # - for scan_to_map: every accepted scan, because map is updated every scan
        # - for scan_to_submap: only on insert events
        if did_insert:
            if matcher_name == "scan_to_map":
                self.last_insert_pose = final_pose
                self.last_insert_time = t
            elif matcher_name != "scan_to_map":
                self.last_insert_pose = final_pose
                self.last_insert_time = t

                # Pose graph insertion only for submap-style matchers
                if self.pose_graph is not None:
                    if hasattr(active_matcher, "get_active_submaps"):
                        active_for_constraints = active_matcher.get_active_submaps()
                        self.pose_graph.add_node_with_intra_constraints(
                            t=t,
                            node_pose_world=final_pose,
                            active_submaps=active_for_constraints,
                        )
                        self.node_count += 1

                        if self.node_count % self.solve_every_n_nodes == 0:
                            self.pose_graph.solve()

        # rolling matched buffer is always AFTER matching
        self.matcher_manager.push_buffered_scan(
            t=t,
            scan_points_local=scan_points_local,
            pose_world=final_pose,
            score=float(result.score),
        )

        self._last_motion_debug = {
            "dtrans": float(dtrans),
            "drot_deg": float(np.rad2deg(drot)),
            "dtime": float(dtime),
            "do_insert": bool(do_insert),
            "did_insert": bool(did_insert),
            "matcher_name": matcher_name,
            "motion_filter_insert": bool(motion_insert),
        }

        return final_pose, result, bool(do_insert), bool(did_insert)

    def finalize(self) -> None:
        if self.pose_graph is not None and self.node_count > 0:
            self.pose_graph.solve()

    def last_motion_debug(self) -> Optional[dict]:
        return self._last_motion_debug

    def last_match_result(self) -> Optional[MatchResult]:
        return self._last_match_result