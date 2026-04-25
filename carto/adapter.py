from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle
from slam_core.matching.core import MatcherManager, MatchResult
from carto.pose_graph.pose_graph_2d import PoseGraph2D


@dataclass
class MotionFilterParams:
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


class CartoLocalSlamAdapter:
    """
    Cartographer-like local SLAM adapter.

    Responsibilities
    ----------------
    - provide odometry samples to the extrapolator
    - obtain the pose prediction from the extrapolator
    - call the active local matcher with that prediction
    - update the extrapolator with the matched local pose
    - apply motion filtering and insertion policy
    """

    def __init__(
        self,
        matcher_manager: MatcherManager,
        extrapolator,
        pose_graph: Optional[PoseGraph2D] = None,
        motion_params: Optional[MotionFilterParams] = None,
        solve_every_n_nodes: int = 30,
        global_slam: Optional[Any] = None,
        heading_calibration_rad: float = 0.0,
        odom_trust: float = 1.0,
    ):
        self.matcher_manager = matcher_manager
        self.extrap = extrapolator
        self.pose_graph = pose_graph
        self.global_slam = global_slam

        self.motion_params = motion_params
        self.solve_every_n_nodes = int(solve_every_n_nodes)

        # Heading calibration is retained as a framework hook, but odometry
        # trust now belongs to the extrapolator, not adapter-side blending.
        self.heading_calibration_rad = float(heading_calibration_rad)
        self.odom_trust = float(np.clip(odom_trust, 0.0, 1.0))

        if hasattr(self.extrap, "odom_trust"):
            self.extrap.odom_trust = float(self.odom_trust)

        self.last_insert_pose: Optional[Pose2] = None
        self.last_insert_time: Optional[float] = None
        self.node_count: int = 0

        self._last_match_result: Optional[MatchResult] = None
        self._last_do_insert: bool = False
        self._last_did_insert: bool = False
        self._last_motion_debug: Optional[dict] = None

    def initialize_extrapolator(self, t0: float, pose0: Pose2) -> None:
        if hasattr(self.extrap, "add_pose"):
            self.extrap.add_pose(float(t0), pose0)
        else:
            self.extrap.update(float(t0), pose0)

    def _predict_world_pose(
        self,
        t: float,
        odom_pose_world: Optional[Pose2],
        odom_alpha: float,
    ) -> Pose2:
        """
        Feed odometry into the extrapolator and request the prediction.

        The adapter no longer blends odometry itself. The extrapolator owns
        the full prior-generation path.
        """
        if odom_pose_world is not None and hasattr(self.extrap, "add_odometry"):
            self.extrap.add_odometry(float(t), odom_pose_world)

        return self.extrap.predict(float(t))

    def process_scan(
        self,
        t: float,
        scan_points_local: np.ndarray,
        odom_pose_world: Optional[Pose2] = None,
        *,
        odom_alpha: float = 0.0,
    ) -> Tuple[Pose2, MatchResult, bool, bool]:
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

        # The extrapolator tracks the matched local pose sequence.
        if hasattr(self.extrap, "add_pose"):
            self.extrap.add_pose(t, final_pose)
        else:
            self.extrap.update(t, final_pose)

        did_insert = False

        if matcher_name == "scan_to_map":
            did_insert = self.matcher_manager.update_active_target(
                pose_world=final_pose,
                scan_points_local=scan_points_local,
                t=t,
            )

        elif do_insert:
            did_insert = self.matcher_manager.update_active_target(
                pose_world=final_pose,
                scan_points_local=scan_points_local,
                t=t,
            )

        self._last_did_insert = bool(did_insert)

        if did_insert:
            self.last_insert_pose = final_pose
            self.last_insert_time = t

            # Only add to the pose graph when the scan matcher found a valid match.
            # FALLBACK poses must not become pose-graph nodes.
            scan_matched = bool(result.success)

            if matcher_name != "scan_to_map" and self.pose_graph is not None and scan_matched:
                if hasattr(active_matcher, "get_last_inserted_submaps"):
                    insertion_submaps = active_matcher.get_last_inserted_submaps()
                elif hasattr(active_matcher, "get_active_submaps"):
                    insertion_submaps = active_matcher.get_active_submaps()
                else:
                    insertion_submaps = []

                node_id = self.pose_graph.add_node_with_intra_constraints(
                    t=t,
                    node_pose_world=final_pose,
                    active_submaps=insertion_submaps,
                )
                self.node_count += 1

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

        self.matcher_manager.push_buffered_scan(
            t=t,
            scan_points_local=scan_points_local,
            pose_world=final_pose,
            score=float(result.score),
        )

        extrap_mode = "pose_velocity"
        if hasattr(self.extrap, "_estimate_velocity_from_queue"):
            has_pose_vel = self.extrap._estimate_velocity_from_queue(self.extrap._pose_queue) is not None
            has_odom_vel = self.extrap._estimate_velocity_from_queue(self.extrap._odom_queue) is not None
            if has_pose_vel and has_odom_vel:
                extrap_mode = "blended_velocity"
            elif has_odom_vel:
                extrap_mode = "odom_velocity"

        self._last_motion_debug = {
            "dtrans": float(dtrans),
            "drot_deg": float(np.rad2deg(drot)),
            "dtime": float(dtime),
            "do_insert": bool(do_insert),
            "did_insert": bool(did_insert),
            "matcher_name": matcher_name,
            "motion_filter_insert": bool(motion_insert),
            "extrap_mode": extrap_mode,
        }

        return final_pose, result, bool(do_insert), bool(did_insert)

    def finalize(self) -> None:
        if self.global_slam is not None:
            self.global_slam.finalize()
        elif self.pose_graph is not None and self.node_count > 0:
            self.pose_graph.solve()

    def apply_optimization_correction(
        self,
        optimized: dict,
        last_node_id: int,
        correction_alpha: float = 0.5,
    ) -> None:
        """
        Nudge the extrapolator toward the most recently optimized node pose.

        This is a best-effort live correction after pose-graph optimization.
        """
        node_key = ("node", int(last_node_id))
        if node_key not in optimized:
            return

        if self.last_insert_time is None:
            return

        opt_pose = optimized[node_key]

        if not hasattr(self, "extrap") or self.extrap is None:
            return

        try:
            curr_pose = self.extrap.predict(float(self.last_insert_time))
        except Exception:
            return

        alpha = float(np.clip(correction_alpha, 0.0, 1.0))
        dx = float(opt_pose.x) - float(curr_pose.x)
        dy = float(opt_pose.y) - float(curr_pose.y)
        dth = wrap_angle(float(opt_pose.theta) - float(curr_pose.theta))

        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dth) < 1e-6:
            return

        corrected = Pose2(
            x=float(curr_pose.x) + alpha * dx,
            y=float(curr_pose.y) + alpha * dy,
            theta=wrap_angle(float(curr_pose.theta) + alpha * dth),
        )

        try:
            if hasattr(self.extrap, "correct_pose"):
                self.extrap.correct_pose(float(self.last_insert_time), corrected)
            elif hasattr(self.extrap, "add_pose"):
                self.extrap.add_pose(float(self.last_insert_time), corrected)
            else:
                self.extrap.update(float(self.last_insert_time), corrected)
        except Exception:
            pass

    def last_motion_debug(self) -> Optional[dict]:
        return self._last_motion_debug