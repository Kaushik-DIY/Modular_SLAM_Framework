from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import wrap_angle
from slam_core.matching.core import MatcherManager, MatchResult


@dataclass
class MotionFilterParams:
    """
    Generic local-SLAM motion filter thresholds in tracking/world frame.
    In Hector orchestration, scan_to_map will ignore sparse insertion and update every scan.
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
    ):
        self.matcher_manager = matcher_manager
        self.extrap = extrapolator
        self.motion_params = motion_params

        self.last_insert_pose: Optional[Pose2] = None
        self.last_insert_time: Optional[float] = None

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
        # - scan_to_map updates every scan
        # - scan_to_submap still uses sparse insertion/update
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

        # Only update the extrapolator on a genuine successful match.
        # Feeding fallback (extrapolated) poses back into the extrapolator causes
        # the velocity estimate to grow unboundedly, eventually placing every
        # prediction outside map bounds -> permanent zero-inlier failure.
        if result.success:
            self.extrap.update(t, final_pose)

        did_insert = False

        if matcher_name == "scan_to_map":
            # Bootstrap rule:
            # The very first scan must seed the map, otherwise scan-to-map can never start.
            # After initialization, only successful matches may be inserted.
            active_matcher = self.matcher_manager.active_matcher
            map_initialized = bool(getattr(active_matcher, "initialized", False))

            if not map_initialized:
                did_insert = self.matcher_manager.update_active_target(
                    pose_world=final_pose,
                    scan_points_local=scan_points_local,
                    t=t,
                )
                if did_insert:
                    self.last_insert_pose = final_pose
                    self.last_insert_time = t

            elif result.success:
                did_insert = self.matcher_manager.update_active_target(
                    pose_world=final_pose,
                    scan_points_local=scan_points_local,
                    t=t,
                )
                if did_insert:
                    self.last_insert_pose = final_pose
                    self.last_insert_time = t

            else:
                did_insert = False

        elif do_insert:
            did_insert = self.matcher_manager.update_active_target(
                pose_world=final_pose,
                scan_points_local=scan_points_local,
                t=t,
            )
            if did_insert:
                self.last_insert_pose = final_pose
                self.last_insert_time = t

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

    def last_motion_debug(self) -> Optional[dict]:
        return self._last_motion_debug

    def last_match_result(self) -> Optional[MatchResult]:
        return self._last_match_result