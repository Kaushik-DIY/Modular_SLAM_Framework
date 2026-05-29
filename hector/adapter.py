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
        motion_filter_skip: bool = False,
        mf_keyframe_params: Optional[MotionFilterParams] = None,
    ):
        self.matcher_manager = matcher_manager
        self.extrap = extrapolator
        self.motion_params = motion_params
        self.use_extrapolator = use_extrapolator

        # Dedicated keyframe thresholds for the scan_to_map motion-filter-skip
        # decision. Kept separate from motion_params so a runner that also drives
        # scan_to_submap (e.g. live-switchable realtime viz) can give submap
        # insertion its own cadence. Falls back to motion_params when None.
        self.mf_keyframe_params = mf_keyframe_params

        # Motion-filter keyframing (scan_to_map, opt-in). When True, a scan whose
        # extrapolated motion since the last keyframe is below motion_params'
        # thresholds is SKIPPED: no GN match, no map insert. Its pose is the
        # dead-reckoned extrapolator prediction. This is the Cartographer
        # "MotionFilter selects keyframes" idea; the win is far fewer GN solves.
        # Requires use_extrapolator=True (otherwise the skipped-scan prediction
        # would just repeat the last matched pose and the map would stagnate).
        self.motion_filter_skip = bool(motion_filter_skip)
        # Diagnostics: how many scans were GN-matched vs dead-reckoned (skipped).
        self.matched_count: int = 0
        self.skipped_count: int = 0

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

        # Motion-filter keyframe-skip applies to BOTH front-ends (scan_to_map and
        # scan_to_submap). When active it uses its own keyframe thresholds
        # (mf_keyframe_params); every other case uses motion_params (which for
        # scan_to_submap is the baseline insertion-cadence filter).
        mf_skip_applies = (
            self.motion_filter_skip
            and matcher_name in ("scan_to_map", "scan_to_submap")
        )
        if mf_skip_applies and self.mf_keyframe_params is not None:
            decision_params = self.mf_keyframe_params
        else:
            decision_params = self.motion_params

        if decision_params is None:
            motion_insert, dtrans, drot, dtime = True, 0.0, 0.0, 0.0
        else:
            motion_insert, dtrans, drot, dtime = motion_filter_decision(
                pred_world=pred_world,
                t=t,
                last_pose=self.last_insert_pose,
                last_time=self.last_insert_time,
                p=decision_params,
            )

        # ------------------------------------------------------------------
        # Motion-filter skip path (scan_to_map AND scan_to_submap, opt-in).
        #
        # Once the front-end is seeded and we already have a keyframe anchor, a
        # sub-threshold scan is dead-reckoned by the extrapolator: NO scan match
        # (no GN / no correlative search), NO insert. This is the efficiency win
        # — the expensive match runs only at keyframes. The dead-reckoned pose is
        # NOT fed back into the extrapolator's matched-pose queue (only matched
        # poses and IMU samples drive velocity), matching Cartographer's design.
        #
        # "Seeded" check: scan_to_map needs its occupancy grid bootstrapped;
        # scan_to_submap just needs one inserted scan, which last_insert_pose
        # already implies — so the keyframe anchor is the sufficient gate there.
        # ------------------------------------------------------------------
        if matcher_name == "scan_to_submap":
            matcher_seeded = self.last_insert_pose is not None
        else:
            matcher_seeded = bool(
                getattr(active_matcher, "initialized", False)
                or getattr(active_matcher, "_is_initialized", False)
            )
        if (
            mf_skip_applies
            and matcher_seeded
            and self.last_insert_pose is not None
            and not motion_insert
        ):
            final_pose = pred_world
            if not np.all(np.isfinite([final_pose.x, final_pose.y, final_pose.theta])):
                raise ValueError(f"Non-finite dead-reckoned pose: {final_pose}")

            skip_result = MatchResult(
                pose_world=final_pose,
                # Score 1.0 = "trusted dead-reckoning" so map-rebuild (min_score
                # >= 0) includes the pose; method tags it for diagnostics.
                score=1.0,
                success=True,
                method="motion_filter_skip",
                debug={
                    "reason": "motion_filter_skip",
                    "dtrans": float(dtrans),
                    "drot_deg": float(np.rad2deg(drot)),
                    "dtime": float(dtime),
                },
            )
            self._last_match_result = skip_result
            self._last_match_pose = final_pose
            self._last_do_insert = False
            self.skipped_count += 1

            # scan_to_submap: STILL insert the dead-reckoned scan into the active
            # submap so it stays dense (the submap is the only local map a keyframe
            # has to match against — skipping inserts makes matches drift). We only
            # skip the expensive correlative+GN MATCH, which is the real cost. The
            # keyframe anchor (last_insert_pose/time) is NOT updated, so the motion
            # threshold keeps accumulating from the last MATCHED keyframe, and no
            # pose-graph node is added (dead-reckoned scans are not keyframes).
            # scan_to_map keeps its global grid crisp, so it inserts nothing here.
            did_insert_skip = False
            if matcher_name == "scan_to_submap":
                did_insert_skip = bool(
                    self.matcher_manager.update_active_target(
                        pose_world=final_pose,
                        scan_points_local=scan_points_local,
                        t=t,
                    )
                )
            self._last_did_insert = did_insert_skip

            # Keep the rolling switch buffer fed so a later matcher switch can
            # still warm-start from a continuous pose stream.
            self.matcher_manager.push_buffered_scan(
                t=t,
                scan_points_local=scan_points_local,
                pose_world=final_pose,
                score=1.0,
            )

            self._last_motion_debug = {
                "matcher_name": matcher_name,
                "dtrans": float(dtrans),
                "drot_deg": float(np.rad2deg(drot)),
                "dtime": float(dtime),
                "motion_filter_insert": False,
                "do_insert": False,
                "did_insert": did_insert_skip,
                "skipped": True,
            }
            return final_pose, skip_result, False, did_insert_skip

        # Hector-style policy:
        # - scan_to_map throttles inserts by map-update cadence, OR (when the
        #   motion filter is enabled) inserts exactly at keyframes
        # - scan_to_submap still uses sparse insertion/update
        if matcher_name == "scan_to_map":
            if self.motion_filter_skip:
                do_insert = bool(motion_insert)
            else:
                map_update_every = max(1, int(getattr(cfg, "MAP_UPDATE_EVERY", 1)))
                do_insert = ((k % map_update_every) == 0)
        else:
            do_insert = bool(motion_insert)

        self.matched_count += 1

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
