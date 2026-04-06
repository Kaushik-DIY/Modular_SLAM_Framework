from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import inverse_pose, pose_compose, wrap_angle
from slam_core.matching.scan_to_submap.backend_base import IScanToSubmapBackend
from slam_core.matching.scan_to_submap.types import (
    ScanToSubmapBackendConfig,
    SubmapMatchDebug,
    SubmapMatchRequest,
    SubmapMatchResponse,
    SubmapSearchWindow,
)
from slam_core.matching.scan_to_submap.precomputation_grid_2d import (
    PrecomputationGrid2D,
    PrecomputationGridStack2D,
)
from slam_core.matching.scan_to_submap.refine import (
    CartoRefinementProblem,
    refine_pose_submap,
)


@dataclass(frozen=True)
class _LinearBounds:
    min_x: int
    max_x: int
    min_y: int
    max_y: int


@dataclass
class _SearchParameters:
    linear_bounds: List[_LinearBounds]
    angular_perturbations: np.ndarray
    resolution: float

    @property
    def num_scans(self) -> int:
        return int(len(self.angular_perturbations))


@dataclass
class _Candidate2D:
    scan_index: int
    x_index_offset: int
    y_index_offset: int
    orientation: float
    score: float = -1.0

    def __lt__(self, other: "_Candidate2D") -> bool:
        return self.score < other.score


def _rotate_points(points_xy: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(points_xy, dtype=float) @ R.T


def _compute_angular_step(points_local: np.ndarray, resolution: float, min_step: float) -> float:
    if points_local.size == 0:
        return float(min_step)

    dmax = float(np.max(np.linalg.norm(points_local, axis=1)))
    if dmax <= 1e-9:
        return float(min_step)

    term = 1.0 - (float(resolution) ** 2) / (2.0 * dmax * dmax)
    term = float(np.clip(term, -1.0, 1.0))
    derived = float(math.acos(term))
    if not np.isfinite(derived) or derived <= 1e-9:
        derived = float(min_step)

    return max(float(min_step), derived)


def _make_search_parameters(
    points_local: np.ndarray,
    resolution: float,
    grid_shape: Tuple[int, int],
    origin_xy: np.ndarray,
    predicted_pose_sub: Pose2,
    linear_search_window: float,
    angular_search_window: float,
    min_rotational_step: float,
) -> _SearchParameters:
    angular_step = _compute_angular_step(points_local, resolution, min_rotational_step)
    num_angular_steps = int(math.ceil(float(angular_search_window) / angular_step))
    angular_perturbations = np.array(
        [k * angular_step for k in range(-num_angular_steps, num_angular_steps + 1)],
        dtype=float,
    )

    linear_bound = int(math.ceil(float(linear_search_window) / float(resolution)))
    linear_bounds: List[_LinearBounds] = []

    h, w = int(grid_shape[0]), int(grid_shape[1])
    gx0 = int(math.floor((predicted_pose_sub.x - float(origin_xy[0])) / resolution))
    gy0 = int(math.floor((predicted_pose_sub.y - float(origin_xy[1])) / resolution))

    for _ in angular_perturbations:
        linear_bounds.append(
            _LinearBounds(
                min_x=max(-linear_bound, -gx0),
                max_x=min(linear_bound, w - 1 - gx0),
                min_y=max(-linear_bound, -gy0),
                max_y=min(linear_bound, h - 1 - gy0),
            )
        )

    return _SearchParameters(
        linear_bounds=linear_bounds,
        angular_perturbations=angular_perturbations,
        resolution=float(resolution),
    )


def _discretize_scans(
    origin_xy: np.ndarray,
    resolution: float,
    predicted_pose_sub: Pose2,
    points_local: np.ndarray,
    angular_perturbations: np.ndarray,
) -> List[np.ndarray]:
    """
    Convert rotated scans into grid indices at the predicted translation.

    Candidate x/y offsets are later applied in grid-cell units.
    """
    scans = []
    tx = float(predicted_pose_sub.x)
    ty = float(predicted_pose_sub.y)

    for dtheta in angular_perturbations:
        theta = float(predicted_pose_sub.theta + dtheta)
        pts_rot = _rotate_points(points_local, theta)

        qx = pts_rot[:, 0] + tx
        qy = pts_rot[:, 1] + ty

        gx = np.floor((qx - float(origin_xy[0])) / float(resolution)).astype(np.int32)
        gy = np.floor((qy - float(origin_xy[1])) / float(resolution)).astype(np.int32)
        scans.append(np.column_stack([gx, gy]))

    return scans


def _shrink_to_fit(
    linear_bounds: List[_LinearBounds],
    discrete_scans: List[np.ndarray],
    grid_shape: Tuple[int, int],
) -> List[_LinearBounds]:
    """
    Tighten candidate translation bounds so shifted scans remain within the grid.

    This mirrors Cartographer's 'ShrinkToFit' idea.
    """
    h, w = int(grid_shape[0]), int(grid_shape[1])
    tightened: List[_LinearBounds] = []

    for bounds, scan in zip(linear_bounds, discrete_scans):
        if scan.size == 0:
            tightened.append(bounds)
            continue

        gx = scan[:, 0]
        gy = scan[:, 1]

        min_dx = max(bounds.min_x, -int(np.min(gx)))
        max_dx = min(bounds.max_x, (w - 1) - int(np.max(gx)))
        min_dy = max(bounds.min_y, -int(np.min(gy)))
        max_dy = min(bounds.max_y, (h - 1) - int(np.max(gy)))

        tightened.append(
            _LinearBounds(
                min_x=int(min_dx),
                max_x=int(max_dx),
                min_y=int(min_dy),
                max_y=int(max_dy),
            )
        )

    return tightened


def _score_candidates(
    precomp_grid: PrecomputationGrid2D,
    discrete_scans: List[np.ndarray],
    candidates: List[_Candidate2D],
) -> None:
    for cand in candidates:
        scan = discrete_scans[cand.scan_index]
        if scan.size == 0:
            cand.score = -1.0
            continue

        gx = scan[:, 0] + int(cand.x_index_offset)
        gy = scan[:, 1] + int(cand.y_index_offset)

        valid = (
            (gx >= 0) & (gx < precomp_grid.values.shape[1]) &
            (gy >= 0) & (gy < precomp_grid.values.shape[0])
        )
        if not np.any(valid):
            cand.score = -1.0
            continue

        vals = precomp_grid.values[gy[valid], gx[valid]]
        cand.score = float(np.mean(vals))

    candidates.sort(reverse=True)


def _generate_lowest_resolution_candidates(
    search_params: _SearchParameters,
    max_depth: int,
) -> List[_Candidate2D]:
    linear_step_size = 1 << int(max_depth)
    candidates: List[_Candidate2D] = []

    for scan_index in range(search_params.num_scans):
        b = search_params.linear_bounds[scan_index]
        orientation = float(search_params.angular_perturbations[scan_index])

        for x_index_offset in range(int(b.min_x), int(b.max_x) + 1, linear_step_size):
            for y_index_offset in range(int(b.min_y), int(b.max_y) + 1, linear_step_size):
                candidates.append(
                    _Candidate2D(
                        scan_index=scan_index,
                        x_index_offset=int(x_index_offset),
                        y_index_offset=int(y_index_offset),
                        orientation=orientation,
                        score=-1.0,
                    )
                )

    return candidates


def _branch_and_bound(
    precomp_stack: PrecomputationGridStack2D,
    discrete_scans: List[np.ndarray],
    search_params: _SearchParameters,
    candidates: List[_Candidate2D],
    candidate_depth: int,
    min_score: float,
) -> _Candidate2D:
    if not candidates:
        return _Candidate2D(0, 0, 0, 0.0, score=float(min_score))

    if candidate_depth == 0:
        return candidates[0]

    best_candidate = _Candidate2D(0, 0, 0, 0.0, score=float(min_score))
    half_width = 1 << (candidate_depth - 1)

    for cand in candidates:
        if cand.score <= best_candidate.score:
            break

        higher_resolution_candidates: List[_Candidate2D] = []
        b = search_params.linear_bounds[cand.scan_index]

        for x_offset in (0, half_width):
            x_new = int(cand.x_index_offset + x_offset)
            if x_new > int(b.max_x):
                continue

            for y_offset in (0, half_width):
                y_new = int(cand.y_index_offset + y_offset)
                if y_new > int(b.max_y):
                    continue

                higher_resolution_candidates.append(
                    _Candidate2D(
                        scan_index=int(cand.scan_index),
                        x_index_offset=x_new,
                        y_index_offset=y_new,
                        orientation=float(cand.orientation),
                        score=-1.0,
                    )
                )

        if not higher_resolution_candidates:
            continue

        _score_candidates(
            precomp_grid=precomp_stack.get(candidate_depth - 1),
            discrete_scans=discrete_scans,
            candidates=higher_resolution_candidates,
        )

        child_best = _branch_and_bound(
            precomp_stack=precomp_stack,
            discrete_scans=discrete_scans,
            search_params=search_params,
            candidates=higher_resolution_candidates,
            candidate_depth=candidate_depth - 1,
            min_score=best_candidate.score,
        )

        if child_best.score > best_candidate.score:
            best_candidate = child_best

    return best_candidate


class BranchAndBoundSubmapBackend(IScanToSubmapBackend):
    """
    Cartographer-style branch-and-bound scan-to-submap backend.

    This backend mirrors the original fast correlative matcher structure:
      1. precompute forward-looking max grids,
      2. generate rotated discrete scans,
      3. score lowest-resolution candidates,
      4. recursively branch-and-bound,
      5. optionally refine the best candidate continuously.
    """

    def __init__(self, config: ScanToSubmapBackendConfig, refine_solver) -> None:
        self.config = config
        self.refine_solver = refine_solver

        self.search_window = config.coarse or SubmapSearchWindow(
            xy_window=0.8,
            theta_window=0.3,
            xy_step=0.05,
            theta_step=0.02,
            level=0,
        )

    def match(self, request: SubmapMatchRequest) -> SubmapMatchResponse:
        submap = request.submap
        submap_pose_world = request.submap_pose_world

        pred_sub = pose_compose(
            inverse_pose(submap_pose_world),
            request.predicted_pose_world,
        )

        points_local = np.asarray(request.scan_points_local, dtype=float)
        n_input = int(points_local.shape[0])

        max_match_pts = int(self.config.max_match_points)
        points_match = points_local
        if points_match.shape[0] > max_match_pts:
            stride = max(1, points_match.shape[0] // max_match_pts)
            points_match = points_match[::stride]

        prob_grid = submap.grid.probability().astype(np.float32)

        precomp_stack = PrecomputationGridStack2D(
            prob_grid=prob_grid,
            branch_and_bound_depth=int(self.config.bnb_depth_limit),
        )

        search_params = _make_search_parameters(
            points_local=points_match,
            resolution=float(submap.grid.res),
            grid_shape=prob_grid.shape,
            origin_xy=np.asarray(submap.grid.origin_world, dtype=float),
            predicted_pose_sub=pred_sub,
            linear_search_window=float(self.search_window.xy_window),
            angular_search_window=float(self.search_window.theta_window),
            min_rotational_step=float(self.config.bnb_min_rotational_step),
        )

        discrete_scans = _discretize_scans(
            origin_xy=np.asarray(submap.grid.origin_world, dtype=float),
            resolution=float(submap.grid.res),
            predicted_pose_sub=pred_sub,
            points_local=points_match,
            angular_perturbations=search_params.angular_perturbations,
        )

        search_params.linear_bounds = _shrink_to_fit(
            linear_bounds=search_params.linear_bounds,
            discrete_scans=discrete_scans,
            grid_shape=prob_grid.shape,
        )

        lowest_resolution_candidates = _generate_lowest_resolution_candidates(
            search_params=search_params,
            max_depth=precomp_stack.max_depth(),
        )
        _score_candidates(
            precomp_grid=precomp_stack.get(precomp_stack.max_depth()),
            discrete_scans=discrete_scans,
            candidates=lowest_resolution_candidates,
        )

        if not lowest_resolution_candidates:
            debug = SubmapMatchDebug(
                backend_type="branch_and_bound",
                coarse_score=-1.0,
                refined=False,
                num_points_match=int(points_match.shape[0]),
                num_points_refine=0,
                extra={"reason": "no_candidates"},
            )
            return SubmapMatchResponse(
                success=False,
                score=-1.0,
                pose_world=request.predicted_pose_world,
                debug=debug,
            )

        best_candidate = _branch_and_bound(
            precomp_stack=precomp_stack,
            discrete_scans=discrete_scans,
            search_params=search_params,
            candidates=lowest_resolution_candidates,
            candidate_depth=precomp_stack.max_depth(),
            min_score=float(self.config.min_score),
        )

        coarse_score = float(best_candidate.score)
        coarse_pose_sub = Pose2(
            float(pred_sub.x + best_candidate.x_index_offset * submap.grid.res),
            float(pred_sub.y + best_candidate.y_index_offset * submap.grid.res),
            float(wrap_angle(pred_sub.theta + best_candidate.orientation)),
        )

        refined = False
        final_pose_sub = coarse_pose_sub
        final_score = coarse_score

        if bool(self.config.do_refine) and coarse_score >= float(self.config.min_score):
            refine_points = points_local
            max_refine_pts = int(self.config.max_refine_points)
            if refine_points.shape[0] > max_refine_pts:
                stride = max(1, refine_points.shape[0] // max_refine_pts)
                refine_points = refine_points[::stride]

            problem = CartoRefinementProblem(
                grid=submap.grid,
                pts_local=refine_points,
                pred_pose_sub=np.array([pred_sub.x, pred_sub.y, pred_sub.theta], dtype=float),
                min_points=int(self.config.refine_min_points),
                w_trans=float(self.config.refine_w_trans),
                w_rot=float(self.config.refine_w_rot),
            )

            final_pose_sub = refine_pose_submap(
                refine_solver=self.refine_solver,
                problem=problem,
                initial_pose_sub=coarse_pose_sub,
            )
            refined = True

        final_pose_world = pose_compose(submap_pose_world, final_pose_sub)
        success = bool(final_score >= float(self.config.min_score))

        debug = SubmapMatchDebug(
            backend_type="branch_and_bound",
            coarse_score=float(coarse_score),
            refined=bool(refined),
            num_points_match=int(points_match.shape[0]),
            num_points_refine=min(n_input, int(self.config.max_refine_points)),
            extra={
                "num_rotations": int(search_params.num_scans),
                "max_depth": int(precomp_stack.max_depth()),
                "best_scan_index": int(best_candidate.scan_index),
                "best_x_index_offset": int(best_candidate.x_index_offset),
                "best_y_index_offset": int(best_candidate.y_index_offset),
                "best_orientation_offset": float(best_candidate.orientation),
                "coarse_pose_sub": coarse_pose_sub,
                "final_pose_sub": final_pose_sub,
            },
        )

        return SubmapMatchResponse(
            success=success,
            score=float(final_score),
            pose_world=final_pose_world,
            debug=debug,
        )