"""
FusionGraph — keyframe-only SE(2) pose graph on top of G2oBackend2D.

RTAB_inspired_implementation_plan.md §9. Wraps the existing g2o backend
(``carto/pose_graph/backends/g2o_backend_2d.py``) unchanged (CLAUDE.md §3) and
exposes a keyframe-only API: spine (neighbour) edges and cross-modal loop
edges, plus ``solve()`` writing optimized poses back into the signatures.

Backend mapping
---------------
The g2o backend only builds EdgeSE2 between a *submap* vertex and a *node*
vertex, so a node-to-node loop edge is not expressible directly. We map each
keyframe ``k`` onto a co-located **submap k + node k** pair joined by a stiff
identity "binding" constraint. An edge between keyframes ``a`` and ``b`` is then
a constraint ``(submap_a -> node_b, z = T_a^{-1} T_b)``:

- neighbour (spine) edge -> ``INTRA_SUBMAP`` (no robust kernel, weight 1e5)
- cross-modal loop edge  -> ``INTER_SUBMAP`` (Huber kernel via the backend)

The optimized keyframe pose is read back from the submap vertex. The first
keyframe's submap is fixed as the anchor.

FusionGraph implements the ``ConstraintSink`` Protocol from
``slam_core/loop_closure.py`` (``add_loop_constraint`` + ``maybe_optimize``), so
the generic ``LoopClosureManager`` can drive it in Phases 8/9.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

import numpy as np

from slam_core.common.types import Pose2
from slam_core.common.se2 import pose_compose, pose_inverse
from slam_core.fusion.signature import Signature
from slam_core.loop_closure import LoopConstraint, LoopClosureConfig

from carto.pose_graph.backends.g2o_backend_2d import G2oBackend2D
from carto.pose_graph.constraint import (
    PoseGraphConstraint,
    ConstraintPose2D,
    INTRA_SUBMAP,
    INTER_SUBMAP,
)
from carto.common.types import Pose2 as _CartoPose2

# Edge weights (plan §9.2).
_BIND_WEIGHT = 1e6      # identity binding submap_k <-> node_k (very stiff)
_SPINE_TRANS_W = 1e5
_SPINE_ROT_W = 1e5
_LOOP_TRANS_W = 1.1e4
_LOOP_ROT_W = 1e5

_TARGET_PREFIX = "kf"


def keyframe_target_id(kf_id: int) -> str:
    """Canonical ``target_id`` string for a keyframe (for LoopConstraint)."""
    return f"{_TARGET_PREFIX}:{int(kf_id)}"


def _parse_target_id(target_id) -> int:
    """Extract the integer keyframe id from a LoopConstraint target_id."""
    if isinstance(target_id, (int, np.integer)):
        return int(target_id)
    m = re.search(r"(\d+)$", str(target_id))
    if not m:
        raise ValueError(f"Cannot parse keyframe id from target_id={target_id!r}")
    return int(m.group(1))


def _as_pose2(p) -> Pose2:
    """Rewrap any (x, y, theta) pose (e.g. carto Pose2) as a slam_core Pose2."""
    return Pose2(float(p.x), float(p.y), float(p.theta))


def _to_carto(p) -> _CartoPose2:
    """The backend's add_submap/add_node enforce isinstance(carto.Pose2)."""
    return _CartoPose2(float(p.x), float(p.y), float(p.theta))


class FusionGraph:
    def __init__(self, backend: Optional[G2oBackend2D] = None, memory=None):
        self.backend = backend if backend is not None else G2oBackend2D()
        self.memory = memory

        self._signatures: Dict[int, Signature] = {}
        self._poses: Dict[int, Pose2] = {}
        self._order: List[int] = []
        self._anchor: Optional[int] = None
        self._loop_count: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, sig: Signature) -> int:
        """Register a keyframe as a co-located submap+node pair."""
        kf = int(sig.id)
        if kf in self._signatures:
            return kf
        self._signatures[kf] = sig
        self._poses[kf] = sig.pose

        self.backend.add_submap(kf, _to_carto(sig.pose))
        self.backend.add_node(kf, _to_carto(sig.pose))
        # Stiff identity binding so submap_k and node_k move together.
        self.backend.add_constraint(
            PoseGraphConstraint(
                submap_id=kf,
                node_id=kf,
                pose=ConstraintPose2D(Pose2(0.0, 0.0, 0.0), _BIND_WEIGHT, _BIND_WEIGHT),
                tag=INTRA_SUBMAP,
            )
        )

        if self._anchor is None:
            self._anchor = kf
            self.backend.set_fixed("submap", kf)
        self._order.append(kf)
        return kf

    def add_neighbor_link(
        self,
        sig_a: Signature,
        sig_b: Signature,
        rel_pose: Optional[Pose2] = None,
        info: Optional[np.ndarray] = None,
    ) -> None:
        """Add a spine edge a->b. ``rel_pose`` defaults to T_a^{-1} T_b."""
        a, b = int(sig_a.id), int(sig_b.id)
        if rel_pose is None:
            rel_pose = pose_compose(pose_inverse(sig_a.pose), sig_b.pose)
        tw, rw = self._weights_from_info(info, _SPINE_TRANS_W, _SPINE_ROT_W)
        self.backend.add_constraint(
            PoseGraphConstraint(
                submap_id=a,
                node_id=b,
                pose=ConstraintPose2D(rel_pose, tw, rw),
                tag=INTRA_SUBMAP,
            )
        )
        if b not in sig_a.neighbors:
            sig_a.neighbors.append(b)
        if a not in sig_b.neighbors:
            sig_b.neighbors.append(a)

    def add_loop_edge(
        self,
        target_id: int,
        source_id: int,
        rel_pose: Pose2,
        translation_weight: float = _LOOP_TRANS_W,
        rotation_weight: float = _LOOP_ROT_W,
    ) -> None:
        """Add a cross-modal loop edge; ``rel_pose`` = T_target^{-1} T_source."""
        t, s = int(target_id), int(source_id)
        if t not in self._signatures or s not in self._signatures:
            raise KeyError(f"loop edge references unknown keyframe(s) {t}, {s}")
        self.backend.add_constraint(
            PoseGraphConstraint(
                submap_id=t,
                node_id=s,
                pose=ConstraintPose2D(rel_pose, translation_weight, rotation_weight),
                tag=INTER_SUBMAP,
            )
        )
        self._loop_count += 1
        ts, ss = self._signatures[t], self._signatures[s]
        if s not in ts.loops:
            ts.loops.append(s)
        if t not in ss.loops:
            ss.loops.append(t)

    # ------------------------------------------------------------------
    # ConstraintSink Protocol (slam_core/loop_closure.py)
    # ------------------------------------------------------------------
    def add_loop_constraint(self, constraint: LoopConstraint) -> None:
        """Insert a verified loop constraint (target-relative form)."""
        target_kf = _parse_target_id(constraint.target_id)
        self.add_loop_edge(
            target_id=target_kf,
            source_id=int(constraint.node_id),
            rel_pose=constraint.relative_pose,
            translation_weight=float(constraint.translation_weight),
            rotation_weight=float(constraint.rotation_weight),
        )

    def maybe_optimize(self, node_count: int, config: LoopClosureConfig) -> None:
        if config.optimize_every_n_nodes <= 0:
            return
        if node_count % config.optimize_every_n_nodes == 0:
            self.solve()

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def solve(self, max_iters: int = 30) -> Dict[int, Pose2]:
        optimized = self.backend.solve(max_iters=max_iters)
        subgraph = self.get_subgraph_for_optimization()
        out: Dict[int, Pose2] = {}
        for kf in self._signatures:
            key = ("submap", kf)
            if key not in optimized:
                continue
            pose = _as_pose2(optimized[key])
            self._poses[kf] = pose
            # Write back into the signature when it is in the optimized subgraph.
            if self.memory is None or kf in subgraph:
                self._signatures[kf].pose = pose
            out[kf] = pose
        return out

    def get_subgraph_for_optimization(self) -> Set[int]:
        """WM (+ reactivated LTM) keyframe ids; all nodes if no memory bound."""
        if self.memory is None:
            return set(self._signatures.keys())
        return {
            kf for kf in self._signatures
            if self.memory.is_in_wm(kf) or self.memory.is_in_stm(kf)
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_pose(self, kf_id: int) -> Optional[Pose2]:
        return self._poses.get(int(kf_id))

    def get_all_poses(self) -> Dict[int, Pose2]:
        return dict(self._poses)

    @property
    def loop_count(self) -> int:
        return self._loop_count

    @property
    def node_count(self) -> int:
        return len(self._signatures)

    @staticmethod
    def _weights_from_info(info, default_tw, default_rw):
        if info is None:
            return default_tw, default_rw
        info = np.asarray(info, dtype=float)
        return float(info[0, 0]), float(info[2, 2])
