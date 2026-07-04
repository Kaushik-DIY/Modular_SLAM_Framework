"""Candidate neighbourhood retrieval shared by the scan verifiers (B&B, ICP)."""
from __future__ import annotations

import fusion_core as fc

from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.backend import SharedMap


def candidate_neighborhood(shared: SharedMap, cfg: FusionV2Config, query_id: int,
                           cand_id: int):
    """Scans grouped around the candidate, query's temporal trail excluded."""
    nb = fc.retrieve_neighborhood(shared.memory, shared.graph, cand_id,
                                  graph_depth=cfg.retrieval_graph_depth,
                                  metric_radius=cfg.retrieval_metric_radius,
                                  scans_only=True)
    sigs, poses = [], []
    for s, p in zip(nb.signatures, nb.poses):
        if abs(s.id - query_id) < cfg.min_kf_separation:
            continue  # never let the query (or its recent trail) verify itself
        sigs.append(s)
        poses.append(p)
    return sigs, poses
