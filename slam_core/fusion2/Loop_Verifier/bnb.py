"""Branch-and-bound (B&B) scan loop verifier.

Candidate-local occupancy grid + bounded search over the query scan.
"""
from __future__ import annotations

import numpy as np

import fusion_core as fc

from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.backend import SharedMap
from slam_core.fusion2.Loop_Verifier.neighborhood import candidate_neighborhood


def verify_candidate_bnb(shared: SharedMap, cfg: FusionV2Config, query_id: int,
                         query_scan: np.ndarray, query_pose: fc.Pose2,
                         cand_id: int):
    """Candidate-local B&B: grid from scans grouped around the candidate
    (query's own temporal neighborhood excluded), then bounded search."""
    sigs, poses = candidate_neighborhood(shared, cfg, query_id, cand_id)
    if len(sigs) < 2:
        return None
    # Candidate neighbourhood becomes the local reference map for the query scan.
    grid = fc.assemble_local_grid(sigs, poses, shared.grid_cfg)
    return fc.bnb_match(grid, query_scan, query_pose, shared.bnb_cfg)
