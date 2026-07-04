"""Proximity loop proposer (LiDAR side).

Bounded pose-driven proposer, RTAB-Map "proximity detection with retrieval".
"""
from __future__ import annotations

from typing import List

import numpy as np

import fusion_core as fc

from slam_core.fusion2.config import FusionV2Config
from slam_core.fusion2.backend import SharedMap


def propose_candidates(shared: SharedMap, cfg: FusionV2Config, query_id: int,
                       query_pose: fc.Pose2) -> List[int]:
    """Bounded pose-driven proximity proposer (RTAB-Map "proximity detection with
    retrieval").

    Searches resident WM AND the relevant slice of LTM: the pose graph retains
    EVERY node (transfer to LTM does not remove its graph pose), so we find the
    nodes within proposal_radius of the current pose over the whole graph, then
    REACTIVATE any chosen node that has aged into LTM (+graph neighbours) back into
    WM so the verifier can load its scan/visual payload. This is the retrieval half
    of RTAB's memory design — without it the WM-only search could never propose a
    GLOBAL loop to a place that had been transferred out of WM. STM (the most
    recent locations) is hidden; the whole step stays bounded by proposal_radius
    and max_candidates_per_query, so it remains real-time."""
    poses = np.asarray(shared.graph.poses())            # [id, x, y, theta], incl. LTM
    if len(poses) == 0:
        return []
    ids = poses[:, 0].astype(np.int64)
    d = np.hypot(poses[:, 1] - query_pose.x, poses[:, 2] - query_pose.y)
    mask = (d <= cfg.proposal_radius_m) & (np.abs(query_id - ids) >= cfg.min_kf_separation)
    if not mask.any():
        return []
    cand_ids = ids[mask]
    order = np.argsort(d[mask])                         # evaluate nearest first
    chosen: List[int] = []
    for idx in order:
        cid = int(cand_ids[idx])
        tier = shared.memory.tier(cid)
        if tier == fc.Tier.STM:                         # ignore recent neighbours
            continue
        if tier == fc.Tier.LTM:                         # bring old place back to WM
            shared.memory.reactivate(cid)
        chosen.append(cid)
        if len(chosen) >= cfg.max_candidates_per_query:
            break
    return chosen
