"""Fusion v2 back end: the single shared SE(2) map.

There is exactly one back end in v1 (the fusion_core SE(2) pose graph plus the
STM/WM/LTM memory hierarchy, signature store, occupancy-grid config, and B&B
config), so it lives in one file rather than a package. ``SharedMap`` bundles the
C++ objects every mode writes into; ``build_shared_map`` constructs them from the
run config. All hot-path state stays in the fusion_core extension module.
"""
from __future__ import annotations

from dataclasses import dataclass

import fusion_core as fc

from slam_core.fusion2.config import FusionV2Config


@dataclass
class SharedMap:
    """Bundle of C++ backend objects shared by all front-end combinations."""

    memory: "fc.MemoryManager"
    store: "fc.InRamLtmStore"
    graph: "fc.FusionGraph2D"
    grid_cfg: "fc.GridConfig"
    bnb_cfg: "fc.BnbConfig"


def build_shared_map(cfg: FusionV2Config) -> SharedMap:
    # Memory config controls STM/WM/LTM residence and rehearsal merging.
    mc = fc.MemoryConfig()
    mc.stm_size = cfg.stm_size
    mc.wm_cap = cfg.wm_cap
    mc.rehearsal_similarity = cfg.rehearsal_similarity
    store = fc.InRamLtmStore()
    memory = fc.MemoryManager(mc, store)

    # Pose graph config stores odometry spine and loop-edge weights.
    gc = fc.GraphConfig()
    gc.huber_scale = cfg.huber_scale
    gc.spine_trans_weight = cfg.spine_trans_weight
    gc.spine_rot_weight = cfg.spine_rot_weight
    graph = fc.FusionGraph2D(gc)

    # Grid config is shared by verification grids and final occupancy rendering.
    grid_cfg = fc.GridConfig()
    grid_cfg.resolution = cfg.grid_resolution
    grid_cfg.l_occ = cfg.grid_l_occ
    grid_cfg.l_free = cfg.grid_l_free

    # B&B config defines the bounded scan loop-search window.
    bnb = fc.BnbConfig()
    bnb.linear_search_window = cfg.bnb_window_xy
    bnb.angular_search_window = cfg.bnb_window_th
    bnb.depth = cfg.bnb_depth
    return SharedMap(memory, store, graph, grid_cfg, bnb)
