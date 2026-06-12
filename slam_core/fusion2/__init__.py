"""Fusion v2 — Python UI/orchestration over the fusion_core C++ shared map.

All hot-path state (signatures, memory tiers, SE(2) graph, retrieval, grids,
B&B verification) lives in the `fusion_core` extension module; this package
only feeds sensors in, proposes loop candidates, and writes outputs.
"""
