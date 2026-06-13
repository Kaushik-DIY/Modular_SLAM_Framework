"""DBoW3 appearance index over Signature descriptors (fusion v3).

Replaces the legacy KeyFrameDatabase (which requires live ORB-SLAM KeyFrame
objects) for native-front-end runs: raw (N,32) uint8 descriptors go straight
into a pydbow3.Database via addFeatures/query. Mirrors OrbLoopProposer's
propose-only contract: candidates above min_score, outside the index-separation
window, capped per query.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from visual_slam.orbslam.slam.bow import DBoW3Vocabulary


class AppearanceIndex:
    def __init__(self, min_score: float = 0.05, min_separation: int = 30,
                 max_candidates: int = 2):
        self.min_score = float(min_score)
        self.min_separation = int(min_separation)
        self.max_candidates = int(max_candidates)
        voc = DBoW3Vocabulary()
        if not getattr(voc, "available", False):
            raise RuntimeError("DBoW3 vocabulary unavailable for AppearanceIndex")
        self._db = voc.pydbow3.Database()
        self._db.setVocabulary(voc.voc, False, 0)
        self._entry_to_kf: List[int] = []   # DBoW entry id -> fusion kf_id

    def add(self, kf_id: int, des: np.ndarray) -> None:
        self._db.addFeatures(np.ascontiguousarray(des, dtype=np.uint8))
        self._entry_to_kf.append(int(kf_id))

    def query(self, kf_id: int, des: np.ndarray) -> List[Tuple[int, float]]:
        """Loop candidates for a (not-yet-added) keyframe: [(kf_id, score)]."""
        if not self._entry_to_kf:
            return []
        results = self._db.query_local_des(np.ascontiguousarray(des, dtype=np.uint8),
                                 self.max_candidates + self.min_separation // 2 + 4)
        out: List[Tuple[int, float]] = []
        for r in results:
            entry, score = int(r.id), float(r.score)
            if entry >= len(self._entry_to_kf) or score < self.min_score:
                continue
            cand = self._entry_to_kf[entry]
            if abs(kf_id - cand) < self.min_separation:
                continue
            out.append((cand, score))
            if len(out) >= self.max_candidates:
                break
        return out

    def __len__(self) -> int:
        return len(self._entry_to_kf)
