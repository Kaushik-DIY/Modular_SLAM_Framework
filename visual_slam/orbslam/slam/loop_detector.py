"""
Loop-candidate retrieval logic.
This module queries the keyframe database and filters candidates by similarity score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.keyframe import KeyFrame
from visual_slam.orbslam.slam.keyframe_database import KeyFrameDatabase


# Store the loop candidates and similarity scores returned by one query.
@dataclass
class LoopDetectorOutput:
    keyframe: KeyFrame
    candidate_keyframes: list[KeyFrame]
    candidate_scores: list[float]
    min_score: float
    unavailable_reason: Optional[str] = None

    @property
    def candidate_idxs(self):
        return [kf.id for kf in self.candidate_keyframes]


# Query the keyframe database and build candidate sets for loop verification.
class LoopDetector:
    def __init__(self, keyframe_database: Optional[KeyFrameDatabase] = None):
        self.keyframe_database = keyframe_database
        self.last_output: Optional[LoopDetectorOutput] = None

    @property
    def available(self) -> bool:
        return self.keyframe_database is not None and self.keyframe_database.available

    def compute_reference_similarity_score(self, keyframe: KeyFrame) -> float:
        if not self.available:
            return 0.0

        self.keyframe_database.compute_bow(keyframe)
        connected = keyframe.get_connected_keyframes()
        if len(connected) == 0:
            return 0.0

        scores = []
        for connected_keyframe in connected:
            if connected_keyframe is None or connected_keyframe.is_bad():
                continue
            self.keyframe_database.compute_bow(connected_keyframe)
            scores.append(self.keyframe_database.score(keyframe.g_des, connected_keyframe.g_des))

        return min(scores) if scores else 0.0

    def detect(self, keyframe: KeyFrame) -> LoopDetectorOutput:
        if not self.available:
            reason = (
                "keyframe database is not configured"
                if self.keyframe_database is None
                else self.keyframe_database.unavailable_reason()
            )
            output = LoopDetectorOutput(keyframe, [], [], 0.0, unavailable_reason=reason)
            self.last_output = output
            return output

        self.keyframe_database.compute_bow(keyframe)
        min_score = self.compute_reference_similarity_score(keyframe)
        candidates = self.keyframe_database.detect_loop_candidates(
            keyframe,
            min_score=min_score,
            min_delta_frames=Parameters.kMinDeltaFrameForMeaningfulLoopClosure,
        )
        scores = [self.keyframe_database.score(keyframe.g_des, candidate.g_des) for candidate in candidates]
        output = LoopDetectorOutput(keyframe, candidates, scores, min_score)
        self.last_output = output
        return output
