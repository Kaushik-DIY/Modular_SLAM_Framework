"""
Keyframe database for loop and relocalization queries.
This module indexes keyframes by visual words and scores candidate retrievals.
"""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Optional

import numpy as np

from visual_slam.orbslam.slam.bow import DBoW3Vocabulary, compute_bow_for_frame
from visual_slam.orbslam.slam.config_parameters import Parameters
from visual_slam.orbslam.slam.frame import Frame
from visual_slam.orbslam.slam.keyframe import KeyFrame


# Index keyframes by visual words for retrieval during relocalization and loop closing.
class KeyFrameDatabase:
    """Inverted-file database used by loop closing and relocalization."""
    def __init__(self, vocabulary: Optional[DBoW3Vocabulary] = None):
        self.voc = vocabulary
        self.inverted_file = defaultdict(list)
        self.mutex = Lock()
        self.dbow_database = None
        self._entry_to_keyframe: dict[int, KeyFrame] = {}
        self._keyframe_to_entry: dict[KeyFrame, int] = {}
        if vocabulary is not None:
            self._reset_dbow_database()

    @property
    def available(self) -> bool:
        return self.voc is not None and bool(getattr(self.voc, "available", False))

    def is_available(self) -> bool:
        return self.available

    def set_vocabulary(self, vocabulary: DBoW3Vocabulary) -> None:
        with self.mutex:
            self.voc = vocabulary
            self.inverted_file.clear()
            self._reset_dbow_database()

    def add(self, keyframe: KeyFrame) -> None:
        if not self.available or keyframe is None:
            return
        self.compute_bow(keyframe)
        with self.mutex:
            for word_id, _ in self._bow_items(keyframe.g_des):
                if keyframe not in self.inverted_file[word_id]:
                    self.inverted_file[word_id].append(keyframe)
            if keyframe not in self._keyframe_to_entry and self.dbow_database is not None:
                entry_id = int(self.dbow_database.addBowVector(self.voc.to_native_bow(keyframe.g_des)))
                self._keyframe_to_entry[keyframe] = entry_id
                self._entry_to_keyframe[entry_id] = keyframe

    def erase(self, keyframe: KeyFrame) -> None:
        if keyframe is None:
            return
        with self.mutex:
            for word_id, _ in self._bow_items(getattr(keyframe, "g_des", None)):
                kf_list = self.inverted_file.get(word_id, [])
                try:
                    kf_list.remove(keyframe)
                except ValueError:
                    pass

    def clear(self) -> None:
        with self.mutex:
            self.inverted_file.clear()
            self._reset_dbow_database()

    def reset(self) -> None:
        self.clear()

    def compute_bow(self, frame_or_keyframe):
        if not self.available:
            raise RuntimeError(self.unavailable_reason())
        if getattr(frame_or_keyframe, "g_des", None) is None:
            return compute_bow_for_frame(frame_or_keyframe, self.voc)
        return getattr(frame_or_keyframe, "g_des"), getattr(frame_or_keyframe, "f_des", None)

    def unavailable_reason(self) -> str:
        if self.voc is None:
            return "BoW vocabulary is not configured"
        return getattr(self.voc, "error", "BoW vocabulary is unavailable")

    def detect_relocalization_candidates(self, frame: Frame) -> list[KeyFrame]:
        if not self.available:
            return []

        self.compute_bow(frame)
        query_id = getattr(frame, "id", getattr(frame, "mn_id", -1))
        keyframes_sharing_words = []

        with self.mutex:
            for word_id, _ in self._bow_items(frame.g_des):
                for keyframe in self.inverted_file.get(word_id, []):
                    if keyframe is None or keyframe.is_bad():
                        continue
                    if keyframe.reloc_query_id != query_id:
                        keyframe.num_reloc_words = 0
                        keyframe.reloc_query_id = query_id
                        keyframes_sharing_words.append(keyframe)
                    keyframe.num_reloc_words += 1

        if len(keyframes_sharing_words) == 0:
            return []

        max_common_words = max(kf.num_reloc_words for kf in keyframes_sharing_words)
        min_common_words = int(max_common_words * 0.8)

        score_and_match = []
        for keyframe in keyframes_sharing_words:
            if keyframe.num_reloc_words > min_common_words:
                score = self.voc.score(frame.g_des, keyframe.g_des)
                keyframe.reloc_score = score
                score_and_match.append((score, keyframe))

        if len(score_and_match) == 0:
            return []

        acc_score_and_match = []
        best_acc_score = 0.0

        for score, keyframe in score_and_match:
            neighbors = keyframe.get_best_covisible_keyframes(10)
            best_score = score
            acc_score = score
            best_keyframe = keyframe

            for neighbor in neighbors:
                if neighbor.reloc_query_id == query_id:
                    acc_score += neighbor.reloc_score
                    if neighbor.reloc_score > best_score:
                        best_keyframe = neighbor
                        best_score = neighbor.reloc_score

            acc_score_and_match.append((acc_score, best_keyframe))
            best_acc_score = max(best_acc_score, acc_score)

        min_score_to_retain = 0.75 * best_acc_score
        already_added = set()
        candidates = []

        for acc_score, keyframe in acc_score_and_match:
            if acc_score > min_score_to_retain and keyframe not in already_added:
                candidates.append(keyframe)
                already_added.add(keyframe)

        return candidates

    def detect_loop_candidates(
        self,
        keyframe: KeyFrame,
        min_score: float,
        min_delta_frames: int = Parameters.kMinDeltaFrameForMeaningfulLoopClosure,
    ) -> list[KeyFrame]:
        if not self.available:
            return []

        self.compute_bow(keyframe)
        dbow_candidates = self._detect_loop_candidates_dbow3(
            keyframe,
            min_score=float(min_score),
            min_delta_frames=int(min_delta_frames),
        )
        if dbow_candidates is not None:
            return dbow_candidates

        connected_keyframes = set(keyframe.get_connected_keyframes())
        keyframes_sharing_words = []

        with self.mutex:
            for word_id, _ in self._bow_items(keyframe.g_des):
                for candidate in self.inverted_file.get(word_id, []):
                    if candidate is None or candidate.is_bad():
                        continue
                    if candidate is keyframe:
                        continue
                    if candidate in connected_keyframes:
                        continue
                    if abs(int(candidate.id) - int(keyframe.id)) <= int(min_delta_frames):
                        continue

                    if candidate.loop_query_id != keyframe.id:
                        candidate.num_loop_words = 0
                        candidate.loop_query_id = keyframe.id
                        keyframes_sharing_words.append(candidate)
                    candidate.num_loop_words += 1

        if len(keyframes_sharing_words) == 0:
            return []

        max_common_words = max(kf.num_loop_words for kf in keyframes_sharing_words)
        min_common_words = int(max_common_words * 0.8)

        score_and_match = []
        for candidate in keyframes_sharing_words:
            if candidate.num_loop_words > min_common_words:
                score = self.voc.score(keyframe.g_des, candidate.g_des)
                candidate.loop_score = score
                if score >= min_score:
                    score_and_match.append((score, candidate))

        if len(score_and_match) == 0:
            return []

        acc_score_and_match = []
        best_acc_score = min_score

        for score, candidate in score_and_match:
            neighbors = candidate.get_best_covisible_keyframes(10)
            best_score = score
            acc_score = score
            best_keyframe = candidate

            for neighbor in neighbors:
                if (
                    neighbor.loop_query_id == keyframe.id
                    and neighbor.num_loop_words > min_common_words
                ):
                    acc_score += neighbor.loop_score
                    if neighbor.loop_score > best_score:
                        best_keyframe = neighbor
                        best_score = neighbor.loop_score

            acc_score_and_match.append((acc_score, best_keyframe))
            best_acc_score = max(best_acc_score, acc_score)

        min_score_to_retain = 0.75 * best_acc_score
        already_added = set()
        candidates = []

        for acc_score, candidate in acc_score_and_match:
            if acc_score > min_score_to_retain and candidate not in already_added:
                candidates.append(candidate)
                already_added.add(candidate)

        return candidates

    def _detect_loop_candidates_dbow3(
        self,
        keyframe: KeyFrame,
        *,
        min_score: float,
        min_delta_frames: int,
    ) -> list[KeyFrame] | None:
        if self.dbow_database is None:
            return None
        try:
            results = self.dbow_database.query(
                self.voc.to_native_bow(keyframe.g_des),
                int(Parameters.kMaxResultsForLoopClosure),
            )
        except Exception:
            return None

        connected = set(keyframe.get_connected_keyframes())
        candidates: list[KeyFrame] = []
        for result in results:
            entry_id = int(getattr(result, "id", -1))
            score = float(getattr(result, "score", 0.0))
            candidate = self._entry_to_keyframe.get(entry_id)
            if candidate is None or candidate is keyframe or candidate.is_bad():
                continue
            if candidate in connected:
                continue
            if abs(int(getattr(candidate, "id", 0)) - int(getattr(keyframe, "id", 0))) <= int(min_delta_frames):
                continue
            if score < min_score:
                continue
            candidate.loop_score = score
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def score(self, bow_a, bow_b) -> float:
        if not self.available:
            return 0.0
        return self.voc.score(bow_a, bow_b)

    def _reset_dbow_database(self) -> None:
        self.dbow_database = None
        self._entry_to_keyframe = {}
        self._keyframe_to_entry = {}
        if self.voc is None or not getattr(self.voc, "available", False):
            return
        pydbow3 = getattr(self.voc, "pydbow3", None)
        native_voc = getattr(self.voc, "voc", None)
        if pydbow3 is None or native_voc is None:
            return
        try:
            database = pydbow3.Database()
            database.setVocabulary(native_voc)
            self.dbow_database = database
        except Exception:
            self.dbow_database = None

    @staticmethod
    def _bow_items(bow) -> list[tuple[int, float]]:
        if bow is None:
            return []
        if hasattr(bow, "toVec"):
            return [(int(word_id), float(weight)) for word_id, weight in bow.toVec()]
        if isinstance(bow, dict):
            return [(int(word_id), float(weight)) for word_id, weight in bow.items()]
        return [(int(word_id), float(weight)) for word_id, weight in list(bow)]
