from __future__ import annotations

import csv
from pathlib import Path

from tools.analyze_gt_loop_retrieval_trace import (
    _pair_key,
    analyze_trace,
    build_funnel,
    build_summary,
    build_top_missed,
)


def _gt_row(
    pair_key: str,
    *,
    kf_i: int,
    kf_j: int,
    stage: str = "NOT_RETRIEVED",
    gt_translation_distance: float = 0.5,
    gt_rotation_angle_deg: float = 10.0,
) -> dict:
    return {
        "pair_key": pair_key,
        "kf_i": str(kf_i),
        "kf_j": str(kf_j),
        "gt_translation_distance": str(gt_translation_distance),
        "gt_rotation_angle_deg": str(gt_rotation_angle_deg),
        "gt_loop_like": "True",
        "gt_near_loop": "True",
        "pipeline_stage": stage,
        "rejection_stage": "",
        "rejection_reason": "",
    }


def test_pair_key_order_independent():
    assert _pair_key(4, 39) == "4-39"
    assert _pair_key(39, 4) == "4-39"


def test_trace_classifies_accepted_pair():
    rows = analyze_trace(
        gt_loop_rows=[_gt_row("15-45", kf_i=15, kf_j=45, stage="ACCEPTED")],
        oracle_by_pair={"15-45": {"accepted": "True", "rejection_stage": "", "rejection_reason": "", "bow_score": "0.1"}},
        profile_by_kf={45: {"num_raw_dbow_candidates": "10"}},
        source_by_kf={45: {"dbow3_candidates_list": [15], "inverted_candidates_list": [15], "chosen_candidates_list": [15]}},
        density_by_pair={},
    )

    assert rows[0]["first_known_loss_stage"] == "ACCEPTED"
    assert rows[0]["diagnostic_confidence"] == "high"


def test_trace_classifies_final_support_failure():
    rows = analyze_trace(
        gt_loop_rows=[_gt_row("8-43", kf_i=8, kf_j=43, stage="FAILED_FINAL_SUPPORT")],
        oracle_by_pair={
            "8-43": {
                "accepted": "False",
                "rejection_stage": "geometry",
                "rejection_reason": "too few matched map points after covisibility expansion (46 < 60)",
                "final_matched_map_points": "46",
            }
        },
        profile_by_kf={43: {"num_raw_dbow_candidates": "10"}},
        source_by_kf={43: {"dbow3_candidates_list": [8], "inverted_candidates_list": [8], "chosen_candidates_list": [8]}},
        density_by_pair={},
    )

    assert rows[0]["first_known_loss_stage"] == "FAILED_FINAL_SUPPORT"


def test_trace_classifies_consistency_failure():
    rows = analyze_trace(
        gt_loop_rows=[_gt_row("4-39", kf_i=4, kf_j=39, stage="FAILED_CONSISTENCY")],
        oracle_by_pair={"4-39": {"accepted": "False", "rejection_stage": "consistency", "rejection_reason": "rejected_by_consistency"}},
        profile_by_kf={39: {"num_raw_dbow_candidates": "10"}},
        source_by_kf={39: {"dbow3_candidates_list": [4], "inverted_candidates_list": [4], "chosen_candidates_list": [4]}},
        density_by_pair={},
    )

    assert rows[0]["first_known_loss_stage"] == "FAILED_CONSISTENCY"


def test_trace_classifies_broad_not_retrieved_when_no_candidate_logs():
    rows = analyze_trace(
        gt_loop_rows=[_gt_row("0-39", kf_i=0, kf_j=39)],
        oracle_by_pair={},
        profile_by_kf={},
        source_by_kf={},
        density_by_pair={},
    )

    assert rows[0]["first_known_loss_stage"] == "NOT_RETRIEVED_BROAD"
    assert rows[0]["diagnostic_confidence"] == "limited"


def test_trace_marks_limited_confidence_when_raw_dbow_identity_missing():
    rows = analyze_trace(
        gt_loop_rows=[_gt_row("1-39", kf_i=1, kf_j=39)],
        oracle_by_pair={},
        profile_by_kf={39: {"num_raw_dbow_candidates": "35", "num_raw_inverted_candidates": "1413", "num_candidates_after_common_words": "1", "num_candidates_after_min_score": "1", "num_candidates_after_accumulation": "1"}},
        source_by_kf={39: {"dbow3_candidates_list": [4], "inverted_candidates_list": [4], "chosen_candidates_list": [4]}},
        density_by_pair={},
    )

    assert rows[0]["raw_dbow_evidence_available"] is False
    assert rows[0]["diagnostic_confidence"] == "limited"
    assert "raw candidate identities are not logged" in rows[0]["missing_data_reason"].lower()


def test_funnel_counts_sum_to_gt_loop_total():
    rows = [
        {"first_known_loss_stage": "ACCEPTED", "actual_candidate_seen": True, "missing_data_reason": ""},
        {"first_known_loss_stage": "FAILED_CONSISTENCY", "actual_candidate_seen": True, "missing_data_reason": ""},
        {"first_known_loss_stage": "FAILED_GEOMETRY", "actual_candidate_seen": True, "missing_data_reason": ""},
        {"first_known_loss_stage": "FAILED_FINAL_SUPPORT", "actual_candidate_seen": True, "missing_data_reason": ""},
        {"first_known_loss_stage": "NOT_RETRIEVED_BROAD", "actual_candidate_seen": False, "missing_data_reason": "raw candidate identities are not logged"},
    ]

    funnel = build_funnel(rows)
    stage_counts = {row["stage"]: row["count"] for row in funnel}

    assert stage_counts["GT_LOOP_LIKE_TOTAL"] == 5
    assert (
        stage_counts["ACCEPTED"]
        + stage_counts["FAILED_CONSISTENCY"]
        + stage_counts["FAILED_GEOMETRY"]
        + stage_counts["FAILED_FINAL_SUPPORT"]
        + stage_counts["NOT_RETRIEVED_BROAD"]
        == stage_counts["GT_LOOP_LIKE_TOTAL"]
    )


def test_top_missed_sorted_by_gt_distance():
    rows = [
        {
            "pair_key": "2-40",
            "kf_i": 2,
            "kf_j": 40,
            "gt_translation_distance": "0.4",
            "gt_rotation_angle_deg": "20.0",
            "first_known_loss_stage": "FAILED_CONSISTENCY",
            "diagnostic_confidence": "high",
            "rejection_reason": "",
            "raw_dbow_present": "UNKNOWN",
            "inverted_candidate_list_contains_pair": False,
            "chosen_candidate_list_contains_pair": False,
            "num_raw_dbow_candidates_for_current": "10",
            "top_candidate_id": "2",
            "top_candidate_score": "0.2",
            "bow_score": "0.2",
            "min_score": "0.1",
            "accumulated_score_ratio": "",
            "final_matched_map_points": "0",
            "suspected_cause": "",
        },
        {
            "pair_key": "0-39",
            "kf_i": 0,
            "kf_j": 39,
            "gt_translation_distance": "0.1",
            "gt_rotation_angle_deg": "30.0",
            "first_known_loss_stage": "NOT_RETRIEVED_BROAD",
            "diagnostic_confidence": "limited",
            "rejection_reason": "",
            "raw_dbow_present": "UNKNOWN",
            "inverted_candidate_list_contains_pair": False,
            "chosen_candidate_list_contains_pair": False,
            "num_raw_dbow_candidates_for_current": "35",
            "top_candidate_id": "4",
            "top_candidate_score": "0.4",
            "bow_score": "",
            "min_score": "",
            "accumulated_score_ratio": "",
            "final_matched_map_points": "",
            "suspected_cause": "",
        },
    ]

    top = build_top_missed(rows)

    assert top[0]["pair_key"] == "0-39"
    assert top[1]["pair_key"] == "2-40"


def test_summary_json_contains_required_fields():
    rows = [
        {"first_known_loss_stage": "ACCEPTED", "actual_candidate_seen": True, "raw_dbow_evidence_available": True, "suspected_cause": "ok", "missing_data_reason": ""},
        {"first_known_loss_stage": "NOT_RETRIEVED_BROAD", "actual_candidate_seen": False, "raw_dbow_evidence_available": False, "suspected_cause": "missing", "missing_data_reason": "raw candidate identities are not logged"},
    ]

    summary = build_summary(rows)

    required = {
        "num_gt_loop_like_pairs",
        "num_actual_candidate_seen",
        "num_accepted",
        "num_not_retrieved_broad",
        "num_with_raw_dbow_identity_available",
        "num_missing_raw_trace",
        "dominant_known_loss_stage",
        "dominant_unknown_reason",
        "top_suspected_causes",
        "recommended_next_checkpoint",
    }
    assert required.issubset(summary.keys())


def test_analysis_does_not_require_slam_imports():
    source = Path("tools/analyze_gt_loop_retrieval_trace.py").read_text(encoding="utf-8")

    assert "visual_slam.orbslam.slam" not in source
