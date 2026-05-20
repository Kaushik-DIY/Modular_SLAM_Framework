#!/usr/bin/env python3
"""Offline GT-positive loop retrieval trace analysis from existing logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-loop-classified", required=True, type=Path)
    parser.add_argument("--gt-loop-all", required=True, type=Path)
    parser.add_argument("--loop-oracle", required=True, type=Path)
    parser.add_argument("--retrieval-profile", required=True, type=Path)
    parser.add_argument("--source-comparison", required=True, type=Path)
    parser.add_argument("--density-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair_key(kf_a: int, kf_b: int) -> str:
    return f"{min(int(kf_a), int(kf_b))}-{max(int(kf_a), int(kf_b))}"


def _parse_list(value: Any) -> list[int]:
    if value in {None, ""}:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    result: list[int] = []
    for item in parsed:
        item_int = _int(item)
        if item_int is not None:
            result.append(item_int)
    return result


def load_gt_loop_pairs(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    return [row for row in rows if _bool(row.get("gt_loop_like"))]


def load_oracle_by_pair(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv(path)
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        current_kf_id = _int(row.get("current_kf_id"))
        candidate_kf_id = _int(row.get("candidate_kf_id"))
        if current_kf_id is None or candidate_kf_id is None:
            continue
        pair_key = _pair_key(current_kf_id, candidate_kf_id)
        existing = by_pair.get(pair_key)
        payload = dict(row)
        payload["pair_key"] = pair_key
        if existing is None:
            by_pair[pair_key] = payload
            continue
        existing_tuple = (
            1 if _bool(existing.get("accepted")) else 0,
            _int(existing.get("final_matched_map_points")) or 0,
            -(_int(existing.get("candidate_rank")) or 999999),
        )
        payload_tuple = (
            1 if _bool(payload.get("accepted")) else 0,
            _int(payload.get("final_matched_map_points")) or 0,
            -(_int(payload.get("candidate_rank")) or 999999),
        )
        if payload_tuple > existing_tuple:
            by_pair[pair_key] = payload
    return by_pair


def load_profile_by_kf(path: Path) -> dict[int, dict[str, Any]]:
    rows = _read_csv(path)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        kf_id = _int(row.get("kf_id"))
        if kf_id is None:
            continue
        result[int(kf_id)] = dict(row)
    return result


def load_source_by_kf(path: Path) -> dict[int, dict[str, Any]]:
    rows = _read_csv(path)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        kf_id = _int(row.get("kf_id"))
        if kf_id is None:
            continue
        payload = dict(row)
        payload["dbow3_candidates_list"] = _parse_list(row.get("dbow3_candidates"))
        payload["inverted_candidates_list"] = _parse_list(row.get("inverted_file_candidates"))
        payload["chosen_candidates_list"] = _parse_list(row.get("chosen_candidates"))
        result[int(kf_id)] = payload
    return result


def load_density_by_pair(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        current_kf_id = _int(row.get("current_kf_id"))
        candidate_kf_id = _int(row.get("candidate_kf_id"))
        if current_kf_id is None or candidate_kf_id is None:
            continue
        result[_pair_key(current_kf_id, candidate_kf_id)] = dict(row)
    return result


def classify_actual_stage(oracle_row: dict[str, Any]) -> str:
    if _bool(oracle_row.get("accepted")):
        return "ACCEPTED"
    reason = str(oracle_row.get("rejection_reason", "") or "").lower()
    rejection_stage = str(oracle_row.get("rejection_stage", "") or "").lower()
    if "consistency" in reason or rejection_stage == "consistency":
        return "FAILED_CONSISTENCY"
    if (
        "final support" in reason
        or "final matched map points" in reason
        or "matched map points after covisibility expansion" in reason
        or "covisibility expansion" in reason
    ):
        return "FAILED_FINAL_SUPPORT"
    return "FAILED_GEOMETRY"


def classify_not_retrieved_stage(
    *,
    source_row: dict[str, Any] | None,
    profile_row: dict[str, Any] | None,
    current_kf_id: int,
    candidate_kf_id: int,
) -> tuple[str, str, str]:
    if source_row is not None:
        chosen_list = set(source_row.get("chosen_candidates_list", []))
        dbow_list = set(source_row.get("dbow3_candidates_list", []))
        inverted_list = set(source_row.get("inverted_candidates_list", []))
        if candidate_kf_id in chosen_list:
            return (
                "CHOSEN_CANDIDATE_CONFIRMED",
                "medium",
                "Pair is present in the retained chosen-candidate list but absent from the oracle row set.",
            )
        if candidate_kf_id in dbow_list:
            return (
                "RAW_DBOW_PRESENT_CONFIRMED",
                "medium",
                "Pair is present in the retained dbow3 candidate list; raw DBOW presence is implied, but raw rank is not logged.",
            )
        if candidate_kf_id in inverted_list:
            return (
                "INVERTED_PRESENT_CONFIRMED",
                "medium",
                "Pair is present in the retained inverted-file candidate list; exact raw shared-word identity trace is not logged.",
            )
    if profile_row is None:
        return (
            "NOT_RETRIEVED_BROAD",
            "limited",
            "Missing retrieval-profile row for this current keyframe.",
        )
    return (
        "NOT_RETRIEVED_BROAD",
        "limited",
        "Raw candidate identities are not logged, so the exact pre-retention loss stage cannot be proven.",
    )


def infer_suspected_cause(
    *,
    first_known_loss_stage: str,
    profile_row: dict[str, Any] | None,
    source_row: dict[str, Any] | None,
    oracle_row: dict[str, Any] | None,
) -> str:
    if first_known_loss_stage == "ACCEPTED":
        return "True loop survived retrieval, consistency, geometry, and final support."
    if first_known_loss_stage == "FAILED_CONSISTENCY":
        return "Candidate survived retrieval but failed multi-frame consistency persistence."
    if first_known_loss_stage == "FAILED_FINAL_SUPPORT":
        return "Candidate survived geometry but failed the final matched-map-point support gate."
    if first_known_loss_stage == "FAILED_GEOMETRY":
        return "Candidate survived consistency but failed geometry verification."
    if first_known_loss_stage == "CHOSEN_CANDIDATE_CONFIRMED":
        return "Chosen-candidate evidence exists, but the current logs do not explain why the oracle row is absent."
    if first_known_loss_stage == "RAW_DBOW_PRESENT_CONFIRMED":
        return "Pair survived into the retained dbow3 candidate list, but the exact downstream omission is not exposed."
    if first_known_loss_stage == "INVERTED_PRESENT_CONFIRMED":
        return "Pair survived into the retained inverted-file candidate list, but final source selection details are insufficient."
    if profile_row is None:
        return "Missing retrieval-profile data for the current keyframe."
    raw_dbow = _int(profile_row.get("num_raw_dbow_candidates"))
    raw_inverted = _int(profile_row.get("num_raw_inverted_candidates"))
    after_common = _int(profile_row.get("num_candidates_after_common_words"))
    after_min = _int(profile_row.get("num_candidates_after_min_score"))
    after_acc = _int(profile_row.get("num_candidates_after_accumulation"))
    if raw_dbow == 0 and raw_inverted == 0:
        return "Current keyframe produced no raw DBOW or shared-word candidates at all."
    if after_common == 0:
        return "Current keyframe had raw retrieval activity, but nothing survived to the retained common-word stage."
    if after_min == 0:
        return "Some candidates survived the common-word stage, but none survived the minScore gate."
    if after_acc == 0:
        return "Some candidates survived minScore, but none survived accumulated-score retention."
    if source_row is not None:
        return "Wrong retained candidates were selected before consistency; raw per-pair traces are missing."
    if oracle_row is None:
        return "Pair is absent from retained candidate traces, and raw per-pair retrieval identities are missing."
    return "Unable to infer a narrower cause from current logs."


def recommended_next_diagnostic(first_known_loss_stage: str) -> str:
    mapping = {
        "FAILED_CONSISTENCY": "Instrument per-group consistency overlap progression for GT-positive candidates.",
        "FAILED_GEOMETRY": "Instrument seed correspondences and geometry inlier evolution for GT-positive candidates.",
        "FAILED_FINAL_SUPPORT": "Instrument post-geometry projection expansion and final support counts for GT-positive candidates.",
        "RAW_DBOW_PRESENT_CONFIRMED": "Add raw DBOW identity/rank tracing to prove later-stage loss for GT-positive pairs.",
        "INVERTED_PRESENT_CONFIRMED": "Add per-source retained-candidate tracing and source-selection diagnostics.",
        "CHOSEN_CANDIDATE_CONFIRMED": "Explain why chosen candidates can be absent from oracle rows for the same current keyframe.",
        "NOT_RETRIEVED_BROAD": "Add raw candidate identity tracing before common-word/minScore/accumulation filters.",
    }
    return mapping.get(first_known_loss_stage, "Add the missing runtime trace at the earliest unproven stage.")


def analyze_trace(
    gt_loop_rows: list[dict[str, Any]],
    oracle_by_pair: dict[str, dict[str, Any]],
    profile_by_kf: dict[int, dict[str, Any]],
    source_by_kf: dict[int, dict[str, Any]],
    density_by_pair: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gt_loop_rows:
        pair_key = str(row["pair_key"])
        current_kf_id = int(row.get("kf_j") or row.get("current_kf_id") or -1)
        candidate_kf_id = int(row.get("kf_i") or row.get("candidate_kf_id") or -1)
        oracle_row = oracle_by_pair.get(pair_key)
        profile_row = profile_by_kf.get(current_kf_id)
        source_row = source_by_kf.get(current_kf_id)
        density_row = density_by_pair.get(pair_key)

        actual_candidate_seen = oracle_row is not None
        accepted = _bool((oracle_row or {}).get("accepted"))
        if actual_candidate_seen:
            first_known_loss_stage = classify_actual_stage(oracle_row or {})
            diagnostic_confidence = "high"
            missing_data_reason = ""
        else:
            first_known_loss_stage, diagnostic_confidence, missing_data_reason = classify_not_retrieved_stage(
                source_row=source_row,
                profile_row=profile_row,
                current_kf_id=current_kf_id,
                candidate_kf_id=candidate_kf_id,
            )

        dbow_contains_pair = bool(source_row and candidate_kf_id in set(source_row.get("dbow3_candidates_list", [])))
        inverted_contains_pair = bool(source_row and candidate_kf_id in set(source_row.get("inverted_candidates_list", [])))
        chosen_contains_pair = bool(source_row and candidate_kf_id in set(source_row.get("chosen_candidates_list", [])))

        raw_dbow_evidence_available = bool(actual_candidate_seen or dbow_contains_pair)
        raw_dbow_present: str = "UNKNOWN"
        if raw_dbow_evidence_available:
            raw_dbow_present = "True"
        elif profile_row is not None and (_int(profile_row.get("num_raw_dbow_candidates")) or 0) == 0:
            raw_dbow_present = "False_for_current_query"

        bow_score = (oracle_row or {}).get("bow_score", row.get("bow_score", ""))
        accumulated_score = (oracle_row or {}).get("accumulated_score", row.get("accumulated_score", ""))
        best_accumulated_score = (oracle_row or {}).get("best_accumulated_score", "")
        accumulated_score_ratio = ""
        acc_value = _float(accumulated_score)
        best_acc_value = _float(best_accumulated_score)
        if acc_value is not None and best_acc_value not in {None, 0.0}:
            accumulated_score_ratio = acc_value / best_acc_value

        suspected_cause = infer_suspected_cause(
            first_known_loss_stage=first_known_loss_stage,
            profile_row=profile_row,
            source_row=source_row,
            oracle_row=oracle_row,
        )

        trace_row = {
            "pair_key": pair_key,
            "kf_i": candidate_kf_id,
            "kf_j": current_kf_id,
            "current_kf_id": current_kf_id,
            "candidate_kf_id": candidate_kf_id,
            "gt_translation_distance": row.get("gt_translation_distance", ""),
            "gt_rotation_angle_deg": row.get("gt_rotation_angle_deg", ""),
            "gt_loop_like": row.get("gt_loop_like", ""),
            "gt_near_loop": row.get("gt_near_loop", ""),
            "actual_candidate_seen": actual_candidate_seen,
            "accepted": accepted,
            "pipeline_stage_from_2_35B": row.get("pipeline_stage", ""),
            "rejection_stage": (oracle_row or {}).get("rejection_stage", row.get("rejection_stage", "")),
            "rejection_reason": (oracle_row or {}).get("rejection_reason", row.get("rejection_reason", "")),
            "raw_dbow_evidence_available": raw_dbow_evidence_available,
            "raw_dbow_present": raw_dbow_present,
            "raw_dbow_rank": "",
            "raw_dbow_score": "",
            "source_comparison_available": source_row is not None,
            "dbow3_candidate_list_contains_pair": dbow_contains_pair,
            "inverted_candidate_list_contains_pair": inverted_contains_pair,
            "chosen_candidate_list_contains_pair": chosen_contains_pair,
            "retrieval_profile_available": profile_row is not None,
            "num_raw_dbow_candidates_for_current": (profile_row or {}).get("num_raw_dbow_candidates", ""),
            "num_raw_inverted_candidates_for_current": (profile_row or {}).get("num_raw_inverted_candidates", ""),
            "num_candidates_after_temporal_filter": (profile_row or {}).get("num_candidates_after_temporal_filter", ""),
            "num_candidates_after_connected_filter": (profile_row or {}).get("num_candidates_after_connected_filter", ""),
            "num_candidates_after_common_words": (profile_row or {}).get("num_candidates_after_common_words", ""),
            "num_candidates_after_min_score": (profile_row or {}).get("num_candidates_after_min_score", ""),
            "num_candidates_after_accumulation": (profile_row or {}).get("num_candidates_after_accumulation", ""),
            "num_candidates_after_consistency": (profile_row or {}).get("num_candidates_after_consistency", ""),
            "top_candidate_id": (profile_row or {}).get("top_candidate_id", ""),
            "top_candidate_score": (profile_row or {}).get("top_candidate_score", ""),
            "top_candidate_acc_score": (profile_row or {}).get("top_candidate_acc_score", ""),
            "top_candidate_consistency": (profile_row or {}).get("top_candidate_consistency", ""),
            "bow_score": bow_score,
            "min_score": (oracle_row or {}).get("min_score", ""),
            "common_words": (oracle_row or {}).get("common_words", ""),
            "max_common_words": (oracle_row or {}).get("max_common_words", ""),
            "common_word_ratio": (oracle_row or {}).get("common_word_ratio", ""),
            "accumulated_score": accumulated_score,
            "best_accumulated_score": best_accumulated_score,
            "accumulated_score_ratio": accumulated_score_ratio,
            "consistency_score": (oracle_row or {}).get("consistency_score", row.get("consistency_score", "")),
            "raw_bow_matches": (oracle_row or {}).get("raw_bow_matches", ""),
            "valid_bow_map_point_matches": (oracle_row or {}).get("valid_bow_map_point_matches", ""),
            "seed_inliers": (oracle_row or {}).get("seed_inliers", ""),
            "refined_inliers": (oracle_row or {}).get("refined_inliers", ""),
            "guided_projection_matches": (oracle_row or {}).get("guided_projection_matches", ""),
            "final_matched_map_points": (oracle_row or {}).get(
                "final_matched_map_points",
                (density_row or {}).get("final_matched_map_points", ""),
            ),
            "first_known_loss_stage": first_known_loss_stage,
            "diagnostic_confidence": diagnostic_confidence,
            "missing_data_reason": missing_data_reason,
            "suspected_cause": suspected_cause,
        }
        rows.append(trace_row)
    rows.sort(
        key=lambda item: (
            _float(item.get("gt_translation_distance")) if _float(item.get("gt_translation_distance")) is not None else 999999.0,
            _float(item.get("gt_rotation_angle_deg")) if _float(item.get("gt_rotation_angle_deg")) is not None else 999999.0,
            int(item.get("kf_i", -1)),
            int(item.get("kf_j", -1)),
        )
    )
    return rows


def build_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    actual_candidate_seen = sum(1 for row in rows if _bool(row.get("actual_candidate_seen")))
    accepted = sum(1 for row in rows if row.get("first_known_loss_stage") == "ACCEPTED")
    failed_final = sum(1 for row in rows if row.get("first_known_loss_stage") == "FAILED_FINAL_SUPPORT")
    failed_geometry = sum(1 for row in rows if row.get("first_known_loss_stage") == "FAILED_GEOMETRY")
    failed_consistency = sum(1 for row in rows if row.get("first_known_loss_stage") == "FAILED_CONSISTENCY")
    not_retrieved_broad = sum(1 for row in rows if row.get("first_known_loss_stage") == "NOT_RETRIEVED_BROAD")
    raw_present = sum(1 for row in rows if row.get("first_known_loss_stage") == "RAW_DBOW_PRESENT_CONFIRMED")
    inverted_present = sum(1 for row in rows if row.get("first_known_loss_stage") == "INVERTED_PRESENT_CONFIRMED")
    chosen_confirmed = sum(1 for row in rows if row.get("first_known_loss_stage") == "CHOSEN_CANDIDATE_CONFIRMED")
    unknown_missing = sum(
        1
        for row in rows
        if row.get("first_known_loss_stage") == "NOT_RETRIEVED_BROAD"
        and "raw candidate identities are not logged" in str(row.get("missing_data_reason", "")).lower()
    )
    funnel = [
        ("GT_LOOP_LIKE_TOTAL", total, "high"),
        ("ACTUAL_CANDIDATE_SEEN", actual_candidate_seen, "high"),
        ("ACCEPTED", accepted, "high"),
        ("FAILED_FINAL_SUPPORT", failed_final, "high"),
        ("FAILED_GEOMETRY", failed_geometry, "high"),
        ("FAILED_CONSISTENCY", failed_consistency, "high"),
        ("NOT_RETRIEVED_BROAD", not_retrieved_broad, "limited"),
        ("RAW_DBOW_PRESENT_CONFIRMED", raw_present, "medium"),
        ("INVERTED_PRESENT_CONFIRMED", inverted_present, "medium"),
        ("CHOSEN_CANDIDATE_CONFIRMED", chosen_confirmed, "medium"),
        ("UNKNOWN_DUE_MISSING_RAW_TRACE", unknown_missing, "limited"),
    ]
    output: list[dict[str, Any]] = []
    for stage, count, confidence in funnel:
        output.append(
            {
                "stage": stage,
                "count": count,
                "percent_of_gt_loop_like": (100.0 * count / total) if total > 0 else 0.0,
                "diagnostic_confidence": confidence,
            }
        )
    return output


def build_top_missed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missed = [row for row in rows if row.get("first_known_loss_stage") != "ACCEPTED"]
    missed.sort(
        key=lambda row: (
            _float(row.get("gt_translation_distance")) if _float(row.get("gt_translation_distance")) is not None else 999999.0,
            _float(row.get("gt_rotation_angle_deg")) if _float(row.get("gt_rotation_angle_deg")) is not None else 999999.0,
            _int(row.get("kf_i")) if _int(row.get("kf_i")) is not None else 999999,
            _int(row.get("kf_j")) if _int(row.get("kf_j")) is not None else 999999,
        )
    )
    output: list[dict[str, Any]] = []
    for row in missed:
        output.append(
            {
                "pair_key": row["pair_key"],
                "kf_i": row["kf_i"],
                "kf_j": row["kf_j"],
                "gt_translation_distance": row["gt_translation_distance"],
                "gt_rotation_angle_deg": row["gt_rotation_angle_deg"],
                "first_known_loss_stage": row["first_known_loss_stage"],
                "diagnostic_confidence": row["diagnostic_confidence"],
                "rejection_reason": row["rejection_reason"],
                "raw_dbow_present": row["raw_dbow_present"],
                "inverted_candidate_list_contains_pair": row["inverted_candidate_list_contains_pair"],
                "chosen_candidate_list_contains_pair": row["chosen_candidate_list_contains_pair"],
                "num_raw_dbow_candidates_for_current": row["num_raw_dbow_candidates_for_current"],
                "top_candidate_id": row["top_candidate_id"],
                "top_candidate_score": row["top_candidate_score"],
                "bow_score": row["bow_score"],
                "min_score": row["min_score"],
                "accumulated_score_ratio": row["accumulated_score_ratio"],
                "final_matched_map_points": row["final_matched_map_points"],
                "suspected_cause": row["suspected_cause"],
                "recommended_next_diagnostic": recommended_next_diagnostic(str(row["first_known_loss_stage"])),
            }
        )
    return output


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts: dict[str, int] = {}
    causes: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("first_known_loss_stage", ""))
        counts[stage] = counts.get(stage, 0) + 1
        cause = str(row.get("suspected_cause", ""))
        if cause:
            causes[cause] = causes.get(cause, 0) + 1
    known_stage_counts = {
        stage: count
        for stage, count in counts.items()
        if stage in {"FAILED_CONSISTENCY", "FAILED_GEOMETRY", "FAILED_FINAL_SUPPORT", "ACCEPTED"}
    }
    dominant_known_loss_stage = ""
    if known_stage_counts:
        dominant_known_loss_stage = max(sorted(known_stage_counts.items()), key=lambda item: item[1])[0]
    dominant_unknown_reason = "raw candidate identities not logged"
    top_suspected_causes = [cause for cause, _ in sorted(causes.items(), key=lambda item: (-item[1], item[0]))[:5]]
    return {
        "num_gt_loop_like_pairs": total,
        "num_actual_candidate_seen": sum(1 for row in rows if _bool(row.get("actual_candidate_seen"))),
        "num_accepted": sum(1 for row in rows if row.get("first_known_loss_stage") == "ACCEPTED"),
        "num_not_retrieved_broad": sum(1 for row in rows if row.get("first_known_loss_stage") == "NOT_RETRIEVED_BROAD"),
        "num_with_raw_dbow_identity_available": sum(1 for row in rows if _bool(row.get("raw_dbow_evidence_available"))),
        "num_missing_raw_trace": sum(
            1
            for row in rows
            if row.get("first_known_loss_stage") == "NOT_RETRIEVED_BROAD"
            and "raw candidate identities are not logged" in str(row.get("missing_data_reason", "")).lower()
        ),
        "dominant_known_loss_stage": dominant_known_loss_stage,
        "dominant_unknown_reason": dominant_unknown_reason,
        "top_suspected_causes": top_suspected_causes,
        "recommended_next_checkpoint": "Checkpoint 2.35D: add raw candidate identity tracing before common-word/minScore/accumulation filters.",
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])


def write_report(
    path: Path,
    *,
    inputs: dict[str, str],
    summary: dict[str, Any],
    funnel: list[dict[str, Any]],
    top_missed: list[dict[str, Any]],
    sparse_density_causal: bool,
) -> None:
    lines = [
        "# GT Positive Retrieval Trace Report",
        "",
        "## 1. Objective",
        "- Trace each GT-loop-like pair through the existing retrieval and verification logs without rerunning SLAM or changing runtime behavior.",
        "",
        "## 2. Input files used",
    ]
    for key, value in inputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## 3. Whether analysis required a SLAM rerun",
        "- No. This checkpoint used only existing 2.35A and 2.35B outputs.",
        "",
        "## 4. GT-loop-like pair count",
        f"- `{summary['num_gt_loop_like_pairs']}`",
        "",
        "## 5. Recall funnel",
        markdown_table(funnel, ["stage", "count", "percent_of_gt_loop_like", "diagnostic_confidence"]),
        "",
        "## 6. Top missed GT-loop pairs",
        markdown_table(
            top_missed[:10],
            [
                "pair_key",
                "gt_translation_distance",
                "gt_rotation_angle_deg",
                "first_known_loss_stage",
                "diagnostic_confidence",
                "suspected_cause",
            ],
        ),
        "",
        "## 7. What can be proven from current logs",
        "- The logs prove which GT-positive pairs reached the actual retained/oracle path.",
        "- They also prove which of those retained pairs failed consistency, geometry, or final support.",
        "- They provide per-current-keyframe counts for raw DBOW totals and post-filter totals.",
        "",
        "## 8. What cannot be proven because raw candidate identities are missing",
        "- No. Current logs do not prove whether a specific GT-missed pair was absent from raw DBOW or removed before retention.",
        "- The runtime logs expose counts at each retrieval stage, but not the per-pair raw candidate identities or raw ranks for the missed GT pairs.",
        "",
        "## 9. Comparison with pySLAM retrieval logic",
        "- The local control flow is structurally close to pySLAM for minScore, common-word filtering, accumulated-score retention, consistency, and projection-based final support.",
        "- The key diagnostic gap is not the presence of these stages, but the lack of pair-level raw retrieval tracing needed to explain where GT-positive misses disappear.",
        "",
        "## 10. Most probable loss stage",
        f"- Dominant known loss stage: `{summary['dominant_known_loss_stage'] or 'none'}`",
        f"- Dominant unknown limitation: `{summary['dominant_unknown_reason']}`",
        "",
        "## 11. Whether sparse keyframe density appears causal",
        f"- `{sparse_density_causal}`",
        "- Existing 2.35B density diagnostics did not implicate sparse keyframe density as the dominant cause.",
        "",
        "## 12. Recommended next implementation checkpoint",
        f"- `{summary['recommended_next_checkpoint']}`",
        "",
        "## Explicit answer",
        "- Do current logs prove whether GT pairs were missing from raw DBOW?",
        "- No. We need runtime raw candidate identity tracing in the next checkpoint.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_loop_rows = load_gt_loop_pairs(args.gt_loop_classified.expanduser().resolve())
    oracle_by_pair = load_oracle_by_pair(args.loop_oracle.expanduser().resolve())
    profile_by_kf = load_profile_by_kf(args.retrieval_profile.expanduser().resolve())
    source_by_kf = load_source_by_kf(args.source_comparison.expanduser().resolve())
    density_by_pair = load_density_by_pair(args.density_profile.expanduser().resolve())

    trace_rows = analyze_trace(
        gt_loop_rows=gt_loop_rows,
        oracle_by_pair=oracle_by_pair,
        profile_by_kf=profile_by_kf,
        source_by_kf=source_by_kf,
        density_by_pair=density_by_pair,
    )
    funnel_rows = build_funnel(trace_rows)
    top_missed_rows = build_top_missed(trace_rows)
    summary = build_summary(trace_rows)

    trace_fields = [
        "pair_key",
        "kf_i",
        "kf_j",
        "current_kf_id",
        "candidate_kf_id",
        "gt_translation_distance",
        "gt_rotation_angle_deg",
        "gt_loop_like",
        "gt_near_loop",
        "actual_candidate_seen",
        "accepted",
        "pipeline_stage_from_2_35B",
        "rejection_stage",
        "rejection_reason",
        "raw_dbow_evidence_available",
        "raw_dbow_present",
        "raw_dbow_rank",
        "raw_dbow_score",
        "source_comparison_available",
        "dbow3_candidate_list_contains_pair",
        "inverted_candidate_list_contains_pair",
        "chosen_candidate_list_contains_pair",
        "retrieval_profile_available",
        "num_raw_dbow_candidates_for_current",
        "num_raw_inverted_candidates_for_current",
        "num_candidates_after_temporal_filter",
        "num_candidates_after_connected_filter",
        "num_candidates_after_common_words",
        "num_candidates_after_min_score",
        "num_candidates_after_accumulation",
        "num_candidates_after_consistency",
        "top_candidate_id",
        "top_candidate_score",
        "top_candidate_acc_score",
        "top_candidate_consistency",
        "bow_score",
        "min_score",
        "common_words",
        "max_common_words",
        "common_word_ratio",
        "accumulated_score",
        "best_accumulated_score",
        "accumulated_score_ratio",
        "consistency_score",
        "raw_bow_matches",
        "valid_bow_map_point_matches",
        "seed_inliers",
        "refined_inliers",
        "guided_projection_matches",
        "final_matched_map_points",
        "first_known_loss_stage",
        "diagnostic_confidence",
        "missing_data_reason",
        "suspected_cause",
    ]
    _write_csv(output_dir / "gt_loop_retrieval_trace.csv", trace_rows, trace_fields)
    _write_csv(
        output_dir / "gt_loop_retrieval_funnel.csv",
        funnel_rows,
        ["stage", "count", "percent_of_gt_loop_like", "diagnostic_confidence"],
    )
    _write_csv(
        output_dir / "gt_loop_top_missed_trace.csv",
        top_missed_rows,
        [
            "pair_key",
            "kf_i",
            "kf_j",
            "gt_translation_distance",
            "gt_rotation_angle_deg",
            "first_known_loss_stage",
            "diagnostic_confidence",
            "rejection_reason",
            "raw_dbow_present",
            "inverted_candidate_list_contains_pair",
            "chosen_candidate_list_contains_pair",
            "num_raw_dbow_candidates_for_current",
            "top_candidate_id",
            "top_candidate_score",
            "bow_score",
            "min_score",
            "accumulated_score_ratio",
            "final_matched_map_points",
            "suspected_cause",
            "recommended_next_diagnostic",
        ],
    )
    (output_dir / "gt_loop_retrieval_trace_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_report(
        output_dir / "GT_POSITIVE_RETRIEVAL_TRACE_REPORT.md",
        inputs={
            "gt_loop_classified": str(args.gt_loop_classified.expanduser().resolve()),
            "gt_loop_all": str(args.gt_loop_all.expanduser().resolve()),
            "loop_oracle": str(args.loop_oracle.expanduser().resolve()),
            "retrieval_profile": str(args.retrieval_profile.expanduser().resolve()),
            "source_comparison": str(args.source_comparison.expanduser().resolve()),
            "density_profile": str(args.density_profile.expanduser().resolve()),
        },
        summary=summary,
        funnel=funnel_rows,
        top_missed=top_missed_rows,
        sparse_density_causal=False,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
