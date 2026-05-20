#!/usr/bin/env python3
"""Analyze checkpoint 2.35D raw loop-retrieval traces for GT-positive pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FUNNEL_STAGES = [
    ("GT_LOOP_LIKE_TOTAL", lambda row: True, "All GT-loop-like pairs from the classified oracle set."),
    ("RAW_DBOW_PRESENT", lambda row: _bool(row.get("raw_dbow_present")), "Present in the raw DBOW query result."),
    (
        "INVERTED_WORD_PRESENT",
        lambda row: _bool(row.get("inverted_word_present")),
        "Visible in the inverted/shared-word candidate set.",
    ),
    (
        "PASSED_CONNECTED_TEMPORAL",
        lambda row: _bool(row.get("passed_connected_filter")) and _bool(row.get("passed_temporal_filter")),
        "Survived connected-keyframe and temporal filtering.",
    ),
    (
        "PASSED_COMMON_WORD",
        lambda row: _bool(row.get("passed_common_word_filter")),
        "Survived the common-word threshold gate.",
    ),
    (
        "PASSED_MIN_SCORE",
        lambda row: _bool(row.get("passed_min_score_filter")),
        "Survived the minScore gate.",
    ),
    (
        "PASSED_ACCUMULATION",
        lambda row: _bool(row.get("passed_accumulated_score_filter")),
        "Survived accumulated-score thresholding.",
    ),
    (
        "RETAINED_CANDIDATE",
        lambda row: _bool(row.get("retained_candidate")),
        "Retained after accumulated-score representative selection.",
    ),
    (
        "PASSED_CONSISTENCY",
        lambda row: _bool(row.get("passed_consistency")),
        "Survived multi-query loop consistency.",
    ),
    (
        "PASSED_GEOMETRY",
        lambda row: _bool(row.get("passed_geometry_if_available")) or _bool(row.get("accepted")),
        "Survived geometry verification and reached final support.",
    ),
    ("ACCEPTED", lambda row: _bool(row.get("accepted")), "Accepted as a loop."),
]

FAILURE_STAGES = [
    "MISSING_FROM_RAW_DBOW",
    "MISSING_FROM_INVERTED_WORD_SET",
    "FAILED_CONNECTED_FILTER",
    "FAILED_TEMPORAL_FILTER",
    "FAILED_COMMON_WORD_FILTER",
    "FAILED_MIN_SCORE_FILTER",
    "FAILED_ACCUMULATION_FILTER",
    "NOT_RETAINED_AFTER_ACCUMULATION",
    "FAILED_CONSISTENCY",
    "FAILED_SEED_GEOMETRY",
    "FAILED_REFINED_GEOMETRY",
    "FAILED_FINAL_SUPPORT",
    "ACCEPTED",
    "UNKNOWN",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--gt-loop-classified", required=True, type=Path)
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


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _pair_key(kf_a: Any, kf_b: Any) -> str:
    return f"{min(int(kf_a), int(kf_b))}-{max(int(kf_a), int(kf_b))}"


def load_gt_loop_classified(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    return [row for row in rows if _bool(row.get("gt_loop_like"))]


def load_gt_positive_trace(path: Path) -> list[dict[str, Any]]:
    return _read_csv(path)


def load_accumulation_trace(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv(path)
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_key = str(row.get("pair_key") or "")
        if not pair_key:
            continue
        existing = by_pair.get(pair_key)
        payload = dict(row)
        if existing is None:
            by_pair[pair_key] = payload
            continue
        payload_retained = int(_bool(payload.get("retained_candidate")))
        existing_retained = int(_bool(existing.get("retained_candidate")))
        payload_acc = _float(payload.get("accumulated_score")) or float("-inf")
        existing_acc = _float(existing.get("accumulated_score")) or float("-inf")
        if (payload_retained, payload_acc) > (existing_retained, existing_acc):
            by_pair[pair_key] = payload
    return by_pair


def select_analysis_rows(trace_rows: list[dict[str, Any]], *, min_kf_gap: int = 10) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in trace_rows:
        if not _bool(row.get("gt_loop_like")):
            continue
        current_kf_id = _int(row.get("current_kf_id"))
        candidate_kf_id = _int(row.get("candidate_kf_id"))
        if current_kf_id is None or candidate_kf_id is None:
            continue
        if abs(int(current_kf_id) - int(candidate_kf_id)) <= int(min_kf_gap):
            continue
        selected.append(dict(row))
    return selected


def build_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    funnel: list[dict[str, Any]] = []
    total = len(rows)
    previous_count = total
    for stage, predicate, reason in FUNNEL_STAGES:
        count = sum(1 for row in rows if predicate(row))
        funnel.append(
            {
                "stage": stage,
                "count": int(count),
                "percent_of_gt_loop_like": (100.0 * count / total) if total > 0 else 0.0,
                "drop_from_previous_stage": int(previous_count - count),
                "drop_reason": reason,
            }
        )
        previous_count = count
    return funnel


def build_failure_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    counts = {stage: 0 for stage in FAILURE_STAGES}
    for row in rows:
        stage = str(row.get("first_failed_stage") or "UNKNOWN")
        counts[stage] = counts.get(stage, 0) + 1
    return [
        {
            "stage": stage,
            "count": int(counts.get(stage, 0)),
            "percent_of_gt_loop_like": (100.0 * counts.get(stage, 0) / total) if total > 0 else 0.0,
            "drop_from_previous_stage": "",
            "drop_reason": "First observed failure stage count.",
        }
        for stage in FAILURE_STAGES
    ]


def dominant_failure_stage(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("first_failed_stage") or "UNKNOWN")
        if stage == "ACCEPTED":
            continue
        counts[stage] = counts.get(stage, 0) + 1
    if not counts:
        return "ACCEPTED"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def recommended_next_action(stage: str) -> str:
    mapping = {
        "MISSING_FROM_RAW_DBOW": "Audit the raw DBOW query pool, database population, and any bounded-query truncation before later filters.",
        "MISSING_FROM_INVERTED_WORD_SET": "Audit BoW word assignment / shared-word visibility against pySLAM before changing thresholds.",
        "FAILED_CONNECTED_FILTER": "Audit whether true loops are still structurally connected when queried and whether exclusion ordering matches pySLAM intent.",
        "FAILED_TEMPORAL_FILTER": "Audit temporal-gap policy versus keyframe scheduling before any threshold tuning.",
        "FAILED_COMMON_WORD_FILTER": "Audit common-word counting and threshold parity with pySLAM.",
        "FAILED_MIN_SCORE_FILTER": "Audit minScore computation, connected-reference score scaling, and pySLAM parity.",
        "FAILED_ACCUMULATION_FILTER": "Audit covisibility accumulation and best-score retention logic against pySLAM.",
        "NOT_RETAINED_AFTER_ACCUMULATION": "Audit representative-keyframe retention ordering and duplicate suppression after accumulation.",
        "FAILED_CONSISTENCY": "Audit consistency-group persistence and whether candidate ordering or group expansion diverges from pySLAM.",
        "FAILED_SEED_GEOMETRY": "Audit seed correspondence quality and early SE3 gating before correction work.",
        "FAILED_REFINED_GEOMETRY": "Audit refined geometry / guided projection expansion before loop correction changes.",
        "FAILED_FINAL_SUPPORT": "Audit final support / matched-map-point thresholds and fusion support, not retrieval.",
        "UNKNOWN": "Investigate unexpected missing trace data before correction changes.",
        "ACCEPTED": "No retrieval correction needed for the accepted pairs.",
    }
    return mapping.get(stage, "Inspect the dominant stage directly in the trace rows.")


def add_false_negative_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        if _bool(row.get("accepted")):
            continue
        stage = str(row.get("first_failed_stage") or "UNKNOWN")
        payload = dict(row)
        payload["suspected_cause"] = stage
        payload["recommended_next_action"] = recommended_next_action(stage)
        annotated.append(payload)
    annotated.sort(
        key=lambda row: (
            str(row.get("first_failed_stage") or ""),
            _float(row.get("gt_translation_distance")) if _float(row.get("gt_translation_distance")) is not None else float("inf"),
            _float(row.get("gt_rotation_angle_deg")) if _float(row.get("gt_rotation_angle_deg")) is not None else float("inf"),
        )
    )
    return annotated


def build_summary(
    rows: list[dict[str, Any]],
    funnel: list[dict[str, Any]],
    *,
    historical_gt_loop_like_count: int,
    min_kf_gap: int,
    group_level_summary_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dominant = dominant_failure_stage(rows)
    group_counts = {
        str(row.get("stage")): int(row.get("count", 0) or 0)
        for row in (group_level_summary_rows or [])
    }
    return {
        "num_gt_loop_like_pairs": int(len(rows)),
        "historical_gt_loop_like_count_2_35B": int(historical_gt_loop_like_count),
        "analysis_min_kf_gap": int(min_kf_gap),
        "accepted_count": int(sum(1 for row in rows if _bool(row.get("accepted")))),
        "dominant_first_failure_stage": dominant,
        "recommended_next_checkpoint": recommended_next_action(dominant),
        "funnel_counts": {row["stage"]: int(row["count"]) for row in funnel},
        "failure_stage_counts": {
            stage: int(sum(1 for row in rows if str(row.get("first_failed_stage") or "UNKNOWN") == stage))
            for stage in FAILURE_STAGES
        },
        "group_level_counts": group_counts,
    }


def build_group_level_false_negative_analysis(
    rows: list[dict[str, Any]],
    accumulation_by_pair: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_pair = {str(row.get("pair_key")): dict(row) for row in rows}
    output: list[dict[str, Any]] = []
    for row in rows:
        pair_key = str(row.get("pair_key") or "")
        current_kf_id = _int(row.get("current_kf_id"))
        candidate_kf_id = _int(row.get("candidate_kf_id"))
        if current_kf_id is None or candidate_kf_id is None:
            continue

        connected_local = not _bool(row.get("passed_connected_filter"))
        temporal_eligible = _bool(row.get("passed_temporal_filter"))
        eligible_for_loop = (not connected_local) and temporal_eligible
        passed_accumulation = _bool(row.get("passed_accumulated_score_filter"))
        retained_exact = _bool(row.get("retained_candidate"))
        accumulation_row = accumulation_by_pair.get(pair_key, {})
        representative_kf_id = _int(accumulation_row.get("best_candidate_id_in_group"))
        representative_pair_key = (
            _pair_key(current_kf_id, representative_kf_id)
            if representative_kf_id is not None and representative_kf_id >= 0
            else ""
        )
        representative_gt_row = rows_by_pair.get(representative_pair_key)
        representative_gt_equivalent = bool(
            representative_gt_row is not None and _bool(representative_gt_row.get("gt_loop_like"))
        )

        classification = "NOT_ELIGIBLE"
        group_recalled = False
        if connected_local:
            classification = "CONNECTED_LOCAL_EXCLUDED"
        elif not temporal_eligible:
            classification = "TEMPORAL_INELIGIBLE"
        elif retained_exact:
            classification = "EXACT_RETAINED"
            group_recalled = True
        elif passed_accumulation:
            if representative_gt_equivalent:
                classification = "NOT_RETAINED_BUT_GT_EQUIVALENT_REPRESENTATIVE"
                group_recalled = True
            else:
                classification = "NOT_RETAINED_AND_LOST"
        else:
            classification = "FAILED_ACCUMULATION_FILTER_OR_EARLIER"

        output.append(
            {
                "pair_key": pair_key,
                "current_kf_id": int(current_kf_id),
                "candidate_kf_id": int(candidate_kf_id),
                "gt_translation_distance": row.get("gt_translation_distance", ""),
                "gt_rotation_angle_deg": row.get("gt_rotation_angle_deg", ""),
                "connected_local": bool(connected_local),
                "temporal_eligible": bool(temporal_eligible),
                "eligible_for_loop": bool(eligible_for_loop),
                "passed_accumulated_score_filter": bool(passed_accumulation),
                "retained_exact": bool(retained_exact),
                "representative_kf_id": representative_kf_id if representative_kf_id is not None else "",
                "representative_pair_key": representative_pair_key,
                "representative_gt_equivalent": bool(representative_gt_equivalent),
                "representative_gt_translation_distance": (
                    representative_gt_row.get("gt_translation_distance", "")
                    if representative_gt_row is not None
                    else ""
                ),
                "representative_gt_rotation_angle_deg": (
                    representative_gt_row.get("gt_rotation_angle_deg", "")
                    if representative_gt_row is not None
                    else ""
                ),
                "group_recalled": bool(group_recalled),
                "classification": classification,
            }
        )

    output.sort(
        key=lambda row: (
            str(row.get("classification") or ""),
            _float(row.get("gt_translation_distance")) if _float(row.get("gt_translation_distance")) is not None else float("inf"),
            _float(row.get("gt_rotation_angle_deg")) if _float(row.get("gt_rotation_angle_deg")) is not None else float("inf"),
        )
    )
    return output


def build_group_level_recall_summary(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(group_rows)
    connected_local = sum(1 for row in group_rows if _bool(row.get("connected_local")))
    eligible = sum(1 for row in group_rows if _bool(row.get("eligible_for_loop")))
    exact_retained = sum(1 for row in group_rows if str(row.get("classification")) == "EXACT_RETAINED")
    gt_equivalent_rep = sum(
        1
        for row in group_rows
        if str(row.get("classification")) == "NOT_RETAINED_BUT_GT_EQUIVALENT_REPRESENTATIVE"
    )
    not_retained_lost = sum(
        1 for row in group_rows if str(row.get("classification")) == "NOT_RETAINED_AND_LOST"
    )
    group_recalled_total = exact_retained + gt_equivalent_rep

    stages = [
        ("GT_LOOP_LIKE_TOTAL", total),
        ("GT_LOOP_LIKE_CONNECTED_LOCAL", connected_local),
        ("GT_LOOP_LIKE_ELIGIBLE_FOR_LOOP", eligible),
        ("GROUP_RECALLED_EXACT", exact_retained),
        ("NOT_RETAINED_BUT_GT_EQUIVALENT_REPRESENTATIVE", gt_equivalent_rep),
        ("GROUP_RECALLED_TOTAL", group_recalled_total),
        ("NOT_RETAINED_AND_LOST", not_retained_lost),
    ]
    output: list[dict[str, Any]] = []
    for stage, count in stages:
        percent_of_eligible = (100.0 * count / eligible) if eligible > 0 else 0.0
        output.append(
            {
                "stage": stage,
                "count": int(count),
                "percent_of_eligible_loop_pairs": percent_of_eligible,
            }
        )
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def build_markdown_report(
    *,
    trace_dir: Path,
    gt_loop_classified: Path,
    rows: list[dict[str, Any]],
    funnel: list[dict[str, Any]],
    summary: dict[str, Any],
    false_negatives: list[dict[str, Any]],
    group_level_summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# GT Raw Retrieval Trace Report",
        "",
        "## Inputs",
        "",
        f"- Trace directory: `{trace_dir}`",
        f"- GT classified pairs: `{gt_loop_classified}`",
        "",
        "## Funnel",
        "",
        markdown_table(
            funnel,
            ["stage", "count", "percent_of_gt_loop_like", "drop_from_previous_stage", "drop_reason"],
        ),
        "",
        "## Summary",
        "",
        f"- GT-loop-like pairs analyzed from current trace: `{summary['num_gt_loop_like_pairs']}`",
        f"- Historical GT-loop-like count from 2.35B input: `{summary['historical_gt_loop_like_count_2_35B']}`",
        f"- Meaningful keyframe-gap filter used for analysis: `>{summary['analysis_min_kf_gap']}`",
        f"- Accepted: `{summary['accepted_count']}`",
        f"- Dominant first-failure stage: `{summary['dominant_first_failure_stage']}`",
        f"- Recommended next checkpoint: `{summary['recommended_next_checkpoint']}`",
        "",
        "## Group-Level Recall",
        "",
        markdown_table(
            group_level_summary_rows,
            ["stage", "count", "percent_of_eligible_loop_pairs"],
        ),
        "",
        "## Top False Negatives",
        "",
        markdown_table(
            false_negatives[:10],
            [
                "pair_key",
                "first_failed_stage",
                "gt_translation_distance",
                "gt_rotation_angle_deg",
                "raw_dbow_rank",
                "shared_words",
                "bow_score",
                "accumulated_score",
                "retained_candidate",
                "passed_consistency",
                "final_matched_map_points",
            ],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_loop_classified = args.gt_loop_classified.expanduser().resolve()
    gt_positive_trace_file = trace_dir / "loop_gt_positive_trace.csv"
    if not gt_positive_trace_file.exists():
        raise FileNotFoundError(f"Missing trace file: {gt_positive_trace_file}")
    accumulation_trace_file = trace_dir / "loop_accumulation_trace.csv"

    classified_rows = load_gt_loop_classified(gt_loop_classified)
    trace_rows = load_gt_positive_trace(gt_positive_trace_file)
    accumulation_by_pair = load_accumulation_trace(accumulation_trace_file) if accumulation_trace_file.exists() else {}
    min_kf_gap = 10
    rows = select_analysis_rows(trace_rows, min_kf_gap=min_kf_gap)
    funnel = build_funnel(rows)
    failure_counts = build_failure_counts(rows)
    false_negatives = add_false_negative_annotations(rows)
    group_level_false_negatives = build_group_level_false_negative_analysis(rows, accumulation_by_pair)
    group_level_summary_rows = build_group_level_recall_summary(group_level_false_negatives)
    summary = build_summary(
        rows,
        funnel,
        historical_gt_loop_like_count=len(classified_rows),
        min_kf_gap=min_kf_gap,
        group_level_summary_rows=group_level_summary_rows,
    )

    funnel_path = output_dir / "gt_retrieval_stage_funnel.csv"
    summary_path = output_dir / "gt_retrieval_stage_summary.json"
    false_negative_path = output_dir / "gt_retrieval_false_negatives_detailed.csv"
    group_level_summary_path = output_dir / "gt_group_level_recall_summary.csv"
    group_level_false_negative_path = output_dir / "gt_group_level_false_negative_analysis.csv"
    report_path = output_dir / "gt_retrieval_stage_report.md"

    _write_csv(
        funnel_path,
        funnel + failure_counts,
        ["stage", "count", "percent_of_gt_loop_like", "drop_from_previous_stage", "drop_reason"],
    )
    _write_csv(
        false_negative_path,
        false_negatives,
        [
            "pair_key",
            "current_kf_id",
            "candidate_kf_id",
            "gt_translation_distance",
            "gt_rotation_angle_deg",
            "first_failed_stage",
            "raw_dbow_present",
            "raw_dbow_rank",
            "raw_dbow_score",
            "inverted_word_present",
            "shared_words",
            "common_word_ratio",
            "bow_score",
            "min_score",
            "score_over_min_score",
            "accumulated_score",
            "best_accumulated_score",
            "accumulated_score_ratio",
            "retained_candidate",
            "consistency_score",
            "final_matched_map_points",
            "suspected_cause",
            "recommended_next_action",
        ],
    )
    _write_csv(
        group_level_summary_path,
        group_level_summary_rows,
        ["stage", "count", "percent_of_eligible_loop_pairs"],
    )
    _write_csv(
        group_level_false_negative_path,
        group_level_false_negatives,
        [
            "pair_key",
            "current_kf_id",
            "candidate_kf_id",
            "gt_translation_distance",
            "gt_rotation_angle_deg",
            "connected_local",
            "temporal_eligible",
            "eligible_for_loop",
            "passed_accumulated_score_filter",
            "retained_exact",
            "representative_kf_id",
            "representative_pair_key",
            "representative_gt_equivalent",
            "representative_gt_translation_distance",
            "representative_gt_rotation_angle_deg",
            "group_recalled",
            "classification",
        ],
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        build_markdown_report(
            trace_dir=trace_dir,
            gt_loop_classified=gt_loop_classified,
            rows=rows,
            funnel=funnel,
            summary=summary,
            false_negatives=false_negatives,
            group_level_summary_rows=group_level_summary_rows,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
