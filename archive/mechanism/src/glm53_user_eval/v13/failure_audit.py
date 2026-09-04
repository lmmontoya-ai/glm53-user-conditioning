"""Row- and generator-level diagnostics after a completed failed V13 gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.glm53_user_eval.v12.fact_validation import FACTORS


def build_failure_audit(
    dataset: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    if analysis.get("technical_passed") is not True:
        raise ValueError("failure audit requires a technically valid cohort")
    if analysis.get("passed") is True:
        raise ValueError("failure audit is not created for a passing cohort")
    source_by_id = {str(row["sample_id"]): row for row in dataset}
    judges = analysis["judges"]
    rows_by_judge = {
        judge_id: {row["sample_id"]: row for row in result["row_results"]}
        for judge_id, result in judges.items()
    }
    generator_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    factor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pair_candidates: dict[str, set[str]] = defaultdict(set)
    diagnostics: list[dict[str, Any]] = []
    for sample_id in sorted(source_by_id):
        source = source_by_id[sample_id]
        luna = rows_by_judge["luna_max"][sample_id]
        terra = rows_by_judge["terra_high"][sample_id]
        luna_failed = bool(luna["issues"])
        terra_failed = bool(terra["issues"])
        if not (luna_failed or terra_failed):
            continue
        if luna_failed and terra_failed:
            category = "both_judges_flagged"
        elif luna_failed:
            category = "luna_only_flagged"
        else:
            category = "terra_only_flagged"
        generator = str(source["generator_family"])
        split = str(source["split"])
        generator_counts[generator][category] += 1
        split_counts[split][category] += 1
        pair_id = str(source.get("pair_id") or sample_id)
        pair_candidates[pair_id].add(sample_id)
        shared_factor_failures: list[str] = []
        for factor in FACTORS:
            luna_wrong = luna.get("factor_matches", {}).get(factor) is False
            terra_wrong = terra.get("factor_matches", {}).get(factor) is False
            if luna_wrong and terra_wrong:
                shared_factor_failures.append(factor)
                factor_counts[factor]["both_wrong"] += 1
            elif luna_wrong:
                factor_counts[factor]["luna_only_wrong"] += 1
            elif terra_wrong:
                factor_counts[factor]["terra_only_wrong"] += 1
        if shared_factor_failures:
            recommendation = "inspect_text_and_expected_facts_then_repair_matched_pair"
        elif luna_failed and terra_failed:
            recommendation = "inspect_evidence_contract_and_matched_pair"
        else:
            recommendation = "judge_sensitive_keep_unless_text_is_independently_defective"
        diagnostics.append(
            {
                "sample_id": sample_id,
                "pair_id": pair_id,
                "split": split,
                "generator_family": generator,
                "latent_class": source.get("latent_class"),
                "category": category,
                "shared_factor_failures": shared_factor_failures,
                "recommendation": recommendation,
                "scenario_text": source["scenario_text"],
                "expected_factors": source.get("factors"),
                "acceptable_judge_labels": source.get("acceptable_judge_labels"),
                "luna": {
                    "derived_label": luna["derived_label"],
                    "factor_predictions": luna["factor_predictions"],
                    "issues": luna["issues"],
                },
                "terra": {
                    "derived_label": terra["derived_label"],
                    "factor_predictions": terra["factor_predictions"],
                    "issues": terra["issues"],
                },
            }
        )
    complete_pair_ids = {
        str(row.get("pair_id") or row["sample_id"]) for row in dataset
    }
    return {
        "schema_version": "glm53_v13_failure_audit_v1",
        "project_id": "glm53_user_eval_codex_judge_cohort_v13",
        "diagnostic_row_count": len(diagnostics),
        "diagnostics": diagnostics,
        "generator_summary": {
            key: dict(value) for key, value in sorted(generator_counts.items())
        },
        "split_summary": {
            key: dict(value) for key, value in sorted(split_counts.items())
        },
        "factor_summary": {
            key: dict(value) for key, value in sorted(factor_counts.items())
        },
        "candidate_pair_ids": sorted(pair_candidates),
        "candidate_pair_count": len(pair_candidates),
        "all_dataset_pair_count": len(complete_pair_ids),
        "repair_constraints": {
            "automatic_row_deletion_allowed": False,
            "outcome_only_deletion_allowed": False,
            "matched_pair_or_generator_level_changes_required": True,
            "class_and_nuisance_balance_must_be_rechecked": True,
            "edited_final_holdout_can_remain_confirmatory": False,
            "fresh_untouched_holdout_required_after_any_final_holdout_edit": True,
            "new_preregistration_and_tag_required_before_rerun": True,
            "gpu_compute_remains_locked": True,
        },
    }


__all__ = ["build_failure_audit"]
