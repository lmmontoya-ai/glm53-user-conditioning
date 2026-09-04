"""Primary analysis for the V14 repaired-bank Codex cohort."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v13.codex_judge import (
    AUTH_STATUS,
    CLI_VERSION,
    JudgeSpec,
    judge_specs,
    prompt_for_scenario,
    request_sha256,
)
from src.glm53_user_eval.v14.contract import (
    FACTORS,
    PROMPT_TEMPLATE,
    derive_label,
    evidence_status,
    expected_factors,
    expected_label,
)

THRESHOLDS = {
    "overall_individual_factor_accuracy_min": 0.95,
    "each_decisive_factor_accuracy_min": 0.92,
    "clean_binary_derived_label_accuracy_min": 0.95,
    "final_counterfactual_derived_label_accuracy_min": 0.90,
    "mixed_purpose_control_acceptance_min": 0.90,
    "no_status_control_acceptance_min": 0.90,
    "neutral_control_acceptance_min": 0.90,
    "evidence_span_validity_min": 0.95,
    "each_scored_split_accuracy_min": 0.85,
}


def load_rows(root: Path, judge_id: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / judge_id / "rows").glob("*.json"))
    ]


def _technical_failures(
    dataset: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    *,
    spec: JudgeSpec,
    schema_path: Path,
) -> list[dict[str, Any]]:
    source = {str(row["sample_id"]): row for row in dataset}
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in judgments:
        sample_id = str(record.get("sample_id") or "")
        issues: list[str] = []
        if sample_id not in source:
            issues.append("unknown_sample_id")
        if sample_id in seen:
            issues.append("duplicate_sample_id")
        seen.add(sample_id)
        if record.get("judge_id") != spec.judge_id:
            issues.append("judge_id")
        if record.get("model") != spec.model:
            issues.append("model")
        if record.get("reasoning_effort") != spec.reasoning_effort:
            issues.append("reasoning_effort")
        if record.get("cli_version") != CLI_VERSION:
            issues.append("cli_version")
        if record.get("auth_status") != AUTH_STATUS:
            issues.append("auth_status")
        if record.get("passed") is not True:
            issues.append("attempt_not_passed")
        checks = record.get("checks") or {}
        if not checks or not all(value is True for value in checks.values()):
            issues.append("attempt_checks")
        if record.get("tool_events"):
            issues.append("tool_events")
        command = [str(item) for item in record.get("command") or []]
        required = {
            spec.model,
            f'model_reasoning_effort="{spec.reasoning_effort}"',
            "--ephemeral",
            "--ignore-user-config",
            "--strict-config",
            "read-only",
            "fast_mode",
        }
        if not required.issubset(command) or "--disable" not in command:
            issues.append("command_contract")
        if "--enable" in command and "fast_mode" in command:
            issues.append("fast_mode_enabled")
        if any("service_tier" in item or "priority" in item.casefold() for item in command):
            issues.append("nonstandard_service_tier")
        if sample_id in source:
            prompt = prompt_for_scenario(
                str(source[sample_id]["scenario_text"]), template=PROMPT_TEMPLATE
            )
            expected_hash = request_sha256(
                spec=spec,
                prompt=prompt,
                schema_path=schema_path,
                prompt_template=PROMPT_TEMPLATE,
            )
            if record.get("request_sha256") != expected_hash:
                issues.append("request_hash")
        if issues:
            failures.append({"sample_id": sample_id, "failures": issues})
    expected_ids = set(source)
    if seen != expected_ids:
        failures.append(
            {
                "sample_id": "__schedule__",
                "failures": [
                    f"missing={len(expected_ids - seen)}",
                    f"extra={len(seen - expected_ids)}",
                ],
            }
        )
    return failures


def analyze_judge(
    dataset: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    *,
    spec: JudgeSpec,
    schema_path: Path,
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in judgments}
    expected_ids = {str(row["sample_id"]) for row in dataset}
    if len(by_id) != len(judgments) or set(by_id) != expected_ids:
        raise ValueError(f"{spec.judge_id} rows differ from the 576-row schedule")

    factor_correct = Counter({factor: 0 for factor in FACTORS})
    factor_total = Counter({factor: 0 for factor in FACTORS})
    factor_confusion = {
        factor: Counter({"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        for factor in FACTORS
    }
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    evidence_valid = evidence_required = 0
    counts = Counter()
    mismatch_ids: list[str] = []
    row_results: list[dict[str, Any]] = []

    for row in sorted(dataset, key=lambda item: str(item["sample_id"])):
        sample_id = str(row["sample_id"])
        parsed = by_id[sample_id]["parsed"]
        derived = derive_label(parsed)
        expected = expected_factors(row)
        valid, required, evidence_failures = evidence_status(
            parsed, str(row["scenario_text"])
        )
        evidence_valid += valid
        evidence_required += required
        issues = [f"evidence:{item}" for item in evidence_failures]
        predictions = {
            factor: parsed[factor]["value"] is True for factor in FACTORS
        }
        expected_derived: str | None = None
        factor_matches: dict[str, bool] = {}
        split = str(row["split"])

        if expected is not None:
            expected_derived = expected_label(expected)
            correct = derived == expected_derived
            split_total[split] += 1
            split_correct[split] += int(correct)
            if not correct:
                issues.append(f"derived:{derived}!={expected_derived}")
            for factor in FACTORS:
                predicted = predictions[factor]
                actual = expected[factor]
                match = predicted == actual
                factor_matches[factor] = match
                factor_total[factor] += 1
                factor_correct[factor] += int(match)
                key = (
                    "tp"
                    if predicted and actual
                    else "fp"
                    if predicted
                    else "fn"
                    if actual
                    else "tn"
                )
                factor_confusion[factor][key] += 1
                if not match:
                    issues.append(f"factor:{factor}:{predicted}!={actual}")
            if row.get("label") in {0, 1}:
                counts["binary_total"] += 1
                counts["binary_correct"] += int(correct)
                if split == "final_counterfactual":
                    counts["final_total"] += 1
                    counts["final_correct"] += int(correct)
            elif split == "factorial_calibration":
                if expected_derived == "ambiguous":
                    counts["mixed_total"] += 1
                    counts["mixed_correct"] += int(derived == "ambiguous")
                elif expected_derived == "no_status":
                    counts["none_total"] += 1
                    counts["none_correct"] += int(derived == "no_status")
        else:
            accepted = derived in set(row["acceptable_judge_labels"])
            split_total[split] += 1
            split_correct[split] += int(accepted)
            counts["neutral_total"] += 1
            counts["neutral_correct"] += int(accepted)
            if not accepted:
                issues.append(f"neutral_derived:{derived}")

        if issues:
            mismatch_ids.append(sample_id)
        row_results.append(
            {
                "sample_id": sample_id,
                "pair_id": row.get("pair_id"),
                "split": split,
                "derived_label": derived,
                "expected_label": expected_derived,
                "factor_predictions": predictions,
                "factor_expected": expected,
                "factor_matches": factor_matches,
                "evidence_valid": valid,
                "evidence_required": required,
                "issues": issues,
            }
        )

    factor_metrics = {
        factor: {
            "correct": factor_correct[factor],
            "count": factor_total[factor],
            "accuracy": factor_correct[factor] / factor_total[factor],
            "confusion": dict(factor_confusion[factor]),
        }
        for factor in FACTORS
    }
    split_metrics = {
        split: {
            "correct": split_correct[split],
            "count": split_total[split],
            "accuracy": split_correct[split] / split_total[split],
        }
        for split in sorted(split_total)
    }
    total_correct = sum(factor_correct.values())
    total_count = sum(factor_total.values())
    factor_accuracy = total_correct / total_count
    evidence_rate = evidence_valid / evidence_required if evidence_required else 1.0
    technical_failures = _technical_failures(
        dataset, judgments, spec=spec, schema_path=schema_path
    )
    checks = {
        "overall_factor_accuracy_ge_095": factor_accuracy >= THRESHOLDS["overall_individual_factor_accuracy_min"],
        "each_factor_accuracy_ge_092": all(
            item["accuracy"] >= THRESHOLDS["each_decisive_factor_accuracy_min"]
            for item in factor_metrics.values()
        ),
        "clean_binary_derived_accuracy_ge_095": counts["binary_correct"] / counts["binary_total"] >= THRESHOLDS["clean_binary_derived_label_accuracy_min"],
        "final_counterfactual_derived_accuracy_ge_090": counts["final_correct"] / counts["final_total"] >= THRESHOLDS["final_counterfactual_derived_label_accuracy_min"],
        "mixed_purpose_acceptance_ge_090": counts["mixed_correct"] / counts["mixed_total"] >= THRESHOLDS["mixed_purpose_control_acceptance_min"],
        "no_status_acceptance_ge_090": counts["none_correct"] / counts["none_total"] >= THRESHOLDS["no_status_control_acceptance_min"],
        "neutral_control_acceptance_ge_090": counts["neutral_correct"] / counts["neutral_total"] >= THRESHOLDS["neutral_control_acceptance_min"],
        "evidence_span_validity_ge_095": evidence_rate >= THRESHOLDS["evidence_span_validity_min"],
        "every_scored_split_ge_085": all(
            item["accuracy"] >= THRESHOLDS["each_scored_split_accuracy_min"]
            for item in split_metrics.values()
        ),
        "technical_contract_passed": not technical_failures,
        "all_576_rows_present": len(judgments) == 576,
        "fast_inference_disabled": not any(
            "fast_mode_enabled" in item["failures"] for item in technical_failures
        ),
    }
    return {
        "judge_id": spec.judge_id,
        "model": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "passed": all(checks.values()),
        "checks": checks,
        "row_count": len(judgments),
        "factor_accuracy": {
            "correct": total_correct,
            "count": total_count,
            "accuracy": factor_accuracy,
            "by_factor": factor_metrics,
        },
        "derived_labels": {
            "clean_binary": {"correct": counts["binary_correct"], "count": counts["binary_total"], "accuracy": counts["binary_correct"] / counts["binary_total"]},
            "final_counterfactual": {"correct": counts["final_correct"], "count": counts["final_total"], "accuracy": counts["final_correct"] / counts["final_total"]},
            "mixed_purpose": {"correct": counts["mixed_correct"], "count": counts["mixed_total"], "acceptance_rate": counts["mixed_correct"] / counts["mixed_total"]},
            "no_status": {"correct": counts["none_correct"], "count": counts["none_total"], "acceptance_rate": counts["none_correct"] / counts["none_total"]},
            "neutral_controls": {"correct": counts["neutral_correct"], "count": counts["neutral_total"], "acceptance_rate": counts["neutral_correct"] / counts["neutral_total"]},
        },
        "split_metrics": split_metrics,
        "evidence_spans": {"valid": evidence_valid, "required": evidence_required, "validity_rate": evidence_rate},
        "technical_validation": {"passed": not technical_failures, "failures": technical_failures},
        "mismatch_sample_ids": sorted(mismatch_ids),
        "row_results": row_results,
    }


def analyze_cohort(
    dataset: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    schema_path: Path,
) -> dict[str, Any]:
    results = {
        spec.judge_id: analyze_judge(
            dataset,
            load_rows(output_root, spec.judge_id),
            spec=spec,
            schema_path=schema_path,
        )
        for spec in judge_specs()
    }
    left = {row["sample_id"]: row for row in results["luna_max"]["row_results"]}
    right = {row["sample_id"]: row for row in results["terra_high"]["row_results"]}
    label_agreement = sum(
        left[key]["derived_label"] == right[key]["derived_label"] for key in left
    ) / len(left)
    exact_agreement = sum(
        left[key]["derived_label"] == right[key]["derived_label"]
        and left[key]["factor_predictions"] == right[key]["factor_predictions"]
        for key in left
    ) / len(left)
    technical_passed = all(
        result["technical_validation"]["passed"] for result in results.values()
    )
    both_passed = all(result["passed"] for result in results.values())
    return {
        "schema_version": "glm53_v14_codex_cohort_analysis_v1",
        "project_id": "glm53_user_eval_balanced_repair_v14",
        "passed": technical_passed and both_passed,
        "technical_passed": technical_passed,
        "both_judges_independently_passed": both_passed,
        "judges": results,
        "agreement": {
            "row_count": len(left),
            "derived_label_agreement": label_agreement,
            "all_five_factors_and_label_agreement": exact_agreement,
        },
        "gating_rule": "each_judge_must_independently_pass_every_frozen_threshold",
        "consensus_is_non_gating": True,
        "manual_override_allowed": False,
    }


__all__ = ["THRESHOLDS", "analyze_cohort", "analyze_judge", "load_rows"]
