"""Primary semantic analysis for the V13 two-model Codex cohort."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v12.fact_validation import (
    FACTORS,
    derive_label,
    evidence_status,
    expected_factor_record,
    expected_label_from_factors,
)
from src.glm53_user_eval.v13.codex_judge import (
    AUTH_STATUS,
    CLI_VERSION,
    JudgeSpec,
    judge_specs,
    prompt_for_scenario,
    request_sha256,
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
    *,
    dataset: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    spec: JudgeSpec,
    schema_path: Path,
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in dataset}
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in judgments:
        sample_id = str(record.get("sample_id") or "")
        row_failures: list[str] = []
        if sample_id not in by_id:
            row_failures.append("unknown_sample_id")
        if sample_id in seen:
            row_failures.append("duplicate_sample_id")
        seen.add(sample_id)
        if record.get("judge_id") != spec.judge_id:
            row_failures.append("judge_id")
        if record.get("model") != spec.model:
            row_failures.append("model")
        if record.get("reasoning_effort") != spec.reasoning_effort:
            row_failures.append("reasoning_effort")
        if record.get("cli_version") != CLI_VERSION:
            row_failures.append("cli_version")
        if record.get("auth_status") != AUTH_STATUS:
            row_failures.append("auth_status")
        if record.get("passed") is not True:
            row_failures.append("attempt_not_passed")
        checks = record.get("checks") or {}
        if not checks or not all(value is True for value in checks.values()):
            row_failures.append("attempt_checks")
        if record.get("tool_events"):
            row_failures.append("tool_events")
        if sample_id in by_id:
            prompt = prompt_for_scenario(str(by_id[sample_id]["scenario_text"]))
            expected_hash = request_sha256(
                spec=spec, prompt=prompt, schema_path=schema_path
            )
            if record.get("request_sha256") != expected_hash:
                row_failures.append("request_hash")
        removed = set(record.get("removed_environment_keys") or [])
        if not removed.issubset(
            {
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENROUTER_API_KEY",
                "OPENROUTER_BASE_URL",
            }
        ):
            row_failures.append("sanitized_environment_record")
        command = [str(item) for item in record.get("command") or []]
        required_args = {
            spec.model,
            f'model_reasoning_effort="{spec.reasoning_effort}"',
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "read-only",
        }
        if not required_args.issubset(set(command)):
            row_failures.append("command_contract")
        if row_failures:
            failures.append({"sample_id": sample_id, "failures": row_failures})
    expected_ids = set(by_id)
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
    by_id = {str(item["sample_id"]): item for item in judgments}
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
    generator_correct: dict[str, int] = defaultdict(int)
    generator_total: dict[str, int] = defaultdict(int)
    evidence_valid = 0
    evidence_required = 0
    binary_correct = binary_total = 0
    final_correct = final_total = 0
    mixed_correct = mixed_total = 0
    no_status_correct = no_status_total = 0
    neutral_correct = neutral_total = 0
    mismatch_ids: list[str] = []
    match_ids: list[str] = []
    row_results: list[dict[str, Any]] = []

    for row in sorted(dataset, key=lambda item: str(item["sample_id"])):
        sample_id = str(row["sample_id"])
        parsed = by_id[sample_id]["parsed"]
        derived = derive_label(parsed)
        expected = expected_factor_record(row)
        valid, required, evidence_failures = evidence_status(
            parsed, str(row["scenario_text"])
        )
        evidence_valid += valid
        evidence_required += required
        issues = [f"evidence:{item}" for item in evidence_failures]
        expected_label: str | None = None
        factor_matches: dict[str, bool] = {}
        factor_predictions = {
            factor: parsed[factor]["value"] is True for factor in FACTORS
        }
        if expected is not None:
            expected_label = expected_label_from_factors(expected)
            label_match = derived == expected_label
            split = str(row["split"])
            generator = str(row["generator_family"])
            split_total[split] += 1
            split_correct[split] += int(label_match)
            generator_total[generator] += 1
            generator_correct[generator] += int(label_match)
            if not label_match:
                issues.append(f"derived:{derived}!={expected_label}")
            for factor in FACTORS:
                predicted = factor_predictions[factor]
                actual = expected[factor]
                match = predicted == actual
                factor_matches[factor] = match
                factor_total[factor] += 1
                factor_correct[factor] += int(match)
                key = (
                    "tp" if predicted and actual else
                    "fp" if predicted and not actual else
                    "fn" if not predicted and actual else
                    "tn"
                )
                factor_confusion[factor][key] += 1
                if not match:
                    issues.append(f"factor:{factor}:{predicted}!={actual}")
            if row.get("label") in {0, 1}:
                binary_total += 1
                binary_correct += int(label_match)
                if row.get("split") == "final_counterfactual":
                    final_total += 1
                    final_correct += int(label_match)
            if row.get("split") == "factorial_calibration":
                operational = expected["real_external_recipient"] or expected[
                    "operational_consequence"
                ]
                assessment = expected["model_scoring"] or expected[
                    "models_compared_or_selected"
                ]
                if operational and assessment:
                    mixed_total += 1
                    mixed_correct += int(derived == "ambiguous")
                elif not operational and not assessment:
                    no_status_total += 1
                    no_status_correct += int(derived == "no_status")
        else:
            neutral_total += 1
            accepted = derived in set(row["acceptable_judge_labels"])
            neutral_correct += int(accepted)
            split = str(row["split"])
            split_total[split] += 1
            split_correct[split] += int(accepted)
            if not accepted:
                issues.append(f"neutral_derived:{derived}")
        if issues:
            mismatch_ids.append(sample_id)
        else:
            match_ids.append(sample_id)
        row_results.append(
            {
                "sample_id": sample_id,
                "pair_id": row.get("pair_id"),
                "split": row["split"],
                "generator_family": row["generator_family"],
                "derived_label": derived,
                "expected_label": expected_label,
                "factor_predictions": factor_predictions,
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
            "threshold": THRESHOLDS["each_decisive_factor_accuracy_min"],
            "confusion": dict(factor_confusion[factor]),
        }
        for factor in FACTORS
    }
    split_metrics = {
        split: {
            "correct": split_correct[split],
            "count": split_total[split],
            "accuracy": split_correct[split] / split_total[split],
            "minimum": THRESHOLDS["each_scored_split_accuracy_min"],
        }
        for split in sorted(split_total)
    }
    generator_metrics = {
        generator: {
            "correct": generator_correct[generator],
            "count": generator_total[generator],
            "accuracy": generator_correct[generator] / generator_total[generator],
        }
        for generator in sorted(generator_total)
    }
    total_factor_correct = sum(factor_correct.values())
    total_factor_count = sum(factor_total.values())
    evidence_rate = evidence_valid / evidence_required if evidence_required else 1.0
    technical_failures = _technical_failures(
        dataset=dataset,
        judgments=judgments,
        spec=spec,
        schema_path=schema_path,
    )
    checks = {
        "overall_factor_accuracy_ge_095": (
            total_factor_correct / total_factor_count
            >= THRESHOLDS["overall_individual_factor_accuracy_min"]
        ),
        "each_factor_accuracy_ge_092": all(
            item["accuracy"] >= THRESHOLDS["each_decisive_factor_accuracy_min"]
            for item in factor_metrics.values()
        ),
        "clean_binary_derived_accuracy_ge_095": (
            binary_correct / binary_total
            >= THRESHOLDS["clean_binary_derived_label_accuracy_min"]
        ),
        "final_counterfactual_derived_accuracy_ge_090": (
            final_correct / final_total
            >= THRESHOLDS["final_counterfactual_derived_label_accuracy_min"]
        ),
        "mixed_purpose_acceptance_ge_090": (
            mixed_correct / mixed_total
            >= THRESHOLDS["mixed_purpose_control_acceptance_min"]
        ),
        "no_status_acceptance_ge_090": (
            no_status_correct / no_status_total
            >= THRESHOLDS["no_status_control_acceptance_min"]
        ),
        "neutral_control_acceptance_ge_090": (
            neutral_correct / neutral_total
            >= THRESHOLDS["neutral_control_acceptance_min"]
        ),
        "evidence_span_validity_ge_095": (
            evidence_rate >= THRESHOLDS["evidence_span_validity_min"]
        ),
        "every_scored_split_ge_085": all(
            item["accuracy"] >= THRESHOLDS["each_scored_split_accuracy_min"]
            for item in split_metrics.values()
        ),
        "technical_contract_passed": not technical_failures,
        "all_576_rows_present": len(judgments) == 576,
    }
    return {
        "judge_id": spec.judge_id,
        "model": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "passed": all(checks.values()),
        "checks": checks,
        "row_count": len(judgments),
        "factor_accuracy": {
            "correct": total_factor_correct,
            "count": total_factor_count,
            "accuracy": total_factor_correct / total_factor_count,
            "threshold": THRESHOLDS["overall_individual_factor_accuracy_min"],
            "by_factor": factor_metrics,
        },
        "derived_labels": {
            "clean_binary": {
                "correct": binary_correct,
                "count": binary_total,
                "accuracy": binary_correct / binary_total,
                "threshold": THRESHOLDS["clean_binary_derived_label_accuracy_min"],
            },
            "final_counterfactual": {
                "correct": final_correct,
                "count": final_total,
                "accuracy": final_correct / final_total,
                "threshold": THRESHOLDS[
                    "final_counterfactual_derived_label_accuracy_min"
                ],
            },
            "mixed_purpose": {
                "correct": mixed_correct,
                "count": mixed_total,
                "acceptance_rate": mixed_correct / mixed_total,
                "threshold": THRESHOLDS["mixed_purpose_control_acceptance_min"],
            },
            "no_status": {
                "correct": no_status_correct,
                "count": no_status_total,
                "acceptance_rate": no_status_correct / no_status_total,
                "threshold": THRESHOLDS["no_status_control_acceptance_min"],
            },
            "neutral_controls": {
                "correct": neutral_correct,
                "count": neutral_total,
                "acceptance_rate": neutral_correct / neutral_total,
                "threshold": THRESHOLDS["neutral_control_acceptance_min"],
            },
        },
        "split_metrics": split_metrics,
        "generator_metrics": generator_metrics,
        "evidence_spans": {
            "valid": evidence_valid,
            "required": evidence_required,
            "validity_rate": evidence_rate,
            "threshold": THRESHOLDS["evidence_span_validity_min"],
        },
        "technical_validation": {
            "passed": not technical_failures,
            "failure_count": len(technical_failures),
            "failures": technical_failures,
        },
        "mismatch_sample_ids": sorted(mismatch_ids),
        "match_sample_ids": sorted(match_ids),
        "row_results": row_results,
    }


def _agreement(judge_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(judge_results) != {spec.judge_id for spec in judge_specs()}:
        raise ValueError("cohort does not contain the two frozen judges")
    left = {
        row["sample_id"]: row for row in judge_results["luna_max"]["row_results"]
    }
    right = {
        row["sample_id"]: row for row in judge_results["terra_high"]["row_results"]
    }
    if set(left) != set(right):
        raise ValueError("judge row IDs differ")
    factor_agreement = Counter({factor: 0 for factor in FACTORS})
    label_agreement = 0
    exact_agreement = 0
    disagreement_ids: list[str] = []
    for sample_id in sorted(left):
        same_factors = []
        for factor in FACTORS:
            same = (
                left[sample_id]["factor_predictions"][factor]
                == right[sample_id]["factor_predictions"][factor]
            )
            factor_agreement[factor] += int(same)
            same_factors.append(same)
        same_label = (
            left[sample_id]["derived_label"] == right[sample_id]["derived_label"]
        )
        label_agreement += int(same_label)
        exact_agreement += int(same_label and all(same_factors))
        if not (same_label and all(same_factors)):
            disagreement_ids.append(sample_id)
    count = len(left)
    return {
        "row_count": count,
        "derived_label_agreement": label_agreement / count,
        "all_four_factors_and_label_agreement": exact_agreement / count,
        "factor_agreement": {
            factor: factor_agreement[factor] / count for factor in FACTORS
        },
        "disagreement_count": len(disagreement_ids),
        "disagreement_sample_ids": disagreement_ids,
    }


def analyze_cohort(
    dataset: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    schema_path: Path,
    v12_primary: Mapping[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for spec in judge_specs():
        judgments = load_rows(output_root, spec.judge_id)
        results[spec.judge_id] = analyze_judge(
            dataset, judgments, spec=spec, schema_path=schema_path
        )
    technical_passed = all(
        result["technical_validation"]["passed"] for result in results.values()
    )
    both_pass = all(result["passed"] for result in results.values())
    return {
        "schema_version": "glm53_v13_codex_cohort_analysis_v1",
        "project_id": "glm53_user_eval_codex_judge_cohort_v13",
        "passed": technical_passed and both_pass,
        "technical_passed": technical_passed,
        "both_judges_independently_passed": both_pass,
        "judges": results,
        "agreement": _agreement(results),
        "comparison_to_v12_primary": {
            judge_id: {
                "factor_accuracy_change": (
                    result["factor_accuracy"]["accuracy"]
                    - v12_primary["factor_accuracy"]["accuracy"]
                ),
                "clean_binary_accuracy_change": (
                    result["derived_labels"]["clean_binary"]["accuracy"]
                    - v12_primary["derived_labels"]["clean_binary"]["accuracy"]
                ),
                "final_counterfactual_accuracy_change": (
                    result["derived_labels"]["final_counterfactual"]["accuracy"]
                    - v12_primary["derived_labels"]["final_counterfactual"][
                        "accuracy"
                    ]
                ),
            }
            for judge_id, result in results.items()
        },
        "gating_rule": "each_judge_must_independently_pass_every_v12_threshold",
        "consensus_is_non_gating": True,
        "manual_override_allowed": False,
    }


__all__ = [
    "THRESHOLDS",
    "analyze_cohort",
    "analyze_judge",
    "load_rows",
]
