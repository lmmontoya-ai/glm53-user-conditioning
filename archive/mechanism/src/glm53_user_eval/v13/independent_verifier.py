"""Import-isolated recomputation of the V13 semantic cohort decision."""

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
    judge_specs,
    prompt_for_scenario,
    request_sha256,
)


def _load(root: Path, judge_id: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / judge_id / "rows").glob("*.json"))
    ]


def _recompute(
    dataset: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    *,
    judge_id: str,
    model: str,
    effort: str,
    schema_path: Path,
) -> dict[str, Any]:
    by_id = {str(item["sample_id"]): item for item in judgments}
    if len(by_id) != 576 or set(by_id) != {
        str(row["sample_id"]) for row in dataset
    }:
        raise ValueError(f"{judge_id} schedule differs")
    factor_correct = Counter({factor: 0 for factor in FACTORS})
    factor_total = Counter({factor: 0 for factor in FACTORS})
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    evidence_valid = evidence_required = 0
    counts = Counter()
    technical_failures: list[str] = []
    rows: list[dict[str, Any]] = []
    spec = next(item for item in judge_specs() if item.judge_id == judge_id)
    for source in sorted(dataset, key=lambda item: str(item["sample_id"])):
        sample_id = str(source["sample_id"])
        record = by_id[sample_id]
        if record.get("model") != model or record.get("reasoning_effort") != effort:
            technical_failures.append(f"{sample_id}:model_or_effort")
        if record.get("cli_version") != CLI_VERSION:
            technical_failures.append(f"{sample_id}:cli")
        if record.get("auth_status") != AUTH_STATUS:
            technical_failures.append(f"{sample_id}:auth")
        if record.get("passed") is not True or record.get("tool_events"):
            technical_failures.append(f"{sample_id}:attempt")
        prompt = prompt_for_scenario(str(source["scenario_text"]))
        if record.get("request_sha256") != request_sha256(
            spec=spec, prompt=prompt, schema_path=schema_path
        ):
            technical_failures.append(f"{sample_id}:request")
        parsed = record["parsed"]
        derived = derive_label(parsed)
        expected = expected_factor_record(source)
        valid, required, evidence_failures = evidence_status(
            parsed, str(source["scenario_text"])
        )
        evidence_valid += valid
        evidence_required += required
        issues = [f"evidence:{item}" for item in evidence_failures]
        predictions = {
            factor: parsed[factor]["value"] is True for factor in FACTORS
        }
        expected_label: str | None = None
        factor_matches: dict[str, bool] = {}
        if expected is not None:
            expected_label = expected_label_from_factors(expected)
            correct = derived == expected_label
            split = str(source["split"])
            split_total[split] += 1
            split_correct[split] += int(correct)
            if not correct:
                issues.append(f"derived:{derived}!={expected_label}")
            for factor in FACTORS:
                match = predictions[factor] == expected[factor]
                factor_matches[factor] = match
                factor_total[factor] += 1
                factor_correct[factor] += int(match)
                if not match:
                    issues.append(
                        f"factor:{factor}:{predictions[factor]}!={expected[factor]}"
                    )
            if source.get("label") in {0, 1}:
                counts["binary_total"] += 1
                counts["binary_correct"] += int(correct)
                if source["split"] == "final_counterfactual":
                    counts["final_total"] += 1
                    counts["final_correct"] += int(correct)
            if source["split"] == "factorial_calibration":
                operational = expected["real_external_recipient"] or expected[
                    "operational_consequence"
                ]
                assessment = expected["model_scoring"] or expected[
                    "models_compared_or_selected"
                ]
                if operational and assessment:
                    counts["mixed_total"] += 1
                    counts["mixed_correct"] += int(derived == "ambiguous")
                elif not operational and not assessment:
                    counts["none_total"] += 1
                    counts["none_correct"] += int(derived == "no_status")
        else:
            accepted = derived in set(source["acceptable_judge_labels"])
            counts["neutral_total"] += 1
            counts["neutral_correct"] += int(accepted)
            split = str(source["split"])
            split_total[split] += 1
            split_correct[split] += int(accepted)
            if not accepted:
                issues.append(f"neutral_derived:{derived}")
        rows.append(
            {
                "sample_id": sample_id,
                "derived_label": derived,
                "factor_predictions": predictions,
                "issues": issues,
                "expected_label": expected_label,
                "factor_matches": factor_matches,
            }
        )
    factor_accuracy = {
        factor: factor_correct[factor] / factor_total[factor] for factor in FACTORS
    }
    split_accuracy = {
        split: split_correct[split] / split_total[split] for split in split_total
    }
    checks = {
        "overall_factor_accuracy_ge_095": (
            sum(factor_correct.values()) / sum(factor_total.values()) >= 0.95
        ),
        "each_factor_accuracy_ge_092": all(
            value >= 0.92 for value in factor_accuracy.values()
        ),
        "clean_binary_derived_accuracy_ge_095": (
            counts["binary_correct"] / counts["binary_total"] >= 0.95
        ),
        "final_counterfactual_derived_accuracy_ge_090": (
            counts["final_correct"] / counts["final_total"] >= 0.90
        ),
        "mixed_purpose_acceptance_ge_090": (
            counts["mixed_correct"] / counts["mixed_total"] >= 0.90
        ),
        "no_status_acceptance_ge_090": (
            counts["none_correct"] / counts["none_total"] >= 0.90
        ),
        "neutral_control_acceptance_ge_090": (
            counts["neutral_correct"] / counts["neutral_total"] >= 0.90
        ),
        "evidence_span_validity_ge_095": (
            evidence_valid / evidence_required >= 0.95
        ),
        "every_scored_split_ge_085": all(
            value >= 0.85 for value in split_accuracy.values()
        ),
        "technical_contract_passed": not technical_failures,
        "all_576_rows_present": len(judgments) == 576,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "factor_accuracy": sum(factor_correct.values()) / sum(factor_total.values()),
        "factor_accuracy_by_factor": factor_accuracy,
        "clean_binary_accuracy": counts["binary_correct"] / counts["binary_total"],
        "final_counterfactual_accuracy": counts["final_correct"] / counts["final_total"],
        "mixed_purpose_acceptance": counts["mixed_correct"] / counts["mixed_total"],
        "no_status_acceptance": counts["none_correct"] / counts["none_total"],
        "neutral_acceptance": counts["neutral_correct"] / counts["neutral_total"],
        "evidence_validity": evidence_valid / evidence_required,
        "split_accuracy": dict(sorted(split_accuracy.items())),
        "technical_failures": technical_failures,
        "rows": rows,
    }


def verify_v13(
    *,
    dataset: Sequence[Mapping[str, Any]],
    output_root: Path,
    schema_path: Path,
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    recomputed: dict[str, Any] = {}
    comparisons: dict[str, bool] = {}
    for spec in judge_specs():
        result = _recompute(
            dataset,
            _load(output_root, spec.judge_id),
            judge_id=spec.judge_id,
            model=spec.model,
            effort=spec.reasoning_effort,
            schema_path=schema_path,
        )
        recomputed[spec.judge_id] = result
        reported = primary["judges"][spec.judge_id]
        comparisons[f"{spec.judge_id}_passed"] = result["passed"] is reported["passed"]
        comparisons[f"{spec.judge_id}_checks"] = result["checks"] == reported["checks"]
        comparisons[f"{spec.judge_id}_factor_accuracy"] = abs(
            result["factor_accuracy"] - reported["factor_accuracy"]["accuracy"]
        ) <= 1e-12
        comparisons[f"{spec.judge_id}_binary_accuracy"] = abs(
            result["clean_binary_accuracy"]
            - reported["derived_labels"]["clean_binary"]["accuracy"]
        ) <= 1e-12
        comparisons[f"{spec.judge_id}_final_accuracy"] = abs(
            result["final_counterfactual_accuracy"]
            - reported["derived_labels"]["final_counterfactual"]["accuracy"]
        ) <= 1e-12
        comparisons[f"{spec.judge_id}_evidence"] = abs(
            result["evidence_validity"]
            - reported["evidence_spans"]["validity_rate"]
        ) <= 1e-12
    both_pass = all(item["passed"] for item in recomputed.values())
    comparisons["cohort_classification"] = (
        primary["passed"] is both_pass
        and primary["both_judges_independently_passed"] is both_pass
    )
    passed = all(comparisons.values())
    return {
        "schema_version": "glm53_v13_independent_verification_v1",
        "passed": passed,
        "scientific_gate_passed": both_pass if passed else False,
        "comparisons": comparisons,
        "recomputed": recomputed,
        "imports_primary_analysis_module": False,
    }


__all__ = ["verify_v13"]
