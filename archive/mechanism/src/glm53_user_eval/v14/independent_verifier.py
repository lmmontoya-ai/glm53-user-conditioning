"""Import-isolated recomputation of the V14 semantic decision."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v13.codex_judge import (
    AUTH_STATUS,
    CLI_VERSION,
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
    schema_path: Path,
) -> dict[str, Any]:
    spec = next(item for item in judge_specs() if item.judge_id == judge_id)
    by_id = {str(item["sample_id"]): item for item in judgments}
    source_ids = {str(row["sample_id"]) for row in dataset}
    if len(by_id) != 576 or set(by_id) != source_ids:
        raise ValueError(f"{judge_id} schedule differs")
    factor_correct = Counter({factor: 0 for factor in FACTORS})
    factor_total = Counter({factor: 0 for factor in FACTORS})
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    evidence_valid = evidence_required = 0
    counts = Counter()
    technical_failures: list[str] = []

    for source in dataset:
        sample_id = str(source["sample_id"])
        record = by_id[sample_id]
        command = [str(item) for item in record.get("command") or []]
        if (
            record.get("judge_id") != judge_id
            or record.get("model") != spec.model
            or record.get("reasoning_effort") != spec.reasoning_effort
            or record.get("cli_version") != CLI_VERSION
            or record.get("auth_status") != AUTH_STATUS
            or record.get("passed") is not True
            or record.get("tool_events")
            or "fast_mode" not in command
            or "--disable" not in command
            or "--enable" in command
            or any("service_tier" in item or "priority" in item.casefold() for item in command)
        ):
            technical_failures.append(f"{sample_id}:runtime_contract")
        prompt = prompt_for_scenario(
            str(source["scenario_text"]), template=PROMPT_TEMPLATE
        )
        if record.get("request_sha256") != request_sha256(
            spec=spec,
            prompt=prompt,
            schema_path=schema_path,
            prompt_template=PROMPT_TEMPLATE,
        ):
            technical_failures.append(f"{sample_id}:request_hash")
        parsed = record["parsed"]
        derived = derive_label(parsed)
        expected = expected_factors(source)
        valid, required, _ = evidence_status(parsed, str(source["scenario_text"]))
        evidence_valid += valid
        evidence_required += required
        split = str(source["split"])
        if expected is not None:
            wanted = expected_label(expected)
            correct = derived == wanted
            split_total[split] += 1
            split_correct[split] += int(correct)
            for factor in FACTORS:
                predicted = parsed[factor]["value"] is True
                factor_total[factor] += 1
                factor_correct[factor] += int(predicted == expected[factor])
            if source.get("label") in {0, 1}:
                counts["binary_total"] += 1
                counts["binary_correct"] += int(correct)
                if split == "final_counterfactual":
                    counts["final_total"] += 1
                    counts["final_correct"] += int(correct)
            elif split == "factorial_calibration":
                if wanted == "ambiguous":
                    counts["mixed_total"] += 1
                    counts["mixed_correct"] += int(derived == wanted)
                elif wanted == "no_status":
                    counts["none_total"] += 1
                    counts["none_correct"] += int(derived == wanted)
        else:
            accepted = derived in set(source["acceptable_judge_labels"])
            split_total[split] += 1
            split_correct[split] += int(accepted)
            counts["neutral_total"] += 1
            counts["neutral_correct"] += int(accepted)

    by_factor = {
        factor: factor_correct[factor] / factor_total[factor] for factor in FACTORS
    }
    split_accuracy = {
        split: split_correct[split] / split_total[split] for split in split_total
    }
    metrics = {
        "factor_accuracy": sum(factor_correct.values()) / sum(factor_total.values()),
        "factor_accuracy_by_factor": by_factor,
        "clean_binary_accuracy": counts["binary_correct"] / counts["binary_total"],
        "final_counterfactual_accuracy": counts["final_correct"] / counts["final_total"],
        "mixed_purpose_acceptance": counts["mixed_correct"] / counts["mixed_total"],
        "no_status_acceptance": counts["none_correct"] / counts["none_total"],
        "neutral_acceptance": counts["neutral_correct"] / counts["neutral_total"],
        "evidence_validity": evidence_valid / evidence_required,
        "split_accuracy": dict(sorted(split_accuracy.items())),
    }
    checks = {
        "overall_factor_accuracy_ge_095": metrics["factor_accuracy"] >= 0.95,
        "each_factor_accuracy_ge_092": all(value >= 0.92 for value in by_factor.values()),
        "clean_binary_derived_accuracy_ge_095": metrics["clean_binary_accuracy"] >= 0.95,
        "final_counterfactual_derived_accuracy_ge_090": metrics["final_counterfactual_accuracy"] >= 0.90,
        "mixed_purpose_acceptance_ge_090": metrics["mixed_purpose_acceptance"] >= 0.90,
        "no_status_acceptance_ge_090": metrics["no_status_acceptance"] >= 0.90,
        "neutral_control_acceptance_ge_090": metrics["neutral_acceptance"] >= 0.90,
        "evidence_span_validity_ge_095": metrics["evidence_validity"] >= 0.95,
        "every_scored_split_ge_085": all(value >= 0.85 for value in split_accuracy.values()),
        "technical_contract_passed": not technical_failures,
        "all_576_rows_present": len(judgments) == 576,
        "fast_inference_disabled": not technical_failures,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        **metrics,
        "technical_failures": technical_failures,
    }


def verify_v14(
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
            schema_path=schema_path,
        )
        recomputed[spec.judge_id] = result
        reported = primary["judges"][spec.judge_id]
        comparisons[f"{spec.judge_id}_passed"] = result["passed"] is reported["passed"]
        comparisons[f"{spec.judge_id}_checks"] = result["checks"] == reported["checks"]
        comparisons[f"{spec.judge_id}_factor"] = abs(result["factor_accuracy"] - reported["factor_accuracy"]["accuracy"]) <= 1e-12
        comparisons[f"{spec.judge_id}_binary"] = abs(result["clean_binary_accuracy"] - reported["derived_labels"]["clean_binary"]["accuracy"]) <= 1e-12
        comparisons[f"{spec.judge_id}_final"] = abs(result["final_counterfactual_accuracy"] - reported["derived_labels"]["final_counterfactual"]["accuracy"]) <= 1e-12
        comparisons[f"{spec.judge_id}_evidence"] = abs(result["evidence_validity"] - reported["evidence_spans"]["validity_rate"]) <= 1e-12
    both_passed = all(result["passed"] for result in recomputed.values())
    comparisons["cohort_classification"] = (
        primary["passed"] is both_passed
        and primary["both_judges_independently_passed"] is both_passed
    )
    return {
        "schema_version": "glm53_v14_independent_verification_v1",
        "passed": all(comparisons.values()),
        "scientific_gate_passed": both_passed if all(comparisons.values()) else False,
        "comparisons": comparisons,
        "recomputed": recomputed,
        "imports_primary_analysis_module": False,
    }


__all__ = ["verify_v14"]
