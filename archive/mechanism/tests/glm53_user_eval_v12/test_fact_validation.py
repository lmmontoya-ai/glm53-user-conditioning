from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from src.glm53_user_eval.v12.fact_validation import (
    PRIMARY_SYSTEM,
    VERIFIER_SYSTEM,
    analyze_primary,
    audit_route_contract,
    blinded_scenario_prompt,
    build_verifier_schedule,
    derive_label,
    evidence_status,
    expected_factor_record,
    load_dataset,
    parse_fact_json,
    request_record,
)

from .conftest import judgment, parsed_for_label, perfect_parsed


def test_frozen_dataset_has_exact_576_unique_rows(
    frozen_rows: list[dict],
) -> None:
    assert len(frozen_rows) == 576
    assert len({row["sample_id"] for row in frozen_rows}) == 576


def test_frozen_dataset_factor_coverage(frozen_rows: list[dict]) -> None:
    factor_rows = [row for row in frozen_rows if expected_factor_record(row) is not None]
    assert len(factor_rows) == 512
    assert sum(row.get("label") in {0, 1} for row in factor_rows) == 480


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("deploy", "deploy"),
        ("eval", "eval"),
        ("ambiguous", "ambiguous"),
        ("no_status", "no_status"),
    ],
)
def test_deterministic_label_rule(label: str, expected: str) -> None:
    assert derive_label(parsed_for_label(label, "scenario")) == expected


def test_parser_accepts_exact_json() -> None:
    parsed = parsed_for_label("ambiguous", "scenario")
    assert parse_fact_json(json.dumps(parsed)) == parsed


def test_parser_accepts_json_fence() -> None:
    parsed = parsed_for_label("deploy", "scenario")
    assert parse_fact_json(f"```json\n{json.dumps(parsed)}\n```") == parsed


def test_parser_rejects_extra_field() -> None:
    parsed = parsed_for_label("deploy", "scenario")
    parsed["hidden_label"] = "deploy"
    with pytest.raises(ValueError, match="fields differ"):
        parse_fact_json(json.dumps(parsed))


@pytest.mark.parametrize("bad", [1, 0, "true", [], {}])
def test_parser_rejects_non_boolean_factor_values(bad: object) -> None:
    parsed = parsed_for_label("no_status", "scenario")
    parsed["model_scoring"]["value"] = bad
    with pytest.raises(ValueError, match="true, false, or null"):
        parse_fact_json(json.dumps(parsed))


def test_primary_prompt_contains_only_scenario(frozen_rows: list[dict]) -> None:
    row = frozen_rows[0]
    prompt = blinded_scenario_prompt(row)
    assert prompt == f"SCENARIO\n{row['scenario_text']}"
    for forbidden_value in (
        row["split"],
        row["generator_family"],
        row["task_id"],
        row["latent_class"],
    ):
        assert str(forbidden_value) not in prompt


def test_system_prompts_do_not_offer_class_candidates() -> None:
    forbidden = re.compile(r"\b(?:eval|deploy|ambiguous|no_status)\b", re.IGNORECASE)
    assert forbidden.search(PRIMARY_SYSTEM) is None
    assert forbidden.search(VERIFIER_SYSTEM) is None


def test_request_omits_sampling_parameters(frozen_rows: list[dict]) -> None:
    request = request_record(frozen_rows[0], pass_kind="primary")
    assert {"temperature", "top_p", "seed"}.isdisjoint(request)
    assert request["extra_body"]["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_primary_and_verifier_prompts_are_independent(frozen_rows: list[dict]) -> None:
    primary = request_record(frozen_rows[0], pass_kind="primary")
    verifier = request_record(frozen_rows[0], pass_kind="verifier")
    assert primary["messages"][0] != verifier["messages"][0]
    assert primary["messages"][1] == verifier["messages"][1]


def test_evidence_requires_exact_scenario_substring() -> None:
    parsed = parsed_for_label("eval", "Exact scenario")
    parsed["model_scoring"]["evidence"] = "not present"
    valid, required, failures = evidence_status(parsed, "Exact scenario")
    assert (valid, required) == (0, 1)
    assert failures == ["model_scoring"]


def test_perfect_primary_analysis_passes(
    frozen_rows: list[dict], tmp_path: Path
) -> None:
    judgments = [judgment(row) for row in frozen_rows]
    report = analyze_primary(frozen_rows, judgments, output_root=tmp_path)
    assert report["passed"] is True
    assert report["factor_accuracy"]["accuracy"] == 1.0
    assert report["derived_labels"]["mixed_purpose"]["count"] == 18
    assert report["derived_labels"]["no_status"]["count"] == 2
    assert report["derived_labels"]["neutral_controls"]["count"] == 64
    assert report["split_metrics"]["neutral_controls"]["accuracy"] == 1.0


def test_mixed_purpose_regression_fails_primary_gate(
    frozen_rows: list[dict], tmp_path: Path
) -> None:
    judgments = []
    for row in frozen_rows:
        parsed = perfect_parsed(row)
        expected = expected_factor_record(row)
        if (
            row["split"] == "factorial_calibration"
            and expected is not None
            and (expected["real_external_recipient"] or expected["operational_consequence"])
            and (expected["model_scoring"] or expected["models_compared_or_selected"])
        ):
            parsed = parsed_for_label("deploy", row["scenario_text"])
        judgments.append(judgment(row, parsed=parsed))
    report = analyze_primary(frozen_rows, judgments, output_root=tmp_path)
    assert report["passed"] is False
    assert report["checks"]["mixed_purpose_acceptance_ge_090"] is False


def test_verifier_schedule_includes_all_mismatches_and_64_matches(
    frozen_rows: list[dict], tmp_path: Path
) -> None:
    primary = analyze_primary(
        frozen_rows, [judgment(row) for row in frozen_rows], output_root=tmp_path
    )
    primary["mismatch_sample_ids"] = [frozen_rows[0]["sample_id"]]
    primary["match_sample_ids"] = [
        row["sample_id"] for row in frozen_rows[1:]
    ]
    schedule = build_verifier_schedule(frozen_rows, primary)
    assert len(schedule["sample_ids"]) == 65
    assert frozen_rows[0]["sample_id"] in schedule["sample_ids"]
    assert schedule["verifier_can_rescue_primary_gate"] is False


def test_route_audit_rejects_provider_fallback(frozen_rows: list[dict]) -> None:
    record = judgment(frozen_rows[0])
    record["response_provider"] = "Novita"
    failures = audit_route_contract([record])
    assert failures == [
        {"sample_id": frozen_rows[0]["sample_id"], "failures": ["response_provider"]}
    ]


def test_load_dataset_rejects_wrong_row_count(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"sample_id": "one"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 576"):
        load_dataset(path)
