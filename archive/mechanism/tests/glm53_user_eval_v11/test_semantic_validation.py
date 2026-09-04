from __future__ import annotations

import json

import pytest
from src.glm53_user_eval.v11.builder import (
    _binary_rows,
    _factorial_calibration_rows,
    _neutral_rows,
)
from src.glm53_user_eval.v11.semantic_validation import (
    analyze_semantic_judgments,
    blinded_judge_prompt,
    parse_judge_json,
    request_record,
)


def all_rows() -> list[dict]:
    rows = _binary_rows("train", 128)
    rows += _binary_rows("validation", 24)
    rows += _binary_rows("ordinary_test", 24)
    rows += _binary_rows("development_counterfactual", 32)
    rows += _binary_rows("final_counterfactual", 32)
    return rows + _neutral_rows() + _factorial_calibration_rows()


def parsed(label: str) -> dict:
    return {
        "label": label,
        "real_or_operational_recipient": None,
        "operational_consequence": None,
        "primary_model_assessment": None,
        "facts_used": ["one fact"],
        "confidence": 0.9,
        "contradictory": False,
    }


def judgment(row: dict, label: str) -> dict:
    return {
        "sample_id": row["sample_id"],
        "parsed": parsed(label),
        "request": request_record(
            row, model="openai/gpt-5.4-mini", max_tokens=2000
        ),
        "response_model": "openai/gpt-5.4-mini",
        "response_provider": "OpenAI",
        "openrouter_metadata": {
            "requested": "openai/gpt-5.4-mini",
            "endpoints": {
                "available": [
                    {
                        "provider": "OpenAI",
                        "model": "openai/gpt-5.4-mini",
                        "selected": True,
                    }
                ]
            },
        },
        "usage": {"cost": 0.001},
    }


def test_judge_prompt_is_blinded_to_stored_label_and_factors() -> None:
    row = next(item for item in all_rows() if item["label"] == 1)
    prompt = blinded_judge_prompt(row)
    assert row["system_prompt"] in prompt
    assert row["user_prompt"] in prompt
    assert "primary_model_assessment" not in prompt
    assert '"label": 1' not in prompt


def test_request_pins_model_provider_and_low_reasoning() -> None:
    request = request_record(all_rows()[0], model="openai/gpt-5.4-mini", max_tokens=2000)
    assert request["model"] == "openai/gpt-5.4-mini"
    assert "temperature" not in request
    assert request["extra_body"]["reasoning"] == {"enabled": True, "effort": "low"}
    assert request["extra_body"]["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_parser_accepts_plain_and_fenced_json() -> None:
    payload = parsed("eval")
    assert parse_judge_json(json.dumps(payload)) == payload
    assert parse_judge_json(f"```json\n{json.dumps(payload)}\n```") == payload


def test_parser_rejects_missing_or_extra_fields() -> None:
    payload = parsed("deploy")
    payload.pop("facts_used")
    with pytest.raises(ValueError, match="fields differ"):
        parse_judge_json(json.dumps(payload))


def test_semantic_analysis_uses_only_acceptable_labels() -> None:
    rows = all_rows()
    judgments = []
    for row in rows:
        label = row["acceptable_judge_labels"][0]
        judgments.append(judgment(row, label))
    report = analyze_semantic_judgments(rows, judgments)
    assert report["passed"] is True
    assert report["binary"]["accuracy"] == 1.0
    assert report["final_counterfactual"]["accuracy"] == 1.0
    assert report["controls"]["acceptance_rate"] == 1.0


def test_semantic_analysis_requires_control_specificity() -> None:
    rows = all_rows()
    judgments = []
    for row in rows:
        label = row["acceptable_judge_labels"][0]
        if row["label"] is None:
            label = "eval"
        judgments.append(judgment(row, label))
    report = analyze_semantic_judgments(rows, judgments)
    assert report["passed"] is False
    assert report["controls"]["acceptance_rate"] < 0.90


def test_semantic_analysis_fails_closed_on_missing_rows() -> None:
    rows = all_rows()
    judgments = [
        judgment(row, row["acceptable_judge_labels"][0])
        for row in rows[:-1]
    ]
    with pytest.raises(ValueError, match="IDs differ"):
        analyze_semantic_judgments(rows, judgments)
