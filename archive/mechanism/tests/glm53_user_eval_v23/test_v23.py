from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.glm53_user_eval.v23.analysis import _holm, interaction, matrices
from src.glm53_user_eval.v23.judges import parse_category_json, parse_content_json
from src.glm53_user_eval.v23.transcripts import (
    _content,
    _hide_confidence,
    _redact,
    _sentence_count,
    select_matched_cells,
)


def test_matched_cell_selection_is_deterministic_and_complete() -> None:
    left = select_matched_cells(count=500, salt="test")
    right = select_matched_cells(count=500, salt="test")
    assert left == right
    assert len(left) == len(set(left)) == 500
    assert len({index for index, _ in left}) == 70
    assert len({task for _, task in left}) == 100


def test_confidence_is_hidden() -> None:
    assert _hide_confidence("I am 85% sure and perhaps 80–90% overall") == (
        "I am <REPORTED_CONFIDENCE> sure and perhaps <REPORTED_CONFIDENCE> overall"
    )


def test_redaction_is_case_insensitive() -> None:
    row = {
        "email": "a@b.com",
        "affiliation": "is at Lab",
        "name": "Jane Doe",
        "organization": "Lab",
    }
    result = _redact("JANE DOE at Lab, a@b.com", row)
    assert "jane" not in result.casefold()
    assert "lab" not in result.casefold()
    assert "a@b.com" not in result


def test_sentence_count() -> None:
    assert _sentence_count("") == 0
    assert _sentence_count("One. Two? Three!") == 3


def test_content_extraction() -> None:
    event = {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "reasoning", "reasoning": "why"},
                            {"type": "text", "text": "yes"},
                        ]
                    }
                }
            ]
        }
    }
    assert _content(event) == ("why", "yes")


def test_content_parser() -> None:
    value = {
        "explicit_evaluation_or_scrutiny_inference": False,
        "deference_to_user_expertise": 0,
        "caution_or_risk_sensitivity": 1,
        "hedging_or_uncertainty": 2,
        "social_desirability_or_impression_management": 0,
        "reasoning_supports_reported_confidence": 2,
        "brief_rationale": "Clear.",
    }
    assert parse_content_json(json.dumps(value)) == value


def test_content_parser_rejects_extra_key() -> None:
    with pytest.raises(ValueError):
        parse_content_json(json.dumps({"extra": True}))


def test_category_parser() -> None:
    value = {"category": "athlete", "brief_rationale": "Professional player."}
    assert parse_category_json(json.dumps(value)) == value


def test_category_parser_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        parse_category_json(json.dumps({"category": "celebrity", "brief_rationale": "x"}))


def test_interaction_known_fixture() -> None:
    source = {
        "famous_ai": np.full((2, 2), 1.0),
        "unknown_ai": np.full((2, 2), 0.5),
        "famous_nonai": np.full((2, 2), 0.25),
        "genpop": np.zeros((2, 2)),
    }
    point, means = interaction(source)
    assert point == pytest.approx(0.25)
    assert means["famous_ai"] == 1.0


def test_matrices_preserve_missingness() -> None:
    rows = [{"group": "famous_ai", "identity_index": 0, "stimulus_id": "dd_0000", "value": 2}]
    result = matrices(rows, lambda row: float(row["value"]))
    assert result["famous_ai"][0, 0] == 2
    assert np.isnan(result["famous_ai"][0, 1])


def test_holm_monotone() -> None:
    result = _holm({"a": 0.01, "b": 0.03, "c": 0.04})
    assert result == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_prereg_declares_no_new_glm_calls() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    prereg = yaml.safe_load(
        (
            root / "pipelines/glm53_user_eval/v23/configs/prereg_v23_behavioral_decomposition.yaml"
        ).read_text(encoding="utf-8")
    )
    assert prereg["new_subject_model_calls"] == 0
    assert prereg["runpod_spend_usd"] == 0
    assert prereg["judges"]["fast_mode"] is False
    assert len(prereg["statistics"]["primary_outcomes"]) == 3


def test_human_review_cannot_be_claimed_by_automation() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    prereg = yaml.safe_load(
        (
            root / "pipelines/glm53_user_eval/v23/configs/prereg_v23_behavioral_decomposition.yaml"
        ).read_text(encoding="utf-8")
    )
    assert prereg["human_audit"]["automation_may_claim_completion"] is False
