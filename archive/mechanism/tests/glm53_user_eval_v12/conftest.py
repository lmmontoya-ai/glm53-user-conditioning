from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v12.fact_validation import (
    FACTORS,
    expected_factor_record,
    expected_label_from_factors,
    load_dataset,
    request_record,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def frozen_rows(repo_root: Path) -> list[dict[str, Any]]:
    return load_dataset(
        repo_root / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl"
    )


def parsed_for_label(label: str, scenario: str) -> dict[str, Any]:
    values = {
        "deploy": (True, True, False, False),
        "eval": (False, False, True, False),
        "ambiguous": (True, False, True, False),
        "no_status": (False, False, False, False),
    }[label]
    quote = scenario[:1]
    parsed = {
        factor: {
            "value": value,
            "evidence": quote if value else None,
        }
        for factor, value in zip(FACTORS, values, strict=True)
    }
    parsed["insufficient_or_contradictory"] = False
    parsed["notes"] = "synthetic exact-contract fixture"
    return parsed


def perfect_parsed(row: Mapping[str, Any]) -> dict[str, Any]:
    expected = expected_factor_record(row)
    if expected is None:
        return parsed_for_label(
            str(row["acceptable_judge_labels"][0]), str(row["scenario_text"])
        )
    quote = str(row["scenario_text"])[:1]
    parsed = {
        factor: {
            "value": expected[factor],
            "evidence": quote if expected[factor] else None,
        }
        for factor in FACTORS
    }
    parsed["insufficient_or_contradictory"] = False
    parsed["notes"] = expected_label_from_factors(expected)
    return parsed


def judgment(
    row: Mapping[str, Any],
    *,
    pass_kind: str = "primary",
    parsed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = request_record(row, pass_kind=pass_kind)  # type: ignore[arg-type]
    return {
        "schema_version": "glm53_v12_fact_judgment_row_v1",
        "sample_id": str(row["sample_id"]),
        "pass_kind": pass_kind,
        "request_sha256": "fixture",
        "request": request,
        "response_model": "openai/gpt-5.4-mini",
        "response_provider": "OpenAI",
        "response_id": f"fixture-{row['sample_id']}",
        "response_text": "{}",
        "parsed": dict(parsed or perfect_parsed(row)),
        "usage": {"cost": 0.0},
        "openrouter_metadata": {
            "requested": "openai/gpt-5.4-mini",
            "endpoints": {
                "available": [{"selected": True, "provider": "OpenAI"}]
            },
        },
    }
