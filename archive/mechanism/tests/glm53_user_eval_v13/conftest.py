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
    load_dataset,
)
from src.glm53_user_eval.v13.codex_judge import (
    AUTH_STATUS,
    CLI_VERSION,
    JudgeSpec,
    prompt_for_scenario,
    request_sha256,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def frozen_rows(repo_root: Path) -> list[dict[str, Any]]:
    return load_dataset(
        repo_root / "artifacts/datasets/contrastive_prompts_v3/samples.jsonl"
    )


@pytest.fixture(scope="session")
def schema_path(repo_root: Path) -> Path:
    return repo_root / (
        "pipelines/glm53_user_eval/v13/configs/fact_judgment.schema.json"
    )


def perfect_parsed(row: Mapping[str, Any]) -> dict[str, Any]:
    expected = expected_factor_record(row)
    if expected is None:
        label = str(row["acceptable_judge_labels"][0])
        values = {
            "deploy": (True, False, False, False),
            "eval": (False, False, True, False),
            "ambiguous": (True, False, True, False),
            "no_status": (False, False, False, False),
        }[label]
        expected = dict(zip(FACTORS, values, strict=True))
    quote = str(row["scenario_text"])[:1]
    parsed: dict[str, Any] = {
        factor: {"value": expected[factor], "evidence": quote if expected[factor] else None}
        for factor in FACTORS
    }
    parsed["insufficient_or_contradictory"] = False
    parsed["notes"] = "fixture"
    return parsed


def judgment(
    row: Mapping[str, Any],
    *,
    spec: JudgeSpec,
    schema_path: Path,
    parsed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = prompt_for_scenario(str(row["scenario_text"]))
    return {
        "schema_version": "glm53_v13_codex_judgment_row_v1",
        "sample_id": str(row["sample_id"]),
        "judge_id": spec.judge_id,
        "model": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "request_sha256": request_sha256(
            spec=spec, prompt=prompt, schema_path=schema_path
        ),
        "prompt_sha256": "fixture",
        "prompt_template_sha256": "fixture",
        "schema_sha256": "fixture",
        "cli_version": CLI_VERSION,
        "auth_status": AUTH_STATUS,
        "command": [
            "codex",
            "exec",
            spec.model,
            f'model_reasoning_effort="{spec.reasoning_effort}"',
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "read-only",
        ],
        "removed_environment_keys": [],
        "checks": {
            "exit_zero": True,
            "not_timed_out": True,
            "stdout_is_jsonl": True,
            "no_tool_events": True,
            "structured_output_parsed": True,
        },
        "passed": True,
        "tool_events": [],
        "parsed": dict(parsed or perfect_parsed(row)),
    }
