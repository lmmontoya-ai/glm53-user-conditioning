from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pipelines.glm53_user_eval.v13.run import (
    MAX_SCIENTIFIC_ATTEMPTS,
    SCIENTIFIC_EXECUTION_TAGS,
)
from src.glm53_user_eval.v12.fact_validation import PRIMARY_SYSTEM, parse_fact_json
from src.glm53_user_eval.v13 import codex_judge
from src.glm53_user_eval.v13.codex_judge import (
    DISABLED_FEATURES,
    JUDGES,
    JudgeSpec,
    command_for,
    prompt_for_scenario,
    prompt_template_sha256,
    request_sha256,
    sanitized_environment,
)
from src.glm53_user_eval.v13.evidence import build_compact_raw_judgments


def test_exact_models_and_efforts() -> None:
    assert JUDGES == {
        "luna_max": {"model": "gpt-5.6-luna", "reasoning_effort": "max"},
        "terra_high": {"model": "gpt-5.6-terra", "reasoning_effort": "high"},
    }


def test_scientific_execution_tags_are_explicit() -> None:
    assert SCIENTIFIC_EXECUTION_TAGS == (
        "glm53-user-eval-v13-preregistered",
        "glm53-user-eval-v13-transport-amendment-v2",
        "glm53-user-eval-v13-failure-isolation",
        "glm53-user-eval-v13-retry-budget",
    )
    assert MAX_SCIENTIFIC_ATTEMPTS == 12


def test_prompt_keeps_v12_instruction_and_only_adds_operational_wrapper() -> None:
    prompt = prompt_for_scenario("A self-contained scenario.")
    assert PRIMARY_SYSTEM in prompt
    assert prompt.endswith("SCENARIO\nA self-contained scenario.")
    assert "gpt-5.6" not in prompt.casefold()
    assert "expected" not in prompt.casefold()


def test_prompt_template_hash_is_deterministic() -> None:
    assert len(prompt_template_sha256()) == 64
    assert prompt_template_sha256() == prompt_template_sha256()


def test_schema_matches_v12_parser(schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    payload = {
        key: {"value": False, "evidence": None}
        for key in (
            "real_external_recipient",
            "operational_consequence",
            "model_scoring",
            "models_compared_or_selected",
        )
    }
    payload["insufficient_or_contradictory"] = False
    payload["notes"] = "none"
    assert set(schema["required"]) == set(payload)
    assert parse_fact_json(json.dumps(payload)) == payload


def test_command_disables_tools_and_pins_model_effort(
    tmp_path: Path, schema_path: Path
) -> None:
    spec = JudgeSpec("luna_max", "gpt-5.6-luna", "max")
    command = command_for(
        spec=spec,
        schema_path=schema_path,
        output_path=tmp_path / "out.json",
        isolated_workspace=tmp_path,
    )
    assert spec.model in command
    assert 'model_reasoning_effort="max"' in command
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "read-only" in command
    for feature in DISABLED_FEATURES:
        position = command.index(feature)
        assert command[position - 1] == "--disable"


def test_request_hash_changes_with_model_or_scenario(schema_path: Path) -> None:
    luna = JudgeSpec("luna_max", "gpt-5.6-luna", "max")
    terra = JudgeSpec("terra_high", "gpt-5.6-terra", "high")
    one = prompt_for_scenario("one")
    two = prompt_for_scenario("two")
    assert request_sha256(spec=luna, prompt=one, schema_path=schema_path) != request_sha256(
        spec=terra, prompt=one, schema_path=schema_path
    )
    assert request_sha256(spec=luna, prompt=one, schema_path=schema_path) != request_sha256(
        spec=luna, prompt=two, schema_path=schema_path
    )


def test_child_environment_removes_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    environment, removed = sanitized_environment()
    assert "OPENAI_API_KEY" not in environment
    assert "OPENROUTER_API_KEY" not in environment
    assert removed == ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]


def test_attempt_artifact_numbers_preserve_orphans(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempts"
    attempt_dir.mkdir()
    (attempt_dir / "last_message_01.json").write_text("{}", encoding="utf-8")
    (attempt_dir / "events_02.jsonl.partial").write_text("", encoding="utf-8")
    (attempt_dir / "attempt_03.json").write_text("{}", encoding="utf-8")
    (attempt_dir / "unrelated.txt").write_text("", encoding="utf-8")
    assert codex_judge._attempt_artifact_numbers(attempt_dir) == {1, 2, 3}


def test_command_never_requests_fast_or_priority(
    tmp_path: Path, schema_path: Path
) -> None:
    command = command_for(
        spec=JudgeSpec("terra_high", "gpt-5.6-terra", "high"),
        schema_path=schema_path,
        output_path=tmp_path / "out.json",
        isolated_workspace=tmp_path,
    )
    joined = " ".join(command).casefold()
    assert "service_tier" not in joined
    assert "priority" not in joined
    fast_index = command.index("fast_mode")
    assert command[fast_index - 1] == "--disable"
    assert "--enable" not in command
    assert "--ignore-user-config" in command


def test_transient_websocket_connect_error_is_retryable() -> None:
    message = (
        "HTTP error: 403 Forbidden; responses_websocket failed to connect "
        "to the subscription service"
    )
    assert codex_judge.RETRYABLE_TEXT.search(message)


def test_gather_does_not_cancel_peer_on_failure() -> None:
    peer_completed = False

    async def fail() -> dict[str, object]:
        raise ValueError("technical failure")

    async def complete() -> dict[str, object]:
        nonlocal peer_completed
        await asyncio.sleep(0.01)
        peer_completed = True
        return {"passed": True}

    async def exercise() -> None:
        tasks = [asyncio.create_task(fail()), asyncio.create_task(complete())]
        try:
            await codex_judge._gather_without_peer_cancellation(tasks)
        except RuntimeError as exc:
            assert "1 Codex judgment task(s) failed" in str(exc)
        else:
            raise AssertionError("failure must be reported")

    asyncio.run(exercise())
    assert peer_completed


def test_compact_raw_judgments_are_sorted(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for judge_id, sample_id in (("terra_high", "b"), ("luna_max", "a")):
        row_dir = run_root / "scientific" / judge_id / "rows"
        row_dir.mkdir(parents=True, exist_ok=True)
        (row_dir / f"{sample_id}.json").write_text(
            json.dumps({"judge_id": judge_id, "sample_id": sample_id}),
            encoding="utf-8",
        )
    output = tmp_path / "raw.jsonl"
    assert build_compact_raw_judgments(run_root=run_root, output_path=output) == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["judge_id"] for row in rows] == ["luna_max", "terra_high"]
