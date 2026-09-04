import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion

from src.glm53_user_eval.api import (
    _acquire_run_lock,
    _completed_ids_for_schedule,
    _load_completion_checkpoint,
    _release_run_lock,
    _save_completion_checkpoint,
    run_one_behavior_sample,
    validate_openrouter_zai_results,
)


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/configs/prereg_v4_api_serverless.yaml"


def _route() -> dict:
    return {
        "requested": "z-ai/glm-5.3-flash",
        "strategy": "direct",
        "endpoints": {
            "available": [
                {
                    "provider": "Z.AI",
                    "model": "z-ai/glm-5.3-flash",
                    "selected": True,
                }
            ]
        },
        "attempts": [
            {"provider": "Z.AI", "model": "z-ai/glm-5.3-flash", "status": 200}
        ],
    }


def _judge_route() -> dict:
    route = _route()
    route["requested"] = "openai/gpt-5.4-mini"
    route["endpoints"]["available"][0].update(
        {"provider": "OpenAI", "model": "openai/gpt-5.4-mini"}
    )
    route["attempts"][0].update(
        {"provider": "OpenAI", "model": "openai/gpt-5.4-mini"}
    )
    return route


def _result() -> dict:
    return {
        "sample_id": "sample",
        "realized_reasoning_tokens": 10,
        "provider_metadata": {
            "provider": "openrouter_zai_first_party",
            "main_response_model": "z-ai/glm-5.3-flash",
            "followup_response_model": "z-ai/glm-5.3-flash",
            "router_metadata": {"main": _route(), "followup": _route()},
            "judge_router_metadata": {
                "binary": _judge_route(),
                "confidence": _judge_route(),
            },
            "reasoning_content": {"main_present": True, "followup_present": True},
            "request_contract": {
                "reasoning_effort": "max",
                "include_reasoning": True,
                "provider_routing": {
                    "order": ["z-ai"],
                    "only": ["z-ai"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            },
        },
    }


def _completion(content: str, *, model: str = "z-ai/glm-5.3-flash") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": f"chatcmpl-{content}",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": content,
                        "refusal": None,
                        "role": "assistant",
                        "annotations": [],
                    },
                }
            ],
            "created": 1,
            "model": model,
            "object": "chat.completion",
        }
    )


def test_v4_preregistration_validates() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipelines/glm53_user_eval/run.py"),
            "validate-prereg",
            "--prereg",
            str(PREREG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_v4_schedule_changes_only_route_fields(tmp_path: Path) -> None:
    source = ROOT / "artifacts/glm53_user_eval/behavior_local/g3_schedule_v3"
    if not source.exists():
        return
    output = tmp_path / "api-schedule"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipelines/glm53_user_eval/run.py"),
            "build-api-v4-schedule",
            "--prereg",
            str(PREREG),
            "--source-schedule-root",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 600
    assert manifest["prompts_sha256"] == manifest["source_prompts_sha256"]
    assert manifest["allowed_source_row_changes"] == ["phase", "provider", "model_id"]


def test_openrouter_route_validation_accepts_only_zai() -> None:
    report = validate_openrouter_zai_results(
        [_result()], expected_model="z-ai/glm-5.3-flash"
    )
    assert report["passed"] is True


def test_openrouter_route_validation_rejects_fallback() -> None:
    result = _result()
    result["provider_metadata"]["router_metadata"]["main"]["attempts"].append(
        {"provider": "Novita", "model": "z-ai/glm-5.3-flash", "status": 200}
    )
    report = validate_openrouter_zai_results(
        [result], expected_model="z-ai/glm-5.3-flash"
    )
    assert report["passed"] is False
    assert report["failures"][0]["failures"] == ["main_provider_fallback"]


def test_reparse_api_results_preserves_original(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    row = {
        "sample_id": "sample",
        "subject_response_main": "No.",
        "binary_answer": None,
        "confidence_p": 90.0,
        "parse_valid": False,
        "provider_metadata": {},
    }
    (run_root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipelines/glm53_user_eval/run.py"),
            "reparse-api-results",
            "--run-root",
            str(run_root),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    original = json.loads((run_root / "results_evaluator_v1.jsonl").read_text())
    corrected = json.loads((run_root / "results.jsonl").read_text())
    assert original["binary_answer"] is None
    assert corrected["binary_answer"] == "no"
    assert corrected["parse_valid"] is True


def test_completion_checkpoint_rejects_changed_inputs(tmp_path: Path) -> None:
    path = tmp_path / "turn.json"
    schedule = {"sample_id": "sample"}
    prompt = {"main_prompt": "question"}
    _save_completion_checkpoint(
        path,
        schedule=schedule,
        prompt=prompt,
        response=_completion("No."),
    )
    loaded = _load_completion_checkpoint(path, schedule=schedule, prompt=prompt)
    assert loaded is not None
    assert loaded.choices[0].message.content == "No."
    with pytest.raises(ValueError, match="checkpoint inputs do not match"):
        _load_completion_checkpoint(
            path,
            schedule={"sample_id": "different"},
            prompt=prompt,
        )


def test_completed_ids_are_scoped_to_bounded_schedule(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "inside"}),
                json.dumps({"sample_id": "outside"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _completed_ids_for_schedule(results, [{"sample_id": "inside"}]) == {
        "inside"
    }


def test_completed_ids_fail_closed_on_duplicates(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"sample_id": "duplicate"}) + "\n" + json.dumps({"sample_id": "duplicate"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="results contain duplicate"):
        _completed_ids_for_schedule(results, [{"sample_id": "duplicate"}])


def test_interrupted_sample_reuses_all_four_turn_checkpoints(tmp_path: Path) -> None:
    schedule = {
        "sample_id": "sample",
        "scenario_id": "scenario",
        "persona_key": "persona",
        "condition": "famous_coherent",
        "reasoning_effort": "max",
        "model_id": "z-ai/glm-5.3-flash",
        "generation_seed": 42,
        "provider": "openrouter_zai_first_party",
    }
    prompt = {
        "system_prompt": "system",
        "main_prompt": "question",
        "followup_prompt": "confidence?",
    }
    checkpoints = {
        "subject_main.json": _completion("No."),
        "subject_followup.json": _completion("90%."),
        "binary_judge.json": _completion(
            "ANSWER = NO", model="openai/gpt-5.4-mini"
        ),
        "confidence_judge.json": _completion(
            "ANSWER = 90%", model="openai/gpt-5.4-mini"
        ),
    }
    for filename, response in checkpoints.items():
        _save_completion_checkpoint(
            tmp_path / filename,
            schedule=schedule,
            prompt=prompt,
            response=response,
        )
    row, _raw = asyncio.run(
        run_one_behavior_sample(
            schedule,
            prompt,
            subject_client=None,  # type: ignore[arg-type]
            judge_client=None,  # type: ignore[arg-type]
            judge_model="openai/gpt-5.4-mini",
            judge_label="gpt-5.4-mini-2026-03-17",
            temperature=1.0,
            top_p=0.95,
            clear_thinking=True,
            max_tokens=8000,
            provider_routing=None,
            send_seed=False,
            send_thinking=False,
            judge_provider_routing=None,
            judge_max_tokens=128,
            checkpoint_root=tmp_path,
            run_id="run",
        )
    )
    assert row.parse_valid is True
    assert row.confidence_p == 90.0
    assert row.provider_metadata["resumed_from_checkpoints"] == [
        "subject_main",
        "subject_followup",
        "binary_judge",
        "confidence_judge",
    ]


def test_behavior_writer_lock_allows_only_one_process_handle(tmp_path: Path) -> None:
    first = _acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another behavior writer"):
            _acquire_run_lock(tmp_path)
    finally:
        _release_run_lock(first)
    second = _acquire_run_lock(tmp_path)
    _release_run_lock(second)


def test_reconcile_ledger_uses_exact_raw_response_match(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    call_root = run_root / "calls" / "sample"
    call_root.mkdir(parents=True)
    first = {"sample_id": "sample", "confidence_p": 10.0}
    second = {"sample_id": "sample", "confidence_p": 90.0}
    (run_root / "results.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    (call_root / "raw_response.json").write_text(
        json.dumps({"result": second}) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "pipelines/glm53_user_eval/scripts/reconcile_behavior_results.py"
            ),
            "--run-root",
            str(run_root),
            "--incident-id",
            "test-incident",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["original_row_count"] == 2
    assert report["canonical_row_count"] == 1
    canonical = json.loads((run_root / "results.jsonl").read_text(encoding="utf-8"))
    original = (
        run_root / "incidents/test-incident/results_original.jsonl"
    ).read_text(encoding="utf-8")
    assert canonical == second
    assert original == json.dumps(first) + "\n" + json.dumps(second) + "\n"
