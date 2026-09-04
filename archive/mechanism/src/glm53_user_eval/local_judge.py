"""Judge immutable local subject generations without keeping the GPU Pod alive."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .api import _completion_with_transport_retries, _message_text, _usage_payload
from .behavior import (
    BINARY_JUDGE_PROMPT,
    CONFIDENCE_JUDGE_PROMPT,
    parse_binary_judge,
    parse_confidence_judge,
)
from .schemas import BehaviorResultRow, LocalSubjectResult, SelfHostedSubjectResult


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def judge_local_subject_schedule(
    *,
    schedule_rows: list[dict[str, Any]],
    subject_root: Path,
    output_root: Path,
    run_id: str,
    judge_model: str,
    judge_api_key_env: str,
    concurrency: int,
) -> dict[str, Any]:
    api_key = os.environ.get(judge_api_key_env)
    if not api_key:
        raise RuntimeError(f"missing required judge credential: {judge_api_key_env}")
    output_root.mkdir(parents=True, exist_ok=True)
    calls_root = output_root / "calls"
    calls_root.mkdir(exist_ok=True)
    results_path = output_root / "results.jsonl"
    completed: set[str] = set()
    if results_path.exists():
        completed = {
            json.loads(line)["sample_id"]
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [row for row in schedule_rows if row["sample_id"] not in completed]
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counts = {
        "scheduled": len(schedule_rows),
        "completed_before": len(completed),
        "new_completed": 0,
        "missing_subject": 0,
        "failed": 0,
    }

    async def worker(schedule: dict[str, Any]) -> None:
        sample_id = str(schedule["sample_id"])
        subject_path = subject_root / "calls" / sample_id / "subject.json"
        if not subject_path.exists():
            counts["missing_subject"] += 1
            return
        async with semaphore:
            try:
                payload = json.loads(subject_path.read_text(encoding="utf-8"))
                subject_payload = payload["subject"]
                if (
                    subject_payload.get("schema_version")
                    == "glm53_self_hosted_subject_result_v1"
                ):
                    subject = SelfHostedSubjectResult.model_validate(subject_payload)
                    runtime_hash = subject.serving_runtime_hash
                    subject_usage = {
                        "main": subject.main_usage,
                        "followup": subject.followup_usage,
                    }
                    provider = "self_hosted_vllm_official_fp8"
                else:
                    subject = LocalSubjectResult.model_validate(subject_payload)
                    runtime_hash = subject.runtime_hash
                    subject_usage = {
                        "main_prompt_tokens": subject.main_prompt_tokens,
                        "main_generated_tokens": subject.main_generated_tokens,
                        "followup_prompt_tokens": subject.followup_prompt_tokens,
                        "followup_generated_tokens": subject.followup_generated_tokens,
                    }
                    provider = "local_huggingface_official_fp8"
                binary_response = await _completion_with_transport_retries(
                    client,
                    model=judge_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": BINARY_JUDGE_PROMPT},
                        {"role": "user", "content": subject.main_text},
                    ],
                )
                confidence_response = await _completion_with_transport_retries(
                    client,
                    model=judge_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": CONFIDENCE_JUDGE_PROMPT},
                        {"role": "user", "content": subject.followup_text},
                    ],
                )
                binary_text = _message_text(binary_response)
                confidence_text = _message_text(confidence_response)
                binary = parse_binary_judge(binary_text)
                confidence = parse_confidence_judge(confidence_text)
                row = BehaviorResultRow(
                    run_id=run_id,
                    sample_id=sample_id,
                    scenario_id=subject.scenario_id,
                    persona_key=subject.persona_key,
                    condition=subject.condition,
                    binary_answer=binary,
                    confidence_p=confidence.value,
                    refused_confidence=confidence.refused,
                    judge_model=judge_model,
                    judge_text=json.dumps(
                        {"binary": binary_text, "confidence": confidence_text},
                        sort_keys=True,
                    ),
                    subject_response_main=subject.main_text,
                    subject_response_followup=subject.followup_text,
                    realized_reasoning_tokens=None,
                    parse_valid=binary is not None and confidence.valid,
                    provider_metadata={
                        "provider": provider,
                        "requested_model": subject.model_id,
                        "model_revision": subject.model_revision,
                        "runtime_hash": runtime_hash,
                        "main_response_model": subject.model_id,
                        "followup_response_model": subject.model_id,
                        "subject_usage": subject_usage,
                        "reasoning_content": {
                            "main_present": None,
                            "main_characters": None,
                            "followup_present": None,
                            "followup_characters": None,
                        },
                        "judge_usage": {
                            "binary": _usage_payload(binary_response),
                            "confidence": _usage_payload(confidence_response),
                        },
                    },
                )
                raw = {
                    "schedule": schedule,
                    "subject_sha256": hashlib.sha256(subject_path.read_bytes()).hexdigest(),
                    "result": row.model_dump(mode="json"),
                    "judge_responses": {
                        "binary": binary_response.model_dump(mode="json"),
                        "confidence": confidence_response.model_dump(mode="json"),
                    },
                }
                _atomic_json(calls_root / sample_id / "judge.json", raw)
                async with write_lock:
                    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(row.model_dump_json() + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    counts["new_completed"] += 1
            except Exception as exc:
                _atomic_json(
                    calls_root / sample_id / "judge.error.json",
                    {
                        "sample_id": sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                counts["failed"] += 1

    await asyncio.gather(*(worker(row) for row in pending))
    await client.close()
    counts["remaining"] = (
        len(schedule_rows) - len(completed) - counts["new_completed"]
    )
    _atomic_json(output_root / "judge_summary.json", counts)
    return counts
