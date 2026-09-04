"""Resumable calls to a self-hosted vLLM server using the official FP8 weights."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .api import _completion_with_transport_retries, _message_text, _usage_payload
from .schemas import SelfHostedSubjectResult


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_self_hosted_schedule(
    *,
    schedule_rows: list[dict[str, Any]],
    prompt_rows: dict[str, dict[str, Any]],
    base_url: str,
    model_id: str,
    model_revision: str,
    image_digest: str,
    serving_config: dict[str, Any],
    generation_config: dict[str, Any],
    output_root: Path,
    run_id: str,
    concurrency: int,
    max_samples: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    calls_root = output_root / "calls"
    calls_root.mkdir(exist_ok=True)
    completed = {
        path.parent.name
        for path in calls_root.glob("*/subject.json")
        if path.is_file()
    }
    pending = [row for row in schedule_rows if row["sample_id"] not in completed]
    if max_samples > 0:
        pending = pending[:max_samples]
    serving_runtime_hash = _sha256_json(serving_config)
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url.rstrip("/") + "/v1")
    semaphore = asyncio.Semaphore(concurrency)
    counts = {
        "scheduled": len(schedule_rows),
        "completed_before": len(completed),
        "new_completed": 0,
        "failed": 0,
    }
    count_lock = asyncio.Lock()
    started = time.perf_counter()

    async def worker(schedule: dict[str, Any]) -> None:
        sample_id = str(schedule["sample_id"])
        prompt = prompt_rows[sample_id]
        if prompt["prompt_hash"] != schedule["prompt_hash"]:
            raise ValueError(f"prompt hash mismatch for {sample_id}")
        common = {
            "model": model_id,
            "temperature": float(generation_config["temperature"]),
            "top_p": float(generation_config["top_p"]),
            "max_tokens": int(generation_config["max_new_tokens"]),
            "seed": int(schedule["generation_seed"]),
            "extra_body": {
                "chat_template_kwargs": {
                    "reasoning_effort": str(generation_config["reasoning_effort"]),
                    "clear_thinking": bool(generation_config["clear_thinking"]),
                }
            },
        }
        main_messages = [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["main_prompt"]},
        ]
        async with semaphore:
            try:
                main_started = time.perf_counter()
                main_response = await _completion_with_transport_retries(
                    client, messages=main_messages, **common
                )
                main_latency = time.perf_counter() - main_started
                main_text = _message_text(main_response)
                followup_messages = [
                    *main_messages,
                    {"role": "assistant", "content": main_text},
                    {"role": "user", "content": prompt["followup_prompt"]},
                ]
                followup_started = time.perf_counter()
                followup_response = await _completion_with_transport_retries(
                    client, messages=followup_messages, **common
                )
                followup_latency = time.perf_counter() - followup_started
                main_raw = main_response.model_dump(mode="json")
                followup_raw = followup_response.model_dump(mode="json")
                result = SelfHostedSubjectResult(
                    run_id=run_id,
                    sample_id=sample_id,
                    scenario_id=str(schedule["scenario_id"]),
                    persona_key=str(schedule["persona_key"]),
                    condition=str(schedule["condition"]),
                    model_id=model_id,
                    model_revision=model_revision,
                    serving_runtime_hash=serving_runtime_hash,
                    serving_engine="vllm_openai",
                    image_digest=image_digest,
                    prompt_hash=str(schedule["prompt_hash"]),
                    main_text=main_text,
                    followup_text=_message_text(followup_response),
                    main_usage=_usage_payload(main_response),
                    followup_usage=_usage_payload(followup_response),
                    main_response_id=str(main_response.id),
                    followup_response_id=str(followup_response.id),
                    main_finish_reason=main_response.choices[0].finish_reason,
                    followup_finish_reason=followup_response.choices[0].finish_reason,
                    main_response_sha256=_sha256_json(main_raw),
                    followup_response_sha256=_sha256_json(followup_raw),
                    generation_seed=int(schedule["generation_seed"]),
                    generation_config=generation_config,
                    latency_seconds={"main": main_latency, "followup": followup_latency},
                    completed_at=_now_iso(),
                )
                _atomic_json(
                    calls_root / sample_id / "subject.json",
                    {
                        "schedule": schedule,
                        "prompt": prompt,
                        "subject": result.model_dump(mode="json"),
                        "responses": {"main": main_raw, "followup": followup_raw},
                    },
                )
                async with count_lock:
                    counts["new_completed"] += 1
            except Exception as exc:
                _atomic_json(
                    calls_root / sample_id / "subject.error.json",
                    {
                        "sample_id": sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "created_at": _now_iso(),
                    },
                )
                async with count_lock:
                    counts["failed"] += 1

    await asyncio.gather(*(worker(row) for row in pending))
    await client.close()
    counts.update(
        {
            "remaining": len(schedule_rows)
            - len(completed)
            - counts["new_completed"],
            "elapsed_seconds": time.perf_counter() - started,
            "concurrency": concurrency,
            "serving_runtime_hash": serving_runtime_hash,
            "completed_at": _now_iso(),
        }
    )
    _atomic_json(output_root / "generation_summary.json", counts)
    return counts
