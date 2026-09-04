"""Pinned-provider two-turn behavior runner with a separate extraction judge."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletion

from .behavior import (
    BINARY_JUDGE_PROMPT,
    CONFIDENCE_JUDGE_PROMPT,
    parse_direct_binary_response,
    parse_binary_judge,
    parse_confidence_judge,
)
from .schemas import BehaviorResultRow


TRANSPORT_ERRORS = (APIConnectionError, APITimeoutError)
OPENROUTER_ZAI_NAMES = {"z.ai", "z-ai", "zai"}


def _acquire_run_lock(output_root: Path) -> BinaryIO:
    """Acquire an OS-backed single-writer lock that is released on process exit."""
    path = output_root / ".behavior_api_writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"another behavior writer holds {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "acquired_at_utc": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _release_run_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _usage_payload(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return dict(usage)


def _message_text(response: Any) -> str:
    message = response.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return "" if content is None else str(content)


def _reasoning_content(response: Any) -> str | None:
    message = response.choices[0].message
    value = getattr(message, "reasoning_content", None)
    if value is None:
        value = getattr(message, "reasoning", None)
    if value is None:
        extra = getattr(message, "model_extra", None) or {}
        value = extra.get("reasoning_content") or extra.get("reasoning")
    return value if isinstance(value, str) else None


def _response_extra(response: Any) -> dict[str, Any]:
    extra = getattr(response, "model_extra", None) or {}
    return dict(extra) if isinstance(extra, dict) else {}


def _router_metadata(response: Any) -> dict[str, Any] | None:
    value = _response_extra(response).get("openrouter_metadata")
    return dict(value) if isinstance(value, dict) else None


def _normalized_provider(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_completion_checkpoint(
    path: Path,
    *,
    schedule: dict[str, Any],
    prompt: dict[str, Any],
) -> ChatCompletion | None:
    """Load a completed API turn only when it belongs to this exact sample."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schedule") != schedule or payload.get("prompt") != prompt:
        raise ValueError(f"checkpoint inputs do not match the scheduled sample: {path}")
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"checkpoint has no valid response object: {path}")
    return ChatCompletion.model_validate(response)


def _save_completion_checkpoint(
    path: Path,
    *,
    schedule: dict[str, Any],
    prompt: dict[str, Any],
    response: Any,
) -> None:
    _atomic_json(
        path,
        {
            "schedule": schedule,
            "prompt": prompt,
            "response": response.model_dump(mode="json"),
        },
    )


def _completed_ids_for_schedule(
    results_path: Path,
    schedule_rows: list[dict[str, Any]],
) -> set[str]:
    """Return completed IDs in this schedule and fail closed on duplicate rows."""
    schedule_ids = [str(row["sample_id"]) for row in schedule_rows]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise ValueError("schedule contains duplicate sample IDs")
    if not results_path.exists():
        return set()
    completed_ids = [
        str(json.loads(line)["sample_id"])
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("results contain duplicate sample IDs")
    return set(completed_ids).intersection(schedule_ids)


def validate_openrouter_zai_results(
    rows: list[dict[str, Any]], *, expected_model: str
) -> dict[str, Any]:
    """Fail closed unless every subject turn used only the first-party Z.AI endpoint."""
    if not rows:
        raise ValueError("route validation requires at least one completed row")
    failures: list[dict[str, Any]] = []
    reasoning_rows = 0
    for row in rows:
        metadata = row.get("provider_metadata") or {}
        router = metadata.get("router_metadata") or {}
        row_failures: list[str] = []
        if metadata.get("provider") != "openrouter_zai_first_party":
            row_failures.append("provider_label")
        if metadata.get("main_response_model") != expected_model:
            row_failures.append("main_model")
        if metadata.get("followup_response_model") != expected_model:
            row_failures.append("followup_model")
        for turn in ("main", "followup"):
            route = router.get(turn)
            if not isinstance(route, dict):
                row_failures.append(f"{turn}_router_metadata")
                continue
            if route.get("requested") != expected_model:
                row_failures.append(f"{turn}_requested_model")
            endpoints = (route.get("endpoints") or {}).get("available") or []
            selected = [item for item in endpoints if item.get("selected")]
            if len(selected) != 1 or _normalized_provider(selected[0].get("provider")) not in OPENROUTER_ZAI_NAMES:
                row_failures.append(f"{turn}_selected_provider")
            attempts = route.get("attempts") or []
            if any(
                _normalized_provider(item.get("provider")) not in OPENROUTER_ZAI_NAMES
                for item in attempts
            ):
                row_failures.append(f"{turn}_provider_fallback")
        judge_router = metadata.get("judge_router_metadata") or {}
        for turn in ("binary", "confidence"):
            route = judge_router.get(turn)
            if not isinstance(route, dict):
                row_failures.append(f"{turn}_judge_router_metadata")
                continue
            if route.get("requested") != "openai/gpt-5.4-mini":
                row_failures.append(f"{turn}_judge_requested_model")
            endpoints = (route.get("endpoints") or {}).get("available") or []
            selected = [item for item in endpoints if item.get("selected")]
            if len(selected) != 1 or _normalized_provider(selected[0].get("provider")) != "openai":
                row_failures.append(f"{turn}_judge_selected_provider")
            attempts = route.get("attempts") or []
            if any(_normalized_provider(item.get("provider")) != "openai" for item in attempts):
                row_failures.append(f"{turn}_judge_provider_fallback")
        reasoning = metadata.get("reasoning_content") or {}
        if row.get("realized_reasoning_tokens") is not None or reasoning.get("main_present") or reasoning.get("followup_present"):
            reasoning_rows += 1
        contract = metadata.get("request_contract") or {}
        routing = contract.get("provider_routing") or {}
        if routing.get("only") != ["z-ai"] or routing.get("allow_fallbacks") is not False:
            row_failures.append("request_provider_contract")
        if contract.get("reasoning_effort") != "max" or contract.get("include_reasoning") is not True:
            row_failures.append("request_reasoning_contract")
        if row_failures:
            failures.append({"sample_id": row.get("sample_id"), "failures": row_failures})
    checks = {
        "completed_rows_present": bool(rows),
        "all_rows_first_party_zai": not failures,
        "all_judges_first_party_openai": not any(
            any("judge_" in reason for reason in item["failures"]) for item in failures
        ),
        "reasoning_observed": reasoning_rows > 0,
    }
    return {
        "schema_version": "glm53_openrouter_route_validation_v1",
        "expected_model": expected_model,
        "row_count": len(rows),
        "reasoning_row_count": reasoning_rows,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": failures,
    }


async def _completion_with_transport_retries(
    client: AsyncOpenAI,
    *,
    attempts: int = 3,
    **kwargs: Any,
) -> Any:
    for attempt in range(attempts):
        try:
            return await client.chat.completions.create(**kwargs)
        except TRANSPORT_ERRORS:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(2**attempt)
        except APIStatusError as exc:
            if exc.status_code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
            await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable retry state")


async def run_one_behavior_sample(
    schedule: dict[str, Any],
    prompt: dict[str, Any],
    *,
    subject_client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_label: str,
    temperature: float,
    top_p: float,
    clear_thinking: bool,
    max_tokens: int,
    provider_routing: dict[str, Any] | None,
    send_seed: bool,
    send_thinking: bool,
    judge_provider_routing: dict[str, Any] | None,
    judge_max_tokens: int,
    checkpoint_root: Path,
    run_id: str,
) -> tuple[BehaviorResultRow, dict[str, Any]]:
    started = time.perf_counter()
    reused_checkpoints: list[str] = []
    subject_extra: dict[str, Any] = {
        "reasoning_effort": schedule["reasoning_effort"],
        "include_reasoning": True,
    }
    if send_thinking:
        subject_extra["thinking"] = {"type": "enabled", "clear_thinking": clear_thinking}
    if provider_routing is not None:
        subject_extra["provider"] = provider_routing
    subject_common: dict[str, Any] = {
        "model": schedule["model_id"],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "extra_body": subject_extra,
    }
    if send_seed:
        subject_common["seed"] = int(schedule["generation_seed"])
    main_messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["main_prompt"]},
    ]
    main_checkpoint = checkpoint_root / "subject_main.json"
    main_response = _load_completion_checkpoint(
        main_checkpoint, schedule=schedule, prompt=prompt
    )
    if main_response is None:
        main_response = await _completion_with_transport_retries(
            subject_client, messages=main_messages, **subject_common
        )
        _save_completion_checkpoint(
            main_checkpoint,
            schedule=schedule,
            prompt=prompt,
            response=main_response,
        )
    else:
        reused_checkpoints.append("subject_main")
    main_text = _message_text(main_response)
    followup_messages = [
        *main_messages,
        {"role": "assistant", "content": main_text},
        {"role": "user", "content": prompt["followup_prompt"]},
    ]
    followup_checkpoint = checkpoint_root / "subject_followup.json"
    followup_response = _load_completion_checkpoint(
        followup_checkpoint, schedule=schedule, prompt=prompt
    )
    if followup_response is None:
        followup_response = await _completion_with_transport_retries(
            subject_client, messages=followup_messages, **subject_common
        )
        _save_completion_checkpoint(
            followup_checkpoint,
            schedule=schedule,
            prompt=prompt,
            response=followup_response,
        )
    else:
        reused_checkpoints.append("subject_followup")
    followup_text = _message_text(followup_response)

    judge_extra = {"provider": judge_provider_routing} if judge_provider_routing else None
    binary_checkpoint = checkpoint_root / "binary_judge.json"
    binary_judge = _load_completion_checkpoint(
        binary_checkpoint, schedule=schedule, prompt=prompt
    )
    if binary_judge is None:
        binary_judge = await _completion_with_transport_retries(
            judge_client,
            model=judge_model,
            max_tokens=judge_max_tokens,
            extra_body=judge_extra,
            messages=[
                {"role": "system", "content": BINARY_JUDGE_PROMPT},
                {"role": "user", "content": main_text},
            ],
        )
        _save_completion_checkpoint(
            binary_checkpoint,
            schedule=schedule,
            prompt=prompt,
            response=binary_judge,
        )
    else:
        reused_checkpoints.append("binary_judge")
    confidence_checkpoint = checkpoint_root / "confidence_judge.json"
    confidence_judge = _load_completion_checkpoint(
        confidence_checkpoint, schedule=schedule, prompt=prompt
    )
    if confidence_judge is None:
        confidence_judge = await _completion_with_transport_retries(
            judge_client,
            model=judge_model,
            max_tokens=judge_max_tokens,
            extra_body=judge_extra,
            messages=[
                {"role": "system", "content": CONFIDENCE_JUDGE_PROMPT},
                {"role": "user", "content": followup_text},
            ],
        )
        _save_completion_checkpoint(
            confidence_checkpoint,
            schedule=schedule,
            prompt=prompt,
            response=confidence_judge,
        )
    else:
        reused_checkpoints.append("confidence_judge")
    binary_judge_text = _message_text(binary_judge)
    confidence_judge_text = _message_text(confidence_judge)
    direct_binary = parse_direct_binary_response(main_text)
    binary = direct_binary or parse_binary_judge(binary_judge_text)
    confidence = parse_confidence_judge(confidence_judge_text)
    subject_usage = {
        "main": _usage_payload(main_response),
        "followup": _usage_payload(followup_response),
    }
    reasoning_tokens = 0
    reasoning_observed = False
    for usage in subject_usage.values():
        details = usage.get("completion_tokens_details") or {}
        value = details.get("reasoning_tokens")
        if value is not None:
            reasoning_tokens += int(value)
            reasoning_observed = True
    provider_metadata = {
        "provider": schedule["provider"],
        "requested_model": schedule["model_id"],
        "main_response_model": getattr(main_response, "model", None),
        "followup_response_model": getattr(followup_response, "model", None),
        "main_response_id": getattr(main_response, "id", None),
        "followup_response_id": getattr(followup_response, "id", None),
        "router_metadata": {
            "main": _router_metadata(main_response),
            "followup": _router_metadata(followup_response),
        },
        "request_contract": {
            "reasoning_effort": schedule["reasoning_effort"],
            "include_reasoning": True,
            "clear_thinking_requested": clear_thinking if send_thinking else None,
            "seed_sent": send_seed,
            "max_tokens": max_tokens,
            "provider_routing": provider_routing,
        },
        "binary_extraction": {
            "source": "deterministic_direct" if direct_binary is not None else "llm_judge",
            "judge_value": parse_binary_judge(binary_judge_text),
            "judge_disagreed": (
                direct_binary is not None
                and parse_binary_judge(binary_judge_text) is not None
                and direct_binary != parse_binary_judge(binary_judge_text)
            ),
        },
        "subject_usage": subject_usage,
        "reasoning_content": {
            "main_present": _reasoning_content(main_response) is not None,
            "main_characters": len(_reasoning_content(main_response) or ""),
            "followup_present": _reasoning_content(followup_response) is not None,
            "followup_characters": len(_reasoning_content(followup_response) or ""),
        },
        "judge_usage": {
            "binary": _usage_payload(binary_judge),
            "confidence": _usage_payload(confidence_judge),
        },
        "judge_router_metadata": {
            "binary": _router_metadata(binary_judge),
            "confidence": _router_metadata(confidence_judge),
        },
        "resumed_from_checkpoints": reused_checkpoints,
        "latency_seconds": time.perf_counter() - started,
    }
    row = BehaviorResultRow(
        run_id=run_id,
        sample_id=schedule["sample_id"],
        scenario_id=schedule["scenario_id"],
        persona_key=schedule["persona_key"],
        condition=schedule["condition"],
        binary_answer=binary,
        confidence_p=confidence.value,
        refused_confidence=confidence.refused,
        judge_model=judge_label,
        judge_text=json.dumps(
            {"binary": binary_judge_text, "confidence": confidence_judge_text},
            sort_keys=True,
        ),
        subject_response_main=main_text,
        subject_response_followup=followup_text,
        realized_reasoning_tokens=reasoning_tokens if reasoning_observed else None,
        parse_valid=binary is not None and confidence.valid,
        provider_metadata=provider_metadata,
    )
    raw = {
        "schedule": schedule,
        "prompt": prompt,
        "result": row.model_dump(mode="json"),
        "subject_responses": {
            "main": main_response.model_dump(mode="json"),
            "followup": followup_response.model_dump(mode="json"),
        },
        "judge_responses": {
            "binary": binary_judge.model_dump(mode="json"),
            "confidence": confidence_judge.model_dump(mode="json"),
        },
    }
    return row, raw


async def run_behavior_schedule(
    schedule_rows: list[dict[str, Any]],
    prompt_rows: dict[str, dict[str, Any]],
    *,
    output_root: Path,
    run_id: str,
    subject_base_url: str,
    subject_api_key_env: str,
    judge_api_key_env: str,
    judge_base_url: str | None,
    judge_model: str,
    judge_label: str,
    temperature: float,
    top_p: float,
    clear_thinking: bool,
    max_tokens: int,
    provider_routing: dict[str, Any] | None,
    metadata_header: bool,
    send_seed: bool,
    send_thinking: bool,
    judge_provider_routing: dict[str, Any] | None,
    judge_metadata_header: bool,
    judge_max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    subject_key = os.environ.get(subject_api_key_env)
    judge_key = os.environ.get(judge_api_key_env)
    if not subject_key:
        raise RuntimeError(f"missing required subject credential: {subject_api_key_env}")
    if not judge_key:
        raise RuntimeError(f"missing required judge credential: {judge_api_key_env}")
    output_root.mkdir(parents=True, exist_ok=True)
    calls_root = output_root / "calls"
    calls_root.mkdir(exist_ok=True)
    results_path = output_root / "results.jsonl"
    completed = _completed_ids_for_schedule(results_path, schedule_rows)
    pending = [row for row in schedule_rows if row["sample_id"] not in completed]
    run_lock = _acquire_run_lock(output_root)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    default_headers = {"X-OpenRouter-Metadata": "enabled"} if metadata_header else None
    subject_client = AsyncOpenAI(
        api_key=subject_key,
        base_url=subject_base_url,
        default_headers=default_headers,
    )
    judge_headers = {"X-OpenRouter-Metadata": "enabled"} if judge_metadata_header else None
    judge_client = AsyncOpenAI(
        api_key=judge_key,
        base_url=judge_base_url,
        default_headers=judge_headers,
    )
    counts = {"completed_before": len(completed), "new_completed": 0, "failed": 0}

    async def worker(schedule: dict[str, Any]) -> None:
        async with semaphore:
            sample_id = schedule["sample_id"]
            call_dir = calls_root / sample_id
            call_dir.mkdir(parents=True, exist_ok=True)
            try:
                row, raw = await run_one_behavior_sample(
                    schedule,
                    prompt_rows[sample_id],
                    subject_client=subject_client,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    judge_label=judge_label,
                    temperature=temperature,
                    top_p=top_p,
                    clear_thinking=clear_thinking,
                    max_tokens=max_tokens,
                    provider_routing=provider_routing,
                    send_seed=send_seed,
                    send_thinking=send_thinking,
                    judge_provider_routing=judge_provider_routing,
                    judge_max_tokens=judge_max_tokens,
                    checkpoint_root=call_dir,
                    run_id=run_id,
                )
                _atomic_json(call_dir / "raw_response.json", raw)
                async with write_lock:
                    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(row.model_dump_json() + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    counts["new_completed"] += 1
            except Exception as exc:
                _atomic_json(
                    call_dir / "error.json",
                    {
                        "sample_id": sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                counts["failed"] += 1

    try:
        await asyncio.gather(*(worker(row) for row in pending))
    finally:
        await subject_client.close()
        await judge_client.close()
        _release_run_lock(run_lock)
    counts["scheduled"] = len(schedule_rows)
    counts["remaining"] = len(schedule_rows) - len(completed) - counts["new_completed"]
    return counts
