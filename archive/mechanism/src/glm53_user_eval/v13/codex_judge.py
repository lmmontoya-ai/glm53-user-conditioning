"""Resumable local-Codex judging with tool-disabled, isolated sessions."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v12.fact_validation import (
    PRIMARY_SYSTEM,
    atomic_json,
    canonical_json,
    parse_fact_json,
    sha256_file,
    sha256_text,
)

CLI_VERSION = "codex-cli 0.151.0"
AUTH_STATUS = "Logged in using ChatGPT"
LOCAL_WRAPPER = (
    "This is a self-contained semantic judgment. Do not use tools, inspect files, "
    "search, or obtain outside context. Apply only the instructions and scenario "
    "below. Return only the requested JSON object.\n\n"
)
PROMPT_TEMPLATE = LOCAL_WRAPPER + PRIMARY_SYSTEM + "\nSCENARIO\n{scenario_text}"

JUDGES: dict[str, dict[str, str]] = {
    "luna_max": {"model": "gpt-5.6-luna", "reasoning_effort": "max"},
    "terra_high": {"model": "gpt-5.6-terra", "reasoning_effort": "high"},
}

DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "fast_mode",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "shell_tool",
    "skill_search",
    "tool_suggest",
    "view_image",
)

RETRYABLE_TEXT = re.compile(
    r"(?:429|rate.?limit|temporar|timeout|timed out|connection|failed to connect|"
    r"403 forbidden.*responses_websocket|overloaded|"
    r"service unavailable|internal server|usage limit.*reset)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JudgeSpec:
    judge_id: str
    model: str
    reasoning_effort: str


def judge_specs() -> list[JudgeSpec]:
    return [
        JudgeSpec(judge_id=judge_id, **settings)
        for judge_id, settings in JUDGES.items()
    ]


def prompt_for_scenario(
    scenario_text: str, *, template: str = PROMPT_TEMPLATE
) -> str:
    if not scenario_text.strip():
        raise ValueError("scenario is empty")
    if template.count("{scenario_text}") != 1:
        raise ValueError("prompt template needs exactly one scenario placeholder")
    return template.replace("{scenario_text}", scenario_text)


def prompt_template_sha256(template: str = PROMPT_TEMPLATE) -> str:
    return sha256_text(template)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def codex_executable() -> str:
    value = shutil.which("codex")
    if not value:
        raise FileNotFoundError("codex CLI is not on PATH")
    return value


def _run_text(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()}"
        )
    stdout = result.stdout.strip()
    return stdout if stdout else result.stderr.strip()


def cli_preflight() -> dict[str, Any]:
    executable = codex_executable()
    version = _run_text([executable, "--version"])
    auth = _run_text([executable, "login", "status"])
    if version != CLI_VERSION:
        raise ValueError(f"Codex CLI version differs: {version!r}")
    if auth != AUTH_STATUS:
        raise ValueError(f"Codex is not using ChatGPT subscription auth: {auth!r}")
    catalog = json.loads(_run_text([executable, "debug", "models"]))
    by_slug = {str(item["slug"]): item for item in catalog["models"]}
    selected: dict[str, Any] = {}
    for spec in judge_specs():
        record = by_slug.get(spec.model)
        if record is None:
            raise ValueError(f"model missing from live Codex catalog: {spec.model}")
        efforts = {
            str(item["effort"]) for item in record.get("supported_reasoning_levels", [])
        }
        if spec.reasoning_effort not in efforts:
            raise ValueError(
                f"{spec.model} lacks reasoning effort {spec.reasoning_effort}"
            )
        selected[spec.judge_id] = {
            "slug": record["slug"],
            "display_name": record.get("display_name"),
            "reasoning_effort": spec.reasoning_effort,
            "supported_reasoning_levels": sorted(efforts),
        }
    return {
        "cli_executable": executable,
        "cli_version": version,
        "auth_status": auth,
        "models": selected,
    }


def command_for(
    *,
    spec: JudgeSpec,
    schema_path: Path,
    output_path: Path,
    isolated_workspace: Path,
) -> list[str]:
    command = [
        codex_executable(),
        "exec",
        "--model",
        spec.model,
        "--config",
        f'model_reasoning_effort="{spec.reasoning_effort}"',
        "--config",
        'model_verbosity="low"',
        "--config",
        'approval_policy="never"',
        "--config",
        'web_search="disabled"',
        "--sandbox",
        "read-only",
        "--cd",
        str(isolated_workspace),
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--json",
    ]
    for feature in DISABLED_FEATURES:
        command.extend(("--disable", feature))
    command.append("-")
    return command


def sanitized_environment() -> tuple[dict[str, str], list[str]]:
    environment = dict(os.environ)
    removed: list[str] = []
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
    ):
        if key in environment:
            removed.append(key)
            environment.pop(key, None)
    environment["NO_COLOR"] = "1"
    return environment, sorted(removed)


def _event_objects(stdout_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    invalid: list[str] = []
    for line in stdout_text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid.append(line)
            continue
        if not isinstance(value, dict):
            invalid.append(line)
            continue
        events.append(value)
    return events, invalid


def _tool_event_types(events: Sequence[Mapping[str, Any]]) -> list[str]:
    allowed_item_types = {"agent_message", "reasoning"}
    found: list[str] = []
    for event in events:
        event_type = str(event.get("type") or "")
        item = event.get("item")
        if isinstance(item, Mapping):
            item_type = str(item.get("type") or "")
            if item_type and item_type not in allowed_item_types:
                found.append(f"{event_type}:{item_type}")
        if any(
            token in event_type.casefold()
            for token in ("tool_call", "command_execution", "web_search", "file_change")
        ):
            found.append(event_type)
    return sorted(set(found))


def request_sha256(
    *,
    spec: JudgeSpec,
    prompt: str,
    schema_path: Path,
    prompt_template: str = PROMPT_TEMPLATE,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "judge_id": spec.judge_id,
                "model": spec.model,
                "reasoning_effort": spec.reasoning_effort,
                "prompt": prompt,
                "prompt_template_sha256": prompt_template_sha256(prompt_template),
                "schema_sha256": sha256_file(schema_path),
                "cli_version": CLI_VERSION,
                "auth_status": AUTH_STATUS,
                "disabled_features": DISABLED_FEATURES,
                "sandbox": "read-only",
                "ephemeral": True,
            }
        )
    )


def _final_path(output_root: Path, spec: JudgeSpec, sample_id: str) -> Path:
    return output_root / spec.judge_id / "rows" / f"{sample_id}.json"


def load_completed_rows(output_root: Path, spec: JudgeSpec) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / spec.judge_id / "rows").glob("*.json"))
    ]


def _attempt_artifact_numbers(attempt_dir: Path) -> set[int]:
    """Return attempt suffixes already used by any durable or partial artifact."""
    numbers: set[int] = set()
    pattern = re.compile(
        r"^(?:attempt|last_message|events|stderr)_(\d{2})"
        r"(?:\.json|\.jsonl|\.txt)(?:\.partial)?$"
    )
    if not attempt_dir.is_dir():
        return numbers
    for path in attempt_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop a child if needed and always reap it without masking cancellation."""
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await process.wait()
    except ProcessLookupError:
        pass


async def _gather_without_peer_cancellation(
    tasks: Sequence[asyncio.Task[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Let every independent judgment settle before reporting task failures."""
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [item for item in outcomes if isinstance(item, BaseException)]
    if failures:
        summaries = "; ".join(
            f"{type(item).__name__}: {item}" for item in failures[:10]
        )
        raise RuntimeError(
            f"{len(failures)} Codex judgment task(s) failed after peers settled: "
            f"{summaries}"
        )
    return [item for item in outcomes if isinstance(item, dict)]


async def _judge_one(
    *,
    row: Mapping[str, Any],
    spec: JudgeSpec,
    output_root: Path,
    schema_path: Path,
    isolated_workspace: Path,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
    timeout_seconds: float,
    prompt_template: str,
    response_parser: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    prompt = prompt_for_scenario(
        str(row["scenario_text"]), template=prompt_template
    )
    expected_request_hash = request_sha256(
        spec=spec,
        prompt=prompt,
        schema_path=schema_path,
        prompt_template=prompt_template,
    )
    final_path = _final_path(output_root, spec, sample_id)
    if final_path.is_file():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            existing.get("sample_id") != sample_id
            or existing.get("judge_id") != spec.judge_id
            or existing.get("request_sha256") != expected_request_hash
        ):
            raise ValueError(f"completed-row checkpoint mismatch: {final_path}")
        response_parser(canonical_json(existing["parsed"]))
        return existing

    attempt_dir = output_root / spec.judge_id / "attempts" / sample_id
    existing_attempts = sorted(attempt_dir.glob("attempt_*.json"))
    for attempt_path in existing_attempts:
        record = json.loads(attempt_path.read_text(encoding="utf-8"))
        if record.get("request_sha256") != expected_request_hash:
            raise ValueError(f"attempt checkpoint mismatch: {attempt_path}")
        if record.get("passed") is True:
            final = dict(record)
            final["schema_version"] = "glm53_v13_codex_judgment_row_v1"
            atomic_json(final_path, final)
            return final

    used_attempt_numbers = _attempt_artifact_numbers(attempt_dir)
    recorded_attempt_numbers = {
        int(path.stem.rsplit("_", maxsplit=1)[-1]) for path in existing_attempts
    }
    orphan_attempt_numbers = sorted(used_attempt_numbers - recorded_attempt_numbers)
    remaining_attempts = max_attempts - len(existing_attempts)
    if remaining_attempts < 1:
        raise RuntimeError(
            f"Codex judge exhausted attempts for {spec.judge_id}/{sample_id}"
        )
    next_attempt_number = max(used_attempt_numbers, default=0) + 1
    environment, removed_environment_keys = sanitized_environment()
    async with semaphore:
        for attempt_number in range(
            next_attempt_number, next_attempt_number + remaining_attempts
        ):
            attempt_dir.mkdir(parents=True, exist_ok=True)
            output_path = attempt_dir / f"last_message_{attempt_number:02d}.json"
            events_path = attempt_dir / f"events_{attempt_number:02d}.jsonl"
            stderr_path = attempt_dir / f"stderr_{attempt_number:02d}.txt"
            events_partial = events_path.with_suffix(events_path.suffix + ".partial")
            stderr_partial = stderr_path.with_suffix(stderr_path.suffix + ".partial")
            command = command_for(
                spec=spec,
                schema_path=schema_path,
                output_path=output_path,
                isolated_workspace=isolated_workspace,
            )
            started_at = time.time()
            timed_out = False
            process: asyncio.subprocess.Process | None = None
            events_partial.parent.mkdir(parents=True, exist_ok=True)
            with (
                events_partial.open("wb") as events_handle,
                stderr_partial.open("wb") as stderr_handle,
            ):
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=events_handle,
                    stderr=stderr_handle,
                    env=environment,
                )
                try:
                    await asyncio.wait_for(
                        process.communicate(prompt.encode("utf-8")),
                        timeout=timeout_seconds,
                    )
                except TimeoutError:
                    timed_out = True
                    await _stop_process(process)
                except asyncio.CancelledError:
                    await _stop_process(process)
                    raise
                finally:
                    events_handle.flush()
                    os.fsync(events_handle.fileno())
                    stderr_handle.flush()
                    os.fsync(stderr_handle.fileno())
            os.replace(events_partial, events_path)
            os.replace(stderr_partial, stderr_path)
            duration_seconds = time.time() - started_at
            stdout_text = events_path.read_text(encoding="utf-8", errors="replace")
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
            events, invalid_event_lines = _event_objects(stdout_text)
            tool_events = _tool_event_types(events)
            response_text = (
                output_path.read_text(encoding="utf-8")
                if output_path.is_file()
                else ""
            )
            parse_error: str | None = None
            parsed: dict[str, Any] | None = None
            try:
                parsed = response_parser(response_text)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
            checks = {
                "exit_zero": process.returncode == 0,
                "not_timed_out": not timed_out,
                "stdout_is_jsonl": not invalid_event_lines,
                "no_tool_events": not tool_events,
                "structured_output_parsed": parsed is not None,
            }
            passed = all(checks.values())
            if process is None:
                raise AssertionError("subprocess was not created")
            record: dict[str, Any] = {
                "schema_version": "glm53_v13_codex_judgment_attempt_v1",
                "sample_id": sample_id,
                "judge_id": spec.judge_id,
                "model": spec.model,
                "reasoning_effort": spec.reasoning_effort,
                "attempt_number": attempt_number,
                "request_sha256": expected_request_hash,
                "prompt_sha256": sha256_text(prompt),
                "prompt_template_sha256": prompt_template_sha256(prompt_template),
                "schema_sha256": sha256_file(schema_path),
                "cli_version": CLI_VERSION,
                "auth_status": AUTH_STATUS,
                "command": command,
                "removed_environment_keys": removed_environment_keys,
                "transport_mode": "file_redirect_v2",
                "orphan_attempt_numbers_before_run": orphan_attempt_numbers,
                "exit_code": process.returncode,
                "timed_out": timed_out,
                "duration_seconds": duration_seconds,
                "checks": checks,
                "passed": passed,
                "parse_error": parse_error,
                "tool_events": tool_events,
                "invalid_event_lines": invalid_event_lines,
                "response_text": response_text,
                "events_path": events_path.relative_to(output_root).as_posix(),
                "events_sha256": sha256_file(events_path),
                "stderr_path": stderr_path.relative_to(output_root).as_posix(),
                "stderr_sha256": sha256_file(stderr_path),
            }
            if parsed is not None:
                record["parsed"] = parsed
            attempt_path = attempt_dir / f"attempt_{attempt_number:02d}.json"
            atomic_json(attempt_path, record)
            if passed:
                final = dict(record)
                final["schema_version"] = "glm53_v13_codex_judgment_row_v1"
                atomic_json(final_path, final)
                return final
            retryable = timed_out or bool(RETRYABLE_TEXT.search(stderr_text))
            if not retryable or attempt_number == max_attempts:
                raise RuntimeError(
                    f"Codex judge failed for {spec.judge_id}/{sample_id}: {checks}; "
                    f"stderr={stderr_text[-500:]!r}"
                )
            await asyncio.sleep(min(30.0, 2.0 ** attempt_number))
    raise RuntimeError(f"Codex judge exhausted attempts for {spec.judge_id}/{sample_id}")


async def run_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    schema_path: Path,
    concurrency_per_judge: int,
    max_attempts: int,
    timeout_seconds: float,
    max_new_per_judge: int | None = None,
    prompt_template: str = PROMPT_TEMPLATE,
    response_parser: Callable[[str], dict[str, Any]] = parse_fact_json,
    concurrency_by_judge: Mapping[str, int] | None = None,
    judge_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if concurrency_per_judge < 1:
        raise ValueError("concurrency must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    cli_preflight()
    isolated_workspace = output_root / "isolated_empty_workspace"
    isolated_workspace.mkdir(parents=True, exist_ok=True)
    if any(isolated_workspace.iterdir()):
        raise ValueError("Codex judge workspace must remain empty")
    ordered_rows = sorted(rows, key=lambda item: str(item["sample_id"]))
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    selected_ids = set(judge_ids or JUDGES)
    if not selected_ids or not selected_ids.issubset(JUDGES):
        raise ValueError("selected judge IDs differ from the frozen cohort")
    for spec in (item for item in judge_specs() if item.judge_id in selected_ids):
        completed = {
            str(item["sample_id"]) for item in load_completed_rows(output_root, spec)
        }
        pending = [row for row in ordered_rows if str(row["sample_id"]) not in completed]
        if max_new_per_judge is not None:
            pending = pending[:max_new_per_judge]
        judge_concurrency = (
            int(concurrency_by_judge[spec.judge_id])
            if concurrency_by_judge is not None
            else concurrency_per_judge
        )
        if judge_concurrency < 1:
            raise ValueError("per-judge concurrency must be positive")
        semaphore = asyncio.Semaphore(judge_concurrency)
        tasks.extend(
            asyncio.create_task(
                _judge_one(
                    row=row,
                    spec=spec,
                    output_root=output_root,
                    schema_path=schema_path,
                    isolated_workspace=isolated_workspace,
                    semaphore=semaphore,
                    max_attempts=max_attempts,
                    timeout_seconds=timeout_seconds,
                    prompt_template=prompt_template,
                    response_parser=response_parser,
                )
            )
            for row in pending
        )
    if tasks:
        await _gather_without_peer_cancellation(tasks)
    results: list[dict[str, Any]] = []
    for spec in judge_specs():
        results.extend(load_completed_rows(output_root, spec))
    return sorted(results, key=lambda item: (item["judge_id"], item["sample_id"]))


__all__ = [
    "AUTH_STATUS",
    "CLI_VERSION",
    "DISABLED_FEATURES",
    "JUDGES",
    "JudgeSpec",
    "cli_preflight",
    "command_for",
    "judge_specs",
    "load_completed_rows",
    "prompt_for_scenario",
    "prompt_template_sha256",
    "request_sha256",
    "run_cohort",
    "sanitized_environment",
]
