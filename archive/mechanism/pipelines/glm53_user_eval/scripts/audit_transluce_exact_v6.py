"""Score-blind contract audit for strict Transluce Inspect shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


SUBJECT = "openrouter/z-ai/glm-5.3-flash"
JUDGE = "openrouter/openai/gpt-5.4-mini"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def newest_success(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("*.eval"), key=lambda path: path.stat().st_mtime)
    for log in reversed(logs):
        if str(read_eval_log(log, header_only=True).status) == "success":
            return log
    return None


def audit_log(path: Path, expected_rows: int) -> dict[str, Any]:
    log = read_eval_log(path)
    failures: list[str] = []
    samples = log.samples or []
    if len(samples) != expected_rows:
        failures.append(f"row_count:{len(samples)}!={expected_rows}")
    task_args = log.eval.task_args or {}
    expected_args = {
        "reasoning_effort": "high",
        "dataset": "dailydilemmas.json",
        "warmup": False,
        "mode": "plain",
        "confidence": True,
        "confidence_style": "confidence",
        "seed": "all",
        "assistant_name": "",
    }
    for key, expected in expected_args.items():
        if task_args.get(key) != expected:
            failures.append(f"task_arg_{key}")
    subject_events = 0
    judge_events = 0
    retry_error_events = 0
    expected_judge_events = 0
    subject_reasoning_rows = 0
    providers: set[str] = set()
    api_cost_usd = 0.0
    for sample in samples:
        if sample.error is not None:
            failures.append(f"sample_error:{sample.id}")
        events = [event for event in (sample.events or []) if getattr(event, "event", "") == "model"]
        per_subject = 0
        per_judge = 0
        generations = (sample.metadata or {}).get("generations") or {}
        expected_sample_judges = sum(
            bool(str(generations.get(turn) or "").strip())
            for turn in ("main", "followup")
        )
        expected_judge_events += expected_sample_judges
        for event in events:
            payload = event.model_dump(mode="json")
            model = payload.get("model")
            config = payload.get("config") or {}
            call = payload.get("call") or {}
            request = call.get("request") or {}
            response = call.get("response") or {}
            api_cost_usd += float((response.get("usage") or {}).get("cost") or 0.0)
            if call.get("error"):
                retry_error_events += 1
                if response.get("model") or response.get("provider"):
                    failures.append(f"errored_attempt_has_response:{sample.id}")
                if model == SUBJECT and request.get("model") != "z-ai/glm-5.3-flash":
                    failures.append(f"errored_subject_request_model:{sample.id}")
                elif model == JUDGE and request.get("model") != "openai/gpt-5.4-mini":
                    failures.append(f"errored_judge_request_model:{sample.id}")
                elif model not in {SUBJECT, JUDGE}:
                    failures.append(f"errored_unexpected_model:{sample.id}:{model}")
                continue
            if model == SUBJECT:
                subject_events += 1
                per_subject += 1
                provider = str(response.get("provider") or "")
                providers.add(provider)
                route = (request.get("extra_body") or {}).get("provider") or {}
                reasoning = (request.get("extra_body") or {}).get("reasoning") or {}
                if request.get("model") != "z-ai/glm-5.3-flash":
                    failures.append(f"subject_model:{sample.id}")
                if response.get("model") != "z-ai/glm-5.3-flash":
                    failures.append(f"subject_response_model:{sample.id}")
                if provider != "Novita":
                    failures.append(f"subject_provider:{sample.id}:{provider}")
                if route != {
                    "require_parameters": True,
                    "order": ["Novita"],
                    "allow_fallbacks": False,
                }:
                    failures.append(f"subject_route:{sample.id}")
                if reasoning != {"effort": "high", "enabled": True}:
                    failures.append(f"subject_reasoning_request:{sample.id}")
                if config.get("max_tokens") != 8000:
                    failures.append(f"subject_max_tokens:{sample.id}")
                if any(config.get(key) is not None for key in ("temperature", "top_p", "seed")):
                    failures.append(f"subject_sampling_defaults:{sample.id}")
                usage = response.get("usage") or {}
                if ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) > 0:
                    subject_reasoning_rows += 1
            elif model == JUDGE:
                judge_events += 1
                per_judge += 1
                if config.get("reasoning_effort") != "low" or config.get("max_tokens") != 2000:
                    failures.append(f"judge_contract:{sample.id}")
            else:
                failures.append(f"unexpected_model:{sample.id}:{model}")
        if per_subject != 2:
            failures.append(f"subject_turn_count:{sample.id}:{per_subject}")
        if per_judge != expected_sample_judges:
            failures.append(
                f"judge_turn_count:{sample.id}:{per_judge}!={expected_sample_judges}"
            )
    checks = {
        "status_success": str(log.status) == "success",
        "expected_rows": len(samples) == expected_rows,
        "all_subject_calls_novita": providers == {"Novita"},
        "two_subject_and_source_expected_judge_calls_per_row": (
            subject_events == 2 * len(samples) and judge_events == expected_judge_events
        ),
        "reasoning_observed": subject_reasoning_rows > 0,
        "no_contract_failures": not failures,
    }
    return {
        "eval_log": str(path),
        "row_count": len(samples),
        "subject_event_count": subject_events,
        "judge_event_count": judge_events,
        "subject_reasoning_event_count": subject_reasoning_rows,
        "retry_error_event_count": retry_error_events,
        "providers": sorted(providers),
        "api_cost_usd": api_cost_usd,
        "checks": checks,
        "failures": failures[:200],
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "schedule_manifest.json").read_text(encoding="utf-8"))
    expected_by_id = {row["shard_id"]: int(row["expected_rows"]) for row in manifest["shards"]}
    reports = []
    for log_dir in sorted((args.run_root / "eval_logs").iterdir()):
        if not log_dir.is_dir():
            continue
        if log_dir.name not in expected_by_id:
            continue
        success = newest_success(log_dir)
        if success is None:
            continue
        reports.append(audit_log(success, expected_by_id[log_dir.name]))
    all_expected_present = len(reports) == len(expected_by_id)
    payload = {
        "schema_version": "glm53_transluce_exact_technical_audit_v1",
        "audited_shards": len(reports),
        "audited_rows": sum(report["row_count"] for report in reports),
        "api_cost_usd": sum(report["api_cost_usd"] for report in reports),
        "retry_error_event_count": sum(report["retry_error_event_count"] for report in reports),
        "expected_shards": len(expected_by_id),
        "require_all": args.require_all,
        "all_expected_present": all_expected_present,
        "passed": (
            bool(reports)
            and all(report["passed"] for report in reports)
            and (all_expected_present or not args.require_all)
        ),
        "reports": reports,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "audited_shards": payload["audited_shards"],
                "audited_rows": payload["audited_rows"],
                "passed": payload["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
