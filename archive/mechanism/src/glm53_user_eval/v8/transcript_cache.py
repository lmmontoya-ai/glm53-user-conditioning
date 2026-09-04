"""Reconstruct immutable v7 first-turn conversations for v8."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

from .artifacts import atomic_jsonl, sha256_file
from .proxy import codebook_id

PRIMARY_GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")


def _sha_text(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _successful_subject_request(sample: Any) -> dict[str, Any]:
    attachments = sample.model_dump(mode="json").get("attachments") or {}
    for event in sample.events or []:
        if getattr(event, "event", "") != "model" or event.model != "openrouter/z-ai/glm-5.3-flash":
            continue
        call = event.model_dump(mode="json").get("call") or {}
        if call.get("error"):
            continue
        request = call.get("request") or {}
        messages = []
        for message in request.get("messages") or []:
            row = dict(message)
            content = row.get("content")
            if isinstance(content, str) and content.startswith("attachment://"):
                row["content"] = attachments[content.removeprefix("attachment://")]
            messages.append(row)
        if len(messages) == 2:
            return {**request, "messages": messages}
    raise ValueError(f"no successful first subject request for {sample.id}")


def build_cache(
    *, raw_scores_path: Path, final_evidence_path: Path, task_ids: set[str], output_path: Path
) -> dict[str, Any]:
    evidence = json.loads(final_evidence_path.read_text(encoding="utf-8"))
    logs_by_shard: dict[str, Path] = {}
    allowed_hashes = {row["sha256"] for row in evidence["successful_eval_logs"]}
    for row in evidence["successful_eval_logs"]:
        path = Path(row["path"])
        shard = path.parent.name
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"source log hash mismatch: {path}")
        logs_by_shard[shard] = path
    wanted: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in raw_scores_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row["group"] in PRIMARY_GROUPS
            and row["persona"] != "anon"
            and row["stimulus"] in task_ids
        ):
            wanted[row["shard_id"]][f"{row['stimulus']}-{row['persona']}-plain"] = row
    cache: list[dict[str, Any]] = []
    for shard_id, samples in sorted(wanted.items()):
        path = logs_by_shard.get(shard_id)
        if path is None:
            raise ValueError(f"v7 final evidence has no successful log for {shard_id}")
        source_hash = sha256_file(path)
        if source_hash not in allowed_hashes:
            raise ValueError("source log is not recognized by v7 final evidence")
        for sample in read_eval_log(path).samples or []:
            sample_id = str(sample.id)
            source = samples.get(sample_id)
            if source is None:
                continue
            request = _successful_subject_request(sample)
            messages = request["messages"]
            generations = (sample.metadata or {}).get("generations") or {}
            first = str(generations.get("main") or "").strip()
            proxy_eligible = bool(first)
            score = source.get("score")
            original = None if score is None else 100.0 * max(float(score), 1.0 - float(score))
            cache.append(
                {
                    "schema_version": "glm53_v8_transcript_cache_row_v1",
                    "sample_id": sample_id,
                    "group": source["group"],
                    "persona_key": source["persona"],
                    "stimulus_id": source["stimulus"],
                    "system_text": messages[0]["content"],
                    "dilemma_text": messages[1]["content"],
                    "first_assistant_text": first,
                    "proxy_eligible": proxy_eligible,
                    "source_error": None if proxy_eligible else "empty_v7_first_assistant_turn",
                    "original_folded_confidence": original,
                    "codebook_id": codebook_id(sample_id),
                    "source_eval_log": str(path.resolve()),
                    "source_eval_sha256": source_hash,
                    "messages_sha256": _sha_text(
                        messages[0]["content"], messages[1]["content"], first
                    ),
                }
            )
    keys = [(row["group"], row["persona_key"], row["stimulus_id"]) for row in cache]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate transcript-cache key")
    expected = len(PRIMARY_GROUPS) * 70 * len(task_ids)
    if len(cache) != expected:
        raise ValueError(f"expected {expected} cache rows, found {len(cache)}")
    atomic_jsonl(
        output_path,
        sorted(cache, key=lambda row: (row["group"], row["persona_key"], row["stimulus_id"])),
    )
    return {
        "schema_version": "glm53_v8_transcript_cache_manifest_v1",
        "row_count": len(cache),
        "task_count": len(task_ids),
        "group_counts": {
            group: sum(row["group"] == group for row in cache) for group in PRIMARY_GROUPS
        },
        "proxy_eligible_rows": sum(row["proxy_eligible"] for row in cache),
        "proxy_ineligible_rows": sum(not row["proxy_eligible"] for row in cache),
        "proxy_ineligible_rate": sum(not row["proxy_eligible"] for row in cache) / len(cache),
        "proxy_ineligible_by_group": {
            group: sum(row["group"] == group and not row["proxy_eligible"] for row in cache)
            for group in PRIMARY_GROUPS
        },
        "cache_sha256": sha256_file(output_path),
        "source_log_count": len({row["source_eval_sha256"] for row in cache}),
    }


def assert_confidence_absent(rendered_text: str, row: dict[str, Any]) -> None:
    original = row.get("original_folded_confidence")
    if original is not None and str(int(original)) + "%" in rendered_text:
        raise ValueError("original confidence leaked into local proxy input")
