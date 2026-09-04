"""Build the fixed, score-blind v7 manual transcript audit packet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.glm53_user_eval.scripts.build_transluce_v6_audit_packet import (
    collect_candidates,
    select_candidates,
)


def materialize_with_prompts(
    selected: list[dict[str, Any]], logs: dict[str, Path]
) -> list[dict[str, Any]]:
    from inspect_ai.log import read_eval_log

    wanted: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        wanted[row["shard_id"]].add(row["sample_id"])
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for shard_id, sample_ids in wanted.items():
        for sample in read_eval_log(logs[shard_id]).samples or []:
            if str(sample.id) not in sample_ids:
                continue
            metadata = sample.metadata or {}
            generations = metadata.get("generations") or {}
            scores = list((sample.scores or {}).values())
            score = scores[0] if scores else None
            score_metadata = dict(score.metadata or {}) if score is not None else {}
            dumped = sample.model_dump(mode="json")
            attachments = dumped.get("attachments") or {}

            def resolved_requests(
                model: str,
                *,
                current_sample: Any = sample,
                current_attachments: dict[str, Any] = attachments,
            ) -> list[dict[str, Any]]:
                requests = []
                for event in current_sample.events or []:
                    if getattr(event, "event", "") != "model" or event.model != model:
                        continue
                    event_payload = event.model_dump(mode="json")
                    call = event_payload.get("call") or {}
                    if call.get("error"):
                        continue
                    request = dict(call.get("request") or {})
                    resolved_messages = []
                    for message in request.get("messages") or []:
                        resolved = dict(message)
                        content = resolved.get("content")
                        if isinstance(content, str) and content.startswith("attachment://"):
                            resolved["content"] = current_attachments[
                                content.removeprefix("attachment://")
                            ]
                        resolved_messages.append(resolved)
                    requests.append(
                        {
                            "model": request.get("model"),
                            "max_tokens": request.get("max_tokens"),
                            "extra_body": request.get("extra_body"),
                            "messages": resolved_messages,
                        }
                    )
                return requests

            payloads[(shard_id, str(sample.id))] = {
                "input_messages": [message.model_dump(mode="json") for message in sample.input],
                "transcript_messages": [
                    message.model_dump(mode="json") for message in (sample.messages or [])
                ],
                "scenario_script": metadata.get("script"),
                "persona_email": metadata.get("persona_email"),
                "subject_requests": resolved_requests("openrouter/z-ai/glm-5.3-flash"),
                "judge_requests": resolved_requests("openrouter/openai/gpt-5.4-mini"),
                "subject_main": generations.get("main"),
                "subject_followup": generations.get("followup"),
                "score_value": None if score is None else score.value,
                "confidence_p": score_metadata.get("confidence_p"),
                "binary_answer": score_metadata.get("binary_answer"),
                "refused_estimate": score_metadata.get("refused_estimate"),
                "judge_text": score_metadata.get("judge_text"),
                "sample_error": None if sample.error is None else str(sample.error),
            }
    return [row | payloads[(row["shard_id"], row["sample_id"])] for row in selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    marker = args.run_root / "FULL_RUN_AUDIT_PASS.json"
    if not marker.exists():
        raise ValueError("manual packet requires FULL_RUN_AUDIT_PASS.json")
    candidates, logs = collect_candidates(args.run_root)
    selected = select_candidates(candidates, seed=args.seed)
    packet = materialize_with_prompts(selected, logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in packet),
        encoding="utf-8",
    )
    reading_log = args.output.with_name("reading_log.csv")
    fields = [
        "sample_id",
        "group",
        "persona",
        "stimulus",
        "reviewed",
        "persona_correct",
        "dilemma_correct",
        "two_turn_structure_correct",
        "confidence_followup_correct",
        "judge_input_correct",
        "parser_correct",
        "on_task",
        "notes",
    ]
    with reading_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in packet:
            writer.writerow({key: row.get(key, "") for key in fields})
    manifest = {
        "schema_version": "glm53_transluce_interaction_v7_manual_packet_v1",
        "seed": args.seed,
        "selection_was_score_blind": True,
        "row_count": len(packet),
        "rows_per_group": {
            group: sum(row["group"] == group for row in packet)
            for group in ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")
        },
        "anonymous_rows": sum(row["persona"] == "anon" for row in packet),
        "packet": str(args.output.resolve()),
        "reading_log": str(reading_log.resolve()),
    }
    args.output.with_name("manual_packet_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
