"""Build a score-blind seeded transcript sample, then materialize it for manual review."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")


def select_candidates(
    candidates: list[dict[str, Any]], *, seed: int, per_group: int = 8
) -> list[dict[str, Any]]:
    """Select one anon and task-block coverage per group without consulting scores."""
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group in GROUPS:
        group_rows = [row for row in candidates if row["group"] == group]
        anon = [row for row in group_rows if row["persona"] == "anon"]
        nonanon = [row for row in group_rows if row["persona"] != "anon"]
        if not anon or len(nonanon) < per_group - 1:
            raise ValueError(f"insufficient audit candidates for {group}")
        selected.append(rng.choice(sorted(anon, key=lambda row: row["sample_id"])))
        by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in nonanon:
            by_block[int(row["task_block"])].append(row)
        if set(by_block) != set(range(5)):
            raise ValueError(f"audit candidates lack task-block coverage for {group}")
        group_selected = [
            rng.choice(sorted(by_block[block], key=lambda row: row["sample_id"]))
            for block in range(5)
        ]
        remaining = [row for row in nonanon if row not in group_selected]
        group_selected.extend(rng.sample(remaining, per_group - 1 - len(group_selected)))
        selected.extend(group_selected)
    return sorted(selected, key=lambda row: (GROUPS.index(row["group"]), row["sample_id"]))


def newest_success(log_dir: Path) -> Path:
    from inspect_ai.log import read_eval_log

    logs = sorted(log_dir.glob("*.eval"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise ValueError(f"missing eval log: {log_dir}")
    path = logs[-1]
    if str(read_eval_log(path, header_only=True).status) != "success":
        raise ValueError(f"newest eval is not successful: {path}")
    return path


def collect_candidates(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    from inspect_ai.log import read_eval_log_sample_summaries

    manifest = json.loads((run_root / "schedule_manifest.json").read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    logs: dict[str, Path] = {}
    for spec in manifest["shards"]:
        shard_id = str(spec["shard_id"])
        path = newest_success(run_root / "eval_logs" / shard_id)
        logs[shard_id] = path
        for sample in read_eval_log_sample_summaries(path):
            metadata = sample.metadata or {}
            stimulus = str(metadata["scenario_id"])
            candidates.append(
                {
                    "sample_id": str(sample.id),
                    "shard_id": shard_id,
                    "group": str(spec["group"]),
                    "persona": str(metadata["persona"]),
                    "stimulus": stimulus,
                    "task_block": int(stimulus.rsplit("_", 1)[1]) // 20,
                }
            )
    return candidates, logs


def materialize(
    selected: list[dict[str, Any]], logs: dict[str, Path]
) -> list[dict[str, Any]]:
    from inspect_ai.log import read_eval_log

    wanted_by_shard: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        wanted_by_shard[row["shard_id"]].add(row["sample_id"])
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for shard_id, wanted in wanted_by_shard.items():
        log = read_eval_log(logs[shard_id])
        for sample in log.samples or []:
            if str(sample.id) not in wanted:
                continue
            metadata = sample.metadata or {}
            generations = metadata.get("generations") or {}
            scores = list((sample.scores or {}).values())
            score = scores[0] if scores else None
            score_metadata = dict(score.metadata or {}) if score is not None else {}
            payloads[(shard_id, str(sample.id))] = {
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    candidates, logs = collect_candidates(args.run_root)
    selected = select_candidates(candidates, seed=args.seed)
    packet = materialize(selected, logs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_root / "audit_packet.jsonl"
    packet_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in packet),
        encoding="utf-8",
    )
    with (args.output_root / "reading_log.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "group",
                "persona",
                "stimulus",
                "reviewed",
                "binary_extraction_correct",
                "confidence_extraction_correct",
                "on_task",
                "notes",
            ],
        )
        writer.writeheader()
        for row in packet:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "group": row["group"],
                    "persona": row["persona"],
                    "stimulus": row["stimulus"],
                }
            )
    manifest = {
        "schema_version": "glm53_transluce_exact_manual_audit_v1",
        "seed": args.seed,
        "selection_was_score_blind": True,
        "row_count": len(packet),
        "rows_per_group": {
            group: sum(row["group"] == group for row in packet) for group in GROUPS
        },
        "anonymous_rows": sum(row["persona"] == "anon" for row in packet),
        "packet": str(packet_path),
        "reading_log": str(args.output_root / "reading_log.csv"),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
