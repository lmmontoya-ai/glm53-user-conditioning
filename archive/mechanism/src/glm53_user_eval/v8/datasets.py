"""Frozen dataset surfaces for the v8 white-box study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_eval_surface(dataset_root: Path) -> list[dict[str, Any]]:
    splits: dict[str, str] = {}
    with (dataset_root / "splits.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            splits[row["sample_id"]] = row["split_id"]
    output: list[dict[str, Any]] = []
    with (dataset_root / "samples.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = 1 if row["context_label"] == "eval_like" else 0
            messages: list[dict[str, str]] = []
            if row["system_prompt"].strip():
                messages.append({"role": "system", "content": row["system_prompt"]})
            messages.append({"role": "user", "content": row["user_prompt"]})
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "family_id": row["family_id"],
                    "slice_id": row["slice_id"],
                    "split": splits[row["sample_id"]],
                    "label": label,
                    "context_label": row["context_label"],
                    "messages": messages,
                }
            )
    if len(output) != 448 or len({row["sample_id"] for row in output}) != len(output):
        raise ValueError("contrastive_prompts_v2 surface is incomplete or duplicated")
    return output


def _system_by_persona(cache_rows: list[dict[str, Any]]) -> dict[str, str]:
    systems: dict[str, str] = {}
    for row in cache_rows:
        key = row["persona_key"]
        previous = systems.setdefault(key, row["system_text"])
        if previous != row["system_text"]:
            raise ValueError(f"persona system prompt drift: {key}")
    return systems


def load_user_surface(
    *,
    personas_path: Path,
    cache_path: Path,
    templates_path: Path,
    schedule_path: Path,
) -> list[dict[str, Any]]:
    personas = json.loads(personas_path.read_text(encoding="utf-8"))
    cache_rows = read_jsonl(cache_path)
    systems = _system_by_persona(cache_rows)
    templates = read_jsonl(templates_path)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    targets = {
        row[key] for row in schedule["pairs"] for key in ("famous_ai", "unknown_ai", "genpop")
    }
    target_orgs = {
        persona["org"]
        for group in ("famous_ai", "unknown_ai")
        for persona in personas[group]
        if persona["key"] in targets
    }
    groups = {
        "F": personas["famous_ai"],
        "U": personas["unknown_ai"],
        "G": personas["genpop"],
        "FN": personas["famous_nonai"],
    }
    if any(len(rows) != 70 for rows in groups.values()):
        raise ValueError("user representation surface requires 70 identities per group")
    output: list[dict[str, Any]] = []
    for group, identities in groups.items():
        for pair_index, persona in enumerate(identities):
            key = persona["key"]
            system_text = systems.get(key)
            if system_text is None:
                raise ValueError(f"no immutable v7 system prompt for {key}")
            for template in templates:
                output.append(
                    {
                        "sample_id": f"neutral__{template['template_id']}__{key}",
                        "group": group,
                        "persona_key": key,
                        "pair_index": pair_index,
                        "organization": persona["org"],
                        "template_id": template["template_id"],
                        "target_identity": key in targets,
                        "target_organization": persona["org"] in target_orgs,
                        "messages": [
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": template["user_text"]},
                        ],
                    }
                )
    if len(output) != 4 * 70 * len(templates):
        raise ValueError("user representation surface has the wrong size")
    return output


def load_proxy_surface(
    cache_path: Path,
    codebooks_path: Path,
    task_ids: set[str],
    personas_path: Path,
) -> list[dict[str, Any]]:
    from .proxy import proxy_messages

    payload = json.loads(codebooks_path.read_text(encoding="utf-8"))
    personas = json.loads(personas_path.read_text(encoding="utf-8"))
    pair_indices = {
        persona["key"]: index
        for group in ("famous_ai", "unknown_ai", "genpop", "famous_nonai")
        for index, persona in enumerate(personas[group])
    }
    output: list[dict[str, Any]] = []
    for row in read_jsonl(cache_path):
        if row["stimulus_id"] not in task_ids or not row["proxy_eligible"]:
            continue
        codebook = payload["codebooks"][row["codebook_id"]]
        output.append(
            row
            | {
                "pair_index": pair_indices[row["persona_key"]],
                "messages": proxy_messages(
                    system_text=row["system_text"],
                    dilemma_text=row["dilemma_text"],
                    first_assistant_text=row["first_assistant_text"],
                    codebook=codebook,
                    followup=payload["followup"],
                    assistant_prefix=payload["assistant_prefix"],
                ),
                "codebook_values": [codebook[label] for label in payload["labels"]],
            }
        )
    return output


def target_proxy_surface(
    all_rows: list[dict[str, Any]], schedule: dict[str, Any], stage: str
) -> list[dict[str, Any]]:
    tasks = set(schedule["tasks"][stage])
    pilot = stage == "pilot"
    pairs = [row for row in schedule["pairs"] if bool(row["pilot"]) == pilot]
    identities = {row[key] for row in pairs for key in ("famous_ai", "unknown_ai")}
    control_count = 4 if pilot else 12
    identities.update(schedule["famous_nonai_controls"][:control_count])
    identities.update(schedule["genpop_controls"][:control_count])
    selected = [
        row for row in all_rows if row["stimulus_id"] in tasks and row["persona_key"] in identities
    ]
    return selected
