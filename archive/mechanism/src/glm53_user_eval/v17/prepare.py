"""Build outcome-independent positive-control and causal schedules."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v17.contract import atomic_json, canonical_sha256, read_json, read_jsonl


def _rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()


def build_positive_control_manifest(
    dataset_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    fresh = [row for row in rows if row.get("split") == "neutral_controls"]
    final = [row for row in rows if row.get("split") == "final_counterfactual"]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fresh:
        label = str(row.get("control_expected_label"))
        if label in {"eval", "deploy"}:
            by_class[label].append(row)
    selected: list[dict[str, Any]] = []
    for label in ("eval", "deploy"):
        fresh_rows = sorted(by_class[label], key=lambda row: row["sample_id"])
        final_rows = sorted(
            [row for row in final if row["latent_class"] == label],
            key=lambda row: _rank("glm53-v17-pc-final", row["sample_id"]),
        )[:16]
        if len(fresh_rows) != 16 or len(final_rows) != 16:
            raise ValueError(f"positive-control {label} count differs from 32")
        for source, source_rows in (("fresh_v15", fresh_rows), ("final_counterfactual", final_rows)):
            selected.extend(
                {
                    "sample_id": row["sample_id"],
                    "source": source,
                    "expected_class": label,
                    "scenario_text": row["scenario_text"],
                    "source_row_sha256": canonical_sha256(row),
                }
                for row in source_rows
            )
    manifest = {
        "schema_version": "glm53_v17_positive_control_manifest_v1",
        "selection_rule": "all 16 fresh clean controls per class plus first 16 final-counterfactual rows per class by sha256(glm53-v17-pc-final|sample_id)",
        "rows": sorted(selected, key=lambda row: (row["expected_class"], row["source"], row["sample_id"])),
        "counts": {
            "total_scenarios": 64,
            "eval": 32,
            "deploy": 32,
            "rendered_rows_with_two_codebooks": 128,
        },
    }
    atomic_json(output_path, manifest)
    return manifest


def build_causal_schedule(
    *,
    proxy_rows: list[dict[str, Any]],
    causal_schedule_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    frozen = read_json(causal_schedule_path)
    eligible = {
        (row["group"], row["persona_key"], row["stimulus_id"])
        for row in proxy_rows
    }
    tasks = sorted({row["stimulus_id"] for row in proxy_rows})
    pilot_pairs = [row for row in frozen["pairs"] if row["pilot"]]
    confirmation_pairs = [row for row in frozen["pairs"] if not row["pilot"]][:8]
    pilot_people = {
        "famous_ai": [row["famous_ai"] for row in pilot_pairs],
        "unknown_ai": [row["unknown_ai"] for row in pilot_pairs],
        "famous_nonai": frozen["famous_nonai_controls"][:4],
        "genpop": frozen["genpop_controls"][:4],
    }
    confirmation_people = {
        "famous_ai": [row["famous_ai"] for row in confirmation_pairs],
        "unknown_ai": [row["unknown_ai"] for row in confirmation_pairs],
        "famous_nonai": frozen["famous_nonai_controls"][4:12],
        "genpop": frozen["genpop_controls"][4:12],
    }

    def complete_tasks(people: dict[str, list[str]], count: int | None, namespace: str) -> list[str]:
        candidates = [
            task
            for task in tasks
            if all((group, person, task) in eligible for group, persons in people.items() for person in persons)
        ]
        ordered = sorted(candidates, key=lambda task: _rank(namespace, task))
        if count is not None and len(ordered) < count:
            raise ValueError(f"only {len(ordered)} technically complete tasks for {namespace}")
        return ordered if count is None else ordered[:count]

    pilot_tasks = complete_tasks(pilot_people, 10, "glm53-v17-pilot-task")
    confirmation_candidates = complete_tasks(
        confirmation_people, None, "glm53-v17-confirmation-task"
    )
    confirmation_tasks = [task for task in confirmation_candidates if task not in pilot_tasks][:25]
    if len(confirmation_tasks) != 25:
        raise ValueError("could not select 25 disjoint confirmation tasks")
    manifest = {
        "schema_version": "glm53_v17_causal_schedule_v1",
        "source_schedule_sha256": canonical_sha256(frozen),
        "technical_eligibility_only": True,
        "pilot": {
            "identities": pilot_people,
            "tasks": pilot_tasks,
            "expected_base_rows": 160,
        },
        "confirmation": {
            "identities": confirmation_people,
            "tasks": confirmation_tasks,
            "expected_base_rows": 800,
        },
        "identity_sets_disjoint": all(
            set(pilot_people[group]).isdisjoint(confirmation_people[group]) for group in pilot_people
        ),
        "task_sets_disjoint": set(pilot_tasks).isdisjoint(confirmation_tasks),
    }
    if not manifest["identity_sets_disjoint"] or not manifest["task_sets_disjoint"]:
        raise ValueError("pilot and confirmation schedules overlap")
    atomic_json(output_path, manifest)
    return manifest


def subset_proxy_rows(
    rows: list[dict[str, Any]], schedule: dict[str, Any], stage: str
) -> list[dict[str, Any]]:
    block = schedule[stage]
    people = {group: set(values) for group, values in block["identities"].items()}
    tasks = set(block["tasks"])
    selected = [
        row for row in rows if row["persona_key"] in people[row["group"]] and row["stimulus_id"] in tasks
    ]
    if len(selected) != int(block["expected_base_rows"]):
        raise ValueError(f"{stage} causal row count differs from frozen schedule")
    return selected


__all__ = [
    "build_causal_schedule",
    "build_positive_control_manifest",
    "subset_proxy_rows",
]
