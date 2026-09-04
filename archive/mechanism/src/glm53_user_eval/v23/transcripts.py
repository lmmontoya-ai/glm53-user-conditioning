from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

from .artifacts import atomic_json, atomic_jsonl, canonical_json, sha256_file, sha256_text

PRIMARY_GROUPS = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
SUBJECT_MODEL = "openrouter/z-ai/glm-5.3-flash"


def _resolve(value: Any, attachments: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("attachment://"):
        key = value.removeprefix("attachment://")
        if key not in attachments:
            raise ValueError(f"missing attachment {key}")
        return attachments[key]
    if isinstance(value, list):
        return [_resolve(item, attachments) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, attachments) for key, item in value.items()}
    return value


def _subject_events(sample: Any) -> list[dict[str, Any]]:
    dumped = sample.model_dump(mode="json")
    attachments = dumped.get("attachments") or {}
    events: list[dict[str, Any]] = []
    for event in sample.events or []:
        if getattr(event, "event", "") != "model" or event.model != SUBJECT_MODEL:
            continue
        value = event.model_dump(mode="json")
        call = value.get("call") or {}
        if call.get("error") or value.get("error"):
            continue
        events.append(_resolve(value, attachments))
    if len(events) != 2:
        raise ValueError(
            f"expected two successful subject turns for {sample.id}, found {len(events)}"
        )
    return events


def _content(event: Mapping[str, Any]) -> tuple[str, str]:
    output = event.get("output") or {}
    choices = output.get("choices") or []
    if len(choices) != 1:
        raise ValueError("subject output must have one choice")
    parts = (choices[0].get("message") or {}).get("content") or []
    if isinstance(parts, str):
        return "", parts.strip()
    reasoning = "\n".join(
        str(part.get("reasoning") or "").strip()
        for part in parts
        if part.get("type") == "reasoning" and str(part.get("reasoning") or "").strip()
    )
    visible = "\n".join(
        str(part.get("text") or "").strip()
        for part in parts
        if part.get("type") == "text" and str(part.get("text") or "").strip()
    )
    return reasoning, visible


def _usage(event: Mapping[str, Any]) -> dict[str, int | None]:
    usage = (event.get("output") or {}).get("usage") or {}
    output_tokens = usage.get("output_tokens")
    reasoning_tokens = usage.get("reasoning_tokens")
    visible_tokens = None
    if output_tokens is not None and reasoning_tokens is not None:
        visible_tokens = max(0, int(output_tokens) - int(reasoning_tokens))
    return {
        "output_tokens": None if output_tokens is None else int(output_tokens),
        "reasoning_tokens": None if reasoning_tokens is None else int(reasoning_tokens),
        "visible_tokens": visible_tokens,
    }


def _first_request(event: Mapping[str, Any]) -> tuple[str, str]:
    messages = ((event.get("call") or {}).get("request") or {}).get("messages") or []
    if [item.get("role") for item in messages] != ["system", "user"]:
        raise ValueError("unexpected first-turn request roles")
    return str(messages[0]["content"]), str(messages[1]["content"])


def _sentence_count(text: str) -> int:
    clean = text.strip()
    if not clean:
        return 0
    return len([item for item in re.split(r"(?<=[.!?])(?:\s+|$)", clean) if item.strip()])


def _literal_present(text: str, literal: str) -> bool:
    return bool(literal.strip()) and literal.casefold() in text.casefold()


def extract_run(
    *,
    run_root: Path,
    raw_scores_path: Path,
    roster_path: Path,
    output_path: Path,
    development_rows_per_group: int | None = None,
    development_salt: str = "",
) -> dict[str, Any]:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    profiles = {
        (group, str(item["key"])): item for group in PRIMARY_GROUPS for item in roster[group]
    }
    raw: dict[tuple[str, str], dict[str, Any]] = {}
    for line in raw_scores_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["group"] in PRIMARY_GROUPS and row["persona"] != "anon":
            raw[(str(row["shard_id"]), f"{row['stimulus']}-{row['persona']}-plain")] = row
    if development_rows_per_group is not None:
        selected: dict[tuple[str, str], dict[str, Any]] = {}
        for group in PRIMARY_GROUPS:
            candidates = [(key, row) for key, row in raw.items() if row["group"] == group]
            candidates.sort(
                key=lambda item: sha256_text(
                    f"{development_salt}|{item[1]['stimulus']}|{item[1]['persona']}"
                )
            )
            selected.update(candidates[:development_rows_per_group])
        raw = selected
    logs = {path.parent.name: path for path in run_root.rglob("*.eval")}
    samples_by_shard: dict[str, dict[str, Any]] = {}
    log_hashes: dict[str, str] = {}
    for shard in sorted({key[0] for key in raw}):
        path = logs.get(shard)
        if path is None:
            raise ValueError(f"missing log for shard {shard}")
        log_hashes[shard] = sha256_file(path)
        samples_by_shard[shard] = {
            str(item.id): item for item in (read_eval_log(path).samples or [])
        }
    rows: list[dict[str, Any]] = []
    for (shard, sample_id), source in sorted(raw.items()):
        path = logs[shard]
        sample = samples_by_shard[shard].get(sample_id)
        if sample is None:
            raise ValueError(f"missing sample {sample_id}")
        first, second = _subject_events(sample)
        first_reasoning, first_visible = _content(first)
        confidence_reasoning, confidence_visible = _content(second)
        system_text, dilemma_text = _first_request(first)
        profile = profiles[(source["group"], source["persona"])]
        full_assistant = (
            f"{first_reasoning}\n{first_visible}\n{confidence_reasoning}\n{confidence_visible}"
        )
        confidence = source.get("confidence_p")
        folded = None if confidence is None else max(float(confidence), 100.0 - float(confidence))
        rows.append(
            {
                "schema_version": "glm53_v23_transcript_row_v1",
                "sample_id": sample_id,
                "group": source["group"],
                "persona_key": source["persona"],
                "stimulus_id": source["stimulus"],
                "shard_id": shard,
                "identity_index": next(
                    i
                    for i, item in enumerate(roster[source["group"]])
                    if item["key"] == source["persona"]
                ),
                "name": profile["name"],
                "affiliation": profile["affiliation"],
                "organization": profile["org"],
                "email": profile["email"],
                "system_text": system_text,
                "dilemma_text": dilemma_text,
                "first_turn_reasoning": first_reasoning,
                "first_turn_answer": first_visible,
                "confidence_turn_reasoning": confidence_reasoning,
                "confidence_turn_answer": confidence_visible,
                "first_turn_choice": source.get("binary_answer"),
                "raw_confidence": confidence,
                "folded_confidence": folded,
                "parse_valid": source.get("binary_answer") in {"yes", "no"}
                and confidence is not None,
                "first_turn_usage": _usage(first),
                "confidence_turn_usage": _usage(second),
                "first_visible_sentence_count": _sentence_count(first_visible),
                "confidence_visible_sentence_count": _sentence_count(confidence_visible),
                "explicit_name_string_present": _literal_present(
                    full_assistant, str(profile["name"])
                ),
                "explicit_affiliation_string_present": any(
                    _literal_present(full_assistant, str(value))
                    for value in (profile["affiliation"], profile["org"], profile["email"])
                ),
                "source_eval_log": str(path.resolve()),
                "source_eval_sha256": log_hashes[shard],
                "transcript_sha256": sha256_text(
                    canonical_json(
                        [
                            system_text,
                            dilemma_text,
                            first_reasoning,
                            first_visible,
                            confidence_reasoning,
                            confidence_visible,
                        ]
                    )
                ),
            }
        )
    expected = (
        28000
        if development_rows_per_group is None
        else development_rows_per_group * len(PRIMARY_GROUPS)
    )
    if len(rows) != expected:
        raise ValueError(f"expected {expected} primary transcript rows, found {len(rows)}")
    keys = {(row["group"], row["persona_key"], row["stimulus_id"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate transcript keys")
    ordered = sorted(
        rows, key=lambda row: (row["group"], row["identity_index"], row["stimulus_id"])
    )
    atomic_jsonl(output_path, ordered)
    manifest = {
        "schema_version": "glm53_v23_transcript_manifest_v1",
        "row_count": len(rows),
        "valid_rows": sum(bool(row["parse_valid"]) for row in rows),
        "group_counts": {
            group: sum(row["group"] == group for row in rows) for group in PRIMARY_GROUPS
        },
        "development_subset": development_rows_per_group is not None,
        "log_count": len({row["source_eval_sha256"] for row in rows}),
        "output_sha256": sha256_file(output_path),
        "raw_scores_sha256": sha256_file(raw_scores_path),
        "roster_sha256": sha256_file(roster_path),
    }
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def _hash_rank(salt: str, index: int, stimulus: str) -> str:
    return hashlib.sha256(f"{salt}|{index}|{stimulus}".encode()).hexdigest()


def select_matched_cells(*, count: int, salt: str) -> list[tuple[int, str]]:
    universe = [(index, f"dd_{task:04d}") for index in range(70) for task in range(100)]
    ranked = sorted(universe, key=lambda item: _hash_rank(salt, *item))
    selected: list[tuple[int, str]] = []
    used: set[tuple[int, str]] = set()
    for index in range(70):
        item = next(value for value in ranked if value[0] == index)
        selected.append(item)
        used.add(item)
    for task in range(100):
        stimulus = f"dd_{task:04d}"
        if any(item[1] == stimulus for item in selected):
            continue
        item = next(value for value in ranked if value[1] == stimulus and value not in used)
        selected.append(item)
        used.add(item)
    for item in ranked:
        if len(selected) >= count:
            break
        if item not in used:
            selected.append(item)
            used.add(item)
    if (
        len(selected) != count
        or len({i for i, _ in selected}) != 70
        or len({s for _, s in selected}) != 100
    ):
        raise ValueError("matched-cell selection coverage failed")
    return sorted(selected)


def _redact(text: str, row: Mapping[str, Any]) -> str:
    values = [
        (str(row["email"]), "<USER_AFFILIATION>"),
        (str(row["affiliation"]), "<USER_AFFILIATION>"),
        (str(row["name"]), "<USER_NAME>"),
        (str(row["organization"]), "<USER_AFFILIATION>"),
    ]
    result = text
    for literal, replacement in sorted(values, key=lambda item: -len(item[0])):
        if not literal.strip():
            continue
        result = re.sub(re.escape(literal), replacement, result, flags=re.IGNORECASE)
    return result


def _hide_confidence(text: str) -> str:
    return re.sub(
        r"(?<!\w)(?:100|\d{1,2})(?:\s*[-–—]\s*(?:100|\d{1,2}))?\s*%",
        "<REPORTED_CONFIDENCE>",
        text,
    )


def build_annotation_packet(
    *, transcript_path: Path, output_path: Path, matched_cells: int, salt: str
) -> dict[str, Any]:
    rows = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()]
    selected = set(select_matched_cells(count=matched_cells, salt=salt))
    packet: list[dict[str, Any]] = []
    for row in rows:
        if (int(row["identity_index"]), str(row["stimulus_id"])) not in selected:
            continue
        pieces = {
            "dilemma": row["dilemma_text"],
            "first_reasoning": row["first_turn_reasoning"],
            "first_answer": row["first_turn_answer"],
            "confidence_reasoning": row["confidence_turn_reasoning"],
            "confidence_explanation": _hide_confidence(str(row["confidence_turn_answer"])),
        }
        blinded = {key: _redact(str(value), row) for key, value in pieces.items()}
        if any(
            str(row[field]).casefold() in canonical_json(blinded).casefold()
            for field in ("name", "email")
            if str(row[field])
        ):
            raise ValueError(f"identity redaction failed for {row['sample_id']}")
        scenario = (
            "DILEMMA\n{dilemma}\n\nFIRST-TURN REASONING\n{first_reasoning}\n\n"
            "FIRST ANSWER\n{first_answer}\n\nCONFIDENCE-TURN REASONING\n{confidence_reasoning}\n\n"
            "CONFIDENCE EXPLANATION (numeric confidence hidden)\n{confidence_explanation}"
        ).format(**blinded)
        packet.append(
            {
                "sample_id": row["sample_id"],
                "scenario_text": scenario,
                "scenario_sha256": sha256_text(scenario),
                "source_transcript_sha256": row["transcript_sha256"],
            }
        )
    expected = matched_cells * 4
    if len(packet) != expected:
        raise ValueError(f"expected {expected} annotation rows, found {len(packet)}")
    atomic_jsonl(output_path, sorted(packet, key=lambda row: sha256_text(salt + row["sample_id"])))
    manifest = {
        "schema_version": "glm53_v23_annotation_packet_manifest_v1",
        "matched_cells": matched_cells,
        "row_count": len(packet),
        "salt": salt,
        "identity_coverage": 70,
        "dilemma_coverage": 100,
        "profile_omitted": True,
        "names_and_affiliations_redacted": True,
        "numeric_confidence_hidden": True,
        "output_sha256": sha256_file(output_path),
    }
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def build_development_packet(
    *, transcript_path: Path, output_path: Path, rows_per_group: int, salt: str
) -> dict[str, Any]:
    rows = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()]
    selected: list[dict[str, Any]] = []
    for group in PRIMARY_GROUPS:
        candidates = [row for row in rows if row["group"] == group and row["parse_valid"]]
        candidates.sort(key=lambda row: sha256_text(f"{salt}|{row['sample_id']}"))
        selected.extend(candidates[:rows_per_group])
    packet: list[dict[str, Any]] = []
    for row in selected:
        pieces = {
            "dilemma": row["dilemma_text"],
            "first_reasoning": row["first_turn_reasoning"],
            "first_answer": row["first_turn_answer"],
            "confidence_reasoning": row["confidence_turn_reasoning"],
            "confidence_explanation": _hide_confidence(str(row["confidence_turn_answer"])),
        }
        blinded = {key: _redact(str(value), row) for key, value in pieces.items()}
        scenario = (
            "DILEMMA\n{dilemma}\n\nFIRST-TURN REASONING\n{first_reasoning}\n\n"
            "FIRST ANSWER\n{first_answer}\n\nCONFIDENCE-TURN REASONING\n{confidence_reasoning}\n\n"
            "CONFIDENCE EXPLANATION (numeric confidence hidden)\n{confidence_explanation}"
        ).format(**blinded)
        packet.append({"sample_id": row["sample_id"], "scenario_text": scenario})
    expected = rows_per_group * len(PRIMARY_GROUPS)
    if len(packet) != expected:
        raise ValueError(f"expected {expected} valid development rows, found {len(packet)}")
    atomic_jsonl(output_path, sorted(packet, key=lambda row: sha256_text(salt + row["sample_id"])))
    manifest = {
        "schema_version": "glm53_v23_development_packet_manifest_v1",
        "source": "v6_only",
        "row_count": len(packet),
        "rows_per_group": rows_per_group,
        "output_sha256": sha256_file(output_path),
    }
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest
