"""Locked v9 feature loading and audit metadata construction."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TASK_RE = re.compile(r"__task_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AuditData:
    features: np.ndarray
    rows: list[dict[str, Any]]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(
            [row["label"] if row["label"] is not None else -1 for row in self.rows],
            dtype=np.int64,
        )


def load_locked_data(
    *,
    feature_root: Path,
    samples_path: Path,
    expected_hashes: dict[str, str],
) -> AuditData:
    fixed_path = feature_root / "fixed_features.npz"
    metadata_path = feature_root / "metadata.jsonl"
    manifest_path = feature_root / "feature_manifest.json"
    actual = {
        "fixed_features": sha256_file(fixed_path),
        "metadata": sha256_file(metadata_path),
        "feature_manifest": sha256_file(manifest_path),
        "samples": sha256_file(samples_path),
    }
    for name, expected in expected_hashes.items():
        if actual[name] != expected.lower():
            raise ValueError(f"{name} hash mismatch: {actual[name]} != {expected}")

    metadata = [
        json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line
    ]
    with samples_path.open("r", encoding="utf-8", newline="") as handle:
        source_by_id = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if len(source_by_id) != 448 or len(metadata) != 448:
        raise ValueError("expected exactly 448 governed rows")

    rows: list[dict[str, Any]] = []
    for row in metadata:
        source = source_by_id.get(row["sample_id"])
        if source is None:
            raise ValueError(f"missing governed source row {row['sample_id']}")
        task_match = TASK_RE.search(source["semantic_task_id"])
        task_number = int(task_match.group(1)) if task_match else None
        cue_location = (
            "system"
            if source["system_prompt"] and source["prompt_role"] == "system_plus_user"
            else "user"
        )
        rows.append(
            row
            | {
                "slice_id": source["slice_id"],
                "prompt_role": source["prompt_role"],
                "system_prompt": source["system_prompt"],
                "user_prompt": source["user_prompt"],
                "semantic_task_id": source["semantic_task_id"],
                "task_number": task_number,
                "cue_location": cue_location,
            }
        )

    with np.load(fixed_path, allow_pickle=False) as archive:
        features = archive["masked_prompt_mean"].astype(np.float32)
    if features.shape != (448, 45, 4096) or not np.isfinite(features).all():
        raise ValueError(f"invalid locked feature matrix {features.shape}")
    validate_core_grid(rows)
    return AuditData(features=features, rows=rows)


def validate_core_grid(rows: list[dict[str, Any]]) -> None:
    core = [row for row in rows if row["slice_id"] == "core_context_pairs"]
    families = sorted({row["family_id"] for row in core})
    if len(families) != 8:
        raise ValueError(f"expected eight core families, got {len(families)}")
    for family in families:
        family_rows = [row for row in core if row["family_id"] == family]
        tasks = {row["task_number"] for row in family_rows}
        if len(family_rows) != 32 or tasks != set(range(1, 17)):
            raise ValueError(f"core family {family} is not a 16-pair grid")
        for task in tasks:
            labels = sorted(row["label"] for row in family_rows if row["task_number"] == task)
            if labels != [0, 1]:
                raise ValueError(f"{family} task {task} is not a complete pair")


def metadata_matrix(
    rows: list[dict[str, Any]], indices: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Build a text-free structural baseline from saved extraction metadata."""
    roles = ["metadata_header", "system_plus_user", "transcript", "user_prefix"]
    locations = ["system", "user"]
    pair_lengths: dict[str, list[int]] = {}
    for row in rows:
        pair_lengths.setdefault(row["pair_id"], []).append(int(row["prompt_tokens"]))
    values: list[list[float]] = []
    for index in indices:
        row = rows[int(index)]
        lengths = pair_lengths[row["pair_id"]]
        pair_gap = float(max(lengths) - min(lengths)) if len(lengths) == 2 else 0.0
        values.append(
            [
                float(row["prompt_tokens"]),
                float(row["valid_token_count"]),
                float(row["retained_token_count"]),
                float(row["masked_token_count"]),
                float(row["cue_span_count"]),
                float(row["masked_span_count"]),
                pair_gap,
            ]
            + [float(row["prompt_role"] == role) for role in roles]
            + [float(row["cue_location"] == location) for location in locations]
        )
    names = (
        [
            "prompt_tokens",
            "valid_token_count",
            "retained_token_count",
            "masked_token_count",
            "cue_span_count",
            "masked_span_count",
            "within_pair_prompt_length_gap",
        ]
        + [f"prompt_role={role}" for role in roles]
        + [f"cue_location={location}" for location in locations]
    )
    return np.asarray(values, dtype=np.float64), names


def deterministic_task_folds() -> list[list[int]]:
    ordered = sorted(
        range(1, 17),
        key=lambda task: hashlib.sha256(f"glm53-v10-task-fold|{task:02d}".encode()).hexdigest(),
    )
    return [ordered[start : start + 2] for start in range(0, 16, 2)]
