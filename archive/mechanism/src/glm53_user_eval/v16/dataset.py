"""Load the locked V15 bank and keep final-control labels out of development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contract import DATASET_SHA256, sha256_file, validate_dataset_counts


def load_rows(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != DATASET_SHA256:
        raise ValueError("contrastive_prompts_v5 hash differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("V15 sample IDs are not unique")
    validate_dataset_counts(rows)
    return rows


def verify_reused_rows(v5_path: Path, v4_path: Path) -> dict[str, Any]:
    v5 = v5_path.read_bytes().splitlines(keepends=True)
    v4 = v4_path.read_bytes().splitlines(keepends=True)
    if len(v5) != 576 or len(v4) != 576 or v5[:512] != v4[:512]:
        raise ValueError("the 512 reused V15 rows are not byte-identical to V14")
    fresh = [json.loads(line) for line in v5[512:]]
    if len(fresh) != 64 or not all(row.get("repair_metadata", {}).get("fresh_text") is True for row in fresh):
        raise ValueError("V15 fresh-control rows violate their provenance contract")
    reused_ids = {json.loads(line)["sample_id"] for line in v4[:512]}
    fresh_ids = {row["sample_id"] for row in fresh}
    if len(fresh_ids) != 64 or reused_ids & fresh_ids:
        raise ValueError("V15 controls do not have 64 fresh IDs")
    return {"passed": True, "reused_rows": 512, "fresh_controls": 64}


def feature_partition(row: dict[str, Any]) -> str:
    split = row["split"]
    if split in {"train", "validation", "development_counterfactual"}:
        return "development"
    if split in {"ordinary_test", "final_counterfactual"}:
        return "final_binary"
    if split == "factorial_calibration":
        return "factorial"
    if split == "neutral_controls":
        return "fresh_controls"
    raise ValueError(f"unknown V16 split {split}")


def feature_metadata(row: dict[str, Any], *, row_index: int, part: str, part_sha256: str) -> dict[str, Any]:
    # control_expected_label and acceptable_judge_labels intentionally stay out.
    keys = (
        "sample_id",
        "pair_id",
        "split",
        "label",
        "latent_class",
        "generator_family",
        "task_id",
        "task_domain",
        "prompt_role",
        "register",
        "factors",
        "nuisance",
        "control_partition",
        "calibration_replicate",
        "calibration_cell",
    )
    result = {key: row.get(key) for key in keys}
    result |= {
        "source_row_index": row_index,
        "feature_partition": feature_partition(row),
        "part": part,
        "part_sha256": part_sha256,
    }
    return result
