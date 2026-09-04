"""Atomic source extraction with final controls in a sealed partition."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .contract import sha256_file
from .dataset import feature_metadata
from .runtime import LoadedV16GLM53

VIEWS = (
    "shared_task_suffix_mean",
    "prompt_final",
    "masked_prompt_mean",
    "decisive_fact_token_mean",
)
PARTITIONS = ("development", "final_binary", "factorial", "fresh_controls")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _valid_part(part: Path, manifest_path: Path, expected: dict[str, Any]) -> bool:
    if not part.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return all(manifest.get(key) == value for key, value in expected.items()) and manifest.get(
        "part_sha256"
    ) == sha256_file(part)


def extract_source_features(
    runtime: LoadedV16GLM53,
    rows: list[dict[str, Any]],
    token_audit: dict[str, Any],
    *,
    output_root: Path,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if len(rows) != 576 or token_audit.get("row_count") != 576:
        raise ValueError("V16 extraction requires all 576 rows")
    token_by_id = {record["sample_id"]: record for record in token_audit["records"]}
    if set(token_by_id) != {row["sample_id"] for row in rows}:
        raise ValueError("tokenizer audit IDs differ from the source bank")
    required_hashes = {
        "dataset_sha256",
        "dataset_manifest_sha256",
        "tokenizer_audit_sha256",
        "prereg_sha256",
        "runtime_config_sha256",
        "downstream_manifest_sha256",
        "v15_decision_sha256",
        "v15_verification_sha256",
        "code_sha256",
        "model_revision",
        "paid_process_nonce",
    }
    if set(source_hashes) != required_hashes or not all(source_hashes.values()):
        raise ValueError("V16 extraction bindings are incomplete")

    part_records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        token_record = token_by_id[sample_id]
        part = output_root / "parts" / f"sample-{row_index:04d}-{sample_id}.npz"
        part_manifest = part.with_suffix(".manifest.json")
        expected = {
            "schema_version": "glm53_v16_source_feature_part_v1",
            "sample_id": sample_id,
            "source_row_index": row_index,
            **source_hashes,
            "rendered_sha256": token_record["rendered_sha256"],
            "representation_schema": list(VIEWS),
        }
        if not _valid_part(part, part_manifest, expected):
            features = runtime.extract(row, token_record)
            arrays = {view: getattr(features, view) for view in VIEWS}
            for view, array in arrays.items():
                if array.shape != (45, 4096):
                    raise ValueError(f"{sample_id}/{view} shape is {array.shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"{sample_id}/{view} contains NaN or Inf")
            atomic_npz(part, **{name: value.astype(np.float16) for name, value in arrays.items()})
            atomic_json(
                part_manifest,
                expected
                | {
                    "part_sha256": sha256_file(part),
                    "shape_by_view": {view: list(value.shape) for view, value in arrays.items()},
                    "dtype_by_view": {view: "float16" for view in arrays},
                    "input_ids_sha256": features.input_ids_sha256,
                    "prompt_tokens": features.prompt_tokens,
                },
            )
        if not _valid_part(part, part_manifest, expected):
            raise ValueError(f"feature part failed validation: {part}")
        manifest = json.loads(part_manifest.read_text(encoding="utf-8"))
        relative = part.relative_to(output_root).as_posix()
        part_records.append(
            {
                "sample_id": sample_id,
                "path": relative,
                "manifest_path": part_manifest.relative_to(output_root).as_posix(),
                "sha256": manifest["part_sha256"],
            }
        )
        metadata.append(
            feature_metadata(
                row,
                row_index=row_index,
                part=relative,
                part_sha256=manifest["part_sha256"],
            )
        )

    consolidated: dict[str, Any] = {}
    expected_counts = {"development": 368, "final_binary": 112, "factorial": 32, "fresh_controls": 64}
    for partition in PARTITIONS:
        indices = [i for i, row in enumerate(metadata) if row["feature_partition"] == partition]
        if len(indices) != expected_counts[partition]:
            raise ValueError(f"{partition} has {len(indices)} rows")
        stacked = {view: [] for view in VIEWS}
        for index in indices:
            with np.load(output_root / part_records[index]["path"]) as archive:
                for view in VIEWS:
                    stacked[view].append(archive[view])
        feature_path = output_root / f"{partition}_features.npz"
        metadata_path = output_root / f"{partition}_metadata.jsonl"
        atomic_npz(
            feature_path,
            **{view: np.stack(values).astype(np.float16) for view, values in stacked.items()},
        )
        atomic_jsonl(metadata_path, [metadata[index] for index in indices])
        consolidated[partition] = {
            "row_count": len(indices),
            "features": feature_path.relative_to(output_root).as_posix(),
            "features_sha256": sha256_file(feature_path),
            "metadata": metadata_path.relative_to(output_root).as_posix(),
            "metadata_sha256": sha256_file(metadata_path),
        }

    if any(
        "control_expected_label" in line or "acceptable_judge_labels" in line
        for line in (output_root / "development_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        raise ValueError("final-control labels leaked into development metadata")
    manifest = {
        "schema_version": "glm53_v16_source_feature_manifest_v1",
        "passed": True,
        "row_count": 576,
        "layer_count": 45,
        "hidden_size": 4096,
        "views": list(VIEWS),
        "primary_view": "shared_task_suffix_mean",
        "partitions": consolidated,
        "source_hashes": source_hashes,
        "parts": part_records,
    }
    atomic_json(output_root / "feature_manifest.json", manifest)
    return manifest


def load_partition(
    feature_root: Path, partition: str, *, view: str = "shared_task_suffix_mean"
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown V16 partition {partition}")
    manifest = json.loads((feature_root / "feature_manifest.json").read_text(encoding="utf-8"))
    record = manifest["partitions"][partition]
    feature_path = feature_root / record["features"]
    metadata_path = feature_root / record["metadata"]
    if sha256_file(feature_path) != record["features_sha256"] or sha256_file(
        metadata_path
    ) != record["metadata_sha256"]:
        raise ValueError(f"{partition} consolidated hashes differ")
    with np.load(feature_path) as archive:
        features = archive[view].astype(np.float32)
    metadata = [
        json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if features.shape != (len(metadata), 45, 4096):
        raise ValueError(f"{partition} feature shape differs: {features.shape}")
    return features, metadata


__all__ = ["PARTITIONS", "VIEWS", "extract_source_features", "load_partition"]
