"""Atomic and hash-bound v11 source-feature extraction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import LoadedV11GLM53

VIEWS = (
    "shared_task_suffix_mean",
    "prompt_final",
    "masked_prompt_mean",
    "decisive_fact_token_mean",
)


def feature_partition(row: dict[str, Any]) -> str:
    split = row["split"]
    if split in {"train", "validation", "development_counterfactual"}:
        return "development"
    if split in {"ordinary_test", "final_counterfactual"}:
        return "final"
    if split == "neutral_controls":
        partition = row.get("control_partition")
        if partition not in {"development", "final"}:
            raise ValueError("neutral control lacks a valid feature partition")
        return str(partition)
    if split == "factorial_calibration":
        return "calibration"
    raise ValueError(f"unknown v11 source split {split}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
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


def _valid_part(
    part_path: Path,
    manifest_path: Path,
    expected: dict[str, Any],
) -> bool:
    if not part_path.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False
    return manifest.get("part_sha256") == sha256_file(part_path)


def extract_source_features(
    runtime: LoadedV11GLM53,
    rows: list[dict[str, Any]],
    token_audit: dict[str, Any],
    *,
    output_root: Path,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if len(rows) != 576 or token_audit.get("row_count") != 576:
        raise ValueError("v11 extraction requires the complete 576-row source surface")
    token_by_id = {record["sample_id"]: record for record in token_audit["records"]}
    if set(token_by_id) != {row["sample_id"] for row in rows}:
        raise ValueError("token-audit IDs differ from source rows")
    required_hashes = {
        "dataset_sha256",
        "dataset_manifest_sha256",
        "tokenizer_audit_sha256",
        "prereg_sha256",
        "runtime_config_sha256",
        "builder_sha256",
        "spec_sha256",
        "text_decision_sha256",
        "model_revision",
        "paid_process_nonce",
    }
    if set(source_hashes) != required_hashes or not all(source_hashes.values()):
        raise ValueError("source extraction hashes are incomplete")

    parts = output_root / "parts"
    part_records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        token_record = token_by_id[sample_id]
        part = parts / f"sample-{row_index:04d}-{sample_id}.npz"
        manifest_path = part.with_suffix(".manifest.json")
        expected = {
            "schema_version": "glm53_v11_source_feature_part_v1",
            "sample_id": sample_id,
            "source_row_index": row_index,
            **source_hashes,
            "rendered_sha256": token_record["rendered_sha256"],
            "representation_schema": list(VIEWS),
        }
        if not _valid_part(part, manifest_path, expected):
            features = runtime.extract(row, token_record)
            arrays = {view: getattr(features, view) for view in VIEWS}
            for view, array in arrays.items():
                if array.shape != (45, 4096):
                    raise ValueError(f"{sample_id}/{view} has shape {array.shape}")
                if (
                    view != "decisive_fact_token_mean" or row["decisive_fact_texts"]
                ) and not np.isfinite(array).all():
                    raise ValueError(f"{sample_id}/{view} contains NaN or Inf")
            atomic_npz(part, **arrays)
            manifest = expected | {
                "part_sha256": sha256_file(part),
                "shape_by_view": {view: list(array.shape) for view, array in arrays.items()},
                "dtype_by_view": {view: str(array.dtype) for view, array in arrays.items()},
                "input_ids_sha256": features.input_ids_sha256,
                "prompt_tokens": features.prompt_tokens,
            }
            atomic_json(manifest_path, manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _valid_part(part, manifest_path, expected):
            raise ValueError(f"completed feature part does not validate: {part}")
        part_records.append(
            {
                "sample_id": sample_id,
                "path": str(part.relative_to(output_root)).replace("\\", "/"),
                "manifest_path": str(manifest_path.relative_to(output_root)).replace("\\", "/"),
                "sha256": manifest["part_sha256"],
            }
        )
        metadata.append(
            {
                "source_row_index": row_index,
                "sample_id": sample_id,
                "pair_id": row["pair_id"],
                "split": row["split"],
                "label": row["label"],
                "latent_class": row["latent_class"],
                "generator_family": row["generator_family"],
                "task_id": row["task_id"],
                "task_domain": row["task_domain"],
                "prompt_role": row["prompt_role"],
                "register": row["register"],
                "factors": row["factors"],
                "nuisance": row["nuisance"],
                "control_partition": row["control_partition"],
                "calibration_replicate": row.get("calibration_replicate"),
                "calibration_cell": row.get("calibration_cell"),
                "feature_partition": feature_partition(row),
                "part": part_records[-1]["path"],
                "part_sha256": manifest["part_sha256"],
            }
        )

    consolidated: dict[str, Any] = {}
    for partition in ("development", "final", "calibration"):
        partition_indices = [
            index for index, row in enumerate(metadata) if row["feature_partition"] == partition
        ]
        stacked: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
        for index in partition_indices:
            with np.load(output_root / part_records[index]["path"]) as archive:
                for view in VIEWS:
                    stacked[view].append(archive[view])
        feature_path = output_root / f"{partition}_features.npz"
        atomic_npz(
            feature_path,
            **{view: np.stack(values).astype(np.float16) for view, values in stacked.items()},
        )
        metadata_path = output_root / f"{partition}_metadata.jsonl"
        metadata_text = "".join(
            json.dumps(metadata[index], sort_keys=True, separators=(",", ":")) + "\n"
            for index in partition_indices
        )
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary.write_text(metadata_text, encoding="utf-8")
        os.replace(temporary, metadata_path)
        consolidated[partition] = {
            "row_count": len(partition_indices),
            "features": str(feature_path.relative_to(output_root)).replace("\\", "/"),
            "features_sha256": sha256_file(feature_path),
            "metadata": str(metadata_path.relative_to(output_root)).replace("\\", "/"),
            "metadata_sha256": sha256_file(metadata_path),
        }
    manifest = {
        "schema_version": "glm53_v11_source_feature_manifest_v1",
        "passed": True,
        "row_count": len(rows),
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


__all__ = ["VIEWS", "extract_source_features", "feature_partition"]
