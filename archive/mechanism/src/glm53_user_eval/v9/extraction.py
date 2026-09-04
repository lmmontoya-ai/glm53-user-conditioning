"""Atomic, resumable v9 prompt-feature extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from src.glm53_user_eval.v8.artifacts import atomic_json, atomic_jsonl, sha256_file

from .datasets import EvalRow
from .masking import validate_mask_contract
from .runtime import LoadedV9GLM53

FIXED_VIEWS = (
    "masked_prompt_mean",
    "prompt_final",
    "last_unmasked_prompt_token",
    "cue_token_mean",
)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _valid_part(part: Path, manifest_path: Path, sample_id: str) -> bool:
    if not part.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("sample_id") == sample_id
        and manifest.get("sha256") == sha256_file(part)
    )


def extract_eval_features(
    runtime: LoadedV9GLM53,
    rows: list[EvalRow],
    *,
    output_root: Path,
) -> dict[str, Any]:
    parts = output_root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    fixed: dict[str, list[np.ndarray]] = {view: [] for view in FIXED_VIEWS}
    part_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        part = parts / f"sample-{row_index:04d}.npz"
        part_manifest_path = parts / f"sample-{row_index:04d}.manifest.json"
        if not _valid_part(part, part_manifest_path, row.sample_id):
            features = runtime.extract(row)
            arrays = {
                "masked_prompt_mean": features.masked_prompt_mean,
                "prompt_final": features.prompt_final,
                "last_unmasked_prompt_token": features.last_unmasked_prompt_token,
                "cue_token_mean": features.cue_token_mean,
            }
            arrays.update(
                {f"bag_l{layer:02d}": bag for layer, bag in features.token_bags.items()}
            )
            atomic_npz(part, **arrays)
            part_manifest = {
                "schema_version": "glm53_v9_feature_part_v1",
                "sample_id": row.sample_id,
                "sha256": sha256_file(part),
                "shape_by_key": {key: list(value.shape) for key, value in arrays.items()},
                "metadata": {
                    "rendered_sha256": features.rendered_sha256,
                    "input_ids_sha256": features.input_ids_sha256,
                    "prompt_tokens": features.prompt_tokens,
                    "cue_mask_status": features.mask.status,
                    "cue_span_count": features.mask.requested_span_count,
                    "masked_span_count": features.mask.masked_span_count,
                    "valid_token_count": int(features.mask.valid.sum()),
                    "masked_token_count": int(features.mask.cue.sum()),
                    "retained_token_count": int(features.mask.retained.sum()),
                },
            }
            atomic_json(part_manifest_path, part_manifest)
        part_manifest = json.loads(part_manifest_path.read_text(encoding="utf-8"))
        archive = np.load(part)
        for view in FIXED_VIEWS:
            fixed[view].append(archive[view])
        row_meta = {
            "sample_id": row.sample_id,
            "pair_id": row.pair_id,
            "family_id": row.family_id,
            "split": row.split,
            "label": row.label,
            "context_label": row.context_label,
            "variant_family": row.variant_family,
            "cue_span_text": row.cue_span_text,
            "cue_spans_json": row.cue_spans_json,
            "part": str(part.relative_to(output_root)),
            "part_sha256": part_manifest["sha256"],
        } | part_manifest["metadata"]
        metadata.append(row_meta)
        part_records.append(
            {"path": str(part.relative_to(output_root)), "sha256": part_manifest["sha256"]}
        )

    fixed_path = output_root / "fixed_features.npz"
    atomic_npz(
        fixed_path,
        **{view: np.stack(values).astype(np.float16) for view, values in fixed.items()},
    )
    metadata_path = output_root / "metadata.jsonl"
    atomic_jsonl(metadata_path, metadata)
    mask_contract = validate_mask_contract([row["cue_mask_status"] for row in metadata])
    manifest = {
        "schema_version": "glm53_v9_feature_manifest_v1",
        "row_count": len(rows),
        "layer_count": 45,
        "hidden_size": 4096,
        "fixed_views": list(FIXED_VIEWS),
        "token_bags": "all_retained_tokens_all_45_layers_float16",
        "fixed_features_sha256": sha256_file(fixed_path),
        "metadata_sha256": sha256_file(metadata_path),
        "mask_contract": mask_contract,
        "parts": part_records,
        "passed": len(rows) == 448 and mask_contract["passed"],
    }
    atomic_json(output_root / "feature_manifest.json", manifest)
    return manifest

