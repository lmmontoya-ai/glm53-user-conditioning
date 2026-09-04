"""Fresh exact-tokenizer audit for every V15 source row."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v11.tokenizer_audit import build_tokenizer_audit

from .contract import DATASET_MANIFEST_SHA256, DATASET_SHA256, canonical_sha256, sha256_file
from .dataset import load_rows, verify_reused_rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_v16_tokenizer_audit(
    *, samples_path: Path, manifest_path: Path, tokenizer_root: Path, v4_path: Path
) -> dict[str, Any]:
    if sha256_file(manifest_path) != DATASET_MANIFEST_SHA256:
        raise ValueError("V15 dataset manifest hash differs")
    rows = load_rows(samples_path)
    reused = verify_reused_rows(samples_path, v4_path)
    first = build_tokenizer_audit(rows, tokenizer_root)
    second = build_tokenizer_audit(rows, tokenizer_root)
    if first["records"] != second["records"]:
        raise ValueError("tokenizer rendering changed across independent passes")
    records = []
    for record in first["records"]:
        value = dict(record)
        value["token_ids_sha256"] = canonical_sha256(value["token_ids"])
        value["attention_mask_count"] = value["rendered_token_count"]
        records.append(value)
    pair = first["pair_contract"]
    if pair["checked_pair_count"] != 240 or pair["singleton_control_count"] != 96:
        raise ValueError("V16 pair/singleton counts differ")
    report = {
        "schema_version": "glm53_v16_tokenizer_audit_v1",
        "dataset_id": "contrastive_prompts_v5",
        "passed": True,
        "samples_path": samples_path.as_posix(),
        "samples_sha256": DATASET_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "tokenizer_json_sha256": first["tokenizer_json_sha256"],
        "chat_template_sha256": first["chat_template_sha256"],
        "deterministic_rendering_passes": 2,
        "row_count": len(records),
        "pair_contract": pair,
        "reused_row_contract": reused,
        "records_sha256": canonical_sha256(records),
        "records": records,
    }
    return report

def write_v16_tokenizer_audit(
    *, samples_path: Path, manifest_path: Path, tokenizer_root: Path, v4_path: Path, output_path: Path
) -> dict[str, Any]:
    report = build_v16_tokenizer_audit(
        samples_path=samples_path,
        manifest_path=manifest_path,
        tokenizer_root=tokenizer_root,
        v4_path=v4_path,
    )
    _atomic_json(output_path, report)
    return report
