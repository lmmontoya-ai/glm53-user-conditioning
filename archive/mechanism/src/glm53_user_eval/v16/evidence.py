"""Compact, hash-bound V16 evidence manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .contract import sha256_file


def _contains_secret(path: Path) -> bool:
    if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".csv"}:
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    secret_values = [
        os.environ.get("AWS_ACCESS_KEY_ID", ""),
        os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        os.environ.get("RUNPOD_API_KEY", ""),
    ]
    return any(value and value in text for value in secret_values)


def build_evidence(
    *, repo_root: Path, roots: list[Path], output_path: Path, terminal: dict[str, Any]
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in candidates:
            if path == output_path or path.name.endswith((".tmp", ".partial")):
                continue
            if _contains_secret(path):
                raise ValueError(f"credential material found in V16 artifact: {path}")
            files.append(
                {
                    "path": path.resolve().relative_to(repo_root.resolve()).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    report = {
        "schema_version": "glm53_v16_final_evidence_v1",
        "project_id": "glm53_user_eval_source_activation_v16",
        "terminal": terminal,
        "file_count": len(files),
        "files": files,
        "credentials_scanned": True,
        "early_cot_executed": False,
        "steering_executed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return report


__all__ = ["build_evidence"]
