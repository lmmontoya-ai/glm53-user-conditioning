"""Immutable evidence manifest for the V12 semantic decision."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:OPENROUTER_API_KEY|OPENAI_API_KEY|RUNPOD_API_KEY|AWS_SECRET_ACCESS_KEY)\s*[:=]\s*['\"]?[A-Za-z0-9_/+.-]{16,}"
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def build_evidence(
    *,
    repo_root: Path,
    fixed_paths: Sequence[Path],
    primary_root: Path,
    verifier_root: Path,
    decision_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Hash every accepted response, attempt, analysis, and source lock."""

    files = {path.resolve() for path in fixed_paths}
    files.add(decision_path.resolve())
    for root in (primary_root, verifier_root):
        files.update(path.resolve() for path in root.rglob("*.json") if path.is_file())
    files.discard(output_path.resolve())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"V12 evidence input is absent: {missing[:3]}")
    credential_hits: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for path in sorted(files):
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="ignore")
        for pattern in HIGH_CONFIDENCE_SECRET_PATTERNS:
            if pattern.search(text):
                credential_hits.append(
                    {"path": _relative(path, repo_root), "pattern": pattern.pattern}
                )
        records.append(
            {
                "path": _relative(path, repo_root),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    if credential_hits:
        raise ValueError(f"credential-like material found: {credential_hits[:3]}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    evidence: dict[str, Any] = {
        "schema_version": "glm53_v12_semantic_evidence_v1",
        "project_id": "glm53_user_eval_fact_validator_v12",
        "decision": decision["decision"],
        "passed": decision["passed"],
        "authorization": decision["authorization"],
        "files": records,
        "file_count": len(records),
        "credential_scan": {
            "passed": True,
            "file_count": len(records),
            "match_count": 0,
        },
    }
    _atomic_json(output_path, evidence)
    return evidence


__all__ = ["build_evidence", "sha256_file"]
