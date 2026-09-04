"""Credential-scanned evidence manifest for V13."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:OPENROUTER_API_KEY|OPENAI_API_KEY|RUNPOD_API_KEY|"
        r"AWS_SECRET_ACCESS_KEY)\s*[:=]\s*['\"]?[A-Za-z0-9_/+.-]{16,}"
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_compact_raw_judgments(*, run_root: Path, output_path: Path) -> int:
    """Combine immutable completed-row records without copying verbose event logs."""
    rows: list[dict[str, Any]] = []
    scientific_root = run_root / "scientific"
    for judge_dir in sorted(path for path in scientific_root.iterdir() if path.is_dir()):
        rows_dir = judge_dir / "rows"
        if not rows_dir.is_dir():
            continue
        for path in sorted(rows_dir.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            rows.append(row)
    rows.sort(key=lambda row: (str(row["judge_id"]), str(row["sample_id"])))
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return len(rows)


def build_evidence(
    *,
    repo_root: Path,
    fixed_paths: Sequence[Path],
    run_root: Path,
    report_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    files = {path.resolve() for path in fixed_paths}
    files.update(path.resolve() for path in run_root.rglob("*") if path.is_file())
    files.update(path.resolve() for path in report_root.rglob("*") if path.is_file())
    files.discard(output_path.resolve())
    records: list[dict[str, Any]] = []
    hits: list[dict[str, str]] = []
    for path in sorted(files):
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(
                    {
                        "path": path.relative_to(repo_root).as_posix(),
                        "pattern": pattern.pattern,
                    }
                )
        records.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if hits:
        raise ValueError(f"credential-like material detected: {hits[:3]}")
    decision_path = report_root / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    evidence = {
        "schema_version": "glm53_v13_codex_cohort_evidence_v1",
        "project_id": "glm53_user_eval_codex_judge_cohort_v13",
        "decision": decision["decision"],
        "passed": decision["passed"],
        "authorization": decision["authorization"],
        "file_count": len(records),
        "files": records,
        "credential_scan": {"passed": True, "match_count": 0},
    }
    _atomic_json(output_path, evidence)
    return evidence


__all__ = ["build_compact_raw_judgments", "build_evidence"]
