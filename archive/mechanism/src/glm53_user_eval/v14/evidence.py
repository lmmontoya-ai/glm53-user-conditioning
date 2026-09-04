"""Compact, credential-scanned V14 evidence bundle."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v12.fact_validation import atomic_json

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


def build_compact_rows(*, run_root: Path, output_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "scientific").glob("*/rows/*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: (str(row["judge_id"]), str(row["sample_id"])))
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(output_path)
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
    for path in sorted(files):
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValueError(f"credential-like material detected in {path}")
        records.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    decision = json.loads((report_root / "decision.json").read_text(encoding="utf-8"))
    evidence = {
        "schema_version": "glm53_v14_balanced_repair_evidence_v1",
        "project_id": "glm53_user_eval_balanced_repair_v14",
        "decision": decision["decision"],
        "passed": decision["passed"],
        "authorization": decision["authorization"],
        "file_count": len(records),
        "files": records,
        "credential_scan": {"passed": True, "match_count": 0},
    }
    atomic_json(output_path, evidence)
    return evidence


__all__ = ["build_compact_rows", "build_evidence"]
