from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.glm53_user_eval.v12.evidence import build_evidence


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_evidence_hashes_primary_and_verifier_rows(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.json"
    decision = tmp_path / "decision.json"
    primary = tmp_path / "primary"
    verifier = tmp_path / "verifier"
    output = tmp_path / "evidence.json"
    _write(fixed, {"fixed": True})
    _write(
        decision,
        {
            "decision": "fact_extracted_semantic_validation_failed_stop_all_experiments",
            "passed": False,
            "authorization": {},
        },
    )
    _write(primary / "rows/a.json", {"row": "a"})
    _write(verifier / "rows/b.json", {"row": "b"})
    evidence = build_evidence(
        repo_root=tmp_path,
        fixed_paths=[fixed],
        primary_root=primary,
        verifier_root=verifier,
        decision_path=decision,
        output_path=output,
    )
    assert evidence["file_count"] == 4
    assert evidence["credential_scan"]["passed"] is True


def test_evidence_rejects_openrouter_secret(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.json"
    decision = tmp_path / "decision.json"
    primary = tmp_path / "primary"
    verifier = tmp_path / "verifier"
    _write(fixed, {"secret": "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"})
    _write(decision, {"decision": "stop", "passed": False, "authorization": {}})
    with pytest.raises(ValueError, match="credential-like"):
        build_evidence(
            repo_root=tmp_path,
            fixed_paths=[fixed],
            primary_root=primary,
            verifier_root=verifier,
            decision_path=decision,
            output_path=tmp_path / "evidence.json",
        )
