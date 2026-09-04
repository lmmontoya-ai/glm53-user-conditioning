from __future__ import annotations

from pathlib import Path

import pytest
from src.glm53_user_eval.v8.artifacts import atomic_json, atomic_jsonl, sha256_file
from src.glm53_user_eval.v8.audit import (
    build_final_evidence,
    build_manual_audit_packet,
    finalize_m8,
    validate_manual_audit,
)

ROOT = Path(__file__).resolve().parents[2]


def _proxy_row(group: str, codebook: str, index: int, arm: str = "baseline") -> dict:
    return {
        "sample_id": f"{group}-{codebook}-{index}",
        "arm_id": arm,
        "group": group,
        "persona_key": f"{group}-{index}",
        "pair_index": index,
        "stimulus_id": f"task-{index % 5}",
        "codebook_id": codebook,
        "expected_folded_confidence": 70.0,
        "allowed_mass": 0.9,
        "proxy_eligible": True,
    }


def _make_packet_artifacts(root: Path) -> None:
    baseline = [
        _proxy_row(group, codebook, index)
        for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
        for codebook in ("0", "1")
        for index in range(6)
    ]
    atomic_jsonl(root / "m5/rows/baseline.jsonl", baseline)
    for arm in ("baseline", "candidate", "opposite"):
        atomic_jsonl(
            root / f"m7/rows/{arm}.jsonl",
            [_proxy_row("famous_ai", str(index % 2), index, arm) for index in range(8)],
        )
    atomic_jsonl(
        root / "m7/rows/random_gaussian_00.jsonl",
        [_proxy_row("famous_ai", str(index % 2), index, "random_gaussian_00") for index in range(8)],
    )
    atomic_json(root / "m7/confirmation_report.json", {"passed": True})
    cache = [
        {
            "sample_id": "missing-first-turn",
            "proxy_eligible": False,
            "source_error": "missing_first_assistant_turn",
        }
    ]
    atomic_jsonl(root / "cache/v7_transcripts_25.jsonl", cache)


def test_build_manual_packet_has_frozen_quotas(tmp_path) -> None:
    _make_packet_artifacts(tmp_path)
    output = tmp_path / "m8/manual_audit_packet.json"
    packet = build_manual_audit_packet(
        repo_root=ROOT,
        artifact_root=tmp_path,
        seed=12,
        output_path=output,
    )
    assert packet["counts"] == {
        "local_proxy": 40,
        "eval_deploy": 20,
        "intervention": 24,
        "technical_errors": 1,
    }
    assert len(packet["required_review_ids"]) == 85
    assert len(set(packet["required_review_ids"])) == 85


def _packet_and_audit(tmp_path: Path, passed: bool = True) -> tuple[Path, Path]:
    packet_path = tmp_path / "packet.json"
    atomic_json(
        packet_path,
        {
            "schema_version": "glm53_v8_manual_audit_packet_v1",
            "required_review_ids": ["row-1", "row-2"],
        },
    )
    audit_path = tmp_path / "audit.json"
    atomic_json(
        audit_path,
        {
            "schema_version": "glm53_v8_manual_audit_v1",
            "packet_sha256": sha256_file(packet_path),
            "reviewer": "researcher",
            "completed_at_utc": "2026-08-30T12:00:00Z",
            "technical_errors_reviewed": True,
            "reviews": [
                {"audit_id": "row-1", "passed": passed, "notes": "checked"},
                {"audit_id": "row-2", "passed": True, "notes": "checked"},
            ],
        },
    )
    return packet_path, audit_path


def _pending_inputs(tmp_path: Path) -> dict:
    source = tmp_path / "m8/hardening_report.json"
    atomic_json(source, {"hardening": True})
    return {
        "hardening_report": {
            "path": source.as_posix(),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
    }


def test_manual_audit_validation_passes_exact_reviews(tmp_path) -> None:
    packet, audit = _packet_and_audit(tmp_path)
    result = validate_manual_audit(packet, audit)
    assert result["passed"] is True


def test_manual_audit_validation_fails_review(tmp_path) -> None:
    packet, audit = _packet_and_audit(tmp_path, passed=False)
    result = validate_manual_audit(packet, audit)
    assert result["passed"] is False
    assert result["checks"]["all_rows_pass"] is False


def test_finalize_m8_requires_manual_pass(tmp_path) -> None:
    packet, audit = _packet_and_audit(tmp_path, passed=False)
    atomic_json(
        tmp_path / "decisions/m8_pending_decision.json",
        {
            "gate": "M8",
            "checks": {"hardening": True, "manual_audit": False},
            "inputs": _pending_inputs(tmp_path),
            "estimates": {},
        },
    )
    decision = finalize_m8(
        artifact_root=tmp_path,
        packet_path=packet,
        audit_path=audit,
        output_path=tmp_path / "decisions/m8_decision.json",
    )
    assert decision["passed"] is False
    assert decision["checks"]["manual_audit"] is False


def test_finalize_m8_unlocks_only_after_manual_pass(tmp_path) -> None:
    packet, audit = _packet_and_audit(tmp_path)
    atomic_json(
        tmp_path / "decisions/m8_pending_decision.json",
        {
            "gate": "M8",
            "checks": {"hardening": True, "manual_audit": False},
            "inputs": _pending_inputs(tmp_path),
            "estimates": {},
        },
    )
    decision = finalize_m8(
        artifact_root=tmp_path,
        packet_path=packet,
        audit_path=audit,
        output_path=tmp_path / "decisions/m8_decision.json",
    )
    assert decision["passed"] is True
    assert decision["checks"]["manual_audit"] is True


def test_finalize_m8_rejects_tampered_pending_input(tmp_path) -> None:
    packet, audit = _packet_and_audit(tmp_path)
    inputs = _pending_inputs(tmp_path)
    atomic_json(
        tmp_path / "decisions/m8_pending_decision.json",
        {
            "gate": "M8",
            "checks": {"hardening": True, "manual_audit": False},
            "inputs": inputs,
            "estimates": {},
        },
    )
    atomic_json(tmp_path / "m8/hardening_report.json", {"hardening": False})
    with pytest.raises(ValueError, match="pending input lineage failed"):
        finalize_m8(
            artifact_root=tmp_path,
            packet_path=packet,
            audit_path=audit,
            output_path=tmp_path / "decisions/m8_decision.json",
        )


def test_final_evidence_refuses_nonterminal_study(tmp_path) -> None:
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text("project: v8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal study"):
        build_final_evidence(
            artifact_root=tmp_path,
            prereg_path=prereg,
            output_path=tmp_path / "final_evidence.json",
        )


def test_final_evidence_hashes_all_files(tmp_path) -> None:
    atomic_json(tmp_path / "decisions/m8_decision.json", {"passed": True})
    atomic_json(tmp_path / "m8/report.json", {"value": 1})
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text("project: v8\n", encoding="utf-8")
    output = tmp_path / "final_evidence.json"
    result = build_final_evidence(
        artifact_root=tmp_path,
        prereg_path=prereg,
        output_path=output,
    )
    assert result["file_count"] == 3
    assert {row["path"] for row in result["files"]} == {
        "decisions/m8_decision.json",
        "m8/report.json",
        "prereg.yaml",
    }
