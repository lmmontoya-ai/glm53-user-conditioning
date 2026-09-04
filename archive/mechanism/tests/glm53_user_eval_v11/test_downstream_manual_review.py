from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from src.glm53_user_eval.v11.downstream_manual_review import (
    build_downstream_review_template,
    validate_completed_downstream_review,
    validate_positive_claim_machine_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
ATTESTATION = "I personally reviewed this row without automated substitution."


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest() -> dict[str, object]:
    return {
        "manual_audit": {
            "proxy_random_rows": 40,
            "recruitment_random_rows": 32,
            "expected_technical_error_rows": 13,
            "pending_review_state": "scientific_decision_complete_manual_audit_pending",
            "completion_contract": {
                "schema_version": "glm53_v11_downstream_completed_review_v1",
                "human_attestation_exact": ATTESTATION,
                "required_checks": {
                    "proxy": ["proxy_check"],
                    "recruitment": ["recruitment_check"],
                    "technical_error": ["error_check"],
                },
            },
        }
    }


def _review_inputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, object]]:
    manifest = _manifest()
    packet = [
        {"surface": "proxy", "sample_id": f"proxy-{index}", "messages": []} for index in range(40)
    ] + [
        {
            "surface": "recruitment",
            "sample_id": f"recruitment-{index}",
            "messages": [],
        }
        for index in range(32)
    ]
    technical = [
        {
            "surface": "proxy_source_transcript",
            "sample_id": f"error-{index}",
            "source_error": "empty_v7_first_assistant_turn",
        }
        for index in range(13)
    ]
    paths = {
        name: tmp_path / name
        for name in (
            "manual_packet.jsonl",
            "technical_errors.jsonl",
            "manual_review_template.jsonl",
            "manual_audit_status.json",
            "completed_human_review.jsonl",
        )
    }
    _jsonl(paths["manual_packet.jsonl"], packet)
    _jsonl(paths["technical_errors.jsonl"], technical)
    template = build_downstream_review_template(packet, technical, manifest=manifest)
    _jsonl(paths["manual_review_template.jsonl"], template)
    status = {
        "schema_version": "glm53_v11_downstream_manual_packet_v1",
        "status": "scientific_decision_complete_manual_audit_pending",
        "human_review_completed": False,
        "final_claim_authorized": False,
        "proxy_rows": 40,
        "recruitment_rows": 32,
        "technical_error_rows": 13,
        "packet_sha256": _canonical(packet),
        "inputs": {
            "manual_packet": _sha256(paths["manual_packet.jsonl"]),
            "technical_errors": _sha256(paths["technical_errors.jsonl"]),
            "review_template": _sha256(paths["manual_review_template.jsonl"]),
        },
    }
    paths["manual_audit_status.json"].write_text(json.dumps(status), encoding="utf-8")
    reviewed_at = datetime.now(UTC).isoformat()
    completed = []
    for row in template:
        complete = dict(row)
        complete |= {
            "reviewer_id": "human-reviewer@example.org",
            "reviewer_is_human": True,
            "human_attestation": ATTESTATION,
            "reviewed_at_utc": reviewed_at,
            "passed": True,
            "checks": {name: True for name in row["checks"]},
        }
        completed.append(complete)
    _jsonl(paths["completed_human_review.jsonl"], completed)
    return paths, manifest


def test_exact_85_row_human_review_passes(tmp_path: Path) -> None:
    paths, manifest = _review_inputs(tmp_path)
    report = validate_completed_downstream_review(
        completed_path=paths["completed_human_review.jsonl"],
        packet_path=paths["manual_packet.jsonl"],
        technical_errors_path=paths["technical_errors.jsonl"],
        template_path=paths["manual_review_template.jsonl"],
        status_path=paths["manual_audit_status.json"],
        manifest=manifest,
    )
    assert report["passed"] is True
    assert report["human_review_completed"] is True
    assert report["final_claim_authorized"] is False
    assert report["review_counts"] == {
        "proxy": 40,
        "recruitment": 32,
        "technical_errors": 13,
        "total": 85,
    }


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "not_human", "string_pass"])
def test_incomplete_or_nonhuman_review_fails_closed(tmp_path: Path, mutation: str) -> None:
    paths, manifest = _review_inputs(tmp_path)
    rows = [
        json.loads(line) for line in paths["completed_human_review.jsonl"].read_text().splitlines()
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "not_human":
        rows[0]["reviewer_is_human"] = False
    else:
        rows[0]["passed"] = "yes"
    _jsonl(paths["completed_human_review.jsonl"], rows)
    with pytest.raises((TypeError, ValueError)):
        validate_completed_downstream_review(
            completed_path=paths["completed_human_review.jsonl"],
            packet_path=paths["manual_packet.jsonl"],
            technical_errors_path=paths["technical_errors.jsonl"],
            template_path=paths["manual_review_template.jsonl"],
            status_path=paths["manual_audit_status.json"],
            manifest=manifest,
        )


def test_human_failure_completes_review_but_blocks_claim(tmp_path: Path) -> None:
    paths, manifest = _review_inputs(tmp_path)
    rows = [
        json.loads(line) for line in paths["completed_human_review.jsonl"].read_text().splitlines()
    ]
    rows[0]["passed"] = False
    rows[0]["checks"] = {name: False for name in rows[0]["checks"]}
    rows[0]["notes"] = "Persona text does not match the frozen roster."
    _jsonl(paths["completed_human_review.jsonl"], rows)
    report = validate_completed_downstream_review(
        completed_path=paths["completed_human_review.jsonl"],
        packet_path=paths["manual_packet.jsonl"],
        technical_errors_path=paths["technical_errors.jsonl"],
        template_path=paths["manual_review_template.jsonl"],
        status_path=paths["manual_audit_status.json"],
        manifest=manifest,
    )
    assert report["human_review_completed"] is True
    assert report["passed"] is False
    assert report["final_claim_authorized"] is False


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_positive_machine_claim_requires_all_hash_bound_verifiers(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "source_decision.json",
            "source_verification.json",
            "proxy_decision.json",
            "proxy_verification.json",
            "recruitment_decision.json",
            "recruitment_verification.json",
            "downstream_manifest.json",
            "preflight.json",
        )
    }
    for name in ("downstream_manifest.json", "preflight.json"):
        _write_json(paths[name], {"name": name})
    for name in (
        "source_verification.json",
        "proxy_verification.json",
        "recruitment_verification.json",
    ):
        _write_json(paths[name], {"passed": True, "scientific_gate_would_pass": True})
    _write_json(
        paths["source_decision.json"],
        {
            "passed": True,
            "decision": "source_instrument_valid_for_frozen_transfer",
            "authorization": {"local_proxy_parity": True},
            "checks": {"independent_verification": True},
            "inputs": {"verification": _sha256(paths["source_verification.json"])},
        },
    )
    common_inputs = {
        "downstream_manifest": _sha256(paths["downstream_manifest.json"]),
        "downstream_preflight": _sha256(paths["preflight.json"]),
        "source_decision": _sha256(paths["source_decision.json"]),
    }
    _write_json(
        paths["proxy_decision.json"],
        {
            "passed": True,
            "scientific_gate_passed": True,
            "decision": "local_proxy_parity_pass_user_recruitment_unlocked",
            "authorization": {"user_recruitment": True},
            "checks": {"independent_verification": True},
            "inputs": common_inputs | {"verification": _sha256(paths["proxy_verification.json"])},
        },
    )
    _write_json(
        paths["recruitment_decision.json"],
        {
            "passed": True,
            "scientific_gate_passed": True,
            "decision": "frozen_eval_readout_recruited_by_ai_specific_user_interaction",
            "checks": {"independent_verification": True},
            "inputs": common_inputs
            | {
                "verification": _sha256(paths["recruitment_verification.json"]),
                "proxy_decision": _sha256(paths["proxy_decision.json"]),
            },
        },
    )
    report = validate_positive_claim_machine_artifacts(
        source_decision_path=paths["source_decision.json"],
        source_verification_path=paths["source_verification.json"],
        proxy_decision_path=paths["proxy_decision.json"],
        proxy_verification_path=paths["proxy_verification.json"],
        recruitment_decision_path=paths["recruitment_decision.json"],
        recruitment_verification_path=paths["recruitment_verification.json"],
        downstream_manifest_path=paths["downstream_manifest.json"],
        downstream_preflight_path=paths["preflight.json"],
    )
    assert report["passed"] is True
    proxy = json.loads(paths["proxy_decision.json"].read_text())
    proxy["inputs"]["verification"] = "0" * 64
    _write_json(paths["proxy_decision.json"], proxy)
    report = validate_positive_claim_machine_artifacts(
        source_decision_path=paths["source_decision.json"],
        source_verification_path=paths["source_verification.json"],
        proxy_decision_path=paths["proxy_decision.json"],
        proxy_verification_path=paths["proxy_verification.json"],
        recruitment_decision_path=paths["recruitment_decision.json"],
        recruitment_verification_path=paths["recruitment_verification.json"],
        downstream_manifest_path=paths["downstream_manifest.json"],
        downstream_preflight_path=paths["preflight.json"],
    )
    assert report["passed"] is False


def test_downstream_claim_cli_requires_completed_human_file() -> None:
    source = (ROOT / "pipelines/glm53_user_eval/v11/run.py").read_text(encoding="utf-8")
    assert '"authorize-downstream-claim": command_authorize_downstream_claim' in source
    assert 'parser.add_argument("--completed-downstream-review", type=Path)' in source
    assert "--completed-downstream-review is required" in source
    assert '"final_claim": True' in source
    assert '"early_cot": False' in source
    assert '"steering": False' in source
