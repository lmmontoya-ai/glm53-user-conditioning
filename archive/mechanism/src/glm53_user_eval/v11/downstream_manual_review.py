"""Human-review gate for the positive V11 downstream claim.

The paid runner writes immutable score-blind packets. This module validates a
separate completed review and the machine artifacts. It never creates reviewer
answers and never mutates the scientific decisions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVIEW_FIELDS = {
    "schema_version",
    "surface",
    "sample_id",
    "source_row_sha256",
    "reviewer_id",
    "reviewer_is_human",
    "human_attestation",
    "reviewed_at_utc",
    "passed",
    "checks",
    "notes",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def build_downstream_review_template(
    packet_rows: list[dict[str, Any]],
    technical_error_rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = manifest["manual_audit"]["completion_contract"]
    rows: list[dict[str, Any]] = []
    for source in [*packet_rows, *technical_error_rows]:
        surface = str(source["surface"])
        review_kind = "technical_error" if surface == "proxy_source_transcript" else surface
        if review_kind not in contract["required_checks"]:
            raise ValueError(f"unexpected downstream review surface: {surface}")
        rows.append(
            {
                "schema_version": contract["schema_version"],
                "surface": surface,
                "sample_id": str(source["sample_id"]),
                "source_row_sha256": _canonical_sha256(source),
                "reviewer_id": "",
                "reviewer_is_human": None,
                "human_attestation": "",
                "reviewed_at_utc": "",
                "passed": None,
                "checks": {name: None for name in contract["required_checks"][review_kind]},
                "notes": "",
            }
        )
    keys = [(row["surface"], row["sample_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("downstream review template has duplicate scientific keys")
    return rows


def validate_completed_downstream_review(
    *,
    completed_path: Path,
    packet_path: Path,
    technical_errors_path: Path,
    template_path: Path,
    status_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    config = manifest["manual_audit"]
    contract = config["completion_contract"]
    packet = _load_jsonl(packet_path)
    technical = _load_jsonl(technical_errors_path)
    template = _load_jsonl(template_path)
    expected_template = build_downstream_review_template(packet, technical, manifest=manifest)
    if template != expected_template:
        raise ValueError("downstream review template differs from immutable source rows")
    packet_counts = {
        "proxy": sum(row.get("surface") == "proxy" for row in packet),
        "recruitment": sum(row.get("surface") == "recruitment" for row in packet),
        "technical_errors": len(technical),
    }
    expected_counts = {
        "proxy": int(config["proxy_random_rows"]),
        "recruitment": int(config["recruitment_random_rows"]),
        "technical_errors": int(config["expected_technical_error_rows"]),
    }
    if packet_counts != expected_counts:
        raise ValueError("downstream manual source quotas differ from the frozen contract")
    if any(row.get("surface") != "proxy_source_transcript" for row in technical):
        raise ValueError("technical-error packet contains another surface")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("schema_version") != "glm53_v11_downstream_manual_packet_v1"
        or status.get("status") != config["pending_review_state"]
        or status.get("human_review_completed") is not False
        or status.get("final_claim_authorized") is not False
        or status.get("proxy_rows") != expected_counts["proxy"]
        or status.get("recruitment_rows") != expected_counts["recruitment"]
        or status.get("technical_error_rows") != expected_counts["technical_errors"]
        or status.get("packet_sha256") != _canonical_sha256(packet)
    ):
        raise ValueError("pending downstream manual status differs from the frozen contract")
    expected_status_inputs = {
        "manual_packet": _file_sha256(packet_path),
        "technical_errors": _file_sha256(technical_errors_path),
        "review_template": _file_sha256(template_path),
    }
    if status.get("inputs") != expected_status_inputs:
        raise ValueError("pending downstream manual status has stale input hashes")

    completed = _load_jsonl(completed_path)
    expected = {(row["surface"], row["sample_id"]): row for row in expected_template}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    reviewers: set[str] = set()
    failed_keys: list[str] = []
    now = datetime.now(UTC)
    for row in completed:
        if set(row) != REVIEW_FIELDS:
            raise ValueError("completed downstream review has an unexpected field set")
        key = (str(row["surface"]), str(row["sample_id"]))
        if key in observed:
            raise ValueError("completed downstream review has duplicate keys")
        if key not in expected:
            raise ValueError("completed downstream review names an unknown row")
        source = expected[key]
        if (
            row["schema_version"] != contract["schema_version"]
            or row["source_row_sha256"] != source["source_row_sha256"]
        ):
            raise ValueError("completed downstream review is not bound to its source row")
        reviewer = row["reviewer_id"]
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("completed downstream review lacks a human reviewer identity")
        if row["reviewer_is_human"] is not True:
            raise ValueError("completed downstream review lacks the human Boolean attestation")
        if row["human_attestation"] != contract["human_attestation_exact"]:
            raise ValueError("completed downstream review lacks the exact human attestation")
        try:
            reviewed_at = datetime.fromisoformat(str(row["reviewed_at_utc"]))
        except ValueError as error:
            raise ValueError("completed downstream review has an invalid timestamp") from error
        if reviewed_at.tzinfo is None or reviewed_at.astimezone(UTC) > now:
            raise ValueError("completed downstream review timestamp is naive or in the future")
        if not isinstance(row["passed"], bool):
            raise TypeError("completed downstream review pass field is not a Boolean")
        review_kind = "technical_error" if key[0] == "proxy_source_transcript" else key[0]
        required_checks = contract["required_checks"][review_kind]
        checks = row["checks"]
        if not isinstance(checks, dict) or set(checks) != set(required_checks):
            raise ValueError("completed downstream review has the wrong checklist")
        if any(not isinstance(value, bool) for value in checks.values()):
            raise ValueError("completed downstream checklist values are not Booleans")
        checks_pass = all(checks.values())
        if row["passed"] is not checks_pass:
            raise ValueError("completed downstream row pass state disagrees with its checklist")
        if not isinstance(row["notes"], str):
            raise TypeError("completed downstream review notes are not text")
        if not row["passed"] and not row["notes"].strip():
            raise ValueError("a failed downstream review row requires notes")
        if not row["passed"]:
            failed_keys.append(f"{key[0]}|{key[1]}")
        reviewers.add(reviewer.strip())
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError("completed downstream review does not cover every frozen row")
    passed = not failed_keys
    return {
        "schema_version": "glm53_v11_downstream_manual_review_decision_v1",
        "passed": passed,
        "human_review_completed": True,
        "all_reviewed_rows_passed": passed,
        "final_claim_authorized": False,
        "review_counts": packet_counts | {"total": sum(packet_counts.values())},
        "reviewer_ids": sorted(reviewers),
        "failed_keys": failed_keys,
        "inputs": expected_status_inputs
        | {
            "pending_status": _file_sha256(status_path),
            "completed_review": _file_sha256(completed_path),
        },
    }


def validate_positive_claim_machine_artifacts(
    *,
    source_decision_path: Path,
    source_verification_path: Path,
    proxy_decision_path: Path,
    proxy_verification_path: Path,
    recruitment_decision_path: Path,
    recruitment_verification_path: Path,
    downstream_manifest_path: Path,
    downstream_preflight_path: Path,
) -> dict[str, Any]:
    paths = {
        "source_decision": source_decision_path,
        "source_verification": source_verification_path,
        "proxy_decision": proxy_decision_path,
        "proxy_verification": proxy_verification_path,
        "recruitment_decision": recruitment_decision_path,
        "recruitment_verification": recruitment_verification_path,
        "downstream_manifest": downstream_manifest_path,
        "downstream_preflight": downstream_preflight_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("positive downstream claim lacks a required machine artifact")
    source = json.loads(source_decision_path.read_text(encoding="utf-8"))
    source_verification = json.loads(source_verification_path.read_text(encoding="utf-8"))
    proxy = json.loads(proxy_decision_path.read_text(encoding="utf-8"))
    proxy_verification = json.loads(proxy_verification_path.read_text(encoding="utf-8"))
    recruitment = json.loads(recruitment_decision_path.read_text(encoding="utf-8"))
    recruitment_verification = json.loads(recruitment_verification_path.read_text(encoding="utf-8"))
    checks = {
        "source_positive": source.get("passed") is True
        and source.get("decision") == "source_instrument_valid_for_frozen_transfer"
        and source.get("authorization", {}).get("local_proxy_parity") is True,
        "source_verified": source_verification.get("passed") is True
        and source_verification.get("scientific_gate_would_pass") is True
        and source.get("checks", {}).get("independent_verification") is True,
        "source_verification_bound": source.get("inputs", {}).get("verification")
        == _file_sha256(source_verification_path),
        "proxy_positive": proxy.get("passed") is True
        and proxy.get("decision") == "local_proxy_parity_pass_user_recruitment_unlocked"
        and proxy.get("scientific_gate_passed") is True
        and proxy.get("authorization", {}).get("user_recruitment") is True,
        "proxy_verified": proxy_verification.get("passed") is True
        and proxy_verification.get("scientific_gate_would_pass") is True
        and proxy.get("checks", {}).get("independent_verification") is True,
        "proxy_verification_bound": proxy.get("inputs", {}).get("verification")
        == _file_sha256(proxy_verification_path),
        "proxy_source_decision_bound": proxy.get("inputs", {}).get("source_decision")
        == _file_sha256(source_decision_path),
        "recruitment_positive": recruitment.get("passed") is True
        and recruitment.get("decision")
        == "frozen_eval_readout_recruited_by_ai_specific_user_interaction"
        and recruitment.get("scientific_gate_passed") is True,
        "recruitment_verified": recruitment_verification.get("passed") is True
        and recruitment_verification.get("scientific_gate_would_pass") is True
        and recruitment.get("checks", {}).get("independent_verification") is True,
        "recruitment_verification_bound": recruitment.get("inputs", {}).get("verification")
        == _file_sha256(recruitment_verification_path),
        "recruitment_source_decision_bound": recruitment.get("inputs", {}).get(
            "source_decision"
        )
        == _file_sha256(source_decision_path),
        "recruitment_proxy_decision_bound": recruitment.get("inputs", {}).get(
            "proxy_decision"
        )
        == _file_sha256(proxy_decision_path),
        "proxy_manifest_bound": proxy.get("inputs", {}).get("downstream_manifest")
        == _file_sha256(downstream_manifest_path),
        "proxy_preflight_bound": proxy.get("inputs", {}).get("downstream_preflight")
        == _file_sha256(downstream_preflight_path),
        "recruitment_manifest_bound": recruitment.get("inputs", {}).get("downstream_manifest")
        == _file_sha256(downstream_manifest_path),
        "recruitment_preflight_bound": recruitment.get("inputs", {}).get("downstream_preflight")
        == _file_sha256(downstream_preflight_path),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "inputs": {name: _file_sha256(path) for name, path in paths.items()},
    }


__all__ = [
    "build_downstream_review_template",
    "validate_completed_downstream_review",
    "validate_positive_claim_machine_artifacts",
]
