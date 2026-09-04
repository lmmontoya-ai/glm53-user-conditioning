"""Fail-closed administrative closure for the stopped V11 text gate."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "glm53_v11_administrative_closure_decision_v1"
EVIDENCE_SCHEMA_VERSION = "glm53_v11_administrative_closure_evidence_v1"
PROJECT_ID = "glm53_user_eval_source_instrument_v11"
TERMINAL_DECISION = "semantic_validation_failed_manual_human_review_unavailable"

NO_AUTHORIZATION = {
    "new_glm_forwards": False,
    "runpod_compute": False,
    "source_activation_extraction": False,
    "steering": False,
    "user_recruitment": False,
}


class AdministrativeClosureError(ValueError):
    """Raised when the closure inputs differ from the frozen failed state."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise AdministrativeClosureError(f"required {name} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdministrativeClosureError(f"{name} is not a JSON object")
    return value


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


def _resolve_locked_path(repo_root: Path, record: Mapping[str, Any]) -> Path:
    relative = record.get("path")
    expected = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise AdministrativeClosureError("amendment lock is incomplete")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise AdministrativeClosureError("amendment lock escapes repository") from exc
    if sha256_file(path) != expected:
        raise AdministrativeClosureError(f"locked artifact hash differs: {relative}")
    return path


def close_v11(
    *,
    repo_root: Path,
    amendment_path: Path,
    diagnostic_path: Path,
    decision_path: Path,
    evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the terminal failed decision and its immutable evidence record."""

    repo_root = repo_root.resolve()
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    if not isinstance(amendment, dict):
        raise AdministrativeClosureError("amendment is not a mapping")
    if amendment.get("schema_version") != (
        "glm53_user_eval_v11_administrative_closure_amendment_v1"
    ):
        raise AdministrativeClosureError("amendment schema differs")
    if amendment.get("project_id") != PROJECT_ID:
        raise AdministrativeClosureError("amendment project differs")
    if amendment.get("terminal_decision") != TERMINAL_DECISION:
        raise AdministrativeClosureError("amendment terminal decision differs")
    if amendment.get("human_review_requirement") != "unmet":
        raise AdministrativeClosureError("amendment misstates human review")
    if amendment.get("paid_compute_authorized") is not False:
        raise AdministrativeClosureError("amendment authorizes paid compute")

    locks = amendment.get("locks")
    if not isinstance(locks, dict) or set(locks) != {
        "preregistration",
        "combined_analysis",
        "semantic_validation",
        "semantic_stop_evidence",
        "semantic_stop_summary",
    }:
        raise AdministrativeClosureError("amendment locks differ")
    locked_paths = {
        name: _resolve_locked_path(repo_root, record)
        for name, record in locks.items()
    }
    analysis = _load_json(locked_paths["combined_analysis"], name="analysis")
    semantic = _load_json(
        locked_paths["semantic_validation"], name="semantic validation"
    )
    diagnostic = _load_json(diagnostic_path, name="AI diagnostic summary")

    if analysis.get("state") != "semantic_validation_failed_manual_review_pending":
        raise AdministrativeClosureError("combined analysis is not at the failed pending state")
    if analysis.get("passed") is not False or analysis.get("authorization") != NO_AUTHORIZATION:
        raise AdministrativeClosureError("combined analysis does not fail closed")
    controls = semantic.get("controls") or {}
    if (
        semantic.get("passed") is not False
        or controls.get("acceptable") != 77
        or controls.get("count") != 96
        or controls.get("acceptance_rate") != 77 / 96
        or controls.get("threshold") != 0.90
    ):
        raise AdministrativeClosureError("semantic failure metrics differ")

    if (
        diagnostic.get("schema_version")
        != "glm53_v11_ai_diagnostic_review_summary_v1"
        or diagnostic.get("reviewer_type") != "ai"
        or diagnostic.get("scientific_role") != "nonhuman_non_gating_diagnostic"
        or diagnostic.get("human_review_requirement_satisfied") is not False
        or diagnostic.get("official_human_merge_eligible") is not False
        or diagnostic.get("authorization") != NO_AUTHORIZATION
    ):
        raise AdministrativeClosureError("AI diagnostic role or authorization differs")
    primary = diagnostic.get("primary") or {}
    supplemental = diagnostic.get("supplemental") or {}
    if (
        primary.get("row_count") != 128
        or sum((primary.get("label_counts") or {}).values()) != 128
        or supplemental.get("row_count") != 6
        or sum((supplemental.get("label_counts") or {}).values()) != 6
    ):
        raise AdministrativeClosureError("AI diagnostic counts differ")

    input_hashes = {
        "amendment": sha256_file(amendment_path),
        "diagnostic_summary": sha256_file(diagnostic_path),
        **{name: sha256_file(path) for name, path in locked_paths.items()},
    }
    decision: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "passed": False,
        "decision": TERMINAL_DECISION,
        "automatic_gate": "failed",
        "automatic_failure": {
            "control_acceptance": 77 / 96,
            "control_acceptance_threshold": 0.90,
            "failed_check": "semantic_control_acceptance",
        },
        "human_review_requirement": "unmet",
        "ai_diagnostic_review": {
            "status": "complete_nonhuman_aggregate_only",
            "scientific_role": "nonhuman_non_gating_diagnostic",
            "primary_reported_agreement": "128/128",
            "row_level_artifacts_hash_verified": False,
        },
        "authorization": NO_AUTHORIZATION,
        "v12": {
            "may_be_preregistered": True,
            "may_reuse_frozen_v11_text": True,
            "must_not_claim_v11_passed": True,
            "must_pass_a_new_automatic_gate_before_paid_compute": True,
        },
        "inputs": input_hashes,
    }
    _atomic_json(decision_path, decision)
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "terminal": True,
        "decision": TERMINAL_DECISION,
        "decision_path": decision_path.resolve().relative_to(repo_root).as_posix(),
        "decision_sha256": sha256_file(decision_path),
        "inputs": input_hashes,
        "authorization": NO_AUTHORIZATION,
    }
    _atomic_json(evidence_path, evidence)
    return decision, evidence


__all__ = [
    "NO_AUTHORIZATION",
    "TERMINAL_DECISION",
    "AdministrativeClosureError",
    "close_v11",
    "sha256_file",
]
