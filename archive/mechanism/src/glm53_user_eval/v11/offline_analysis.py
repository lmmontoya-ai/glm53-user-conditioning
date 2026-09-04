"""Build the combined V11 offline analysis record.

The preregistration names ``offline_audit/analysis.json`` as the combined
analysis artifact. Earlier V11 commands wrote only component reports. This
module closes that bookkeeping gap without changing a gate or granting paid
compute. It accepts a pending human review as an explicit state and binds the
validated manual reports once they exist.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "glm53_v11_combined_offline_analysis_v1"
PROJECT_ID = "glm53_user_eval_source_instrument_v11"

_NO_PAID_AUTHORIZATION = {
    "new_glm_forwards": False,
    "runpod_compute": False,
    "source_activation_extraction": False,
    "user_recruitment": False,
    "steering": False,
}


class OfflineAnalysisError(ValueError):
    """Raised when an input is missing, malformed, or not hash-bound."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise OfflineAnalysisError(f"required {name} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OfflineAnalysisError(f"{name} is not a JSON object: {path}")
    return value


def _require_schema(value: Mapping[str, Any], expected: str, *, name: str) -> None:
    if value.get("schema_version") != expected:
        raise OfflineAnalysisError(f"{name} schema differs")


def _require_bool(value: Mapping[str, Any], field: str, *, name: str) -> bool:
    result = value.get(field)
    if not isinstance(result, bool):
        raise OfflineAnalysisError(f"{name}.{field} is not Boolean")
    return result


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


def _registered_output(
    *, repo_root: Path, prereg_path: Path
) -> tuple[dict[str, Any], Path]:
    if not prereg_path.is_file():
        raise OfflineAnalysisError(f"preregistration is absent: {prereg_path}")
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if not isinstance(prereg, dict):
        raise OfflineAnalysisError("preregistration is not a mapping")
    if prereg.get("schema_version") != "glm53_user_eval_v11_source_prereg_v1":
        raise OfflineAnalysisError("preregistration schema differs")
    if prereg.get("project_id") != PROJECT_ID:
        raise OfflineAnalysisError("preregistration project ID differs")
    offline_gate = prereg.get("offline_only_gate")
    if not isinstance(offline_gate, dict):
        raise OfflineAnalysisError("preregistration offline gate is absent")
    analysis_path = offline_gate.get("analysis_path")
    if not isinstance(analysis_path, str) or not analysis_path:
        raise OfflineAnalysisError("preregistration analysis path is absent")
    output = (repo_root / analysis_path).resolve()
    try:
        output.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise OfflineAnalysisError(
            "preregistered analysis path escapes the repository"
        ) from exc
    return prereg, output


def _bind_primary_manual_review(audit_root: Path) -> dict[str, Any]:
    packet_path = audit_root / "manual_packet.csv"
    lock_path = audit_root / "manual_packet_lock.json"
    manifest_path = audit_root / "manual_packet_manifest.json"
    for path, name in (
        (packet_path, "manual packet"),
        (lock_path, "manual packet lock"),
        (manifest_path, "manual packet manifest"),
    ):
        if not path.is_file():
            raise OfflineAnalysisError(f"required {name} is absent: {path}")
    lock = _load_json(lock_path, name="manual packet lock")
    manifest = _load_json(manifest_path, name="manual packet manifest")
    _require_schema(
        lock,
        "contrastive_prompts_v3_manual_audit_lock_v1",
        name="manual packet lock",
    )
    _require_schema(
        manifest,
        "contrastive_prompts_v3_manual_packet_v1",
        name="manual packet manifest",
    )
    packet_sha256 = sha256_file(packet_path)
    row_count = int(lock.get("row_count", -1))
    if row_count <= 0 or int(manifest.get("row_count", -1)) != row_count:
        raise OfflineAnalysisError("manual packet authorities disagree on row count")
    if lock.get("packet_sha256") != packet_sha256:
        raise OfflineAnalysisError("manual packet hash differs from its lock")
    if manifest.get("packet_sha256") != packet_sha256:
        raise OfflineAnalysisError("manual packet hash differs from its manifest")

    completed_path = audit_root / "manual_completed.csv"
    audit_path = audit_root / "manual_audit.json"
    completed_present = completed_path.is_file()
    audit_present = audit_path.is_file()
    if audit_present and not completed_present:
        raise OfflineAnalysisError("manual audit exists without manual_completed.csv")

    inputs = {
        "manual_packet": packet_sha256,
        "manual_packet_lock": sha256_file(lock_path),
        "manual_packet_manifest": sha256_file(manifest_path),
    }
    passed: bool | None = None
    failure_count: int | None = None
    reviewer_disagreement_count: int | None = None
    if not completed_present:
        status = "pending_two_human_review"
    elif not audit_present:
        status = "completed_pending_validation"
        inputs["manual_completed"] = sha256_file(completed_path)
    else:
        manual_audit = _load_json(audit_path, name="manual audit")
        _require_schema(
            manual_audit,
            "contrastive_prompts_v3_manual_audit_v1",
            name="manual audit",
        )
        passed = _require_bool(manual_audit, "passed", name="manual audit")
        completed_sha256 = sha256_file(completed_path)
        if manual_audit.get("completed_sha256") != completed_sha256:
            raise OfflineAnalysisError(
                "manual audit does not bind manual_completed.csv"
            )
        if manual_audit.get("lock_sha256") != sha256_file(lock_path):
            raise OfflineAnalysisError("manual audit does not bind its packet lock")
        if int(manual_audit.get("row_count", -1)) != row_count:
            raise OfflineAnalysisError("manual audit row count differs")
        status = "complete_passed" if passed else "complete_failed"
        failure_count = int(manual_audit.get("failure_count", -1))
        reviewer_disagreement_count = int(
            manual_audit.get("reviewer_disagreement_count", -1)
        )
        inputs |= {
            "manual_completed": completed_sha256,
            "manual_audit": sha256_file(audit_path),
        }

    return {
        "status": status,
        "gating": True,
        "row_count": row_count,
        "completed_review_present": completed_present,
        "audit_present": audit_present,
        "passed": passed,
        "failure_count": failure_count,
        "reviewer_disagreement_count": reviewer_disagreement_count,
        "inputs": inputs,
    }


def _bind_supplemental_review(audit_root: Path, samples_path: Path) -> dict[str, Any]:
    packet_path = audit_root / "supplemental_semantic_disagreements.csv"
    manifest_path = (
        audit_root / "supplemental_semantic_disagreements_manifest.json"
    )
    digest_path = (
        audit_root / "supplemental_semantic_disagreements_manifest.sha256"
    )
    for path, name in (
        (packet_path, "supplemental packet"),
        (manifest_path, "supplemental packet manifest"),
        (digest_path, "supplemental packet manifest digest"),
    ):
        if not path.is_file():
            raise OfflineAnalysisError(f"required {name} is absent: {path}")
    manifest = _load_json(manifest_path, name="supplemental packet manifest")
    _require_schema(
        manifest,
        "contrastive_prompts_v3_supplemental_manual_packet_v1",
        name="supplemental packet manifest",
    )
    if (
        manifest.get("scientific_role")
        != "supplemental_non_gating_human_review"
        or manifest.get("changes_preregistered_gate") is not False
        or manifest.get("changes_semantic_metrics") is not False
        or manifest.get("changes_paid_authorization") is not False
    ):
        raise OfflineAnalysisError("supplemental packet is not diagnostic-only")
    expected_digest = f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
    if digest_path.read_text(encoding="ascii") != expected_digest:
        raise OfflineAnalysisError("supplemental manifest digest differs")
    packet_sha256 = sha256_file(packet_path)
    if manifest.get("packet_sha256") != packet_sha256:
        raise OfflineAnalysisError("supplemental packet hash differs from its manifest")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise OfflineAnalysisError("supplemental packet source hashes are absent")
    if source_hashes.get("samples_jsonl") != sha256_file(samples_path):
        raise OfflineAnalysisError("supplemental packet does not bind samples.jsonl")

    completed_path = (
        audit_root / "supplemental_semantic_disagreements_completed.csv"
    )
    report_path = audit_root / "supplemental_semantic_review_report.json"
    report_manifest_path = (
        audit_root / "supplemental_semantic_review_manifest.json"
    )
    completed_present = completed_path.is_file()
    report_present = report_path.is_file()
    report_manifest_present = report_manifest_path.is_file()
    if report_present != report_manifest_present:
        raise OfflineAnalysisError(
            "supplemental report and report manifest must appear together"
        )
    if report_present and not completed_present:
        raise OfflineAnalysisError(
            "supplemental review report exists without its completed packet"
        )

    inputs = {
        "supplemental_packet": packet_sha256,
        "supplemental_packet_manifest": sha256_file(manifest_path),
        "supplemental_packet_manifest_digest": sha256_file(digest_path),
    }
    integrity_passed: bool | None = None
    if not completed_present:
        status = "pending_two_human_review"
    elif not report_present:
        status = "completed_pending_validation"
        inputs["supplemental_completed"] = sha256_file(completed_path)
    else:
        report = _load_json(report_path, name="supplemental review report")
        report_manifest = _load_json(
            report_manifest_path, name="supplemental review manifest"
        )
        _require_schema(
            report,
            "contrastive_prompts_v3_supplemental_manual_review_v1",
            name="supplemental review report",
        )
        _require_schema(
            report_manifest,
            "contrastive_prompts_v3_supplemental_review_manifest_v1",
            name="supplemental review manifest",
        )
        integrity_passed = _require_bool(
            report, "passed_review_integrity", name="supplemental review report"
        )
        if report_manifest.get("report_sha256") != sha256_file(report_path):
            raise OfflineAnalysisError(
                "supplemental review manifest does not bind its report"
            )
        if report_manifest.get("passed_review_integrity") is not integrity_passed:
            raise OfflineAnalysisError(
                "supplemental report and manifest disagree on integrity"
            )
        if (
            report.get("changes_preregistered_gate") is not False
            or report.get("changes_semantic_metrics") is not False
            or report.get("changes_paid_authorization") is not False
            or report_manifest.get("changes_preregistered_gate") is not False
            or report_manifest.get("changes_semantic_metrics") is not False
            or report_manifest.get("changes_paid_authorization") is not False
        ):
            raise OfflineAnalysisError(
                "supplemental completed review is not diagnostic-only"
            )
        completed_sha256 = sha256_file(completed_path)
        report_inputs = report.get("input_hashes")
        if not isinstance(report_inputs, dict) or report_inputs.get(
            "completed_packet"
        ) != completed_sha256:
            raise OfflineAnalysisError(
                "supplemental review report does not bind its completed packet"
            )
        status = (
            "complete_integrity_passed"
            if integrity_passed
            else "complete_integrity_failed"
        )
        inputs |= {
            "supplemental_completed": completed_sha256,
            "supplemental_review_report": sha256_file(report_path),
            "supplemental_review_manifest": sha256_file(report_manifest_path),
        }

    return {
        "status": status,
        "scientific_role": "supplemental_non_gating_diagnostic",
        "gating": False,
        "row_count": int(manifest.get("row_count", -1)),
        "completed_review_present": completed_present,
        "report_present": report_present,
        "passed_review_integrity": integrity_passed,
        "changes_preregistered_gate": False,
        "changes_semantic_metrics": False,
        "changes_paid_authorization": False,
        "inputs": inputs,
    }


def _combined_state(
    *,
    prerequisite_checks: Mapping[str, bool],
    semantic_passed: bool,
    manual: Mapping[str, Any],
) -> str:
    manual_status = str(manual["status"])
    manual_complete = manual_status in {"complete_passed", "complete_failed"}
    if not semantic_passed:
        suffix = "complete" if manual_complete else "pending"
        return f"semantic_validation_failed_manual_review_{suffix}"
    if not manual_complete:
        return "manual_review_pending"
    if manual.get("passed") is not True:
        return "manual_validation_failed"
    if not all(prerequisite_checks.values()):
        return "offline_component_failure"
    return "offline_analysis_complete_all_component_gates_passed"


def build_offline_analysis(
    *,
    repo_root: Path,
    prereg_path: Path,
    dataset_root: Path,
    audit_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically write the preregistered combined analysis.

    This artifact is descriptive. Its authorization fields remain false even
    when all component analyses pass. The separate text-decision command owns
    paid-compute authorization.
    """

    repo_root = repo_root.resolve()
    prereg_path = prereg_path.resolve()
    dataset_root = dataset_root.resolve()
    audit_root = audit_root.resolve()
    prereg, registered_output = _registered_output(
        repo_root=repo_root, prereg_path=prereg_path
    )
    output_path = (output_path or registered_output).resolve()
    if output_path != registered_output:
        raise OfflineAnalysisError(
            "combined analysis output differs from the preregistered path"
        )

    samples_path = dataset_root / "samples.jsonl"
    dataset_manifest_path = dataset_root / "manifest.json"
    tokenizer_path = dataset_root / "tokenizer_audit.json"
    structural_path = audit_root / "structural_audit.json"
    development_path = audit_root / "development_analysis.json"
    final_text_path = audit_root / "final_text_analysis.json"
    final_marker_path = audit_root / "FINAL_TEXT_HOLDOUT_OPENED.json"
    lexical_path = audit_root / "lexical_decision.json"
    semantic_path = audit_root / "semantic_validation.json"

    dataset_manifest = _load_json(dataset_manifest_path, name="dataset manifest")
    structural = _load_json(structural_path, name="structural audit")
    tokenizer = _load_json(tokenizer_path, name="tokenizer audit")
    development = _load_json(development_path, name="development analysis")
    final_text = _load_json(final_text_path, name="final text analysis")
    final_marker = _load_json(final_marker_path, name="final holdout marker")
    lexical = _load_json(lexical_path, name="lexical decision")
    semantic = _load_json(semantic_path, name="semantic validation")

    _require_schema(
        structural,
        "contrastive_prompts_v3_combined_structure_audit_v1",
        name="structural audit",
    )
    _require_schema(tokenizer, "glm53_v11_tokenizer_audit_v1", name="tokenizer audit")
    _require_schema(
        development,
        "contrastive_prompts_v3_development_text_audit_v1",
        name="development analysis",
    )
    _require_schema(
        final_text,
        "contrastive_prompts_v3_final_text_audit_v1",
        name="final text analysis",
    )
    _require_schema(
        final_marker,
        "glm53_v11_final_text_holdout_open_v1",
        name="final holdout marker",
    )
    _require_schema(
        lexical,
        "glm53_v11_lexical_gate_decision_v1",
        name="lexical decision",
    )
    _require_schema(
        semantic,
        "contrastive_prompts_v3_semantic_validation_v1",
        name="semantic validation",
    )

    expected_rows = int(prereg["dataset"]["expected"]["total_rows"])
    samples_sha256 = sha256_file(samples_path)
    if dataset_manifest.get("samples_sha256") != samples_sha256:
        raise OfflineAnalysisError("dataset manifest does not bind samples.jsonl")
    if structural.get("samples_sha256") != samples_sha256:
        raise OfflineAnalysisError("structural audit does not bind samples.jsonl")
    if tokenizer.get("samples_sha256") != samples_sha256:
        raise OfflineAnalysisError("tokenizer audit does not bind samples.jsonl")
    if int(tokenizer.get("row_count", -1)) != expected_rows:
        raise OfflineAnalysisError("tokenizer audit row count differs")
    primary_structure = structural.get("primary")
    if not isinstance(primary_structure, dict) or int(
        primary_structure.get("row_count", -1)
    ) != expected_rows:
        raise OfflineAnalysisError("structural audit row count differs")
    if int(semantic.get("row_count", -1)) != expected_rows:
        raise OfflineAnalysisError("semantic validation row count differs")

    development_lock = development.get("development_lock_sha256")
    if not isinstance(development_lock, str) or not development_lock:
        raise OfflineAnalysisError("development lock is absent")
    if final_text.get("development_lock_sha256") != development_lock:
        raise OfflineAnalysisError("final text analysis uses a different development lock")
    if (
        final_marker.get("opened_once") is not True
        or final_marker.get("status") != "complete"
        or final_marker.get("development_lock_sha256") != development_lock
        or final_marker.get("final_analysis_sha256") != sha256_file(final_text_path)
        or final_marker.get("samples_sha256") != samples_sha256
    ):
        raise OfflineAnalysisError("final holdout marker does not bind the completed run")
    lexical_inputs = lexical.get("inputs")
    expected_lexical_inputs = {
        "development": sha256_file(development_path),
        "final_text": sha256_file(final_text_path),
        "samples": samples_sha256,
        "tokenizer_audit": sha256_file(tokenizer_path),
    }
    if lexical_inputs != expected_lexical_inputs:
        raise OfflineAnalysisError("lexical decision input hashes differ")

    structural_passed = _require_bool(structural, "passed", name="structural audit")
    tokenizer_passed = _require_bool(tokenizer, "passed", name="tokenizer audit")
    lexical_passed = _require_bool(lexical, "passed", name="lexical decision")
    semantic_passed = _require_bool(
        semantic, "passed", name="semantic validation"
    )
    lexical_checks = lexical.get("checks")
    if not isinstance(lexical_checks, dict) or any(
        not isinstance(value, bool) for value in lexical_checks.values()
    ):
        raise OfflineAnalysisError("lexical decision checks are not Boolean")

    manual = _bind_primary_manual_review(audit_root)
    supplemental = _bind_supplemental_review(audit_root, samples_path)
    prerequisite_checks = {
        "structural_audit": structural_passed,
        "tokenizer_audit": tokenizer_passed,
        "lexical_gate": lexical_passed,
    }
    manual_complete = manual["status"] in {"complete_passed", "complete_failed"}
    passed = bool(
        all(prerequisite_checks.values())
        and semantic_passed
        and manual_complete
        and manual["passed"] is True
    )

    inputs = {
        "preregistration": sha256_file(prereg_path),
        "samples": samples_sha256,
        "dataset_manifest": sha256_file(dataset_manifest_path),
        "structural_audit": sha256_file(structural_path),
        "tokenizer_audit": sha256_file(tokenizer_path),
        "development_analysis": sha256_file(development_path),
        "final_text_analysis": sha256_file(final_text_path),
        "final_holdout_marker": sha256_file(final_marker_path),
        "lexical_decision": sha256_file(lexical_path),
        "semantic_validation": sha256_file(semantic_path),
    }
    inputs |= dict(manual["inputs"])
    inputs |= dict(supplemental["inputs"])

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project_id": PROJECT_ID,
        "analysis_role": "combined_offline_analysis_no_authorization",
        "registered_output": prereg["offline_only_gate"]["analysis_path"],
        "passed": passed,
        "state": _combined_state(
            prerequisite_checks=prerequisite_checks,
            semantic_passed=semantic_passed,
            manual=manual,
        ),
        "checks": prerequisite_checks
        | {
            "semantic_validation": semantic_passed,
            "manual_review_complete": manual_complete,
            "manual_validation": manual["passed"] is True,
        },
        "components": {
            "structure": {
                "schema_version": structural["schema_version"],
                "passed": structural_passed,
                "row_count": expected_rows,
            },
            "tokenizer": {
                "schema_version": tokenizer["schema_version"],
                "passed": tokenizer_passed,
                "row_count": int(tokenizer["row_count"]),
                "checked_pair_count": int(
                    tokenizer.get("pair_contract", {}).get("checked_pair_count", -1)
                ),
            },
            "development_text": {
                "schema_version": development["schema_version"],
                "development_lock_sha256": development_lock,
                "model_bundle_sha256": development.get("model_bundle_sha256"),
                "final_holdout_evaluated": development.get(
                    "final_holdout_evaluated"
                ),
                "fit_splits": development.get("fit_splits"),
                "selection_splits": development.get("selection_splits"),
                "post_selection_report_splits": development.get(
                    "post_selection_report_splits"
                ),
            },
            "final_lexical": {
                "schema_version": final_text["schema_version"],
                "evaluated_split": final_text.get("evaluated_split"),
                "row_count": final_text.get("row_count"),
                "selection_performed": final_text.get("selection_performed"),
                "holdout_opened_once": final_marker.get("opened_once"),
            },
            "lexical_decision": {
                "schema_version": lexical["schema_version"],
                "passed": lexical_passed,
                "decision": lexical.get("decision"),
                "checks": lexical_checks,
            },
            "semantic_validation": {
                "schema_version": semantic["schema_version"],
                "passed": semantic_passed,
                "row_count": int(semantic["row_count"]),
                "binary": semantic.get("binary"),
                "final_counterfactual": semantic.get("final_counterfactual"),
                "controls": semantic.get("controls"),
                "route_validation": semantic.get("route_validation"),
                "realized_cost_usd": semantic.get("realized_cost_usd"),
            },
            "primary_manual_review": manual,
            "supplemental_manual_review": supplemental,
        },
        "authorization": dict(_NO_PAID_AUTHORIZATION),
        "authorization_note": (
            "This analysis artifact cannot authorize paid work. The separate "
            "machine text-decision command owns that decision."
        ),
        "inputs": dict(sorted(inputs.items())),
    }
    if any(report["authorization"].values()):  # pragma: no cover, static guard
        raise AssertionError("combined offline analysis authorized paid work")
    _atomic_json(output_path, report)
    return report


__all__ = [
    "PROJECT_ID",
    "SCHEMA_VERSION",
    "OfflineAnalysisError",
    "build_offline_analysis",
    "sha256_file",
]
