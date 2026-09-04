"""Score-blind audit packets, manual review validation, and final evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .datasets import load_eval_surface
from .supervisor import decision_payload


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rank(seed: int, section: str, row: dict[str, Any]) -> str:
    identity = "|".join(
        str(row.get(key, ""))
        for key in ("sample_id", "arm_id", "group", "codebook_id", "split")
    )
    return hashlib.sha256(f"glm53-v8-audit|{seed}|{section}|{identity}".encode()).hexdigest()


def _audit_row(section: str, row: dict[str, Any]) -> dict[str, Any]:
    arm = str(row.get("arm_id", "none"))
    audit_id = f"{section}|{row['sample_id']}|{arm}"
    return {"audit_id": audit_id, "section": section, **row}


def _select(rows: list[dict[str, Any]], count: int, seed: int, section: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"{section} needs {count} rows, found {len(rows)}")
    return sorted(rows, key=lambda row: _rank(seed, section, row))[:count]


def _technical_errors(cache_rows: list[dict[str, Any]], scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = [row for row in cache_rows if not row.get("proxy_eligible", True)]
    for row in scored:
        numeric = (row.get("expected_folded_confidence"), row.get("allowed_mass"))
        invalid_numeric = any(
            value is not None and not math.isfinite(float(value)) for value in numeric
        )
        if row.get("error") or invalid_numeric:
            errors.append(row)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in errors:
        unique[(str(row.get("sample_id")), str(row.get("arm_id", "none")))] = row
    return [unique[key] for key in sorted(unique)]


def build_manual_audit_packet(
    *,
    repo_root: Path,
    artifact_root: Path,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    """Build the frozen post-result review packet without selecting on scores."""

    proxy_rows = _rows(artifact_root / "m5/rows/baseline.jsonl")
    proxy_selected: list[dict[str, Any]] = []
    for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop"):
        for codebook in ("0", "1"):
            stratum = [
                row
                for row in proxy_rows
                if row["group"] == group and str(row["codebook_id"]) == codebook
            ]
            proxy_selected.extend(_select(stratum, 5, seed, f"proxy-{group}-{codebook}"))

    eval_rows = load_eval_surface(repo_root / "artifacts/datasets/contrastive_prompts_v2")
    eval_selected: list[dict[str, Any]] = []
    splits = ("train", "val", "test", "holdout_realism", "holdout_hardneg")
    for split in splits:
        eval_selected.extend(
            _select([row for row in eval_rows if row["split"] == split], 4, seed, f"eval-{split}")
        )

    intervention_selected: list[dict[str, Any]] = []
    for category in ("baseline", "candidate", "opposite"):
        intervention_selected.extend(
            _select(
                _rows(artifact_root / f"m7/rows/{category}.jsonl"),
                6,
                seed,
                f"intervention-{category}",
            )
        )
    random_rows = [
        row
        for path in sorted((artifact_root / "m7/rows").glob("random_*.jsonl"))
        for row in _rows(path)
    ]
    intervention_selected.extend(
        _select(random_rows, 6, seed, "intervention-random")
    )

    cache_rows = _rows(artifact_root / "cache/v7_transcripts_25.jsonl")
    all_scored = proxy_rows + [
        row
        for path in sorted((artifact_root / "m7/rows").glob("*.jsonl"))
        for row in _rows(path)
    ]
    technical = _technical_errors(cache_rows, all_scored)

    sections = {
        "local_proxy": [_audit_row("local_proxy", row) for row in proxy_selected],
        "eval_deploy": [_audit_row("eval_deploy", row) for row in eval_selected],
        "intervention": [_audit_row("intervention", row) for row in intervention_selected],
        "technical_errors": [_audit_row("technical_errors", row) for row in technical],
    }
    review_ids = [row["audit_id"] for rows in sections.values() for row in rows]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("manual audit packet contains duplicate audit IDs")
    packet = {
        "schema_version": "glm53_v8_manual_audit_packet_v1",
        "selection_seed": seed,
        "selection_is_score_blind": True,
        "counts": {key: len(value) for key, value in sections.items()},
        "required_review_ids": sorted(review_ids),
        "sections": sections,
        "source_hashes": {
            "m5_baseline": sha256_file(artifact_root / "m5/rows/baseline.jsonl"),
            "m7_confirmation": sha256_file(artifact_root / "m7/confirmation_report.json"),
            "transcript_cache": sha256_file(
                artifact_root / "cache/v7_transcripts_25.jsonl"
            ),
        },
    }
    atomic_json(output_path, packet)
    return packet


def validate_manual_audit(packet_path: Path, audit_path: Path) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = set(packet["required_review_ids"])
    reviews = audit.get("reviews", [])
    observed = [str(row.get("audit_id")) for row in reviews]
    checks = {
        "schema": audit.get("schema_version") == "glm53_v8_manual_audit_v1",
        "packet_hash": audit.get("packet_sha256") == sha256_file(packet_path),
        "reviewer": bool(str(audit.get("reviewer", "")).strip()),
        "completed_at": bool(str(audit.get("completed_at_utc", "")).strip()),
        "review_ids_exact": set(observed) == required and len(observed) == len(required),
        "all_rows_pass": bool(reviews)
        and all(row.get("passed") is True for row in reviews),
        "technical_errors_reviewed": audit.get("technical_errors_reviewed") is True,
    }
    return {
        "schema_version": "glm53_v8_manual_audit_validation_v1",
        "packet_sha256": sha256_file(packet_path),
        "audit_sha256": sha256_file(audit_path),
        "required_reviews": len(required),
        "observed_reviews": len(observed),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _resolve_pending_input(artifact_root: Path, name: str, stored_path: str) -> Path:
    source = Path(stored_path)
    if source.is_file():
        return source
    fixed = {
        "hardening_report": artifact_root / "m8/hardening_report.json",
        "independent_verification": artifact_root / "m8/independent_verification.json",
        "geometry_report": artifact_root / "m8/geometry_report.json",
        "m7_decision": artifact_root / "decisions/m7_decision.json",
    }
    if name in fixed:
        return fixed[name]
    if name.startswith("hardening_arm_"):
        return artifact_root / f"m8/rows/{name.removeprefix('hardening_arm_')}.jsonl"
    return source


def verify_pending_input_lineage(
    artifact_root: Path, inputs: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    if not inputs:
        raise ValueError("M8 pending decision has no hashed inputs")
    checks = {}
    for name, record in inputs.items():
        source = _resolve_pending_input(artifact_root, name, str(record["path"]))
        checks[name] = (
            source.is_file()
            and source.stat().st_size == int(record["size_bytes"])
            and sha256_file(source) == record["sha256"]
        )
    return checks


def finalize_m8(
    *,
    artifact_root: Path,
    packet_path: Path,
    audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    pending_path = artifact_root / "decisions/m8_pending_decision.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    if pending.get("gate") != "M8" or pending.get("checks", {}).get("manual_audit") is not False:
        raise ValueError("M8 pending decision is missing the manual-audit lock")
    lineage = verify_pending_input_lineage(artifact_root, pending.get("inputs", {}))
    if not all(lineage.values()):
        raise ValueError(f"M8 pending input lineage failed: {lineage}")
    validation = validate_manual_audit(packet_path, audit_path)
    validation["pending_input_checks"] = lineage
    validation["pending_inputs_passed"] = True
    validation_path = artifact_root / "m8/manual_audit_validation.json"
    atomic_json(validation_path, validation)
    checks = dict(pending["checks"])
    checks["manual_audit"] = validation["passed"]
    checks["pending_input_hashes"] = True
    inputs = dict(pending["inputs"])
    inputs.update(
        {
            "manual_audit_packet": {
                "path": packet_path.as_posix(),
                "sha256": sha256_file(packet_path),
                "size_bytes": packet_path.stat().st_size,
            },
            "manual_audit": {
                "path": audit_path.as_posix(),
                "sha256": sha256_file(audit_path),
                "size_bytes": audit_path.stat().st_size,
            },
            "manual_audit_validation": {
                "path": validation_path.as_posix(),
                "sha256": sha256_file(validation_path),
                "size_bytes": validation_path.stat().st_size,
            },
            "pending_decision": {
                "path": pending_path.as_posix(),
                "sha256": sha256_file(pending_path),
                "size_bytes": pending_path.stat().st_size,
            },
        }
    )
    estimates = dict(pending.get("estimates", {})) | {
        "manual_audit_validation": validation,
        "status": "complete" if all(checks.values()) else "manual_audit_failed",
    }
    decision = decision_payload("M8", checks, inputs, estimates)
    atomic_json(output_path, decision)
    return decision


def build_final_evidence(
    *, artifact_root: Path, prereg_path: Path, output_path: Path
) -> dict[str, Any]:
    from .reporting import terminal_state

    state = terminal_state(artifact_root)
    if not state["terminal"]:
        raise ValueError(f"final evidence requires a terminal study: {state['reason']}")
    decision_path = artifact_root / "decisions/m8_decision.json"
    decision_sha256 = sha256_file(decision_path) if decision_path.is_file() else None
    files = [
        path
        for path in artifact_root.rglob("*")
        if path.is_file()
        and path != output_path
        and not path.name.endswith(".tmp")
    ]
    manifest = [
        {
            "path": path.relative_to(artifact_root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(files)
    ]
    evidence = {
        "schema_version": "glm53_v8_final_evidence_v1",
        "project_id": "glm53_user_eval_mechanism_v8",
        "preregistration": {
            "path": prereg_path.as_posix(),
            "sha256": sha256_file(prereg_path),
        },
        "terminal_state": state,
        "final_m8_decision_sha256": decision_sha256,
        "file_count": len(manifest),
        "files": manifest,
    }
    atomic_json(output_path, evidence)
    return evidence
