"""Build deterministic evidence for the V11 semantic-validation stop."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "glm53_v11_semantic_stop_evidence_v1"
SUMMARY_SCHEMA_VERSION = "glm53_v11_semantic_stop_summary_v1"
DEFAULT_PREREG_TAG = "glm53-user-eval-v11-preregistered"

PROVENANCE_WARNINGS: tuple[dict[str, Any], ...] = (
    {
        "code": "initial_rows_precede_preserved_attempt_logs",
        "affected_valid_rows": 27,
        "message": (
            "The first 27 valid row files predate the preserved bounded-resume attempt logs. "
            "Their files and request records are hashed, but no saved attempt log covers their "
            "creation."
        ),
    },
    {
        "code": "failed_malformed_response_cost_unavailable",
        "message": (
            "The saved artifacts do not contain the API cost of failed malformed responses. "
            "The realized semantic cost covers the 576 preserved valid rows only."
        ),
    },
)

_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "named_api_key_assignment",
        re.compile(
            r"(?i)(?:OPENROUTER_API_KEY|OPENAI_API_KEY|RUNPOD_API_KEY)"
            r"\s*[:=]\s*[\"']?[^\s\"',}\]]{16,}"
        ),
    ),
    (
        "aws_credential_assignment",
        re.compile(
            r"(?i)(?:AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)"
            r"\s*[:=]\s*[\"']?[^\s\"',}\]]{12,}"
        ),
    ),
    ("openrouter_key", re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}")),
    ("aws_access_key", re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    (
        "authorization_bearer",
        re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{20,}"),
    ),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)


class EvidenceError(ValueError):
    """Raised when evidence cannot be bound without ambiguity."""


class CredentialPatternError(EvidenceError):
    """Raised when a high-confidence credential pattern appears in an input."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise EvidenceError(f"evidence path is outside the repository: {path}") from error


def _required_file(path: Path, *, name: str) -> Path:
    if not path.is_file():
        raise EvidenceError(f"missing {name}: {path}")
    return path


def _artifact_record(
    *,
    name: str,
    path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    _required_file(path, name=name)
    return {
        "name": name,
        "path": _repo_relative(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _entries_sha256(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            json.dumps(
                dict(entry),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _credential_matches(text: str) -> list[str]:
    return [name for name, pattern in _CREDENTIAL_PATTERNS if pattern.search(text)]


def _scan_credentials(paths: Iterable[Path], *, repo_root: Path) -> dict[str, Any]:
    scanned = 0
    for path in paths:
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = _credential_matches(text)
        if matches:
            relative = _repo_relative(path, repo_root)
            raise CredentialPatternError(
                f"credential pattern found in {relative}: {', '.join(matches)}"
            )
    return {
        "passed": True,
        "patterns_checked": [name for name, _pattern in _CREDENTIAL_PATTERNS],
        "scanned_file_count": scanned,
    }


def _git_output(repo_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repo_root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceError(f"git {' '.join(arguments)} failed: {message}") from error


def _git_lock(repo_root: Path, *, tag: str, prereg_path: Path) -> dict[str, Any]:
    tag_object_sha = _git_output(repo_root, "rev-parse", tag).decode().strip()
    commit_sha = _git_output(repo_root, "rev-parse", f"{tag}^{{}}").decode().strip()
    object_type = _git_output(repo_root, "cat-file", "-t", tag).decode().strip()
    if object_type != "tag":
        raise EvidenceError(f"preregistration ref is not an annotated tag: {tag}")

    fields = (
        _git_output(
            repo_root,
            "show",
            "-s",
            "--format=%H%x00%T%x00%P%x00%aI%x00%cI%x00%s",
            commit_sha,
        )
        .decode("utf-8")
        .strip()
        .split("\x00")
    )
    if len(fields) != 6:
        raise EvidenceError("unexpected preregistration commit metadata")
    commit, tree, parents, author_time, committer_time, subject = fields

    prereg_relative = _repo_relative(prereg_path, repo_root)
    prereg_at_tag = _git_output(repo_root, "show", f"{tag}:{prereg_relative}")
    prereg_at_tag_sha256 = hashlib.sha256(prereg_at_tag).hexdigest()
    current_prereg_sha256 = sha256_file(prereg_path)
    if prereg_at_tag_sha256 != current_prereg_sha256:
        raise EvidenceError("current preregistration differs from the tagged preregistration")

    return {
        "annotated_tag": tag,
        "tag_object_sha": tag_object_sha,
        "tag_object_type": object_type,
        "target_commit_sha": commit,
        "commit_tree_sha": tree,
        "parent_commit_shas": parents.split() if parents else [],
        "author_time": author_time,
        "committer_time": committer_time,
        "commit_subject": subject,
        "prereg_at_tag_sha256": prereg_at_tag_sha256,
        "current_prereg_sha256": current_prereg_sha256,
        "prereg_matches_tag": True,
    }


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    _required_file(path, name=name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvidenceError(f"invalid JSON in {name}: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must contain one JSON object")
    return value


def _sample_ids(samples_path: Path) -> set[str]:
    sample_ids: set[str] = set()
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceError(
                    f"invalid samples JSONL at line {line_number}"
                ) from error
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise EvidenceError(f"samples line {line_number} lacks sample_id")
            if sample_id in sample_ids:
                raise EvidenceError(f"duplicate sample_id in samples: {sample_id}")
            sample_ids.add(sample_id)
    return sample_ids


def _semantic_row_records(
    rows_root: Path,
    *,
    repo_root: Path,
    expected_sample_ids: set[str],
    expected_row_count: int,
) -> tuple[list[dict[str, Any]], Decimal]:
    row_paths = sorted(path for path in rows_root.glob("*.json") if path.is_file())
    if len(row_paths) != expected_row_count:
        raise EvidenceError(
            f"expected {expected_row_count} semantic rows, found {len(row_paths)}"
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_cost = Decimal(0)
    for path in row_paths:
        row = _load_json(path, name="semantic judge row")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id != path.stem:
            raise EvidenceError(f"row sample_id does not match filename: {path}")
        if sample_id in seen:
            raise EvidenceError(f"duplicate semantic row sample_id: {sample_id}")
        seen.add(sample_id)

        usage = row.get("usage")
        if not isinstance(usage, dict) or usage.get("cost") is None:
            raise EvidenceError(f"semantic row lacks usage.cost: {path}")
        try:
            total_cost += Decimal(str(usage["cost"]))
        except Exception as error:
            raise EvidenceError(f"invalid semantic row cost: {path}") from error

        records.append(
            {
                "path": _repo_relative(path, repo_root),
                "sample_id": sample_id,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    if seen != expected_sample_ids:
        missing = sorted(expected_sample_ids - seen)
        extra = sorted(seen - expected_sample_ids)
        raise EvidenceError(
            f"semantic rows do not match samples; missing={missing[:5]}, extra={extra[:5]}"
        )
    return records, total_cost


def _attempt_records(attempt_root: Path, *, repo_root: Path) -> list[dict[str, Any]]:
    paths = sorted(path for path in attempt_root.iterdir() if path.is_file())
    if not paths:
        raise EvidenceError(f"no preserved attempt logs found: {attempt_root}")
    return [
        {
            "path": _repo_relative(path, repo_root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _manual_packet_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _supplemental_packet_validation(
    *,
    packet_path: Path,
    manifest_path: Path,
    digest_path: Path,
    samples_path: Path,
    original_packet_path: Path,
    semantic_validation_path: Path,
    row_records: list[dict[str, Any]],
    expected_sample_ids: set[str],
) -> dict[str, Any]:
    supplemental = _load_json(manifest_path, name="supplemental manual manifest")
    required_false_fields = (
        "changes_preregistered_gate",
        "changes_semantic_metrics",
        "changes_paid_authorization",
    )
    if supplemental.get("scientific_role") != "supplemental_non_gating_human_review":
        raise EvidenceError("supplemental manual packet is not marked non-gating")
    if any(supplemental.get(field) is not False for field in required_false_fields):
        raise EvidenceError("supplemental manual packet changes a frozen gate or authorization")
    if supplemental.get("row_count") != 6:
        raise EvidenceError("supplemental manual packet must contain exactly six rows")

    packet_sha256 = sha256_file(packet_path)
    if supplemental.get("packet_sha256") != packet_sha256:
        raise EvidenceError("supplemental packet hash differs from its manifest")
    with packet_path.open("r", encoding="utf-8-sig", newline="") as handle:
        packet_rows = list(csv.DictReader(handle))
    packet_ids = [str(row.get("sample_id") or "") for row in packet_rows]
    if len(packet_ids) != 6 or any(not sample_id for sample_id in packet_ids):
        raise EvidenceError("supplemental packet does not contain six identified rows")
    if len(packet_ids) != len(set(packet_ids)):
        raise EvidenceError("supplemental packet contains duplicate sample IDs")
    if not set(packet_ids) <= expected_sample_ids:
        raise EvidenceError("supplemental packet contains IDs absent from the samples")
    if supplemental.get("selected_sample_ids") != packet_ids:
        raise EvidenceError("supplemental packet order differs from its manifest")

    judgment_hash_rows = sorted(
        (
            {"sample_id": item["sample_id"], "sha256": item["sha256"]}
            for item in row_records
        ),
        key=lambda item: item["sample_id"],
    )
    judgment_set_sha256 = hashlib.sha256(
        json.dumps(
            judgment_hash_rows,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    expected_source_hashes = {
        "samples_jsonl": sha256_file(samples_path),
        "original_manual_packet": sha256_file(original_packet_path),
        "semantic_validation": sha256_file(semantic_validation_path),
        "semantic_judgment_set": judgment_set_sha256,
    }
    if supplemental.get("source_hashes") != expected_source_hashes:
        raise EvidenceError("supplemental manifest source hashes differ from current inputs")

    manifest_sha256 = sha256_file(manifest_path)
    expected_digest = f"{manifest_sha256}  {manifest_path.name}\n"
    try:
        actual_digest = digest_path.read_text(encoding="ascii")
    except (UnicodeDecodeError, OSError) as error:
        raise EvidenceError("supplemental manifest digest is unreadable") from error
    if actual_digest != expected_digest:
        raise EvidenceError("supplemental manifest digest does not match the manifest")

    return {
        "digest_matches_manifest": True,
        "manifest_sha256": manifest_sha256,
        "non_gating": True,
        "packet_sha256": packet_sha256,
        "row_count": 6,
        "source_hashes_verified": True,
        "status": "pending_two_human_review",
    }


def _offline_chain_validation(
    *,
    core_paths: Mapping[str, Path],
    expected_row_count: int,
) -> dict[str, Any]:
    dataset_manifest = _load_json(
        core_paths["dataset_manifest"], name="dataset manifest"
    )
    tokenizer = _load_json(core_paths["tokenizer_audit"], name="tokenizer audit")
    structural = _load_json(core_paths["structural_audit"], name="structural audit")
    development = _load_json(
        core_paths["development_analysis"], name="development analysis"
    )
    final_text = _load_json(core_paths["final_text_analysis"], name="final text analysis")
    marker = _load_json(
        core_paths["final_text_holdout_marker"], name="final text holdout marker"
    )
    lexical = _load_json(core_paths["lexical_decision"], name="lexical decision")

    samples_sha256 = sha256_file(core_paths["samples"])
    if dataset_manifest.get("row_count") != expected_row_count:
        raise EvidenceError("dataset manifest row count differs from expected rows")
    if dataset_manifest.get("samples_sha256") != samples_sha256:
        raise EvidenceError("dataset manifest does not bind the current samples")
    if tokenizer.get("passed") is not True:
        raise EvidenceError("tokenizer audit did not pass")
    if tokenizer.get("row_count") != expected_row_count:
        raise EvidenceError("tokenizer audit row count differs from expected rows")
    if tokenizer.get("samples_sha256") != samples_sha256:
        raise EvidenceError("tokenizer audit does not bind the current samples")
    pair_contract = tokenizer.get("pair_contract")
    if not isinstance(pair_contract, dict) or pair_contract.get("passed") is not True:
        raise EvidenceError("tokenizer pair contract did not pass")
    if structural.get("passed") is not True:
        raise EvidenceError("structural audit did not pass")
    if structural.get("samples_sha256") != samples_sha256:
        raise EvidenceError("structural audit does not bind the current samples")

    model_bundle_sha256 = sha256_file(core_paths["development_models"])
    if development.get("model_bundle_sha256") != model_bundle_sha256:
        raise EvidenceError("development analysis does not bind the model bundle")
    if development.get("final_holdout_evaluated") is not False:
        raise EvidenceError("development analysis opened the final holdout")
    development_lock = development.get("development_lock_sha256")
    if not isinstance(development_lock, str) or len(development_lock) != 64:
        raise EvidenceError("development analysis lacks a valid lock hash")
    if final_text.get("development_lock_sha256") != development_lock:
        raise EvidenceError("final text analysis does not bind the development lock")
    if final_text.get("evaluated_split") != "final_counterfactual":
        raise EvidenceError("final text analysis used an unexpected split")
    if final_text.get("selection_performed") is not False:
        raise EvidenceError("final text analysis reports post-holdout selection")

    final_text_sha256 = sha256_file(core_paths["final_text_analysis"])
    if marker.get("opened_once") is not True or marker.get("status") != "complete":
        raise EvidenceError("final text holdout marker is not complete")
    if marker.get("development_lock_sha256") != development_lock:
        raise EvidenceError("final text holdout marker has the wrong development lock")
    if marker.get("final_analysis_sha256") != final_text_sha256:
        raise EvidenceError("final text holdout marker has the wrong analysis hash")
    if marker.get("samples_sha256") != samples_sha256:
        raise EvidenceError("final text holdout marker has the wrong samples hash")

    expected_lexical_inputs = {
        "development": sha256_file(core_paths["development_analysis"]),
        "final_text": final_text_sha256,
        "samples": samples_sha256,
        "tokenizer_audit": sha256_file(core_paths["tokenizer_audit"]),
    }
    if lexical.get("inputs") != expected_lexical_inputs:
        raise EvidenceError("lexical decision input hashes differ from the offline chain")

    return {
        "dataset_manifest_bound": True,
        "development_holdout_closed_during_fit": True,
        "development_model_bundle_bound": True,
        "final_holdout_marker_bound": True,
        "lexical_inputs_bound": True,
        "structural_audit_passed": True,
        "tokenizer_audit_passed": True,
    }


def _combined_analysis_validation(
    *,
    core_paths: Mapping[str, Path],
) -> dict[str, Any]:
    analysis = _load_json(
        core_paths["combined_offline_analysis"], name="combined offline analysis"
    )
    semantic = _load_json(core_paths["semantic_validation"], name="semantic validation")
    lexical = _load_json(core_paths["lexical_decision"], name="lexical decision")
    if analysis.get("state") != "semantic_validation_failed_manual_review_pending":
        raise EvidenceError("combined offline analysis has an unexpected state")
    if analysis.get("passed") is not False:
        raise EvidenceError("combined offline analysis is not a failed gate")

    expected_authorization = {
        "new_glm_forwards": False,
        "runpod_compute": False,
        "source_activation_extraction": False,
        "steering": False,
        "user_recruitment": False,
    }
    if analysis.get("authorization") != expected_authorization:
        raise EvidenceError("combined offline analysis contains an authorization")

    expected_inputs = {
        "dataset_manifest": sha256_file(core_paths["dataset_manifest"]),
        "development_analysis": sha256_file(core_paths["development_analysis"]),
        "final_holdout_marker": sha256_file(core_paths["final_text_holdout_marker"]),
        "final_text_analysis": sha256_file(core_paths["final_text_analysis"]),
        "lexical_decision": sha256_file(core_paths["lexical_decision"]),
        "manual_packet": sha256_file(core_paths["manual_packet"]),
        "manual_packet_lock": sha256_file(core_paths["manual_packet_lock"]),
        "manual_packet_manifest": sha256_file(core_paths["manual_packet_manifest"]),
        "preregistration": sha256_file(core_paths["preregistration"]),
        "samples": sha256_file(core_paths["samples"]),
        "semantic_validation": sha256_file(core_paths["semantic_validation"]),
        "structural_audit": sha256_file(core_paths["structural_audit"]),
        "supplemental_packet": sha256_file(core_paths["supplemental_manual_packet"]),
        "supplemental_packet_manifest": sha256_file(
            core_paths["supplemental_manual_packet_manifest"]
        ),
        "supplemental_packet_manifest_digest": sha256_file(
            core_paths["supplemental_manual_packet_digest"]
        ),
        "tokenizer_audit": sha256_file(core_paths["tokenizer_audit"]),
    }
    if analysis.get("inputs") != expected_inputs:
        raise EvidenceError("combined offline analysis input hashes differ from current artifacts")

    components = analysis.get("components")
    if not isinstance(components, dict):
        raise EvidenceError("combined offline analysis lacks components")
    semantic_component = components.get("semantic_validation")
    if not isinstance(semantic_component, dict):
        raise EvidenceError("combined offline analysis lacks semantic metrics")
    semantic_fields = (
        "binary",
        "controls",
        "final_counterfactual",
        "passed",
        "realized_cost_usd",
        "route_validation",
        "row_count",
        "schema_version",
    )
    if any(semantic_component.get(field) != semantic.get(field) for field in semantic_fields):
        raise EvidenceError("combined offline semantic metrics differ from semantic validation")
    lexical_component = components.get("lexical_decision")
    if not isinstance(lexical_component, dict):
        raise EvidenceError("combined offline analysis lacks the lexical decision")
    lexical_fields = ("checks", "decision", "passed", "schema_version")
    if any(lexical_component.get(field) != lexical.get(field) for field in lexical_fields):
        raise EvidenceError("combined offline lexical decision differs from its source")

    return {
        "all_authorizations_false": True,
        "input_hashes_verified": True,
        "lexical_component_verified": True,
        "semantic_component_verified": True,
        "state": "semantic_validation_failed_manual_review_pending",
    }


def build_semantic_stop_evidence(
    *,
    repo_root: Path,
    audit_root: Path,
    samples_path: Path,
    prereg_path: Path,
    tag: str = DEFAULT_PREREG_TAG,
    expected_row_count: int = 576,
    summary_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build the immutable-input manifest and compact terminal summary."""
    repo_root = repo_root.resolve()
    audit_root = audit_root.resolve()
    samples_path = samples_path.resolve()
    prereg_path = prereg_path.resolve()
    summary_path = summary_path or audit_root / "semantic_stop_summary.json"
    manifest_path = manifest_path or audit_root / "semantic_stop_evidence_manifest.json"

    core_paths = {
        "samples": samples_path,
        "dataset_manifest": samples_path.parent / "manifest.json",
        "tokenizer_audit": samples_path.parent / "tokenizer_audit.json",
        "preregistration": prereg_path,
        "structural_audit": audit_root / "structural_audit.json",
        "development_analysis": audit_root / "development_analysis.json",
        "development_models": audit_root / "development_models.joblib",
        "final_text_analysis": audit_root / "final_text_analysis.json",
        "final_text_holdout_marker": audit_root / "FINAL_TEXT_HOLDOUT_OPENED.json",
        "combined_offline_analysis": audit_root / "analysis.json",
        "semantic_validation": audit_root / "semantic_validation.json",
        "lexical_decision": audit_root / "lexical_decision.json",
        "manual_packet": audit_root / "manual_packet.csv",
        "manual_packet_lock": audit_root / "manual_packet_lock.json",
        "manual_packet_manifest": audit_root / "manual_packet_manifest.json",
        "supplemental_manual_packet": (
            audit_root / "supplemental_semantic_disagreements.csv"
        ),
        "supplemental_manual_packet_manifest": (
            audit_root / "supplemental_semantic_disagreements_manifest.json"
        ),
        "supplemental_manual_packet_digest": (
            audit_root / "supplemental_semantic_disagreements_manifest.sha256"
        ),
    }
    core_artifacts = [
        _artifact_record(name=name, path=path, repo_root=repo_root)
        for name, path in core_paths.items()
    ]

    samples_ids = _sample_ids(samples_path)
    if len(samples_ids) != expected_row_count:
        raise EvidenceError(
            f"expected {expected_row_count} samples, found {len(samples_ids)}"
        )

    semantic = _load_json(core_paths["semantic_validation"], name="semantic validation")
    lexical = _load_json(core_paths["lexical_decision"], name="lexical decision")
    manual_lock = _load_json(core_paths["manual_packet_lock"], name="manual packet lock")
    manual_manifest = _load_json(
        core_paths["manual_packet_manifest"], name="manual packet manifest"
    )
    if semantic.get("row_count") != expected_row_count:
        raise EvidenceError("semantic validation row count differs from expected rows")
    if semantic.get("passed") is not False:
        raise EvidenceError("semantic validation is not a failed terminal gate")
    if lexical.get("passed") is not True:
        raise EvidenceError("lexical gate did not pass")
    offline_chain = _offline_chain_validation(
        core_paths=core_paths,
        expected_row_count=expected_row_count,
    )

    row_records, row_cost = _semantic_row_records(
        audit_root / "semantic_judge/rows",
        repo_root=repo_root,
        expected_sample_ids=samples_ids,
        expected_row_count=expected_row_count,
    )
    attempt_records = _attempt_records(
        audit_root / "semantic_judge/attempt_logs", repo_root=repo_root
    )

    expected_cost = Decimal(str(semantic.get("realized_cost_usd")))
    if abs(row_cost - expected_cost) > Decimal("1e-12"):
        raise EvidenceError(
            "semantic validation cost does not equal the preserved valid-row cost"
        )

    packet_sha256 = sha256_file(core_paths["manual_packet"])
    packet_rows = _manual_packet_row_count(core_paths["manual_packet"])
    if manual_lock.get("packet_sha256") != packet_sha256:
        raise EvidenceError("manual packet hash differs from its lock")
    if manual_manifest.get("packet_sha256") != packet_sha256:
        raise EvidenceError("manual packet hash differs from its manifest")
    if manual_lock.get("row_count") != packet_rows:
        raise EvidenceError("manual packet row count differs from its lock")
    if manual_manifest.get("row_count") != packet_rows:
        raise EvidenceError("manual packet row count differs from its manifest")

    supplemental_review = _supplemental_packet_validation(
        packet_path=core_paths["supplemental_manual_packet"],
        manifest_path=core_paths["supplemental_manual_packet_manifest"],
        digest_path=core_paths["supplemental_manual_packet_digest"],
        samples_path=samples_path,
        original_packet_path=core_paths["manual_packet"],
        semantic_validation_path=core_paths["semantic_validation"],
        row_records=row_records,
        expected_sample_ids=samples_ids,
    )
    combined_analysis = _combined_analysis_validation(core_paths=core_paths)

    scanned_paths = list(core_paths.values())
    scanned_paths.extend(
        audit_root / "semantic_judge/rows" / f"{item['sample_id']}.json"
        for item in row_records
    )
    scanned_paths.extend(
        repo_root / str(record["path"]) for record in attempt_records
    )
    credential_scan = _scan_credentials(scanned_paths, repo_root=repo_root)

    git_lock = _git_lock(repo_root, tag=tag, prereg_path=prereg_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project_id": "glm53_user_eval_source_instrument_v11",
        "scope": "semantic_validation_terminal_stop",
        "git_lock": git_lock,
        "core_artifacts": core_artifacts,
        "semantic_rows": {
            "actual_count": len(row_records),
            "expected_count": expected_row_count,
            "files": row_records,
            "files_sha256": _entries_sha256(row_records),
            "preserved_valid_row_cost_usd": str(row_cost),
        },
        "attempt_logs": {
            "count": len(attempt_records),
            "files": attempt_records,
            "files_sha256": _entries_sha256(attempt_records),
        },
        "manual_review": {
            "completed_review_present": (
                audit_root / "manual_completed.csv"
            ).is_file(),
            "packet_row_count": packet_rows,
            "packet_sha256": packet_sha256,
            "status": "pending_two_human_review",
        },
        "supplemental_review": supplemental_review,
        "offline_chain": offline_chain,
        "combined_offline_analysis": combined_analysis,
        "credential_scan": credential_scan,
        "provenance_warnings": list(PROVENANCE_WARNINGS),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_matches = _credential_matches(manifest_bytes.decode("utf-8"))
    if manifest_matches:
        raise CredentialPatternError(
            "credential pattern found in evidence manifest: "
            + ", ".join(manifest_matches)
        )
    _atomic_write(manifest_path, manifest_bytes)
    manifest_sha256 = sha256_file(manifest_path)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "project_id": "glm53_user_eval_source_instrument_v11",
        "state": "semantic_gate_failed_paid_compute_locked",
        "passed": False,
        "authorization": {
            "new_glm_forwards": False,
            "runpod_compute": False,
        },
        "preregistration": {
            "annotated_tag": tag,
            "target_commit_sha": git_lock["target_commit_sha"],
            "prereg_sha256": git_lock["current_prereg_sha256"],
        },
        "lexical_gate": {
            "decision": lexical.get("decision"),
            "passed": True,
        },
        "semantic_gate": {
            "binary": semantic.get("binary"),
            "controls": semantic.get("controls"),
            "final_counterfactual": semantic.get("final_counterfactual"),
            "passed": False,
            "realized_valid_row_cost_usd": semantic.get("realized_cost_usd"),
            "route_validation": semantic.get("route_validation"),
            "row_count": semantic.get("row_count"),
        },
        "manual_review": manifest["manual_review"],
        "supplemental_review": supplemental_review,
        "offline_chain": offline_chain,
        "combined_offline_analysis": combined_analysis,
        "evidence_manifest": {
            "path": _repo_relative(manifest_path, repo_root),
            "sha256": manifest_sha256,
        },
        "credential_scan": credential_scan,
        "provenance_warnings": list(PROVENANCE_WARNINGS),
    }
    summary_bytes = _canonical_json_bytes(summary)
    summary_matches = _credential_matches(summary_bytes.decode("utf-8"))
    if summary_matches:
        raise CredentialPatternError(
            "credential pattern found in evidence summary: "
            + ", ".join(summary_matches)
        )
    _atomic_write(summary_path, summary_bytes)

    return {
        "manifest_path": _repo_relative(manifest_path, repo_root),
        "manifest_sha256": manifest_sha256,
        "summary_path": _repo_relative(summary_path, repo_root),
        "summary_sha256": sha256_file(summary_path),
    }


__all__ = [
    "DEFAULT_PREREG_TAG",
    "PROVENANCE_WARNINGS",
    "CredentialPatternError",
    "EvidenceError",
    "build_semantic_stop_evidence",
    "sha256_file",
]
