"""Integrity checks for the V21 post-gate exploratory continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.glm53_user_eval.v20.contract import read_json, read_yaml, sha256_file


def validate_v21_prereg(repo_root: Path, prereg_path: Path) -> dict[str, Any]:
    prereg = read_yaml(prereg_path)
    if prereg.get("schema_version") != "glm53_user_eval_v21_prereg_v1":
        raise ValueError("unexpected V21 preregistration schema")
    if prereg.get("project_id") != "glm53_user_eval_hua_exploratory_continuation_v21":
        raise ValueError("unexpected V21 project ID")
    if prereg.get("status", {}).get("confirmatory") is not False:
        raise ValueError("V21 must remain explicitly non-confirmatory")
    if prereg.get("execution", {}).get("baseline_rows_reused_without_rescoring") != 1404:
        raise ValueError("V21 baseline reuse count changed")
    if prereg.get("execution", {}).get("total_new_prompt_evaluations") != 8620:
        raise ValueError("V21 intervention row plan changed")

    checked: dict[str, str] = {}
    for name, record in prereg["immutable_inputs"].items():
        path = repo_root / record["path"]
        if not path.is_file():
            raise ValueError(f"V21 immutable input is absent: {name}")
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise ValueError(f"V21 immutable input hash mismatch: {name}")
        checked[name] = observed

    decision = read_json(
        repo_root / prereg["immutable_inputs"]["baseline_decision"]["path"]
    )
    verification = read_json(
        repo_root / prereg["immutable_inputs"]["baseline_verification"]["path"]
    )
    analysis = read_json(
        repo_root / prereg["immutable_inputs"]["baseline_analysis"]["path"]
    )
    if decision.get("decision") != "stop_local_parity_failed" or decision.get("passed"):
        raise ValueError("V21 requires the preserved failed V20 parity decision")
    if decision.get("authorization", {}).get("direct_user_intervention") is not False:
        raise ValueError("V20 must not authorize the V21 intervention")
    if verification.get("passed") is not True:
        raise ValueError("V20 parity recomputation did not pass")
    checks = analysis.get("checks", {})
    if checks.get("components") is not False:
        raise ValueError("V21 justification expects only the V20 component-pattern failure")
    if not all(bool(value) for key, value in checks.items() if key != "components"):
        raise ValueError("V20 had another failed local-parity check")

    baseline_path = repo_root / prereg["immutable_inputs"]["baseline_raw_scores"]["path"]
    row_count = sum(1 for line in baseline_path.read_text(encoding="utf-8").splitlines() if line)
    if row_count != 1404:
        raise ValueError("V20 baseline row count changed")
    return {
        "schema_version": "glm53_v21_prereg_validation_v1",
        "passed": True,
        "prereg_sha256": sha256_file(prereg_path),
        "immutable_inputs": checked,
        "baseline_rows": row_count,
        "scope": "exploratory_post_failed_local_parity_gate",
    }


__all__ = ["validate_v21_prereg"]
