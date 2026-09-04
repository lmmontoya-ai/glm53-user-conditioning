"""Immutable inputs and fail-closed validation for V19."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from src.glm53_user_eval.v17.contract import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    canonical_sha256,
    read_json,
    read_jsonl,
    read_yaml,
    sha256_file,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_v19_prereg(root: Path, path: Path, *, verify_git: bool = True) -> dict[str, Any]:
    prereg = read_yaml(path)
    _require(prereg.get("schema_version") == "glm53_user_eval_v19_prereg_v1", "wrong V19 schema")
    _require(prereg.get("project_id") == "glm53_user_eval_hua_lean_v19", "wrong V19 project")
    for name, record in prereg["immutable_inputs"].items():
        source = root / record["path"]
        _require(source.is_file(), f"missing V19 input: {name}")
        _require(sha256_file(source) == record["sha256"], f"V19 input hash mismatch: {name}")

    parent = prereg["parent"]
    if verify_git:
        tagged = subprocess.check_output(
            ["git", "rev-list", "-n", "1", parent["final_tag"]], cwd=root, text=True
        ).strip()
        _require(tagged == parent["final_commit"], "V18 final tag moved")
    parent_decision = read_json(root / prereg["immutable_inputs"]["v18_decision"]["path"])
    _require(parent_decision["decision"] == "runtime_calibration_failed_before_hua_positive_control", "wrong V18 parent result")
    _require(parent_decision["completed_science"]["positive_control_rows"] == 0, "V18 contains positive-control rows")

    design = read_json(root / prereg["immutable_inputs"]["design"]["path"])
    _require(len(design["tasks"]) == 22 and len(set(design["tasks"])) == 22, "V19 r6 needs 22 tasks")
    _require(all(len(values) == 16 for values in design["identities"].values()), "V19 needs 16 identities per group")
    _require(design["planned_base_rows"] == 1408, "V19 planned base-row count changed")
    _require(design["reconstructable_base_rows"] == 1404, "V19 reconstructable row count changed")
    _require(design["total_user_model_forwards"] == 7020, "V19 user-forward count changed")
    _require(
        design["conditions"]
        == ["baseline", "all16_deploy", "all16_eval", "status_deploy", "formality"],
        "V19 condition order changed",
    )

    source = read_json(root / "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json")
    expected_tasks = source["tasks"]["pilot"] + source["tasks"]["confirmation"] + source["tasks"]["hardening"]
    _require(design["tasks"] == expected_tasks[:22], "V19 tasks differ from the first 22 outcome-independent V8 tasks")
    expected_identities = {
        "famous_ai": [row["famous_ai"] for row in source["pairs"]],
        "unknown_ai": [row["unknown_ai"] for row in source["pairs"]],
        "famous_nonai": source["famous_nonai_controls"],
        "genpop": source["genpop_controls"],
    }
    _require(design["identities"] == expected_identities, "V19 identities differ from the outcome-independent V8 set")

    selection = read_json(root / prereg["immutable_inputs"]["positive_control_selection"]["path"])
    _require(selection["scenario_count"] == 32, "V19 positive-control count changed")
    _require(len(selection["sample_ids"]) == len(set(selection["sample_ids"])) == 32, "V19 positive-control IDs are invalid")
    parent_pc = read_json(root / prereg["immutable_inputs"]["positive_control_manifest"]["path"])
    selected = [row for row in parent_pc["rows"] if row["sample_id"] in set(selection["sample_ids"])]
    _require(len(selected) == 32, "V19 positive-control selection does not resolve")
    counts: dict[tuple[str, str], int] = {}
    for row in selected:
        key = (row["expected_class"], row["source"])
        counts[key] = counts.get(key, 0) + 1
    _require(set(counts.values()) == {8} and len(counts) == 4, "V19 positive controls are unbalanced")

    runtime = read_yaml(root / prereg["immutable_inputs"]["runtime"]["path"])
    _require(runtime["forward"]["primary_batch_size"] == 1, "V19 must use batch size one")
    _require(runtime["runpod"]["gpu_id"] == "NVIDIA B300 SXM6 AC", "V19 hardware changed")
    _require(runtime["runpod"]["gpu_count"] == 2, "V19 r6 must use two B300s")
    _require(runtime["runpod"]["allow_gpu_type_fallback"] is False, "V19 GPU fallback is forbidden")
    _require(prereg["statistics"]["bootstrap_reps"] == 20000, "V19 bootstrap count changed")
    return {
        "schema_version": "glm53_v19_prereg_validation_v1",
        "passed": True,
        "prereg_sha256": sha256_file(path),
        "immutable_input_count": len(prereg["immutable_inputs"]),
        "planned_model_forwards": int(runtime["throughput_gate"]["planned_model_forwards"]),
    }


__all__ = [
    "atomic_json",
    "atomic_jsonl",
    "atomic_npz",
    "canonical_sha256",
    "read_json",
    "read_jsonl",
    "read_yaml",
    "sha256_file",
    "validate_v19_prereg",
]
