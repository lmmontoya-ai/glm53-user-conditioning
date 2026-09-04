"""Immutable inputs and fail-closed validation for V16."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ID = "glm53_user_eval_source_activation_v16"
SCHEMA_VERSION = "glm53_user_eval_v16_prereg_v1"
PARENT_COMMIT = "d2e468b6b277e7a6b3ba10c044004c9d1299fc23"
PARENT_TAG = "glm53-user-eval-v15-final"
MODEL_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
DATASET_SHA256 = "a9c29d1a10a3eb7c431d8097808e9df10f888aa1585cc27638f44bb0aca1352e"
DATASET_MANIFEST_SHA256 = "f06b5d01727f29d5ccb55de2aa18de442aef6fff8aabd7e4443ebb54589c4e5b"
V15_DECISION_SHA256 = "6546853ea1be45f179a4a396c621be375f15ba0ff5412d9775fbadbb4725b9fc"
V15_VERIFICATION_SHA256 = "da4f4d7d3e6fd5d33a34f14f99eae33f05d43f9ca6af18dea062646daf25509e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()


def validate_parent(repo_root: Path) -> dict[str, Any]:
    paths = {
        "dataset": repo_root / "artifacts/datasets/contrastive_prompts_v5/samples.jsonl",
        "dataset_manifest": repo_root / "artifacts/datasets/contrastive_prompts_v5/manifest.json",
        "v15_decision": repo_root
        / "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json",
        "v15_verification": repo_root
        / "artifacts/glm53_user_eval/v15/reports/codex_cohort/verification.json",
    }
    expected = {
        "dataset": DATASET_SHA256,
        "dataset_manifest": DATASET_MANIFEST_SHA256,
        "v15_decision": V15_DECISION_SHA256,
        "v15_verification": V15_VERIFICATION_SHA256,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        raise ValueError(f"V15 parent hash mismatch: {observed}")
    tag_commit = git_output(repo_root, "rev-list", "-n", "1", PARENT_TAG)
    if tag_commit != PARENT_COMMIT:
        raise ValueError(f"{PARENT_TAG} resolves to {tag_commit}, not {PARENT_COMMIT}")
    decision = json.loads(paths["v15_decision"].read_text(encoding="utf-8"))
    verification = json.loads(paths["v15_verification"].read_text(encoding="utf-8"))
    required_authorization = {
        "exact_fp8_source_extraction": True,
        "runpod_compute": True,
        "local_proxy_parity": False,
        "prompt_recruitment": False,
        "first_cot_transfer": False,
        "steering": False,
    }
    if (
        decision.get("passed") is not True
        or decision.get("decision")
        != "fresh_control_bank_validated_by_both_codex_judges"
        or any(decision.get("authorization", {}).get(k) is not v for k, v in required_authorization.items())
        or verification.get("passed") is not True
    ):
        raise ValueError("V15 does not authorize exact-FP8 source extraction")
    return {
        "passed": True,
        "tag_commit": tag_commit,
        "hashes": observed,
        "decision": decision["decision"],
        "authorization": required_authorization,
    }


def validate_prereg(repo_root: Path, prereg_path: Path) -> dict[str, Any]:
    parent = validate_parent(repo_root)
    config = load_yaml(prereg_path)
    checks = {
        "schema": config.get("schema_version") == SCHEMA_VERSION,
        "project": config.get("project_id") == PROJECT_ID,
        "parent_commit": config.get("parent", {}).get("commit") == PARENT_COMMIT,
        "parent_tag": config.get("parent", {}).get("tag") == PARENT_TAG,
        "dataset": config.get("source", {}).get("dataset_sha256") == DATASET_SHA256,
        "dataset_manifest": config.get("source", {}).get("manifest_sha256")
        == DATASET_MANIFEST_SHA256,
        "model_revision": config.get("subject", {}).get("revision") == MODEL_REVISION,
        "model_identity": config.get("subject", {}).get("model_id")
        == "zai-org/GLM-5.3-Flash",
        "model_shape": config.get("subject", {}).get("weight_shards") == 62
        and config.get("subject", {}).get("weight_bytes") == 328337455672
        and config.get("subject", {}).get("text_layers") == 45
        and config.get("subject", {}).get("hidden_size") == 4096
        and config.get("subject", {}).get("hc_streams") == 4,
        "primary_view": config.get("source_readout", {}).get("primary_view")
        == "shared_task_suffix_mean",
        "permutations": config.get("source_readout", {}).get("permutation_reps") == 1000,
        "stability": config.get("source_readout", {}).get("stability_bootstrap_reps") == 1000,
        "hardware": config.get("infrastructure", {}).get("gpu_type")
        == "NVIDIA B300 SXM6 AC"
        and config.get("infrastructure", {}).get("gpu_count") == 2
        and config.get("infrastructure", {}).get("allow_gpu_fallback") is False,
        "budget": config.get("infrastructure", {}).get("compute_cap_usd") == 25.0
        and config.get("infrastructure", {}).get("reserve_usd") == 15.0
        and config.get("infrastructure", {}).get("storage_allowance_usd") == 0.10,
        "scope": config.get("scope", {}).get("early_cot") is False
        and config.get("scope", {}).get("steering") is False
        and config.get("scope", {}).get("dataset_repair") is False,
    }
    infrastructure = config.get("infrastructure", {})
    checks["infrastructure_code"] = all(
        (repo_root / str(infrastructure.get(f"{name}_path", ""))).is_file()
        and infrastructure.get(f"{name}_sha256")
        == sha256_file(repo_root / str(infrastructure[f"{name}_path"]))
        for name in ("launcher", "bootstrap", "watchdog")
    )
    tokenizer_record = config.get("source", {}).get("tokenizer_audit", {})
    tokenizer_path = repo_root / str(tokenizer_record.get("path", ""))
    checks["tokenizer_audit"] = (
        tokenizer_path.is_file()
        and tokenizer_record.get("sha256") == sha256_file(tokenizer_path)
    )
    if not all(checks.values()):
        raise ValueError(f"V16 preregistration validation failed: {checks}")
    return {"passed": True, "checks": checks, "parent": parent, "prereg_sha256": sha256_file(prereg_path)}


def validate_dataset_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = Counter(str(row.get("split")) for row in rows)
    labels = Counter(row.get("label") for row in rows)
    controls = Counter(
        str(row.get("control_expected_label"))
        for row in rows
        if row.get("split") == "neutral_controls"
    )
    expected_splits = {
        "train": 256,
        "validation": 48,
        "ordinary_test": 48,
        "development_counterfactual": 64,
        "final_counterfactual": 64,
        "factorial_calibration": 32,
        "neutral_controls": 64,
    }
    if len(rows) != 576 or dict(splits) != expected_splits:
        raise ValueError(f"V15 dataset split counts differ: {splits}")
    if labels != Counter({0: 240, 1: 240, None: 96}):
        raise ValueError(f"V15 dataset label counts differ: {labels}")
    if controls != Counter({"eval": 16, "deploy": 16, "ambiguous": 16, "no_status": 16}):
        raise ValueError(f"V15 control counts differ: {controls}")
    return {
        "row_count": len(rows),
        "split_counts": dict(splits),
        "label_counts": {str(key): value for key, value in labels.items()},
        "control_counts": dict(controls),
    }
