"""Immutable inputs, serialization, and fail-closed V17 validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}")
    return value


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def validate_v17_prereg(repo_root: Path, prereg_path: Path) -> dict[str, Any]:
    prereg = read_yaml(prereg_path)
    if prereg.get("schema_version") != "glm53_user_eval_v17_prereg_v1":
        raise ValueError("unexpected V17 preregistration schema")
    if prereg.get("project_id") != "glm53_user_eval_hua_causal_v17":
        raise ValueError("unexpected V17 project ID")
    checks: dict[str, bool] = {}
    for name, record in prereg["immutable_inputs"].items():
        path = repo_root / record["path"]
        checks[f"input_{name}"] = (
            path.is_file()
            and isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
            and sha256_file(path) == record["sha256"]
        )
    v16 = read_json(repo_root / prereg["immutable_inputs"]["v16_decision"]["path"])
    checks["v16_stopped"] = (
        v16.get("machine_decision") == "stop_before_local_parity"
        and v16.get("source_passed") is False
    )
    v7 = read_json(repo_root / prereg["immutable_inputs"]["v7_decision"]["path"])
    checks["v7_behavior_confirmed"] = (
        v7.get("decision") == "confirmed_target_sized_interaction"
        and v7.get("whitebox_green_light") is True
    )
    checks["v16_readout_excluded"] = prereg["science"]["direction_source"] == (
        "hua_table12_exact_only"
    )
    checks["no_cot"] = prereg["scope"]["early_cot"] is False
    checks["no_generation"] = prereg["scope"]["autoregressive_generation"] is False
    checks["budget"] = (
        float(prereg["budget"]["soft_cap_usd"]) == 10.0
        and float(prereg["budget"]["hard_cap_usd"]) == 15.0
        and float(prereg["budget"]["minimum_post_run_reserve_usd"]) == 8.0
    )
    if not all(checks.values()):
        raise ValueError(f"V17 preregistration validation failed: {checks}")
    return {
        "schema_version": "glm53_v17_prereg_validation_v1",
        "passed": True,
        "prereg_sha256": sha256_file(prereg_path),
        "checks": checks,
    }


__all__ = [
    "atomic_json",
    "atomic_jsonl",
    "atomic_npz",
    "canonical_sha256",
    "git_commit",
    "read_json",
    "read_jsonl",
    "read_yaml",
    "sha256_file",
    "validate_v17_prereg",
]
