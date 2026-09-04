from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file


def load_prereg(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("preregistration must be a mapping")
    return value


def validate_prereg(repo: Path, prereg_path: Path) -> dict[str, Any]:
    prereg = load_prereg(prereg_path)
    failures: list[str] = []
    if prereg.get("project_id") != "glm53_user_eval_behavioral_decomposition_v23":
        failures.append("project_id")
    for parent, record in prereg["locked_inputs"].items():
        path = repo / record["path"]
        if not path.is_file():
            failures.append(f"missing:{parent}")
        elif sha256_file(path) != record["sha256"]:
            failures.append(f"hash:{parent}")
    if prereg["data_roles"] != {"v6": "development_only", "v7": "locked_analysis"}:
        failures.append("data_roles")
    judges = prereg["judges"]
    if judges["inference_tier"] != "standard" or judges["fast_mode"] is not False:
        failures.append("judge_inference")
    if prereg["new_subject_model_calls"] != 0 or prereg["runpod_spend_usd"] != 0:
        failures.append("fresh_compute")
    if prereg["annotation"]["matched_cells"] != 500:
        failures.append("annotation_sample")
    if prereg["statistics"]["bootstrap_reps"] != 20000:
        failures.append("bootstrap_reps")
    if len(prereg["statistics"]["primary_outcomes"]) != 3:
        failures.append("primary_count")
    result = {
        "schema_version": "glm53_v23_prereg_validation_v1",
        "passed": not failures,
        "failures": failures,
        "prereg_sha256": sha256_file(prereg_path),
        "input_hashes": {
            name: sha256_file(repo / record["path"])
            for name, record in prereg["locked_inputs"].items()
            if (repo / record["path"]).is_file()
        },
    }
    if failures:
        raise ValueError(f"V23 preregistration failed: {failures}")
    return result


def require_preregistered_tag(repo: Path, prereg_path: Path) -> dict[str, str]:
    tag = "glm53-user-eval-v23-preregistered"
    tag_commit = subprocess.check_output(
        ["git", "rev-list", "-n", "1", tag], cwd=repo, text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tagged = subprocess.check_output(
        ["git", "show", f"{tag}:{prereg_path.relative_to(repo).as_posix()}"],
        cwd=repo,
    )
    import hashlib

    if hashlib.sha256(tagged).hexdigest() != sha256_file(prereg_path):
        raise ValueError("working preregistration differs from preregistered tag")
    return {"tag": tag, "tag_commit": tag_commit, "head": head}


def write_validation(repo: Path, prereg: Path, output: Path) -> dict[str, Any]:
    result = validate_prereg(repo, prereg)
    atomic_json(output, result)
    return result
