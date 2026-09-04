"""Parent-evidence, model, and source-lock verification."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import sha256_file


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def verify_hashes(repo: Path, locks: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in locks.items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(f"hash mismatch for {relative}: {actual} != {expected}")
        observed[relative] = actual
    return observed


def verify_parent(repo: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    parent = prereg["parent_result"]
    tag_commit = git_output(repo, "rev-list", "-n", "1", parent["final_tag"])
    if tag_commit != parent["final_commit"]:
        raise ValueError("v7 final tag does not resolve to the locked commit")
    observed = verify_hashes(repo, parent["artifact_hashes"])
    decision = json.loads((repo / parent["decision_path"]).read_text(encoding="utf-8"))
    if decision.get("decision") != "confirmed_target_sized_interaction":
        raise ValueError("v7 decision is not the required confirmed state")
    if not decision.get("whitebox_green_light"):
        raise ValueError("v7 white-box green light is false")
    return {"tag_commit": tag_commit, "artifact_hashes": observed, "decision": decision}


def verify_log_manifest(repo: Path, final_evidence_path: Path) -> dict[str, str]:
    payload = json.loads(final_evidence_path.read_text(encoding="utf-8"))
    logs: dict[str, str] = {}
    for row in payload["successful_eval_logs"]:
        path = Path(row["path"])
        if not path.is_absolute():
            path = repo / path
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ValueError(f"v7 eval-log hash mismatch: {path}")
        logs[str(path.resolve())] = actual
    if len(logs) != 100:
        raise ValueError(f"expected 100 v7 logs, found {len(logs)}")
    return logs


def verify_direction_dataset(repo: Path, split_config_path: Path) -> dict[str, str]:
    """Verify the exact governed dataset bytes frozen for M3."""

    config = json.loads(split_config_path.read_text(encoding="utf-8"))
    dataset_root = repo / "artifacts/datasets" / config["dataset_id"]
    locks = {
        dataset_root / "samples.csv": config["samples_sha256"],
        dataset_root / "splits.csv": config["splits_sha256"],
        dataset_root / "summary.json": config["summary_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected in locks.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(f"direction dataset hash mismatch for {path}: {actual} != {expected}")
        observed[str(path.relative_to(repo)).replace("\\", "/")] = actual
    return observed
