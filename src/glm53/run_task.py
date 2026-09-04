"""Wrapper around the pinned Transluce runner used for the completed runs.

The shard builder, Inspect command, and shard orchestration come unchanged from the archived
runner. This module builds the positive-control schedule, points the Inspect command at the
context-block task wrapper, and reads rows back with the archived extractor.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from . import REPO_ROOT
from .io import read_json, repo_path, sha256_file

ARCHIVE_SCRIPTS = REPO_ROOT / "archive/mechanism/pipelines/glm53_user_eval/scripts"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ARCHIVE_SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runner_module() -> ModuleType:
    """The archived shard runner (`run_transluce_exact_v6.py`), loaded by path."""
    return _load("glm53_archived_runner", "run_transluce_exact_v6.py")


def extractor_module() -> ModuleType:
    """The archived row extractor (`analyze_transluce_exact_v6.py`), loaded by path."""
    return _load("glm53_archived_extractor", "analyze_transluce_exact_v6.py")


def source_root(task: Mapping[str, Any], override: Path | None = None) -> Path:
    env = os.environ.get("GLM53_TRANSLUCE_ROOT")
    if override is not None:
        return Path(override).resolve()
    if env:
        return Path(env).resolve()
    return repo_path(task["reference_contract"]["local_checkout"]).resolve()


def verify_source(task: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Commit and file-hash checks of the pinned checkout; never raises, reports each check."""
    runner = runner_module()
    result: dict[str, Any] = {"root": str(root), "exists": root.exists(), "checks": {}}
    if not root.exists():
        return result
    try:
        head = runner.git_head(root)
    except Exception as exc:  # noqa: BLE001
        head = f"error: {type(exc).__name__}"
    result["commit"] = head
    result["checks"]["commit"] = head == task["reference_contract"]["commit"]
    for relative, expected in task["reference_contract"]["files"].items():
        path = root / relative
        result["checks"][relative] = path.exists() and sha256_file(path) == expected
    result["passed"] = all(result["checks"].values())
    return result


def prereg_view(task: Mapping[str, Any], sharding: Mapping[str, Any]) -> dict[str, Any]:
    """The mapping shape the archived runner expects, filled from task.yaml."""
    subject = task["subject"]
    return {
        "schema_version": "glm53_user_eval_prereg_v6",
        "subject": {
            "replacement_model": subject["inspect_model"],
            "provider": subject["provider"],
            "reasoning_effort": subject["reasoning_effort"],
            "credential_env": subject["credential_env"],
        },
        "execution": {
            "dilemmas_per_shard": int(sharding["dilemmas_per_shard"]),
            "shard_count": 0,
            "total_max_connections": int(task["execution"]["total_max_connections"]),
        },
        "population": {"dilemma_count": int(task["task_settings"]["dilemma_count"]), "expected_scientific_rows": 0},
    }


def family_for(salt: str, condition: str, persona_key: str, stimulus_id: str, families: int) -> int:
    digest = hashlib.sha256(f"{salt}|{condition}|{persona_key}|{stimulus_id}".encode()).hexdigest()
    return int(digest, 16) % families


def synthetic_key(persona_key: str, condition: str, family: int) -> str:
    return f"{persona_key}__{condition}__f{family}"


def positive_control_shards(task: Mapping[str, Any], control: Mapping[str, Any], roster: Mapping[str, list[dict[str, Any]]]) -> list[Any]:
    """One shard per (condition, dilemma) with one synthetic persona per genpop identity."""
    runner = runner_module()
    families = len(read_json(repo_path(control["context_blocks_file"]))["families"])
    salt = str(control["paraphrase_salt"])
    per_shard = int(control["sharding"]["dilemmas_per_shard"])
    if per_shard != 1:
        raise ValueError("per-dilemma paraphrase assignment requires one dilemma per shard")
    dilemmas = int(control["population"]["dilemmas"])
    shards = []
    for condition in control["conditions"]:
        for offset in range(dilemmas):
            stimulus = f"dd_{offset:04d}"
            keys = []
            for group in control["population"]["groups"]:
                for row in roster[group]:
                    keys.append(synthetic_key(str(row["key"]), condition, family_for(salt, condition, str(row["key"]), stimulus, families)))
            if control["population"].get("include_anonymous"):
                keys.append("anon")
            shards.append(runner.Shard(group=condition, offset=offset, limit=1, persona_keys=tuple(keys)))
    return shards


def manifest(task: Mapping[str, Any], control: Mapping[str, Any], shards: list[Any], root: Path) -> dict[str, Any]:
    return {
        "schema_version": "glm53_positive_control_plan_v1",
        "subject_model": task["subject"]["inspect_model"],
        "provider": task["subject"]["provider"],
        "reasoning_effort": task["subject"]["reasoning_effort"],
        "judge_model": task["judge"]["inspect_model"],
        "conditions": list(control["conditions"]),
        "context_blocks_sha256": sha256_file(repo_path(control["context_blocks_file"])),
        "paraphrase_salt": control["paraphrase_salt"],
        "transluce_root": str(root),
        "shard_count": len(shards),
        "total_expected_rows": sum(s.expected_rows for s in shards),
        "shards": [
            {"shard_id": s.shard_id, "group": s.group, "offset": s.offset, "limit": s.limit, "persona_keys": list(s.persona_keys), "expected_rows": s.expected_rows}
            for s in shards
        ],
    }


def run(
    task: Mapping[str, Any], control: Mapping[str, Any], shards: list[Any], *, root: Path, output_root: Path, parallel: int, connections: int
) -> dict[str, Any]:
    """Execute shards with the archived orchestrator; requires the subject credential in the environment."""
    runner = runner_module()
    prereg = prereg_view(task, control["sharding"])
    prereg["execution"]["shard_count"] = len(shards)
    prereg["population"]["expected_scientific_rows"] = sum(s.expected_rows for s in shards)
    blocks_path = repo_path(control["context_blocks_file"]).resolve()
    os.environ["GLM53_CONTEXT_BLOCKS"] = str(blocks_path)
    os.environ["GLM53_TRANSLUCE_ROOT"] = str(root)
    original_command = runner.inspect_command
    runner.build_shards = lambda _prereg, _root: shards

    def patched_command(**kw: Any) -> list[str]:
        command = original_command(**kw)
        command[2] = f"{REPO_ROOT / 'src/glm53/transluce_context_task.py'}@pmisaligned_context"
        return command

    runner.inspect_command = patched_command
    return runner.run_shards(
        prereg_path=REPO_ROOT / "configs/task.yaml",
        prereg=prereg,
        source_root=root,
        output_root=output_root,
        shards=shards,
        parallel_shards=parallel,
        connections_per_shard=connections,
    )


def extract_rows(output_root: Path) -> list[dict[str, Any]]:
    """Rows from the completed shard logs, via the archived extractor."""
    rows, _shards = extractor_module().extract_rows(output_root)
    return rows


def split_key_from_persona(key: str) -> tuple[str, str | None, int | None]:
    """(original key, condition, family) for a synthetic persona key; (key, None, None) otherwise."""
    if "__" not in key:
        return key, None, None
    base, condition, family = key.rsplit("__", 2)
    return base, condition, int(family.removeprefix("f"))
