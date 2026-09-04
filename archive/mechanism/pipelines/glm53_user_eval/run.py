"""Command line entrypoint for the staged GLM-5.3 user/evaluation project."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.audits import projected_budget_ok
from src.glm53_user_eval.analysis import (
    analyze_g1_behavior,
    analyze_g3_local_behavior,
    analyze_roster_behavior,
    decide_roster_result,
)
from src.glm53_user_eval.api import run_behavior_schedule, validate_openrouter_zai_results
from src.glm53_user_eval.decisions import (
    build_g0_decision,
    build_g1_decision,
    build_g2_decision,
    build_g3_decision,
    build_g3_api_decision,
    build_roster_v5_decision,
)
from src.glm53_user_eval.local_inference import run_local_subject_schedule
from src.glm53_user_eval.local_judge import judge_local_subject_schedule
from src.glm53_user_eval.personas import (
    load_glm52_deltas,
    load_roster,
    reproduce_delta_cache,
    select_personas,
    validate_all_twin_indices,
)
from src.glm53_user_eval.behavior import (
    CONFIDENCE_FOLLOWUP,
    parse_direct_binary_response,
    render_daily_dilemma_binary_prompt,
)
from src.glm53_user_eval.prompts import prompt_sha256, render_plain_system
from src.glm53_user_eval.schemas import BehaviorPromptRecord, BehaviorScheduleRow
from src.glm53_user_eval.runtime import validate_glm53_config
from src.glm53_user_eval.runtime_doctor import run_runtime_doctor, stage_model_snapshot
from src.glm53_user_eval.vllm_inference import run_self_hosted_schedule
from src.glm53_user_eval.source_locks import (
    load_source_locks,
    sha256_file,
    validate_model_metadata,
    verify_git_checkout,
)


CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
DEFAULT_PREREG = CONFIG_ROOT / "prereg_v3.yaml"
DEFAULT_BEHAVIOR = CONFIG_ROOT / "behavior_v3.yaml"
DEFAULT_SERVING = CONFIG_ROOT / "serving_v1.yaml"
DEFAULT_API_PREREG = CONFIG_ROOT / "prereg_v4_api_serverless.yaml"
DEFAULT_ROSTER_PREREG = CONFIG_ROOT / "prereg_v5_roster.yaml"
DEFAULT_API_BEHAVIOR = CONFIG_ROOT / "behavior_v4.yaml"
DEFAULT_LOCKS = ROOT / "reference/source_locks_glm53_user_eval_v1.json"
DEFAULT_TRANSLUCE = ROOT.parent / "reference/transluce-user-awareness"
DEFAULT_HUA = ROOT.parent / "reference/hua-steering"
DEFAULT_TRANSFORMERS = ROOT.parent / "reference/transformers-glm53"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/glm53_user_eval"
EXPECTED_CACHE = {
    "entry_count": 339,
    "valid_100_counts": {
        "famous_ai": 43,
        "famous_ai_real": 44,
        "famous_nonai": 44,
        "genpop": 48,
        "unknown_ai": 44,
    },
    "valid_100_group_means": {
        "famous_ai": -1.4054885820790375,
        "famous_ai_real": -1.692269766011384,
        "famous_nonai": 0.12250296126134347,
        "genpop": 0.043222658231040455,
        "unknown_ai": -0.600451584193202,
    },
}


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def validate_prereg(path: Path) -> dict[str, Any]:
    prereg = read_yaml(path)
    if prereg.get("schema_version") not in {
        "glm53_user_eval_prereg_v1",
        "glm53_user_eval_prereg_v2",
        "glm53_user_eval_prereg_v3",
        "glm53_user_eval_prereg_v4",
        "glm53_user_eval_prereg_v5",
        "glm53_user_eval_prereg_v6",
        "glm53_user_eval_prereg_v7",
    }:
        raise ValueError("unexpected preregistration schema")
    locks_path = resolve_repo_path(prereg["source_locks"])
    locks = load_source_locks(locks_path)
    if prereg["subject"]["model_revision"] != locks.model.revision:
        raise ValueError("preregistered model revision differs from source lock")
    is_local_first = prereg["schema_version"] == "glm53_user_eval_prereg_v3"
    is_api_v4 = prereg["schema_version"] == "glm53_user_eval_prereg_v4"
    is_roster_v5 = prereg["schema_version"] == "glm53_user_eval_prereg_v5"
    is_transluce_v6 = prereg["schema_version"] == "glm53_user_eval_prereg_v6"
    is_transluce_v7 = prereg["schema_version"] == "glm53_user_eval_prereg_v7"
    if is_local_first:
        retired = prereg.get("retired_api_gate", {})
        if retired.get("executed_calls") != 0 or retired.get("scientific_rows") != 0:
            raise ValueError("local-first amendment contradicts the immutable API null record")
        local = prereg.get("local_behavior", {})
        if local.get("sample_count") != 600 or local.get("identity_pairs") != 4:
            raise ValueError("local behavioral grid differs from the v3 design")
        if local.get("conditions") != [
            "famous_coherent",
            "unknown_same_org",
            "unknown_general",
        ]:
            raise ValueError("local behavioral conditions differ from the v3 design")
    elif is_transluce_v6 or is_transluce_v7:
        reference = prereg.get("reference_contract", {})
        if reference.get("commit") != locks.transluce.commit:
            raise ValueError("Transluce commit differs from the source lock")
        expected_files = {
            "core/personas2.json",
            "core/solver.py",
            "evals/pmisaligned/task.py",
            "evals/pmisaligned/data/dailydilemmas.json",
            "scripts/run/run_pop_plain.sh",
            "scripts/plot/plot_s2_boxplot.py",
        }
        files = reference.get("files", {})
        if set(files) != expected_files or any(len(str(value)) != 64 for value in files.values()):
            raise ValueError("Transluce reference file hashes are incomplete")
        subject = prereg.get("subject", {})
        if subject.get("replacement_model") != "openrouter/z-ai/glm-5.3-flash":
            raise ValueError("Transluce replacement model changed")
        if subject.get("provider") != "Novita" or subject.get("provider_slug") != "novita":
            raise ValueError("Transluce run must use the Novita endpoint")
        if subject.get("reasoning_effort") != "high":
            raise ValueError("Transluce run must use high reasoning")
        if subject.get("allow_provider_fallbacks") is not False:
            raise ValueError("Transluce provider fallbacks must be disabled")
        population = prereg.get("population", {})
        if population.get("expected_scientific_rows") != 34400:
            raise ValueError("Transluce population row count changed")
        if population.get("dilemma_count") != 100:
            raise ValueError("Transluce run must use all 100 reference dilemmas")
        execution = prereg.get("execution", {})
        if execution.get("total_max_connections") != 200:
            raise ValueError("Transluce total concurrency must match the reference run")
        if execution.get("scientific_change_from_reference") != "subject_model_only":
            raise ValueError("Transluce scientific-change declaration changed")
        judge_v6 = prereg.get("judge", {})
        if judge_v6.get("model") != "openrouter/openai/gpt-5.4-mini":
            raise ValueError("Transluce judge differs from the pinned source")
        if judge_v6.get("reasoning_effort") != "low" or judge_v6.get("max_tokens") != 2000:
            raise ValueError("Transluce judge generation contract changed")
        if is_transluce_v6:
            previous = resolve_repo_path(prereg["amendment"]["preserves_prior_result"]["evidence_path"])
            if sha256_file(previous) != prereg["amendment"]["preserves_prior_result"]["evidence_sha256"]:
                raise ValueError("v6 no longer points to the immutable v5 evidence")
        else:
            parent_lock = prereg.get("parent_generation_contract", {})
            parent_path = resolve_repo_path(parent_lock.get("prereg_path", ""))
            if sha256_file(parent_path) != parent_lock.get("prereg_sha256"):
                raise ValueError("v7 parent preregistration hash changed")
            parent = read_yaml(parent_path)
            for section in parent_lock.get("equal_sections", []):
                if prereg.get(section) != parent.get(section):
                    raise ValueError(f"v7 generation section differs from v6: {section}")
            discovery = prereg.get("discovery_result", {})
            for field in ("analysis", "decision"):
                artifact = resolve_repo_path(discovery[f"{field}_path"])
                if sha256_file(artifact) != discovery[f"{field}_sha256"]:
                    raise ValueError(f"v7 discovery {field} artifact hash changed")
            if discovery.get("role") != "exploratory_hypothesis_generation_only":
                raise ValueError("v7 must label v6 as discovery-only")
            uncertainty = prereg.get("analysis", {}).get("uncertainty", {})
            if uncertainty.get("bootstrap_reps") != 20000:
                raise ValueError("v7 bootstrap repetition count changed")
            if uncertainty.get("bootstrap_seed") != 20260830:
                raise ValueError("v7 bootstrap seed changed")
            if prereg.get("technical_validity", {}).get("required_scientific_rows") != 34400:
                raise ValueError("v7 technical row requirement changed")
    else:
        if prereg["api"].get("allow_provider_fallback"):
            raise ValueError("claim-grade API runs cannot load-balance across providers")
        if is_api_v4 or is_roster_v5:
            api = prereg["api"]
            expected_routing = {
                "order": ["z-ai"],
                "only": ["z-ai"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            if api.get("provider") != "openrouter_zai_first_party":
                raise ValueError("API preregistration must label the first-party route explicitly")
            if api.get("model_id") != "z-ai/glm-5.3-flash":
                raise ValueError("OpenRouter model differs from the frozen model slug")
            if api.get("base_url") != "https://openrouter.ai/api/v1":
                raise ValueError("the API run must use the OpenRouter endpoint")
            if api.get("routing") != expected_routing:
                raise ValueError("provider routing differs from the pinned Z.AI route")
            if api.get("router_metadata") is not True:
                raise ValueError("the API run requires router metadata")
            if api.get("send_seed") is not False or api.get("send_thinking_object") is not False:
                raise ValueError("API parameters contradict the first-party endpoint boundary")
            if is_api_v4:
                behavior = prereg.get("api_behavior", {})
                if behavior.get("sample_count") != 600 or behavior.get("identity_pairs") != 4:
                    raise ValueError("v4 API behavior grid differs from the frozen design")
                if behavior.get("source_schedule_sha256") != "dbe671c69b216b9a7839d80c7ec6c59e722b89e47eea87a3e788b794945b82cd":
                    raise ValueError("v4 source schedule hash differs from the stopped local design")
                if behavior.get("source_prompts_sha256") != "7646c0a4ad3acd01e40b0e7e326629987e782f1e3c662f020b7bd8f2e416b776":
                    raise ValueError("v4 prompt hash differs from the stopped local design")
            else:
                roster = prereg.get("roster", {})
                tasks = prereg.get("tasks", {})
                behavior = prereg.get("behavior", {})
                if roster.get("pair_count") != 70 or roster.get("condition_count") != 280:
                    raise ValueError("v5 roster must contain all 70 indices and four conditions")
                if roster.get("conditions") != [
                    "famous_coherent",
                    "unknown_same_org",
                    "unknown_general",
                    "famous_nonai_control",
                ]:
                    raise ValueError("v5 roster conditions changed")
                if tasks.get("discovery_split") != "behavior_hardening_25":
                    raise ValueError("v5 discovery must use the untouched hardening split")
                if tasks.get("confirmation_split") != "behavior_causal_25":
                    raise ValueError("v5 confirmation must use the untouched causal split")
                if behavior.get("rows_per_stage") != 7000 or behavior.get("total_rows") != 14000:
                    raise ValueError("v5 row counts changed")
                frozen_schedules = behavior.get("frozen_schedules", {})
                if set(frozen_schedules) != {
                    "discovery_schedule_sha256",
                    "discovery_prompts_sha256",
                    "confirmation_schedule_sha256",
                    "confirmation_prompts_sha256",
                } or any(len(str(value)) != 64 for value in frozen_schedules.values()):
                    raise ValueError("v5 frozen schedule hashes are incomplete")
                previous = resolve_repo_path(prereg["amendment"]["preserves_prior_result"]["decision_path"])
                if sha256_file(previous) != prereg["amendment"]["preserves_prior_result"]["decision_sha256"]:
                    raise ValueError("v5 no longer points to the immutable v4 decision")
                task_manifest = resolve_repo_path(tasks["manifest"])
                if sha256_file(task_manifest) != prereg["amendment"]["untouched_task_split_sha256"]:
                    raise ValueError("v5 task split manifest changed")
            judge_v4 = prereg.get("judge", {})
            if judge_v4.get("provider") != "openrouter_openai_first_party":
                raise ValueError("v4 extraction judge must use first-party OpenAI through OpenRouter")
            if judge_v4.get("api_model_id") != "openai/gpt-5.4-mini":
                raise ValueError("v4 extraction judge API slug changed")
            if judge_v4.get("routing") != {
                "order": ["openai"],
                "only": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }:
                raise ValueError("v4 extraction judge is not pinned to first-party OpenAI")
    judge = prereg.get("judge")
    if judge is not None and not (is_transluce_v6 or is_transluce_v7):
        if judge.get("allow_provider_fallback"):
            raise ValueError("claim-grade judge runs cannot load-balance across providers")
        if not str(judge.get("model_id", "")).endswith("2026-03-17"):
            raise ValueError("extraction judge must use the frozen dated snapshot")
    serialized = json.dumps(prereg, default=str)
    if "PINNED" in serialized or "RESOLVE_" in serialized:
        raise ValueError("preregistration contains an unresolved placeholder")
    if not (is_transluce_v6 or is_transluce_v7) and float(prereg["budget"]["project_hard_cap_usd"]) != 125.0:
        raise ValueError("project hard cap differs from supplied preregistration")
    if (is_transluce_v6 or is_transluce_v7) and float(prereg["budget"]["incremental_api_cap_usd"]) != 50.0:
        raise ValueError("Transluce incremental API cap changed")
    return {
        "schema_version": prereg["schema_version"],
        "project_id": prereg["project_id"],
        "prereg_sha256": sha256_file(path),
        "source_lock_sha256": sha256_file(locks_path),
        "model_revision": locks.model.revision,
        "execution_backend": prereg["subject"].get("backend", "api"),
        "api_provider": (
            None
            if is_local_first
            else prereg["subject"]["provider"]
            if (is_transluce_v6 or is_transluce_v7)
            else prereg["api"]["provider"]
        ),
        "api_model_id": (
            None
            if is_local_first
            else prereg["subject"]["api_model_id"]
            if (is_transluce_v6 or is_transluce_v7)
            else prereg["api"]["model_id"]
        ),
    }


def command_validate_prereg(args: argparse.Namespace) -> int:
    result = validate_prereg(Path(args.prereg))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_reproduce_transluce(args: argparse.Namespace) -> int:
    source = Path(args.source_root).resolve()
    deltas_path = source / "cache/aggregates/s2glm52_deltas_conf.json"
    deltas = load_glm52_deltas(deltas_path)
    report = reproduce_delta_cache(deltas)
    checks = {
        "entry_count": report["entry_count"] == EXPECTED_CACHE["entry_count"],
        "person_means": report["max_person_mean_error"] <= 1e-9,
        "valid_counts": report["valid_100_counts"] == EXPECTED_CACHE["valid_100_counts"],
        "group_means": all(
            abs(report["valid_100_group_means"][key] - value) <= 1e-9
            for key, value in EXPECTED_CACHE["valid_100_group_means"].items()
        ),
    }
    payload = {
        "schema_version": "glm53_transluce_reproduction_v1",
        "source_commit": git_output_from(source),
        "source_file": str(deltas_path),
        "source_sha256": sha256_file(deltas_path),
        "report": report,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output = output / "reproduction.json" if output.suffix == "" else output
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def git_output_from(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def command_select_personas(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    source = Path(args.source_root).resolve()
    roster = load_roster(source / "core/personas2.json")
    deltas = load_glm52_deltas(source / prereg["selection"]["source_metric"])
    spec = prereg["selection"]
    pairs, controls = select_personas(
        roster,
        deltas,
        seed=int(spec["seed"]),
        enriched_count=int(spec["enriched_pairs"]),
        primary_count=int(spec["primary_intervention_pairs"]),
        prospective_count=int(spec["prospective_pairs"]),
        famous_nonai_count=int(spec["famous_nonai_controls"]),
        genpop_count=int(spec["genpop_controls"]),
        required_delta_count=int(spec["require_complete_delta_count"]),
    )
    payload = {
        "schema_version": "glm53_identity_selection_v1",
        "selection_seed": int(spec["seed"]),
        "source_commit": git_output_from(source),
        "pairs": [pair.model_dump(mode="json") for pair in pairs],
        "controls": [control.model_dump(mode="json") for control in controls],
    }
    write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pairs": len(pairs), "controls": len(controls)}))
    return 0


def _stratified_task_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("domain") or row.get("source") or "unstratified")
        strata.setdefault(key, []).append(row)
    rng = random.Random(seed)
    for values in strata.values():
        values.sort(key=lambda row: str(row["id"]))
        rng.shuffle(values)
    ordered: list[dict[str, Any]] = []
    keys = sorted(strata)
    while any(strata.values()):
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop())
    return ordered


def command_build_task_splits(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    rows = json.loads(Path(args.dilemmas).read_text(encoding="utf-8"))
    spec = prereg["tasks"]
    total = (
        int(spec["behavior_main"]) + int(spec["behavior_hardening"]) + int(spec["behavior_causal"])
    )
    if len(rows) < total or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("dilemma source lacks enough unique rows")
    ordered = _stratified_task_order(rows, int(spec["seed"]))[:total]
    main_end = int(spec["behavior_main"])
    hard_end = main_end + int(spec["behavior_hardening"])
    payload = {
        "schema_version": "glm53_task_splits_v1",
        "seed": int(spec["seed"]),
        "source_sha256": sha256_file(Path(args.dilemmas)),
        "splits": {
            "behavior_main_50": [row["id"] for row in ordered[:main_end]],
            "behavior_hardening_25": [row["id"] for row in ordered[main_end:hard_end]],
            "behavior_causal_25": [row["id"] for row in ordered[hard_end:]],
        },
    }
    write_json(Path(args.output), payload)
    print(json.dumps({key: len(value) for key, value in payload["splits"].items()}))
    return 0


def command_verify_sources(args: argparse.Namespace) -> int:
    locks = load_source_locks(Path(args.source_locks))
    verify_git_checkout(Path(args.transluce_root), locks.transluce.commit)
    verify_git_checkout(Path(args.hua_root), locks.hua.commit)
    verify_git_checkout(Path(args.transformers_root), locks.transformers.commit)
    roster = load_roster(Path(args.transluce_root) / "core/personas2.json")
    validate_all_twin_indices(roster)
    from huggingface_hub import HfApi, hf_hub_download

    model_info = HfApi().model_info(
        locks.model.repo,
        revision=locks.model.revision,
        files_metadata=True,
    )
    shards = [item for item in model_info.siblings if item.rfilename.endswith(".safetensors")]
    validate_model_metadata(
        locks,
        observed_revision=str(model_info.sha),
        safetensor_shards=len(shards),
        safetensor_bytes=sum(int(item.size or 0) for item in shards),
    )
    config_path = Path(
        hf_hub_download(locks.model.repo, "config.json", revision=locks.model.revision)
    )
    runtime_contract = validate_glm53_config(json.loads(config_path.read_text(encoding="utf-8")))
    payload = {
        "source_locks_complete": True,
        "transluce_commit": locks.transluce.commit,
        "hua_commit": locks.hua.commit,
        "transformers_commit": locks.transformers.commit,
        "model_revision": locks.model.revision,
        "model_safetensor_shards": len(shards),
        "model_safetensor_bytes": sum(int(item.size or 0) for item in shards),
        "roster_counts": {
            group: len(roster[group]) for group in roster if group != "famous_ai_real"
        },
        "twin_mapping_exact": True,
        "runtime_contract": runtime_contract,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_check_prompt_parity(args: argparse.Namespace) -> int:
    source = Path(args.source_root)
    sys.path.insert(0, str(source))
    from core.personas import PERSONAS  # type: ignore[import-not-found]

    solver_path = source / "core/solver.py"
    tree = ast.parse(solver_path.read_text(encoding="utf-8"), filename=str(solver_path))
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_plain_system"
    ]
    if len(function_nodes) != 1:
        raise ValueError("could not isolate exactly one Transluce _plain_system function")
    module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"PERSONAS": PERSONAS}
    exec(compile(module, str(solver_path), "exec"), namespace)
    plain_system = namespace["_plain_system"]

    selected = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    keys = (
        [
            key
            for pair in selected["pairs"]
            for key in (pair["famous_key"], pair["unknown_ai_key"], pair["genpop_key"])
        ]
        + [control["key"] for control in selected["controls"]]
        + ["anon"]
    )
    mismatches = []
    for key in keys:
        expected = plain_system(key, args.assistant_name)
        observed = render_plain_system(PERSONAS[key], args.assistant_name)
        if observed.encode("utf-8") != expected.encode("utf-8"):
            mismatches.append(key)
    payload = {"checked": len(keys), "mismatches": mismatches, "passed": not mismatches}
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not mismatches else 1


def command_build_behavior_schedule(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    behavior = read_yaml(Path(args.behavior_config))
    source = Path(args.source_root)
    roster = load_roster(source / "core/personas2.json")
    personas = {
        row["key"]: row
        for group in ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
        for row in roster[group]
    }
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    split_manifest = json.loads(Path(args.task_splits).read_text(encoding="utf-8"))
    selected_task_ids = split_manifest["splits"][args.task_split]
    dilemma_rows = json.loads(Path(args.dilemmas).read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in dilemma_rows}
    if any(task_id not in by_id for task_id in selected_task_ids):
        raise ValueError("task split references a missing dilemma")

    conditions: list[tuple[str, str, int | None]] = []
    for pair in selection["pairs"]:
        conditions.extend(
            [
                (pair["famous_key"], "famous_coherent", int(pair["twin_index"])),
                (pair["unknown_ai_key"], "unknown_same_org", int(pair["twin_index"])),
                (pair["genpop_key"], "unknown_general", int(pair["twin_index"])),
            ]
        )
    for control in selection["controls"]:
        conditions.append((control["key"], f"{control['group']}_control", None))
    conditions.append(("anon", "anonymous", None))

    schedule: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    seed = int(behavior["generation_seed"])
    for persona_key, condition, pair_index in conditions:
        persona = None if persona_key == "anon" else personas[persona_key]
        system_prompt = render_plain_system(persona, behavior["assistant_display_name"])
        for order, task_id in enumerate(selected_task_ids):
            dilemma = by_id[task_id]
            main_prompt = render_daily_dilemma_binary_prompt(
                str(dilemma["scenario"]), str(dilemma["action"])
            )
            digest = prompt_sha256(
                system_prompt,
                main_prompt + "\n\0\n" + CONFIDENCE_FOLLOWUP,
            )
            sample_id = f"{args.task_split}__{persona_key}__{task_id}__seed{seed}"
            schedule_row = BehaviorScheduleRow(
                sample_id=sample_id,
                scenario_id=task_id,
                persona_key=persona_key,
                pair_index=pair_index,
                condition=condition,
                phase=args.phase,
                reasoning_effort=prereg["subject"]["reasoning_effort"],
                generation_seed=seed,
                prompt_hash=digest,
                provider=prereg["api"]["provider"],
                model_id=prereg["api"]["model_id"],
            )
            prompt_row = BehaviorPromptRecord(
                sample_id=sample_id,
                system_prompt=system_prompt,
                main_prompt=main_prompt,
                followup_prompt=CONFIDENCE_FOLLOWUP,
                prompt_hash=digest,
            )
            row = schedule_row.model_dump(mode="json")
            row["task_order"] = order
            schedule.append(row)
            prompts.append(prompt_row.model_dump(mode="json"))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "schedule.jsonl", schedule)
    write_jsonl(output / "prompts.jsonl", prompts)
    schedule_hash = sha256_file(output / "schedule.jsonl")
    prompts_hash = sha256_file(output / "prompts.jsonl")
    manifest = {
        "schema_version": "glm53_behavior_schedule_manifest_v1",
        "phase": args.phase,
        "task_split": args.task_split,
        "task_count": len(selected_task_ids),
        "persona_condition_count": len(conditions),
        "sample_count": len(schedule),
        "schedule_sha256": schedule_hash,
        "prompts_sha256": prompts_hash,
        "provider": prereg["api"]["provider"],
        "model_id": prereg["api"]["model_id"],
        "judge_provider": prereg.get("judge", {}).get("provider"),
        "judge_model_id": prereg.get("judge", {}).get("model_id"),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_build_local_behavior_schedule(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    behavior = read_yaml(Path(args.behavior_config))
    if prereg.get("schema_version") != "glm53_user_eval_prereg_v3":
        raise ValueError("local schedule requires preregistration v3")
    source = Path(args.source_root)
    roster = load_roster(source / "core/personas2.json")
    personas = {
        row["key"]: row
        for group in ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
        for row in roster[group]
    }
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    primary_pairs = [pair for pair in selection["pairs"] if pair["primary_intervention"]]
    if len(primary_pairs) != 4:
        raise ValueError("local schedule requires exactly four frozen primary pairs")
    split_manifest = json.loads(Path(args.task_splits).read_text(encoding="utf-8"))
    selected_task_ids = split_manifest["splits"][args.task_split]
    dilemma_rows = json.loads(Path(args.dilemmas).read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in dilemma_rows}
    if any(task_id not in by_id for task_id in selected_task_ids):
        raise ValueError("task split references a missing dilemma")

    schedule: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    seed = int(behavior["generation_seed"])
    model_id = str(prereg["subject"]["model_id"])
    revision = str(prereg["subject"]["model_revision"])
    for task_order, task_id in enumerate(selected_task_ids):
        dilemma = by_id[task_id]
        main_prompt = render_daily_dilemma_binary_prompt(
            str(dilemma["scenario"]), str(dilemma["action"])
        )
        for pair in primary_pairs:
            conditions = (
                (pair["famous_key"], "famous_coherent"),
                (pair["unknown_ai_key"], "unknown_same_org"),
                (pair["genpop_key"], "unknown_general"),
            )
            for persona_key, condition in conditions:
                system_prompt = render_plain_system(
                    personas[persona_key], behavior["assistant_display_name"]
                )
                digest = prompt_sha256(
                    system_prompt,
                    main_prompt + "\n\0\n" + CONFIDENCE_FOLLOWUP,
                )
                sample_id = f"{args.task_split}__{persona_key}__{task_id}__seed{seed}"
                schedule_row = BehaviorScheduleRow(
                    sample_id=sample_id,
                    scenario_id=task_id,
                    persona_key=persona_key,
                    pair_index=int(pair["twin_index"]),
                    condition=condition,
                    phase="local_main",
                    reasoning_effort=str(prereg["subject"]["reasoning_effort"]),
                    generation_seed=seed,
                    prompt_hash=digest,
                    provider="local_official_fp8",
                    model_id=f"{model_id}@{revision}",
                )
                row = schedule_row.model_dump(mode="json")
                row["task_order"] = task_order
                row["analysis_block"] = task_order % int(
                    prereg["local_behavior"]["heterogeneity_blocks"]
                )
                row["task_source"] = str(dilemma.get("source") or "unknown")
                schedule.append(row)
                prompts.append(
                    BehaviorPromptRecord(
                        sample_id=sample_id,
                        system_prompt=system_prompt,
                        main_prompt=main_prompt,
                        followup_prompt=CONFIDENCE_FOLLOWUP,
                        prompt_hash=digest,
                    ).model_dump(mode="json")
                )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "schedule.jsonl", schedule)
    write_jsonl(output / "prompts.jsonl", prompts)
    schedule_hash = sha256_file(output / "schedule.jsonl")
    prompts_hash = sha256_file(output / "prompts.jsonl")
    expected = prereg["local_behavior"]
    for key, observed in (("schedule_sha256", schedule_hash), ("prompts_sha256", prompts_hash)):
        expected_value = str(expected[key])
        if not expected_value.startswith("FILL_") and expected_value != observed:
            raise ValueError(f"generated {key} differs from preregistration")
    manifest = {
        "schema_version": "glm53_local_behavior_schedule_manifest_v1",
        "phase": "local_main",
        "task_split": args.task_split,
        "task_count": len(selected_task_ids),
        "identity_pair_count": len(primary_pairs),
        "persona_condition_count": len(primary_pairs) * 3,
        "sample_count": len(schedule),
        "schedule_sha256": schedule_hash,
        "prompts_sha256": prompts_hash,
        "provider": "local_official_fp8",
        "model_id": model_id,
        "model_revision": revision,
        "judge_provider": prereg["judge"]["provider"],
        "judge_model_id": prereg["judge"]["model_id"],
    }
    if manifest["sample_count"] != int(prereg["local_behavior"]["sample_count"]):
        raise ValueError("local schedule size differs from preregistration")
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_build_api_v4_schedule(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    if prereg.get("schema_version") != "glm53_user_eval_prereg_v4":
        raise ValueError("the API behavior derivation requires preregistration v4")
    source_root = Path(args.source_schedule_root)
    source_schedule_path = source_root / "schedule.jsonl"
    source_prompts_path = source_root / "prompts.jsonl"
    expected = prereg["api_behavior"]
    if sha256_file(source_schedule_path) != expected["source_schedule_sha256"]:
        raise ValueError("source schedule differs from the frozen 600-row schedule")
    if sha256_file(source_prompts_path) != expected["source_prompts_sha256"]:
        raise ValueError("source prompts differ from the frozen prompt set")
    source_rows = read_jsonl(source_schedule_path)
    prompts = read_jsonl(source_prompts_path)
    if len(source_rows) != 600 or len(prompts) != 600:
        raise ValueError("v4 requires exactly 600 schedule and prompt rows")
    if {row["sample_id"] for row in source_rows} != {row["sample_id"] for row in prompts}:
        raise ValueError("source schedule and prompts have different sample IDs")
    derived: list[dict[str, Any]] = []
    for row in source_rows:
        if row.get("generation_seed") != 42 or row.get("reasoning_effort") != "max":
            raise ValueError("source schedule generation contract changed")
        updated = dict(row)
        updated["phase"] = "api_main_v4"
        updated["provider"] = prereg["api"]["provider"]
        updated["model_id"] = prereg["api"]["model_id"]
        derived.append(updated)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "schedule.jsonl", derived)
    write_jsonl(output / "prompts.jsonl", prompts)
    manifest = {
        "schema_version": "glm53_api_behavior_schedule_manifest_v4",
        "phase": "api_main_v4",
        "sample_count": len(derived),
        "source_schedule_sha256": sha256_file(source_schedule_path),
        "source_prompts_sha256": sha256_file(source_prompts_path),
        "schedule_sha256": sha256_file(output / "schedule.jsonl"),
        "prompts_sha256": sha256_file(output / "prompts.jsonl"),
        "allowed_source_row_changes": ["phase", "provider", "model_id"],
        "provider": prereg["api"]["provider"],
        "model_id": prereg["api"]["model_id"],
        "provider_routing": prereg["api"]["routing"],
        "judge_provider": prereg["judge"]["provider"],
        "judge_model_id": prereg["judge"]["model_id"],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_build_roster_v5_schedule(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    if prereg.get("schema_version") != "glm53_user_eval_prereg_v5":
        raise ValueError("the roster schedule requires preregistration v5")
    validate_prereg(Path(args.prereg))
    behavior = read_yaml(Path(args.behavior_config))
    source = Path(args.source_root)
    roster = load_roster(source / "core/personas2.json")
    validate_all_twin_indices(roster)
    split_manifest = json.loads(Path(args.task_splits).read_text(encoding="utf-8"))
    allowed_splits = {
        str(prereg["tasks"]["discovery_split"]),
        str(prereg["tasks"]["confirmation_split"]),
    }
    if args.task_split not in allowed_splits:
        raise ValueError("v5 schedule may use only the two untouched task splits")
    selected_task_ids = split_manifest["splits"][args.task_split]
    if len(selected_task_ids) != 25:
        raise ValueError("v5 roster stages require exactly 25 tasks")
    dilemma_rows = json.loads(Path(args.dilemmas).read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in dilemma_rows}
    if any(task_id not in by_id for task_id in selected_task_ids):
        raise ValueError("task split references a missing dilemma")

    conditions_by_index: list[tuple[int, str, str]] = []
    for pair_index in range(70):
        conditions_by_index.extend(
            [
                (pair_index, roster["famous_ai"][pair_index]["key"], "famous_coherent"),
                (
                    pair_index,
                    roster["unknown_ai"][pair_index]["key"],
                    "unknown_same_org",
                ),
                (pair_index, roster["genpop"][pair_index]["key"], "unknown_general"),
                (
                    pair_index,
                    roster["famous_nonai"][pair_index]["key"],
                    "famous_nonai_control",
                ),
            ]
        )
    personas = {
        row["key"]: row
        for group in ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
        for row in roster[group]
    }
    seed = int(behavior["generation_seed"])
    schedule: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for task_order, task_id in enumerate(selected_task_ids):
        dilemma = by_id[task_id]
        main_prompt = render_daily_dilemma_binary_prompt(
            str(dilemma["scenario"]), str(dilemma["action"])
        )
        for pair_index, persona_key, condition in conditions_by_index:
            system_prompt = render_plain_system(
                personas[persona_key], behavior["assistant_display_name"]
            )
            digest = prompt_sha256(
                system_prompt,
                main_prompt + "\n\0\n" + CONFIDENCE_FOLLOWUP,
            )
            sample_id = f"{args.task_split}__{persona_key}__{task_id}__seed{seed}"
            row = BehaviorScheduleRow(
                sample_id=sample_id,
                scenario_id=task_id,
                persona_key=persona_key,
                pair_index=pair_index,
                condition=condition,
                phase=args.phase,
                reasoning_effort=str(prereg["subject"]["reasoning_effort"]),
                generation_seed=seed,
                prompt_hash=digest,
                provider=str(prereg["api"]["provider"]),
                model_id=str(prereg["api"]["model_id"]),
            ).model_dump(mode="json")
            row["task_order"] = task_order
            row["analysis_block"] = task_order % 5
            row["task_source"] = str(dilemma.get("source") or "unknown")
            schedule.append(row)
            prompts.append(
                BehaviorPromptRecord(
                    sample_id=sample_id,
                    system_prompt=system_prompt,
                    main_prompt=main_prompt,
                    followup_prompt=CONFIDENCE_FOLLOWUP,
                    prompt_hash=digest,
                ).model_dump(mode="json")
            )
    if len(schedule) != int(prereg["behavior"]["rows_per_stage"]):
        raise ValueError("generated v5 roster row count differs from preregistration")
    if len({row["sample_id"] for row in schedule}) != len(schedule):
        raise ValueError("v5 roster schedule contains duplicate sample IDs")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "schedule.jsonl", schedule)
    write_jsonl(output / "prompts.jsonl", prompts)
    schedule_hash = sha256_file(output / "schedule.jsonl")
    prompts_hash = sha256_file(output / "prompts.jsonl")
    frozen = prereg["behavior"]["frozen_schedules"]
    prefix = (
        "discovery"
        if args.task_split == prereg["tasks"]["discovery_split"]
        else "confirmation"
    )
    if schedule_hash != frozen[f"{prefix}_schedule_sha256"]:
        raise ValueError(f"{prefix} schedule differs from the preregistered hash")
    if prompts_hash != frozen[f"{prefix}_prompts_sha256"]:
        raise ValueError(f"{prefix} prompts differ from the preregistered hash")
    manifest = {
        "schema_version": "glm53_roster_behavior_schedule_manifest_v5",
        "phase": args.phase,
        "task_split": args.task_split,
        "task_count": len(selected_task_ids),
        "pair_count": 70,
        "conditions": prereg["roster"]["conditions"],
        "sample_count": len(schedule),
        "schedule_sha256": schedule_hash,
        "prompts_sha256": prompts_hash,
        "task_split_manifest_sha256": sha256_file(Path(args.task_splits)),
        "provider": prereg["api"]["provider"],
        "model_id": prereg["api"]["model_id"],
        "judge_provider": prereg["judge"]["provider"],
        "judge_model_id": prereg["judge"]["model_id"],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_behavior_api(args: argparse.Namespace) -> int:
    validate_prereg(Path(args.prereg))
    prereg = read_yaml(Path(args.prereg))
    behavior = read_yaml(Path(args.behavior_config))
    schedule_root = Path(args.schedule_root)
    schedule_rows = read_jsonl(schedule_root / "schedule.jsonl")
    prompt_rows = {row["sample_id"]: row for row in read_jsonl(schedule_root / "prompts.jsonl")}
    if set(prompt_rows) != {row["sample_id"] for row in schedule_rows}:
        raise ValueError("schedule and prompt record IDs differ")
    if args.max_samples:
        schedule_rows = schedule_rows[: int(args.max_samples)]
    configured_judge = str(behavior["confidence_judge_model"])
    judge_config = prereg.get("judge", {})
    judge_label = str(judge_config.get("model_id", configured_judge))
    if configured_judge != judge_label:
        raise ValueError("behavior judge differs from preregistered judge snapshot")
    judge_api_model = str(judge_config.get("api_model_id") or judge_label.rsplit("/", 1)[-1])
    counts = asyncio.run(
        run_behavior_schedule(
            schedule_rows,
            prompt_rows,
            output_root=Path(args.output),
            run_id=args.run_id,
            subject_base_url=str(prereg["api"]["base_url"]),
            subject_api_key_env=str(prereg["api"]["credential_env"]),
            judge_api_key_env=str(judge_config.get("credential_env", "OPENAI_API_KEY")),
            judge_base_url=judge_config.get("base_url"),
            judge_model=judge_api_model,
            judge_label=judge_label,
            temperature=float(behavior["temperature"]),
            top_p=float(behavior["top_p"]),
            clear_thinking=bool(behavior["clear_thinking"]),
            max_tokens=int(behavior["max_tokens"]),
            provider_routing=dict(prereg["api"]["routing"]),
            metadata_header=bool(prereg["api"]["router_metadata"]),
            send_seed=bool(prereg["api"]["send_seed"]),
            send_thinking=bool(prereg["api"]["send_thinking_object"]),
            judge_provider_routing=judge_config.get("routing"),
            judge_metadata_header=bool(judge_config.get("router_metadata", False)),
            judge_max_tokens=int(judge_config.get("max_tokens", 128)),
            concurrency=int(args.concurrency),
        )
    )
    write_json(Path(args.output) / "run_summary.json", counts)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0 if counts["failed"] == 0 and counts["remaining"] == 0 else 1


def command_validate_api_route(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    report = validate_openrouter_zai_results(
        read_jsonl(Path(args.results)), expected_model=str(prereg["api"]["model_id"])
    )
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def command_reparse_api_results(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    results_path = run_root / "results.jsonl"
    original_path = run_root / "results_evaluator_v1.jsonl"
    if original_path.exists():
        raise ValueError("the original evaluator output is already preserved")
    rows = read_jsonl(results_path)
    original_text = results_path.read_text(encoding="utf-8")
    atomic_write(original_path, original_text)
    corrected: list[dict[str, Any]] = []
    changed = 0
    for row in rows:
        direct = parse_direct_binary_response(str(row.get("subject_response_main") or ""))
        updated = dict(row)
        metadata = dict(updated.get("provider_metadata") or {})
        extraction = dict(metadata.get("binary_extraction") or {})
        extraction.update(
            {
                "source": "deterministic_direct" if direct is not None else "llm_judge",
                "evaluator_version": "glm53_binary_extractor_v2",
                "original_binary_answer": row.get("binary_answer"),
            }
        )
        metadata["binary_extraction"] = extraction
        updated["provider_metadata"] = metadata
        if direct is not None and direct != row.get("binary_answer"):
            updated["binary_answer"] = direct
            updated["parse_valid"] = updated.get("confidence_p") is not None
            changed += 1
        corrected.append(updated)
    write_jsonl(results_path, corrected)
    report = {
        "schema_version": "glm53_evaluator_correction_v1",
        "source_results": str(original_path),
        "corrected_results": str(results_path),
        "row_count": len(rows),
        "changed_rows": changed,
        "subject_calls_rerun": 0,
        "judge_calls_rerun": 0,
        "rule": "exact yes/no subject replies use deterministic extraction before judge fallback",
        "source_sha256": sha256_file(original_path),
        "corrected_sha256": sha256_file(results_path),
    }
    write_json(run_root / "evaluator_correction_v2.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_analyze_behavior(args: argparse.Namespace) -> int:
    validation = validate_prereg(Path(args.prereg))
    results = read_jsonl(Path(args.results))
    schedule = read_jsonl(Path(args.schedule))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    reading_log = Path(args.reading_log) if args.reading_log else None
    estimates, checks = analyze_g1_behavior(
        results,
        schedule,
        selection,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed),
        reading_log=reading_log,
    )
    decision = build_g1_decision(
        run_id=args.run_id,
        prereg_sha256=validation["prereg_sha256"],
        source_lock_sha256=validation["source_lock_sha256"],
        model_revision=validation["model_revision"],
        estimates=estimates,
        checks=checks,
        inputs=(str(args.results), str(args.schedule), str(args.selection)),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "analysis.json",
        {"estimates": estimates, "checks": checks, "passed": decision.passed},
    )
    write_json(output / "decision.json", decision.model_dump(mode="json"))
    print(decision.model_dump_json(indent=2))
    return 0 if decision.passed else 1


def command_decide_g0(args: argparse.Namespace) -> int:
    validation = validate_prereg(Path(args.prereg))
    source_report = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    reproduction = json.loads(Path(args.reproduction).read_text(encoding="utf-8"))
    prompt_parity = json.loads(Path(args.prompt_parity).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    task_splits = json.loads(Path(args.task_splits).read_text(encoding="utf-8"))
    tag_commit = ""
    try:
        tag_commit = git_output("rev-list", "-n", "1", args.tag)
    except subprocess.CalledProcessError:
        pass
    head = git_output("rev-parse", "HEAD")
    checks = {
        "source_locks_complete": bool(source_report.get("source_locks_complete")),
        "roster_counts_exact": all(
            source_report.get("roster_counts", {}).get(group) == 70
            for group in ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
        ),
        "twin_mapping_exact": bool(source_report.get("twin_mapping_exact")),
        "prompt_parity_exact": bool(prompt_parity.get("passed")),
        "glm52_cache_reproduced": bool(reproduction.get("passed")),
        "parser_fixtures_passed": bool(args.parser_fixtures_passed),
        "selection_frozen": len(selection.get("pairs", [])) == 16
        and len(selection.get("controls", [])) == 16,
        "task_splits_frozen": {key: len(value) for key, value in task_splits["splits"].items()}
        == {
            "behavior_main_50": 50,
            "behavior_hardening_25": 25,
            "behavior_causal_25": 25,
        },
        "prereg_committed_and_tagged": bool(tag_commit) and tag_commit == head,
        "budget_within_cap": projected_budget_ok(0.0, 0.0, 125.0),
    }
    decision = build_g0_decision(
        run_id=args.run_id,
        prereg_sha256=validation["prereg_sha256"],
        source_lock_sha256=validation["source_lock_sha256"],
        model_revision=validation["model_revision"],
        checks=checks,
        estimates={
            "selected_pairs": len(selection["pairs"]),
            "selected_controls": len(selection["controls"]),
            "reproduction_max_error": reproduction["report"]["max_person_mean_error"],
        },
        inputs=tuple(
            str(path)
            for path in (
                args.source_report,
                args.reproduction,
                args.prompt_parity,
                args.selection,
                args.task_splits,
            )
        ),
    )
    write_json(Path(args.output), decision.model_dump(mode="json"))
    print(decision.model_dump_json(indent=2))
    return 0 if decision.passed else 1


def command_stage_model(args: argparse.Namespace) -> int:
    runtime = read_yaml(Path(args.runtime_config))
    locks = load_source_locks(Path(args.source_locks))
    if runtime["model_id"] != locks.model.repo or runtime["revision"] != locks.model.revision:
        raise ValueError("runtime model identity differs from the source lock")
    report = stage_model_snapshot(
        model_id=str(runtime["model_id"]),
        revision=str(runtime["revision"]),
        output_root=Path(args.output_root),
        expected_shards=int(runtime["weight_shards"]),
        expected_bytes=int(runtime["weight_bytes"]),
        max_workers=int(args.max_workers),
    )
    output = Path(args.output)
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_runtime_doctor(args: argparse.Namespace) -> int:
    runtime = read_yaml(Path(args.runtime_config))
    locks = load_source_locks(Path(args.source_locks))
    if runtime["model_id"] != locks.model.repo or runtime["revision"] != locks.model.revision:
        raise ValueError("runtime model identity differs from the source lock")
    prompt_rows = read_jsonl(Path(args.prompts))
    if len(prompt_rows) < 20:
        raise ValueError("G2 runtime doctor requires at least 20 frozen prompts")
    messages = [
        [
            {"role": "system", "content": str(row["system_prompt"])},
            {"role": "user", "content": str(row["main_prompt"])},
        ]
        for row in prompt_rows[:20]
    ]
    model_path = Path(args.model_root) / str(runtime["revision"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"staged model directory not found: {model_path}")
    report = run_runtime_doctor(
        model_path=model_path,
        revision=str(runtime["revision"]),
        prompts=messages,
        reasoning_effort=str(args.reasoning_effort),
        clear_thinking=bool(args.clear_thinking),
        deadline_minutes=int(args.deadline_minutes),
        expected_transformers_commit=str(runtime["transformers_commit"]),
    )
    report["run_id"] = args.run_id
    report["runtime_config_sha256"] = sha256_file(Path(args.runtime_config))
    report["source_lock_sha256"] = sha256_file(Path(args.source_locks))
    report["prompts_sha256"] = sha256_file(Path(args.prompts))
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def command_decide_g2(args: argparse.Namespace) -> int:
    validation = validate_prereg(Path(args.prereg))
    runtime_config = read_yaml(Path(args.runtime_config))
    stage = json.loads(Path(args.stage_manifest).read_text(encoding="utf-8"))
    doctor = json.loads(Path(args.doctor_report).read_text(encoding="utf-8"))
    expected_revision = validation["model_revision"]
    expected_transformers = str(runtime_config["transformers_commit"])
    expected_shards = int(runtime_config["weight_shards"])
    expected_bytes = int(runtime_config["weight_bytes"])
    doctor_checks = doctor.get("checks", {})
    extraction_shapes = doctor.get("extraction_shapes", {})
    checks = {
        "official_revision_loaded": stage.get("revision") == expected_revision
        and doctor.get("revision") == expected_revision,
        "all_weight_shards_verified": stage.get("safetensor_shards") == expected_shards
        and stage.get("safetensor_bytes") == expected_bytes
        and len(stage.get("safetensor_sha256", {})) == expected_shards,
        "transformers_commit_exact": doctor.get("transformers_commit") == expected_transformers,
        "twenty_prompt_forwards": doctor.get("prompt_count") == 20
        and bool(doctor_checks.get("twenty_prompts")),
        "mhc_shape_contract": doctor.get("loaded_layer_count") == 45
        and bool(doctor_checks.get("layer_shape_contract")),
        "hyper_head_mean_exact": bool(doctor_checks.get("hyper_head_exact")),
        "prompt_vectors_extracted": len(extraction_shapes) == 60,
        "alpha_zero_logits_exact": bool(doctor_checks.get("zero_logits_exact")),
        "alpha_zero_generation_exact": bool(doctor_checks.get("zero_greedy_exact")),
        "additive_hook_local": bool(
            doctor_checks.get("additive_delta_within_bf16_tolerance")
        ),
        "hooks_removed": bool(doctor_checks.get("hooks_removed")),
        "deadline_respected": float(doctor.get("elapsed_seconds", float("inf")))
        <= int(doctor.get("deadline_minutes", 0)) * 60,
    }
    runtime_hash = sha256_file(Path(args.doctor_report))
    decision = build_g2_decision(
        run_id=args.run_id,
        prereg_sha256=validation["prereg_sha256"],
        source_lock_sha256=validation["source_lock_sha256"],
        model_revision=expected_revision,
        runtime_hash=runtime_hash,
        estimates={
            "weight_shards": stage.get("safetensor_shards"),
            "weight_bytes": stage.get("safetensor_bytes"),
            "load_seconds": doctor.get("load_seconds"),
            "elapsed_seconds": doctor.get("elapsed_seconds"),
            "gpu_count": doctor.get("cuda", {}).get("device_count"),
            "zero_hook_logit_max_error": doctor.get("zero_hook_logit_max_error"),
            "hyper_head_max_error": doctor.get("hyper_head_max_error"),
        },
        checks=checks,
        inputs=(str(args.stage_manifest), str(args.doctor_report)),
    )
    write_json(Path(args.output), decision.model_dump(mode="json"))
    print(decision.model_dump_json(indent=2))
    return 0 if decision.passed else 1


def command_behavior_local(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    behavior = read_yaml(Path(args.behavior_config))
    runtime = read_yaml(Path(args.runtime_config))
    validation = validate_prereg(Path(args.prereg))
    schedule_root = Path(args.schedule_root)
    if sha256_file(schedule_root / "schedule.jsonl") != prereg["local_behavior"][
        "schedule_sha256"
    ]:
        raise ValueError("local behavior schedule differs from preregistration")
    if sha256_file(schedule_root / "prompts.jsonl") != prereg["local_behavior"][
        "prompts_sha256"
    ]:
        raise ValueError("local behavior prompts differ from preregistration")
    schedule_rows = read_jsonl(schedule_root / "schedule.jsonl")
    prompt_rows = {row["sample_id"]: row for row in read_jsonl(schedule_root / "prompts.jsonl")}
    if set(prompt_rows) != {row["sample_id"] for row in schedule_rows}:
        raise ValueError("schedule and prompt record IDs differ")
    model_path = Path(args.model_root) / validation["model_revision"]
    if not model_path.is_dir():
        raise FileNotFoundError(f"staged model directory not found: {model_path}")
    generation_config = {
        "reasoning_effort": behavior["reasoning_effort"],
        "clear_thinking": bool(behavior["clear_thinking"]),
        "do_sample": bool(behavior["do_sample"]),
        "temperature": float(behavior["temperature"]),
        "top_p": float(behavior["top_p"]),
        "max_new_tokens": int(behavior["max_new_tokens"]),
    }
    summary = run_local_subject_schedule(
        schedule_rows=schedule_rows,
        prompt_rows=prompt_rows,
        model_path=model_path,
        model_id=str(prereg["subject"]["model_id"]),
        revision=validation["model_revision"],
        transformers_commit=str(runtime["transformers_commit"]),
        g2_decision_path=Path(args.g2_decision),
        generation_config=generation_config,
        output_root=Path(args.output),
        run_id=args.run_id,
        deadline_minutes=int(args.deadline_minutes),
        max_samples=int(args.max_samples),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


def command_behavior_vllm(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    behavior = read_yaml(Path(args.behavior_config))
    serving = read_yaml(Path(args.serving_config))
    validation = validate_prereg(Path(args.prereg))
    schedule_root = Path(args.schedule_root)
    if sha256_file(schedule_root / "schedule.jsonl") != prereg["local_behavior"][
        "schedule_sha256"
    ]:
        raise ValueError("self-hosted behavior schedule differs from preregistration")
    if sha256_file(schedule_root / "prompts.jsonl") != prereg["local_behavior"][
        "prompts_sha256"
    ]:
        raise ValueError("self-hosted behavior prompts differ from preregistration")
    decision = json.loads(Path(args.g2_decision).read_text(encoding="utf-8"))
    if decision.get("gate") != "G2" or decision.get("passed") is not True:
        raise ValueError("self-hosted behavior requires a passing G2 decision")
    if decision.get("model_revision") != validation["model_revision"]:
        raise ValueError("G2 model revision differs from the serving subject")
    schedule_rows = read_jsonl(schedule_root / "schedule.jsonl")
    prompt_rows = {
        row["sample_id"]: row for row in read_jsonl(schedule_root / "prompts.jsonl")
    }
    if set(prompt_rows) != {row["sample_id"] for row in schedule_rows}:
        raise ValueError("schedule and prompt record IDs differ")
    base_url = args.base_url or os.environ.get("GLM53_LOCAL_BASE_URL")
    if not base_url:
        raise ValueError("pass --base-url or set GLM53_LOCAL_BASE_URL")
    generation_config = {
        "reasoning_effort": behavior["reasoning_effort"],
        "clear_thinking": bool(behavior["clear_thinking"]),
        "do_sample": bool(behavior["do_sample"]),
        "temperature": float(behavior["temperature"]),
        "top_p": float(behavior["top_p"]),
        "max_new_tokens": int(behavior["max_new_tokens"]),
    }
    image_digest = str(serving["image"]).split("@", 1)[1]
    summary = asyncio.run(
        run_self_hosted_schedule(
            schedule_rows=schedule_rows,
            prompt_rows=prompt_rows,
            base_url=str(base_url),
            model_id=str(serving["served_model_name"]),
            model_revision=validation["model_revision"],
            image_digest=image_digest,
            serving_config=serving,
            generation_config=generation_config,
            output_root=Path(args.output),
            run_id=args.run_id,
            concurrency=int(args.concurrency),
            max_samples=int(args.max_samples),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


def command_judge_local(args: argparse.Namespace) -> int:
    prereg = read_yaml(Path(args.prereg))
    schedule_rows = read_jsonl(Path(args.schedule_root) / "schedule.jsonl")
    counts = asyncio.run(
        judge_local_subject_schedule(
            schedule_rows=schedule_rows,
            subject_root=Path(args.subject_root),
            output_root=Path(args.output),
            run_id=args.run_id,
            judge_model=str(prereg["judge"]["model_id"]),
            judge_api_key_env=str(prereg["judge"]["credential_env"]),
            concurrency=int(args.concurrency),
        )
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0 if counts["failed"] == 0 and counts["missing_subject"] == 0 else 1


def command_analyze_local(args: argparse.Namespace) -> int:
    validation = validate_prereg(Path(args.prereg))
    results = read_jsonl(Path(args.results))
    schedule = read_jsonl(Path(args.schedule))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    estimates, checks = analyze_g3_local_behavior(
        results,
        schedule,
        selection,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed),
        reading_log=Path(args.reading_log) if args.reading_log else None,
    )
    runtime_hashes = {
        str((row.get("provider_metadata") or {}).get("runtime_hash") or "")
        for row in results
    }
    if len(runtime_hashes) != 1 or len(next(iter(runtime_hashes), "")) != 64:
        raise ValueError("G3 results do not share one valid G2 runtime hash")
    decision = build_g3_decision(
        run_id=args.run_id,
        prereg_sha256=validation["prereg_sha256"],
        source_lock_sha256=validation["source_lock_sha256"],
        model_revision=validation["model_revision"],
        runtime_hash=next(iter(runtime_hashes)),
        estimates=estimates,
        checks=checks,
        inputs=(str(args.results), str(args.schedule), str(args.selection)),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", {"estimates": estimates, "checks": checks})
    write_json(output / "decision.json", decision.model_dump(mode="json"))
    print(decision.model_dump_json(indent=2))
    return 0 if decision.passed else 1


def command_analyze_api_g3(args: argparse.Namespace) -> int:
    validation = validate_prereg(Path(args.prereg))
    results = read_jsonl(Path(args.results))
    schedule = read_jsonl(Path(args.schedule))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    estimates, checks = analyze_g3_local_behavior(
        results,
        schedule,
        selection,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed),
        reading_log=Path(args.reading_log) if args.reading_log else None,
        metadata_mode="api",
    )
    checks["api_route_metadata_complete"] = checks.pop("local_runtime_metadata_complete")
    decision = build_g3_api_decision(
        run_id=args.run_id,
        prereg_sha256=validation["prereg_sha256"],
        source_lock_sha256=validation["source_lock_sha256"],
        model_revision=validation["model_revision"],
        estimates=estimates,
        checks=checks,
        inputs=(str(args.results), str(args.schedule), str(args.selection)),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", {"estimates": estimates, "checks": checks})
    write_json(output / "decision.json", decision.model_dump(mode="json"))
    print(decision.model_dump_json(indent=2))
    return 0 if decision.passed else 1


def command_analyze_roster_v5(args: argparse.Namespace) -> int:
    validation = validate_prereg(Path(args.prereg))
    if validation["schema_version"] != "glm53_user_eval_prereg_v5":
        raise ValueError("roster analysis requires preregistration v5")
    results = [row for path in args.results for row in read_jsonl(Path(path))]
    schedule = [row for path in args.schedules for row in read_jsonl(Path(path))]
    if len({row["sample_id"] for row in results}) != len(results):
        raise ValueError("roster results contain duplicate sample IDs")
    if len({row["sample_id"] for row in schedule}) != len(schedule):
        raise ValueError("roster schedules contain duplicate sample IDs")
    prereg = read_yaml(Path(args.prereg))
    estimates, checks = analyze_roster_behavior(
        results,
        schedule,
        bootstrap_reps=int(args.bootstrap_reps),
        sign_flip_reps=int(prereg["estimands"]["identity_discovery"]["sign_flip_reps"]),
        seed=int(args.seed),
        reading_log=Path(args.reading_log) if args.reading_log else None,
        manual_minimum=int(args.manual_minimum),
    )
    payload = {
        "schema_version": "glm53_roster_analysis_v5",
        "run_id": args.run_id,
        "prereg_sha256": validation["prereg_sha256"],
        "input_results": list(args.results),
        "input_schedules": list(args.schedules),
        "estimates": estimates,
        "checks": checks,
        "passed_integrity": all(checks.values()),
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed_integrity"] else 1


def command_decide_roster_v5(args: argparse.Namespace) -> int:
    validation = validate_prereg(Path(args.prereg))
    discovery = json.loads(Path(args.discovery_analysis).read_text(encoding="utf-8"))
    confirmation = json.loads(Path(args.confirmation_analysis).read_text(encoding="utf-8"))
    combined = json.loads(Path(args.combined_analysis).read_text(encoding="utf-8"))
    for label, payload in (
        ("discovery", discovery),
        ("confirmation", confirmation),
        ("combined", combined),
    ):
        if payload.get("prereg_sha256") != validation["prereg_sha256"]:
            raise ValueError(f"{label} analysis uses a different preregistration")
        if not payload.get("passed_integrity"):
            raise ValueError(f"{label} analysis failed its integrity checks")
    discovery_estimates = discovery["estimates"]
    confirmation_estimates = confirmation["estimates"]
    combined_estimates = combined["estimates"]
    checks, decision_name = decide_roster_result(
        discovery_estimates,
        confirmation_estimates,
        combined_estimates,
    )
    concise = {
        "discovery": {
            key: discovery_estimates[key]
            for key in (
                "sample_count",
                "parse_rate",
                "name_effect_pp",
                "name_ci95_pp",
                "negative_name_identity_count",
                "affiliation_effect_pp",
                "affiliation_ci95_pp",
                "generic_fame_effect_pp",
                "generic_fame_ci95_pp",
                "discovery_candidate_pair_indices",
            )
        },
        "confirmation": {
            key: confirmation_estimates[key]
            for key in (
                "sample_count",
                "parse_rate",
                "name_effect_pp",
                "name_ci95_pp",
                "negative_name_identity_count",
                "affiliation_effect_pp",
                "affiliation_ci95_pp",
                "generic_fame_effect_pp",
                "generic_fame_ci95_pp",
            )
        },
        "combined": {
            key: combined_estimates[key]
            for key in (
                "sample_count",
                "parse_rate",
                "name_effect_pp",
                "name_ci95_pp",
                "negative_name_identity_count",
                "affiliation_effect_pp",
                "affiliation_ci95_pp",
                "generic_fame_effect_pp",
                "generic_fame_ci95_pp",
                "between_identity_name_effect_sd_pp",
                "replicated_identity_pair_indices",
            )
        },
    }
    decision = build_roster_v5_decision(
        run_id=args.run_id,
        prereg_sha256=validation["prereg_sha256"],
        source_lock_sha256=validation["source_lock_sha256"],
        model_revision=validation["model_revision"],
        estimates=concise,
        checks=checks,
        decision=decision_name,
        inputs=(
            str(args.discovery_analysis),
            str(args.confirmation_analysis),
            str(args.combined_analysis),
        ),
    )
    write_json(Path(args.output), decision.model_dump(mode="json"))
    print(decision.model_dump_json(indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-prereg")
    validate.add_argument("--prereg", default=str(DEFAULT_PREREG))
    validate.set_defaults(func=command_validate_prereg)

    reproduce = sub.add_parser("reproduce-transluce")
    reproduce.add_argument("--source-root", default=str(DEFAULT_TRANSLUCE))
    reproduce.add_argument("--output", required=True)
    reproduce.set_defaults(func=command_reproduce_transluce)

    select = sub.add_parser("select-personas")
    select.add_argument("--source-root", default=str(DEFAULT_TRANSLUCE))
    select.add_argument("--prereg", default=str(DEFAULT_PREREG))
    select.add_argument("--output", required=True)
    select.set_defaults(func=command_select_personas)

    splits = sub.add_parser("build-task-splits")
    splits.add_argument(
        "--dilemmas", default=str(DEFAULT_TRANSLUCE / "evals/pmisaligned/data/dailydilemmas.json")
    )
    splits.add_argument("--prereg", default=str(DEFAULT_PREREG))
    splits.add_argument("--output", required=True)
    splits.set_defaults(func=command_build_task_splits)

    sources = sub.add_parser("verify-sources")
    sources.add_argument("--source-locks", default=str(DEFAULT_LOCKS))
    sources.add_argument("--transluce-root", default=str(DEFAULT_TRANSLUCE))
    sources.add_argument("--hua-root", default=str(DEFAULT_HUA))
    sources.add_argument("--transformers-root", default=str(DEFAULT_TRANSFORMERS))
    sources.add_argument("--output", required=True)
    sources.set_defaults(func=command_verify_sources)

    parity = sub.add_parser("check-prompt-parity")
    parity.add_argument("--source-root", default=str(DEFAULT_TRANSLUCE))
    parity.add_argument("--selection", required=True)
    parity.add_argument("--assistant-name", default="GLM-5.3-Flash")
    parity.add_argument("--output", required=True)
    parity.set_defaults(func=command_check_prompt_parity)

    schedule = sub.add_parser("build-behavior-schedule")
    schedule.add_argument("--prereg", default=str(DEFAULT_API_PREREG))
    schedule.add_argument("--behavior-config", default=str(DEFAULT_API_BEHAVIOR))
    schedule.add_argument("--source-root", default=str(DEFAULT_TRANSLUCE))
    schedule.add_argument("--selection", default=str(CONFIG_ROOT / "identity_selection_v1.json"))
    schedule.add_argument("--task-splits", default=str(CONFIG_ROOT / "task_splits_v1.json"))
    schedule.add_argument(
        "--dilemmas", default=str(DEFAULT_TRANSLUCE / "evals/pmisaligned/data/dailydilemmas.json")
    )
    schedule.add_argument("--task-split", default="behavior_main_50")
    schedule.add_argument("--phase", default="api_main")
    schedule.add_argument("--output", required=True)
    schedule.set_defaults(func=command_build_behavior_schedule)

    local_schedule = sub.add_parser("build-local-behavior-schedule")
    local_schedule.add_argument("--prereg", default=str(DEFAULT_PREREG))
    local_schedule.add_argument("--behavior-config", default=str(DEFAULT_BEHAVIOR))
    local_schedule.add_argument("--source-root", default=str(DEFAULT_TRANSLUCE))
    local_schedule.add_argument("--selection", default=str(CONFIG_ROOT / "identity_selection_v1.json"))
    local_schedule.add_argument("--task-splits", default=str(CONFIG_ROOT / "task_splits_v1.json"))
    local_schedule.add_argument(
        "--dilemmas", default=str(DEFAULT_TRANSLUCE / "evals/pmisaligned/data/dailydilemmas.json")
    )
    local_schedule.add_argument("--task-split", default="behavior_main_50")
    local_schedule.add_argument("--output", required=True)
    local_schedule.set_defaults(func=command_build_local_behavior_schedule)

    api_v4_schedule = sub.add_parser("build-api-v4-schedule")
    api_v4_schedule.add_argument("--prereg", default=str(DEFAULT_API_PREREG))
    api_v4_schedule.add_argument("--source-schedule-root", required=True)
    api_v4_schedule.add_argument("--output", required=True)
    api_v4_schedule.set_defaults(func=command_build_api_v4_schedule)

    roster_schedule = sub.add_parser("build-roster-v5-schedule")
    roster_schedule.add_argument("--prereg", default=str(DEFAULT_ROSTER_PREREG))
    roster_schedule.add_argument("--behavior-config", default=str(DEFAULT_API_BEHAVIOR))
    roster_schedule.add_argument("--source-root", default=str(DEFAULT_TRANSLUCE))
    roster_schedule.add_argument(
        "--task-splits", default=str(CONFIG_ROOT / "task_splits_v1.json")
    )
    roster_schedule.add_argument(
        "--dilemmas", default=str(DEFAULT_TRANSLUCE / "evals/pmisaligned/data/dailydilemmas.json")
    )
    roster_schedule.add_argument("--task-split", required=True)
    roster_schedule.add_argument("--phase", required=True)
    roster_schedule.add_argument("--output", required=True)
    roster_schedule.set_defaults(func=command_build_roster_v5_schedule)

    behavior_api = sub.add_parser("behavior-api")
    behavior_api.add_argument("--prereg", default=str(DEFAULT_API_PREREG))
    behavior_api.add_argument("--behavior-config", default=str(DEFAULT_API_BEHAVIOR))
    behavior_api.add_argument("--schedule-root", required=True)
    behavior_api.add_argument("--output", required=True)
    behavior_api.add_argument("--run-id", required=True)
    behavior_api.add_argument("--concurrency", type=int, default=32)
    behavior_api.add_argument("--max-samples", type=int, default=0)
    behavior_api.set_defaults(func=command_behavior_api)

    validate_api = sub.add_parser("validate-api-route")
    validate_api.add_argument("--prereg", default=str(DEFAULT_API_PREREG))
    validate_api.add_argument("--results", required=True)
    validate_api.add_argument("--output", required=True)
    validate_api.set_defaults(func=command_validate_api_route)

    reparse_api = sub.add_parser("reparse-api-results")
    reparse_api.add_argument("--run-root", required=True)
    reparse_api.set_defaults(func=command_reparse_api_results)

    analyze = sub.add_parser("analyze-behavior")
    analyze.add_argument("--prereg", default=str(DEFAULT_API_PREREG))
    analyze.add_argument("--results", required=True)
    analyze.add_argument("--schedule", required=True)
    analyze.add_argument("--selection", default=str(CONFIG_ROOT / "identity_selection_v1.json"))
    analyze.add_argument("--reading-log")
    analyze.add_argument("--bootstrap-reps", type=int, default=20000)
    analyze.add_argument("--seed", type=int, default=20260828)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(func=command_analyze_behavior)

    decide = sub.add_parser("decide-g0")
    decide.add_argument("--prereg", default=str(DEFAULT_PREREG))
    decide.add_argument("--source-report", required=True)
    decide.add_argument("--reproduction", required=True)
    decide.add_argument("--prompt-parity", required=True)
    decide.add_argument("--selection", required=True)
    decide.add_argument("--task-splits", required=True)
    decide.add_argument("--parser-fixtures-passed", action="store_true")
    decide.add_argument("--tag", default="glm53-user-eval-prereg-v1")
    decide.add_argument("--run-id", default="glm53-g0-source-parity")
    decide.add_argument("--output", required=True)
    decide.set_defaults(func=command_decide_g0)

    stage = sub.add_parser("stage-model")
    stage.add_argument("--runtime-config", default=str(CONFIG_ROOT / "runtime_v1.yaml"))
    stage.add_argument("--source-locks", default=str(DEFAULT_LOCKS))
    stage.add_argument("--output-root", required=True)
    stage.add_argument("--max-workers", type=int, default=8)
    stage.add_argument("--output", required=True)
    stage.set_defaults(func=command_stage_model)

    doctor = sub.add_parser("runtime-doctor")
    doctor.add_argument("--runtime-config", default=str(CONFIG_ROOT / "runtime_v1.yaml"))
    doctor.add_argument("--source-locks", default=str(DEFAULT_LOCKS))
    doctor.add_argument("--model-root", required=True)
    doctor.add_argument("--prompts", required=True)
    doctor.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="max")
    doctor.add_argument("--clear-thinking", action=argparse.BooleanOptionalAction, default=True)
    doctor.add_argument("--deadline-minutes", type=int, default=110)
    doctor.add_argument("--run-id", default="glm53-g2-runtime-doctor")
    doctor.add_argument("--output", required=True)
    doctor.set_defaults(func=command_runtime_doctor)

    decide_g2 = sub.add_parser("decide-g2")
    decide_g2.add_argument("--prereg", default=str(DEFAULT_PREREG))
    decide_g2.add_argument("--runtime-config", default=str(CONFIG_ROOT / "runtime_v1.yaml"))
    decide_g2.add_argument("--stage-manifest", required=True)
    decide_g2.add_argument("--doctor-report", required=True)
    decide_g2.add_argument("--run-id", default="glm53-g2-runtime-doctor")
    decide_g2.add_argument("--output", required=True)
    decide_g2.set_defaults(func=command_decide_g2)

    local = sub.add_parser("behavior-local")
    local.add_argument("--prereg", default=str(DEFAULT_PREREG))
    local.add_argument("--behavior-config", default=str(DEFAULT_BEHAVIOR))
    local.add_argument("--runtime-config", default=str(CONFIG_ROOT / "runtime_v1.yaml"))
    local.add_argument("--schedule-root", required=True)
    local.add_argument("--model-root", required=True)
    local.add_argument("--g2-decision", required=True)
    local.add_argument("--output", required=True)
    local.add_argument("--run-id", default="glm53-g3-local-subject")
    local.add_argument("--deadline-minutes", type=int, default=720)
    local.add_argument("--max-samples", type=int, default=0)
    local.set_defaults(func=command_behavior_local)

    vllm_local = sub.add_parser("behavior-vllm")
    vllm_local.add_argument("--prereg", default=str(DEFAULT_PREREG))
    vllm_local.add_argument("--behavior-config", default=str(DEFAULT_BEHAVIOR))
    vllm_local.add_argument("--serving-config", default=str(DEFAULT_SERVING))
    vllm_local.add_argument("--schedule-root", required=True)
    vllm_local.add_argument("--g2-decision", required=True)
    vllm_local.add_argument("--base-url")
    vllm_local.add_argument("--output", required=True)
    vllm_local.add_argument("--run-id", default="glm53-g3-vllm-subject")
    vllm_local.add_argument("--concurrency", type=int, default=16)
    vllm_local.add_argument("--max-samples", type=int, default=0)
    vllm_local.set_defaults(func=command_behavior_vllm)

    judge_local = sub.add_parser("judge-local-behavior")
    judge_local.add_argument("--prereg", default=str(DEFAULT_PREREG))
    judge_local.add_argument("--schedule-root", required=True)
    judge_local.add_argument("--subject-root", required=True)
    judge_local.add_argument("--output", required=True)
    judge_local.add_argument("--run-id", default="glm53-g3-local-judged")
    judge_local.add_argument("--concurrency", type=int, default=16)
    judge_local.set_defaults(func=command_judge_local)

    analyze_local = sub.add_parser("analyze-local-behavior")
    analyze_local.add_argument("--prereg", default=str(DEFAULT_PREREG))
    analyze_local.add_argument("--results", required=True)
    analyze_local.add_argument("--schedule", required=True)
    analyze_local.add_argument("--selection", default=str(CONFIG_ROOT / "identity_selection_v1.json"))
    analyze_local.add_argument("--reading-log")
    analyze_local.add_argument("--bootstrap-reps", type=int, default=20000)
    analyze_local.add_argument("--seed", type=int, default=20260828)
    analyze_local.add_argument("--run-id", default="glm53-g3-local-behavior")
    analyze_local.add_argument("--output", required=True)
    analyze_local.set_defaults(func=command_analyze_local)

    analyze_api_g3 = sub.add_parser("analyze-api-g3")
    analyze_api_g3.add_argument("--prereg", default=str(DEFAULT_API_PREREG))
    analyze_api_g3.add_argument("--results", required=True)
    analyze_api_g3.add_argument("--schedule", required=True)
    analyze_api_g3.add_argument(
        "--selection", default=str(CONFIG_ROOT / "identity_selection_v1.json")
    )
    analyze_api_g3.add_argument("--reading-log")
    analyze_api_g3.add_argument("--bootstrap-reps", type=int, default=20000)
    analyze_api_g3.add_argument("--seed", type=int, default=20260828)
    analyze_api_g3.add_argument("--run-id", required=True)
    analyze_api_g3.add_argument("--output", required=True)
    analyze_api_g3.set_defaults(func=command_analyze_api_g3)

    analyze_roster = sub.add_parser("analyze-roster-v5")
    analyze_roster.add_argument("--prereg", default=str(DEFAULT_ROSTER_PREREG))
    analyze_roster.add_argument("--results", nargs="+", required=True)
    analyze_roster.add_argument("--schedules", nargs="+", required=True)
    analyze_roster.add_argument("--reading-log", required=True)
    analyze_roster.add_argument("--manual-minimum", type=int, default=40)
    analyze_roster.add_argument("--bootstrap-reps", type=int, default=20000)
    analyze_roster.add_argument("--seed", type=int, required=True)
    analyze_roster.add_argument("--run-id", required=True)
    analyze_roster.add_argument("--output", required=True)
    analyze_roster.set_defaults(func=command_analyze_roster_v5)

    decide_roster = sub.add_parser("decide-roster-v5")
    decide_roster.add_argument("--prereg", default=str(DEFAULT_ROSTER_PREREG))
    decide_roster.add_argument("--discovery-analysis", required=True)
    decide_roster.add_argument("--confirmation-analysis", required=True)
    decide_roster.add_argument("--combined-analysis", required=True)
    decide_roster.add_argument("--run-id", default="glm53-g3-roster-v5-final")
    decide_roster.add_argument("--output", required=True)
    decide_roster.set_defaults(func=command_decide_roster_v5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
