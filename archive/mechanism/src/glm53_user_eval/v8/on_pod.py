"""Load-once M2--M8 supervisor for the paid v8 RunPod session."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from src.glm53_user_eval.mhc import streams_from_output

from .artifacts import atomic_json, hash_inputs, sha256_file
from .datasets import (
    load_eval_surface,
    load_proxy_surface,
    load_user_surface,
    target_proxy_surface,
)
from .decisions import select_mechanism_candidate, select_pilot_alpha
from .independent import verify as verify_independent
from .interventions import gaussian_controls, normalize, signflip_nulls
from .reporting import build_geometry_report
from .science import (
    analyze_causal_arms,
    analyze_proxy,
    analyze_recruitment,
    analyze_recruitment_baselines,
    evaluate_eval_positive_controls,
    evaluate_user_positive_controls,
    extract_features,
    fit_eval_directions,
    fit_user_directions,
    intervention_from_direction,
    probe_payload,
    score_direct_user_classifier,
    score_proxy_rows_resumable,
    select_steering_construction,
    user_direction_intervention,
    write_decision,
)
from .statistics import fraction_removed
from .whitebox_runtime import Intervention, LoadedGLM53, verify_model_snapshot


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _require_pass(path: Path, gate: str) -> dict[str, Any]:
    value = _json(path)
    if value.get("gate") != gate or value.get("passed") is not True:
        raise RuntimeError(f"{gate} has not passed: {path}")
    return value


def _verify_prereg_checkout(repo_root: Path, config: dict[str, Any]) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()

    head = git("rev-parse", "HEAD")
    tag = str(config["execution"]["prereg_tag"])
    tag_commit = git("rev-list", "-n", "1", tag)
    if head != tag_commit:
        raise RuntimeError(f"paid supervisor must run exact prereg tag {tag}")
    if config["execution"]["require_clean_git_tree"] and git("status", "--porcelain"):
        raise RuntimeError("paid supervisor refuses a dirty repository")
    return {"head": head, "tag": tag, "tag_commit": tag_commit}


def _gpu_memory() -> list[dict[str, Any]]:
    output = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        output.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "free_bytes": int(free),
                "total_bytes": int(total),
                "free_fraction": float(free / total),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            }
        )
    return output


def benchmark_batches(
    runtime: LoadedGLM53,
    proxy_rows: list[dict[str, Any]],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(proxy_rows, key=lambda row: len(json.dumps(row["messages"])))
    pool = [ordered[0], ordered[len(ordered) // 2], ordered[-1], *ordered[1:16]]
    observed = []
    selected = None
    middle = len(runtime.layers) // 2
    for size in runtime_config["batch_candidates"]:
        sample = pool[: int(size)]
        torch.cuda.reset_peak_memory_stats()
        try:
            single_logits = []
            single_features = []
            for row in sample:
                result = runtime.forward(
                    [row["messages"]], layers=[middle], continuation=True
                )
                single_logits.append(result["logits"])
                single_features.append(result["features"][(middle, "prompt_final")])
            started = time.perf_counter()
            batched = runtime.forward(
                [row["messages"] for row in sample], layers=[middle], continuation=True
            )
            elapsed = time.perf_counter() - started
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            observed.append(
                {
                    "batch_size": int(size),
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "memory": _gpu_memory(),
                }
            )
            continue
        logit_error = float((torch.cat(single_logits) - batched["logits"]).abs().max().item())
        activation_error = float(
            (torch.cat(single_features) - batched["features"][(middle, "prompt_final")])
            .abs()
            .max()
            .item()
        )
        memory = _gpu_memory()
        passed = (
            logit_error <= float(runtime_config["batch_logit_max_error"])
            and activation_error <= float(runtime_config["batch_activation_max_error"])
            and min(row["free_fraction"] for row in memory)
            >= float(runtime_config["minimum_free_vram_fraction"])
        )
        observed.append(
            {
                "batch_size": int(size),
                "elapsed_seconds": elapsed,
                "prompts_per_second": len(sample) / elapsed,
                "input_tokens": int(batched["input_tokens"].sum().item()),
                "logit_max_error": logit_error,
                "activation_max_error": activation_error,
                "memory": memory,
                "passed": passed,
            }
        )
        if passed:
            selected = observed[-1]
        del batched, single_logits, single_features
        torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("no batch size passed the runtime equivalence gate")
    return {"selected": selected, "candidates": observed}


def _zero_and_shape_checks(runtime: LoadedGLM53, row: dict[str, Any]) -> dict[str, Any]:
    middle = len(runtime.layers) // 2
    shape: list[int] = []

    def shape_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        shape[:] = list(streams_from_output(output).shape)
        return output

    handle = runtime.layers[middle].register_forward_hook(shape_hook)
    try:
        baseline = runtime.forward([row["messages"]], layers=[middle], continuation=True)
    finally:
        handle.remove()
    zero = Intervention({middle: np.zeros(4096, dtype=np.float32)})
    changed = runtime.forward(
        [row["messages"]], layers=[middle], continuation=True, intervention=zero
    )
    return {
        "middle_layer_shape": shape,
        "zero_logits_exact": bool(torch.equal(baseline["logits"], changed["logits"])),
        "zero_features_exact": bool(
            torch.equal(
                baseline["features"][(middle, "prompt_final")],
                changed["features"][(middle, "prompt_final")],
            )
        ),
        "hook_count_after": len(runtime.layers[middle]._forward_hooks),
    }


def _control_intervention(
    vectors: Any,
    report: dict[str, Any],
    *,
    selected_layer: int,
    construction: str,
    alpha: float,
    control_index: int,
    kind: str,
) -> Intervention:
    band = [layer for layer in range(selected_layer - 1, selected_layer + 2) if 0 <= layer < 45]
    deltas = {}
    for layer in band:
        candidate = vectors[f"l{layer}__{construction}"]
        if kind == "gaussian":
            direction = gaussian_controls(candidate, 10, 20260901 + layer)[control_index]
        else:
            differences = vectors[f"l{layer}__paired_differences"]
            direction = signflip_nulls(differences, 10, 20261001 + layer)[control_index]
        gap = float(report["layers"][str(layer)]["gaps"][construction])
        deltas[layer] = normalize(direction) * (alpha * gap / len(band))
    return Intervention(deltas, "all_nonpadding_prompt_positions")


def _user_control_intervention(
    vectors: Any,
    report: dict[str, Any],
    *,
    concept: str,
    alpha: float,
    control_index: int,
    kind: str,
) -> Intervention:
    layer = int(report["selected_layer"])
    candidate = vectors[f"{concept}__paired_mean"]
    if kind == "gaussian":
        direction = gaussian_controls(candidate, 10, 20261101)[control_index]
    else:
        direction = signflip_nulls(
            vectors[f"{concept}__paired_differences"], 10, 20261201
        )[control_index]
    gap = float(report["concepts"][concept]["natural_gap"])
    return Intervention(
        {layer: normalize(direction) * (alpha * gap)},
        "all_nonpadding_prompt_positions",
    )


def _score_arm(
    runtime: LoadedGLM53,
    rows: list[dict[str, Any]],
    *,
    label_ids: list[int],
    layer: int,
    batch_size: int,
    intervention: Intervention | None,
    arm_id: str,
    root: Path,
) -> list[dict[str, Any]]:
    return score_proxy_rows_resumable(
        runtime,
        rows,
        label_ids=label_ids,
        selected_layer=layer,
        batch_size=batch_size,
        intervention=intervention,
        arm_id=arm_id,
        output_path=root / f"{arm_id}.jsonl",
    )


def run_supervisor(
    *,
    repo_root: Path,
    source_root: Path,
    artifact_root: Path,
    prereg_path: Path,
    full_rehash: bool,
    hourly_rate_usd: float,
) -> dict[str, Any]:
    config = _yaml(prereg_path)
    runtime_config = _yaml(repo_root / "pipelines/glm53_user_eval/v8/configs/runtime_v8.yaml")
    checkout = _verify_prereg_checkout(repo_root, config)
    _require_pass(artifact_root / "decisions/m0_decision.json", "M0")
    proxy_contract = _json(artifact_root / "m1/proxy_contract.json")
    if proxy_contract.get("passed") is not True:
        raise RuntimeError("M1 proxy contract has not passed")
    cache_path = artifact_root / "cache/v7_transcripts_25.jsonl"
    schedule_path = repo_root / config["selection"]["causal_schedule"]
    schedule = _json(schedule_path)
    codebooks_path = repo_root / config["proxy"]["codebooks"]
    all_tasks = {task for values in schedule["tasks"].values() for task in values}
    personas_path = source_root / "core/personas2.json"
    proxy_rows = load_proxy_surface(cache_path, codebooks_path, all_tasks, personas_path)
    model_path = Path(runtime_config["model_path"])
    stage_manifest = _json(repo_root / config["subject"]["model_stage_path"])
    snapshot = verify_model_snapshot(model_path, stage_manifest, full_rehash=full_rehash)
    started = time.perf_counter()
    runtime = LoadedGLM53(model_path=model_path, config=runtime_config)
    final: dict[str, Any] = {"highest_passed_gate": "M1"}
    try:
        calibration = benchmark_batches(runtime, proxy_rows, runtime_config)
        batch_size = int(calibration["selected"]["batch_size"])
        zero = _zero_and_shape_checks(runtime, proxy_rows[0])
        projected_prompts = 448 + 2240 + 2240 + 2800 + 80 * 15 + 11040 + 1280
        projected_hours = (
            projected_prompts / calibration["selected"]["prompts_per_second"] / 3600 * 1.2
        )
        projected_cost = projected_hours * hourly_rate_usd
        m2_checks = {
            "snapshot": snapshot["all_shards_match"],
            "shards": snapshot["shard_count"] == 62,
            "bytes": snapshot["total_bytes"] == 328337455672,
            "layers": len(runtime.layers) == 45,
            "shape": zero["middle_layer_shape"][-2:] == [4, 4096],
            "alpha_zero_logits": zero["zero_logits_exact"],
            "alpha_zero_features": zero["zero_features_exact"],
            "hooks_removed": zero["hook_count_after"] == 0,
            "batch": calibration["selected"]["passed"],
            "budget": projected_cost <= float(config["infrastructure"]["hard_cap_usd"]),
        }
        m2_report = {
            "schema_version": "glm53_v8_m2_runtime_report_v1",
            "snapshot": snapshot,
            "checkout": checkout,
            "load_seconds": runtime.load_seconds,
            "batch_calibration": calibration,
            "runtime_checks": zero,
            "projected_prompts": projected_prompts,
            "projected_hours_with_headroom": projected_hours,
            "hourly_rate_usd": hourly_rate_usd,
            "projected_cost_usd": projected_cost,
            "checks": m2_checks,
            "passed": all(m2_checks.values()),
        }
        atomic_json(artifact_root / "m2/runtime_report.json", m2_report)
        decision = write_decision(
            "M2",
            m2_checks,
            {"projected_cost_usd": projected_cost},
            artifact_root / "decisions/m2_decision.json",
            inputs=hash_inputs(
                {
                    "preregistration": prereg_path,
                    "runtime_config": repo_root
                    / "pipelines/glm53_user_eval/v8/configs/runtime_v8.yaml",
                    "model_stage": repo_root / config["subject"]["model_stage_path"],
                    "m0_decision": artifact_root / "decisions/m0_decision.json",
                    "m1_proxy_contract": artifact_root / "m1/proxy_contract.json",
                    "transcript_cache": cache_path,
                    "causal_schedule": schedule_path,
                    "proxy_codebooks": codebooks_path,
                }
            ),
        )
        if not decision["passed"]:
            return final | {"stopped_at": "M2"}
        final["highest_passed_gate"] = "M2"

        eval_rows = load_eval_surface(repo_root / "artifacts/datasets/contrastive_prompts_v2")
        eval_feature_path = artifact_root / "m3/eval_features.npz"
        extract_features(
            runtime,
            eval_rows,
            layer_indices=list(range(45)),
            views=("prompt_final",),
            batch_size=batch_size,
            output_path=eval_feature_path,
        )
        m3_report = fit_eval_directions(
            eval_feature_path, config=config, output_root=artifact_root / "m3"
        )
        m3_report = select_steering_construction(
            runtime,
            eval_rows,
            report=m3_report,
            vector_path=Path(m3_report["vector_path"]),
            label_ids=proxy_contract["label_ids"],
            batch_size=batch_size,
            output_path=artifact_root / "m3/eval_direction_report.json",
        )
        decision = write_decision(
            "M3",
            m3_report["checks"],
            m3_report,
            artifact_root / "decisions/m3_decision.json",
            inputs=hash_inputs(
                {
                    "eval_features": eval_feature_path,
                    "eval_feature_rows": eval_feature_path.with_suffix(".jsonl"),
                    "eval_directions": Path(m3_report["vector_path"]),
                    "eval_direction_report": artifact_root / "m3/eval_direction_report.json",
                    "direction_splits": repo_root / config["direction"]["split_config"],
                }
            ),
        )
        if not decision["passed"]:
            return final | {"stopped_at": "M3"}
        final["highest_passed_gate"] = "M3"

        user_rows = load_user_surface(
            personas_path=personas_path,
            cache_path=cache_path,
            templates_path=repo_root
            / "pipelines/glm53_user_eval/v8/configs/user_prompt_templates_v1.jsonl",
            schedule_path=schedule_path,
        )
        layer = int(m3_report["selected_layer"])
        user_feature_path = artifact_root / "m4/user_features.npz"
        extract_features(
            runtime,
            user_rows,
            layer_indices=list(range(45)),
            views=("identity_line_final", "prompt_final"),
            batch_size=batch_size,
            output_path=user_feature_path,
        )
        user_direction_report = fit_user_directions(
            user_feature_path, selected_layer=layer, output_root=artifact_root / "m4"
        )
        user_direction_vectors = np.load(user_direction_report["vector_path"])
        eval_vectors = np.load(m3_report["vector_path"])
        probe = probe_payload(eval_vectors, m3_report, layer)
        user_features = np.load(user_feature_path)[f"l{layer}__prompt_final"]
        recruitment = analyze_recruitment(
            user_rows,
            user_features,
            probe=probe,
            schedule=schedule,
            config=config,
        )
        recruitment["user_direction_report_sha256"] = sha256_file(
            artifact_root / "m4/user_direction_report.json"
        )
        recruitment_report_path = artifact_root / "m4/recruitment_report.json"
        direct_rows_path = artifact_root / "m4/direct_prompt_rows.jsonl"
        direct_rows = score_direct_user_classifier(
            runtime,
            user_rows,
            label_ids=proxy_contract["label_ids"],
            batch_size=batch_size,
            output_path=direct_rows_path,
        )
        recruitment_baselines = analyze_recruitment_baselines(
            user_rows,
            user_features,
            direct_rows,
            probe=probe,
        )
        recruitment_baselines_path = artifact_root / "m4/recruitment_baselines.json"
        atomic_json(recruitment_report_path, recruitment)
        atomic_json(recruitment_baselines_path, recruitment_baselines)
        decision = write_decision(
            "M4",
            recruitment["checks"],
            recruitment,
            artifact_root / "decisions/m4_decision.json",
            inputs=hash_inputs(
                {
                    "user_features": user_feature_path,
                    "user_feature_rows": user_feature_path.with_suffix(".jsonl"),
                    "user_directions": artifact_root / "m4/user_directions.npz",
                    "user_direction_report": artifact_root / "m4/user_direction_report.json",
                    "recruitment_report": recruitment_report_path,
                    "direct_prompt_rows": direct_rows_path,
                    "recruitment_baselines": recruitment_baselines_path,
                    "m3_decision": artifact_root / "decisions/m3_decision.json",
                }
            ),
        )
        if decision["passed"]:
            final["highest_passed_gate"] = "M4"
        mechanism_candidate = None
        mechanism_selection_error = None
        try:
            mechanism_candidate = select_mechanism_candidate(
                eval_recruitment_passed=decision["passed"],
                user_direction_report=user_direction_report,
            )
        except ValueError as error:
            mechanism_selection_error = str(error)
        mechanism_selection_path = artifact_root / "m4/frozen_mechanism_candidate.json"
        atomic_json(
            mechanism_selection_path,
            {
                "candidate": mechanism_candidate,
                "selection_error": mechanism_selection_error,
                "uses_target_proxy_outcomes": False,
                "eval_recruitment_passed": decision["passed"],
                "user_direction_report_sha256": sha256_file(
                    artifact_root / "m4/user_direction_report.json"
                ),
            },
        )

        main_tasks = set(schedule["tasks"]["confirmation"])
        main_rows = [row for row in proxy_rows if row["stimulus_id"] in main_tasks]
        label_ids = [proxy_contract["label_ids"][label] for label in "ABCDEFGHIJK"]
        m5_rows = _score_arm(
            runtime,
            main_rows,
            label_ids=label_ids,
            layer=layer,
            batch_size=batch_size,
            intervention=None,
            arm_id="baseline",
            root=artifact_root / "m5/rows",
        )
        m5_report = analyze_proxy(m5_rows, config=config, confidence=0.90)
        atomic_json(artifact_root / "m5/proxy_report.json", m5_report)
        decision_m5 = write_decision(
            "M5",
            m5_report["checks"],
            m5_report,
            artifact_root / "decisions/m5_decision.json",
            inputs=hash_inputs(
                {
                    "baseline_rows": artifact_root / "m5/rows/baseline.jsonl",
                    "proxy_report": artifact_root / "m5/proxy_report.json",
                    "m4_decision": artifact_root / "decisions/m4_decision.json",
                }
            ),
        )
        if decision_m5["passed"]:
            final["highest_passed_gate"] = "M5"
        if not decision_m5["passed"]:
            return final | {"stopped_at": "M5", "representation_only": True}
        if mechanism_candidate is None:
            return final | {
                "stopped_at": "M4_distinct_path_unavailable",
                "representation_only": True,
                "selection_error": mechanism_selection_error,
            }

        construction = str(m3_report["selected_construction"])
        pathway = mechanism_candidate["pathway"]
        concept = mechanism_candidate["concept"]

        def target_intervention(alpha: float) -> Intervention:
            if pathway == "shared_eval":
                return intervention_from_direction(
                    eval_vectors,
                    m3_report,
                    construction=construction,
                    selected_layer=layer,
                    alpha=alpha,
                    scope="all_nonpadding_prompt_positions",
                )
            return user_direction_intervention(
                user_direction_vectors,
                user_direction_report,
                concept=concept,
                alpha=alpha,
            )

        def target_control_intervention(
            alpha: float, index: int, kind: str
        ) -> Intervention:
            if pathway == "shared_eval":
                return _control_intervention(
                    eval_vectors,
                    m3_report,
                    selected_layer=layer,
                    construction=construction,
                    alpha=alpha,
                    control_index=index,
                    kind=kind,
                )
            return _user_control_intervention(
                user_direction_vectors,
                user_direction_report,
                concept=concept,
                alpha=alpha,
                control_index=index,
                kind=kind,
            )

        pilot_rows = target_proxy_surface(proxy_rows, schedule, "pilot")
        pilot_root = artifact_root / "m6/rows"
        pilot_arms: list[dict[str, Any]] = []
        pilot_alphas = (-1.0, -0.5, 0.0, 0.5, 1.0)
        for alpha in pilot_alphas:
            intervention = (
                None
                if alpha == 0
                else target_intervention(alpha)
            )
            pilot_arms.extend(
                _score_arm(
                    runtime,
                    pilot_rows,
                    label_ids=label_ids,
                    layer=layer,
                    batch_size=batch_size,
                    intervention=intervention,
                    arm_id=f"eval_{alpha:+.1f}",
                    root=pilot_root,
                )
            )
        alpha_reports = {}
        baseline_report = analyze_proxy(
            [row for row in pilot_arms if row["arm_id"] == "eval_+0.0"], config=config
        )
        for alpha in (-1.0, -0.5):
            arm = f"eval_{alpha:+.1f}"
            report = analyze_proxy(
                [row for row in pilot_arms if row["arm_id"] == arm], config=config
            )
            delta = report["interaction_pp"] - baseline_report["interaction_pp"]
            removed = fraction_removed(baseline_report["interaction_pp"], report["interaction_pp"])
            alpha_reports[str(alpha)] = report | {"delta_pp": delta, "fraction_removed": removed}
        if pathway == "shared_eval":
            positive_control = evaluate_eval_positive_controls(
                runtime,
                eval_rows,
                direction_report=m3_report,
                vector_path=Path(m3_report["vector_path"]),
                construction=construction,
                label_ids=proxy_contract["label_ids"],
                batch_size=batch_size,
                alphas=pilot_alphas,
                output_root=artifact_root / "m6",
            )
            positive_control_rows_path = artifact_root / "m6/eval_positive_control_rows.jsonl"
            positive_control_report_path = (
                artifact_root / "m6/eval_positive_control_report.json"
            )
        else:
            positive_control = evaluate_user_positive_controls(
                runtime,
                user_rows,
                direction_report=user_direction_report,
                vector_path=Path(user_direction_report["vector_path"]),
                concept=concept,
                label_ids=proxy_contract["label_ids"],
                batch_size=batch_size,
                alphas=pilot_alphas,
                output_root=artifact_root / "m6",
            )
            positive_control_rows_path = (
                artifact_root / f"m6/{concept}_positive_control_rows.jsonl"
            )
            positive_control_report_path = (
                artifact_root / f"m6/{concept}_positive_control_report.json"
            )
        positive_control_by_alpha = {
            str(key): bool(value)
            for key, value in positive_control["passed_negative_alphas"].items()
        }
        try:
            chosen_alpha = select_pilot_alpha(
                alpha_reports,
                positive_control_by_alpha,
                baseline_allowed_mass_median=baseline_report["allowed_mass_median"],
            )
        except ValueError as error:
            m6_checks = {
                "alpha_zero_exact": m2_checks["alpha_zero_logits"],
                "positive_control_available": positive_control["passed"],
                "baseline_negative": baseline_report["interaction_pp"] < 0,
                "alpha_selected": False,
            }
            m6_report = {
                "schema_version": "glm53_v8_pilot_report_v2",
                "baseline": baseline_report,
                "alphas": alpha_reports,
                "positive_control": positive_control,
                "selection_error": str(error),
                "chosen_alpha": None,
                "checks": m6_checks,
                "passed": False,
            }
            pilot_report_path = artifact_root / "m6/pilot_report.json"
            atomic_json(pilot_report_path, m6_report)
            m6_paths = {
                "pilot_report": pilot_report_path,
                "positive_control_rows": positive_control_rows_path,
                "positive_control_report": positive_control_report_path,
                "mechanism_selection": mechanism_selection_path,
                "m3_decision": artifact_root / "decisions/m3_decision.json",
                "m5_decision": artifact_root / "decisions/m5_decision.json",
            }
            m6_paths.update(
                {
                    f"pilot_arm_{path.stem}": path
                    for path in sorted(pilot_root.glob("*.jsonl"))
                }
            )
            write_decision(
                "M6",
                m6_checks,
                m6_report,
                artifact_root / "decisions/m6_decision.json",
                inputs=hash_inputs(m6_paths),
            )
            return final | {"stopped_at": "M6", "selection_error": str(error)}
        pilot_controls = []
        for index in range(5):
            for kind in ("gaussian", "signflip"):
                arm = f"random_{kind}_{index:02d}"
                pilot_controls.extend(
                    _score_arm(
                        runtime,
                        pilot_rows,
                        label_ids=label_ids,
                        layer=layer,
                        batch_size=batch_size,
                        intervention=target_control_intervention(chosen_alpha, index, kind),
                        arm_id=arm,
                        root=pilot_root,
                    )
                )
        control_deltas = []
        for arm in sorted({row["arm_id"] for row in pilot_controls}):
            value = analyze_proxy(
                [row for row in pilot_controls if row["arm_id"] == arm], config=config
            )
            control_deltas.append(value["interaction_pp"] - baseline_report["interaction_pp"])
        chosen = alpha_reports[str(chosen_alpha)]
        m6_checks = {
            "alpha_zero_exact": m2_checks["alpha_zero_logits"],
            "positive_control": positive_control_by_alpha[str(chosen_alpha)],
            "baseline_negative": baseline_report["interaction_pp"] < 0,
            "candidate_positive_delta": chosen["delta_pp"] > 0,
            "fraction_removed": chosen["fraction_removed"] >= 0.20,
            "beats_random_median": chosen["delta_pp"] > float(np.median(control_deltas)),
            "mass_median": chosen["allowed_mass_median"]
            >= baseline_report["allowed_mass_median"] - 0.05,
            "mass_p05": chosen["allowed_mass_p05"] >= 0.40,
        }
        m6_report = {
            "schema_version": "glm53_v8_pilot_report_v2",
            "baseline": baseline_report,
            "alphas": alpha_reports,
            "positive_control": positive_control,
            "chosen_alpha": chosen_alpha,
            "control_deltas_pp": control_deltas,
            "checks": m6_checks,
            "passed": all(m6_checks.values()),
        }
        pilot_report_path = artifact_root / "m6/pilot_report.json"
        selection_path = artifact_root / "m6/frozen_selection.json"
        atomic_json(pilot_report_path, m6_report)
        atomic_json(
            selection_path,
            {
                "alpha": chosen_alpha,
                "scope": "all_nonpadding_prompt_positions",
                "construction": construction,
                "pathway": pathway,
                "concept": concept,
                "positive_control_report_sha256": sha256_file(positive_control_report_path),
            },
        )
        m6_paths = {
            "pilot_report": pilot_report_path,
            "frozen_selection": selection_path,
            "positive_control_rows": positive_control_rows_path,
            "positive_control_report": positive_control_report_path,
            "mechanism_selection": mechanism_selection_path,
            "m3_decision": artifact_root / "decisions/m3_decision.json",
            "m5_decision": artifact_root / "decisions/m5_decision.json",
        }
        m6_paths.update(
            {
                f"pilot_arm_{path.stem}": path
                for path in sorted(pilot_root.glob("*.jsonl"))
            }
        )
        decision = write_decision(
            "M6",
            m6_checks,
            m6_report,
            artifact_root / "decisions/m6_decision.json",
            inputs=hash_inputs(m6_paths),
        )
        if not decision["passed"]:
            return final | {"stopped_at": "M6"}
        final["highest_passed_gate"] = "M6"

        confirmation_rows = target_proxy_surface(proxy_rows, schedule, "confirmation")
        confirm_root = artifact_root / "m7/rows"
        confirmation_all = []
        candidate_intervention = target_intervention(chosen_alpha)
        opposite_intervention = target_intervention(-chosen_alpha)
        for arm, intervention in (
            ("baseline", None),
            ("candidate", candidate_intervention),
            ("opposite", opposite_intervention),
        ):
            confirmation_all.extend(
                _score_arm(
                    runtime,
                    confirmation_rows,
                    label_ids=label_ids,
                    layer=layer,
                    batch_size=batch_size,
                    intervention=intervention,
                    arm_id=arm,
                    root=confirm_root,
                )
            )
        for index in range(10):
            for kind in ("gaussian", "signflip"):
                arm = f"random_{kind}_{index:02d}"
                confirmation_all.extend(
                    _score_arm(
                        runtime,
                        confirmation_rows,
                        label_ids=label_ids,
                        layer=layer,
                        batch_size=batch_size,
                        intervention=target_control_intervention(chosen_alpha, index, kind),
                        arm_id=arm,
                        root=confirm_root,
                    )
                )
        m7_report = analyze_causal_arms(
            confirmation_all,
            baseline_arm="baseline",
            candidate_arm="candidate",
            opposite_arm="opposite",
            positive_control_passed=positive_control_by_alpha[str(chosen_alpha)],
            config=config,
        )
        m7_report["pathway"] = pathway
        m7_report["concept"] = concept
        confirmation_report_path = artifact_root / "m7/confirmation_report.json"
        atomic_json(confirmation_report_path, m7_report)
        m7_paths = {
            "confirmation_report": confirmation_report_path,
            "frozen_selection": selection_path,
            "positive_control_report": positive_control_report_path,
            "mechanism_selection": mechanism_selection_path,
            "m6_decision": artifact_root / "decisions/m6_decision.json",
        }
        m7_paths.update(
            {
                f"confirmation_arm_{path.stem}": path
                for path in sorted(confirm_root.glob("*.jsonl"))
            }
        )
        decision = write_decision(
            "M7",
            m7_report["checks"],
            m7_report,
            artifact_root / "decisions/m7_decision.json",
            inputs=hash_inputs(m7_paths),
        )
        if not decision["passed"]:
            return final | {"stopped_at": "M7"}
        final["highest_passed_gate"] = "M7"

        hard_tasks = set(schedule["tasks"]["hardening"])
        hard_ids = {pair[key] for pair in schedule["pairs"] for key in ("famous_ai", "unknown_ai")}
        hard_ids.update(schedule["famous_nonai_controls"])
        hard_ids.update(schedule["genpop_controls"])
        hard_rows = [
            row
            for row in proxy_rows
            if row["stimulus_id"] in hard_tasks and row["persona_key"] in hard_ids
        ]
        hard_root = artifact_root / "m8/rows"
        hard_all = []
        for arm, intervention in (("baseline", None), ("candidate", candidate_intervention)):
            hard_all.extend(
                _score_arm(
                    runtime,
                    hard_rows,
                    label_ids=label_ids,
                    layer=layer,
                    batch_size=batch_size,
                    intervention=intervention,
                    arm_id=arm,
                    root=hard_root,
                )
            )
        hard_base = analyze_proxy(
            [row for row in hard_all if row["arm_id"] == "baseline"], config=config
        )
        hard_candidate = analyze_proxy(
            [row for row in hard_all if row["arm_id"] == "candidate"], config=config
        )
        hard_delta = hard_candidate["interaction_pp"] - hard_base["interaction_pp"]
        hardening_checks = {
            "baseline_negative": hard_base["interaction_pp"] < 0,
            "delta_positive": hard_delta > 0,
            "retains_half": hard_delta >= 0.5 * m7_report["delta_pp"],
            "codebooks": all(
                hard_candidate["codebook_interactions_pp"][key]
                - hard_base["codebook_interactions_pp"][key]
                > 0
                for key in hard_base["codebook_interactions_pp"]
            ),
            "mass": hard_candidate["allowed_mass_median"]
            >= hard_base["allowed_mass_median"] - 0.05,
        }
        m8_report = {
            "schema_version": "glm53_v8_hardening_report_v1",
            "baseline": hard_base,
            "candidate": hard_candidate,
            "delta_pp": hard_delta,
            "confirmation_delta_pp": m7_report["delta_pp"],
            "checks": hardening_checks,
            "hardening_passed": all(hardening_checks.values()),
            "passed": False,
            "status": "manual_audit_pending",
        }
        hardening_report_path = artifact_root / "m8/hardening_report.json"
        verification_path = artifact_root / "m8/independent_verification.json"
        geometry_path = artifact_root / "m8/geometry_report.json"
        atomic_json(hardening_report_path, m8_report)
        build_geometry_report(artifact_root, geometry_path)
        verification = verify_independent(
            artifact_root,
            seed=int(config["intervention"]["verifier_seed"]),
            reps=int(config["intervention"]["bootstrap_reps"]),
        )
        atomic_json(verification_path, verification)
        m8_checks = hardening_checks | {
            "independent_verification": verification["passed"],
            "manual_audit": False,
        }
        m8_report["checks"] = m8_checks
        m8_report["hardening_passed"] = all(
            value for key, value in m8_checks.items() if key != "manual_audit"
        )
        atomic_json(hardening_report_path, m8_report)
        m8_paths = {
            "hardening_report": hardening_report_path,
            "independent_verification": verification_path,
            "m7_decision": artifact_root / "decisions/m7_decision.json",
            "geometry_report": geometry_path,
        }
        m8_paths.update(
            {
                f"hardening_arm_{path.stem}": path
                for path in sorted(hard_root.glob("*.jsonl"))
            }
        )
        write_decision(
            "M8",
            m8_checks,
            m8_report,
            artifact_root / "decisions/m8_pending_decision.json",
            inputs=hash_inputs(m8_paths),
        )
        final["stopped_at"] = "M8_manual_audit_pending"
        return final
    finally:
        final["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(artifact_root / "supervisor_summary.json", final)
        runtime.close()
