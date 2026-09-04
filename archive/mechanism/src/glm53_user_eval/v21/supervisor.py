"""Paid execution of the V21 exploratory intervention continuation."""

from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from src.glm53_user_eval.v11.downstream import (
    validate_downstream_assets,
    validate_runtime_proxy_token_contract,
)
from src.glm53_user_eval.v17.runtime import LoadedV17GLM53, raw_layer_deltas
from src.glm53_user_eval.v20.analysis import analyze_causal_rows
from src.glm53_user_eval.v20.contract import (
    atomic_json,
    atomic_jsonl,
    read_json,
    read_jsonl,
    read_yaml,
    sha256_file,
)
from src.glm53_user_eval.v20.supervisor import (
    _direction_bundle,
    _loader_compatible_runtime_config,
    _score_arm,
    _selected_proxy_rows,
)
from src.glm53_user_eval.v20.verification import verify_causal
from src.glm53_user_eval.v21.contract import validate_v21_prereg


def _deadline() -> dt.datetime:
    raw = os.environ.get("GLM53_V20_DEADLINE_UTC", "")
    if not raw:
        raise ValueError("V21 deadline is required")
    value = dt.datetime.fromisoformat(raw).astimezone(dt.UTC)
    if value <= dt.datetime.now(dt.UTC):
        raise ValueError("V21 deadline has passed")
    return value


def _remaining_seconds() -> float:
    return (_deadline() - dt.datetime.now(dt.UTC)).total_seconds()


def _require_time(seconds: float, reserve: float = 600.0) -> None:
    if float(seconds) + reserve > _remaining_seconds():
        raise RuntimeError("V21 continuation does not fit before the backup reserve")


def _validate_baseline(
    baseline_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]
) -> None:
    if len(baseline_rows) != 1404:
        raise ValueError("V21 baseline does not contain 1,404 rows")
    baseline = {str(row["sample_id"]): row for row in baseline_rows}
    selected = {str(row["sample_id"]): row for row in selected_rows}
    if len(baseline) != 1404 or set(baseline) != set(selected):
        raise ValueError("V21 baseline and reconstructed target keys differ")
    for sample_id, row in baseline.items():
        source = selected[sample_id]
        if (
            row["messages_sha256"] != source["messages_sha256"]
            or row["codebook_id"] != source["codebook_id"]
            or row["codebook_values"] != source["codebook_values"]
            or int(row["stage_index"]) != int(source["stage_index"])
        ):
            raise ValueError(f"V21 baseline binding differs for {sample_id}")


def runtime_smoke(
    runtime: LoadedV17GLM53,
    selected_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    label_ids: list[int],
    all16_direction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline = {str(row["sample_id"]): row for row in baseline_rows}
    smoke_count = int(config["forward"]["smoke_rows"])
    indices = np.linspace(0, len(selected_rows) - 1, smoke_count).round().astype(int)
    smoke = [selected_rows[index] for index in indices]
    live = []
    for row in smoke:
        forward = runtime.forward_intervened_batch(
            [row["messages"]],
            allowed_token_ids=label_ids,
            continuation=True,
            return_full_logits=True,
        )[0]
        expected = baseline[str(row["sample_id"])]
        live.append((row, forward, expected))
    allowed_error = max(
        float(np.max(np.abs(forward.allowed_logits - np.asarray(expected["allowed_logits"]))))
        for _, forward, expected in live
    )
    logsum_error = max(
        abs(float(forward.full_logsumexp) - float(expected["full_logsumexp"]))
        for _, forward, expected in live
    )
    metadata_equal = all(
        forward.prompt_sha256 == expected["prompt_sha256"]
        and int(forward.prompt_tokens) == int(expected["prompt_tokens"])
        and int(forward.full_argmax_token_id) == int(expected["full_argmax_token_id"])
        for _, forward, expected in live
    )
    layers = [10, 14, 18, 22, 26, 30]
    zero = {layer: np.zeros(4096, dtype=np.float32) for layer in layers}
    alpha_zero = runtime.forward_intervened_batch(
        [smoke[0]["messages"]],
        allowed_token_ids=label_ids,
        layer_deltas=zero,
        scope="user_content",
        continuation=True,
        return_full_logits=True,
    )[0]
    if alpha_zero.full_logits is None or live[0][1].full_logits is None:
        raise ValueError("V21 alpha-zero smoke omitted full logits")
    alpha_zero_error = float(np.max(np.abs(alpha_zero.full_logits - live[0][1].full_logits)))

    benchmark_count = int(config["throughput_gate"]["benchmark_rows"])
    benchmark = selected_rows[:benchmark_count]
    deltas = raw_layer_deltas(all16_direction, layers, 0.6)
    started = time.perf_counter()
    for row in benchmark:
        runtime.forward_intervened_batch(
            [row["messages"]],
            allowed_token_ids=label_ids,
            layer_deltas=deltas,
            scope="user_content",
            continuation=True,
        )
    elapsed = time.perf_counter() - started
    rate = benchmark_count / elapsed
    projected = (
        float(config["throughput_gate"]["projection_headroom_multiplier"])
        * int(config["throughput_gate"]["planned_new_prompt_evaluations"])
        / rate
    )
    reserve = float(config["throughput_gate"]["backup_reserve_seconds"])
    passed = bool(
        allowed_error <= float(config["forward"]["allowed_logit_tolerance_against_v20"])
        and logsum_error <= float(config["forward"]["full_logsumexp_tolerance_against_v20"])
        and metadata_equal
        and alpha_zero_error == float(config["forward"]["alpha_zero_full_logit_tolerance"])
        and projected + reserve <= _remaining_seconds()
    )
    return {
        "schema_version": "glm53_v21_runtime_smoke_v1",
        "passed": passed,
        "v20_baseline_max_allowed_logit_error": allowed_error,
        "v20_baseline_max_full_logsumexp_error": logsum_error,
        "v20_baseline_metadata_equal": metadata_equal,
        "alpha_zero_max_full_logit_error": alpha_zero_error,
        "benchmark_rows": benchmark_count,
        "benchmark_seconds": elapsed,
        "prompts_per_second": rate,
        "projected_new_science_seconds_with_headroom": projected,
        "fits_deadline_with_backup_reserve": projected + reserve <= _remaining_seconds(),
        "sample_ids": [row["sample_id"] for row in smoke],
    }


def run_exploratory_continuation(
    *,
    repo_root: Path,
    prereg_path: Path,
    runtime_path: Path,
    v20_runtime_path: Path,
    downstream_manifest_path: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    validate_v21_prereg(repo_root, prereg_path)
    prereg = read_yaml(prereg_path)
    config = read_yaml(runtime_path)
    v20_prereg = read_yaml(
        repo_root / prereg["immutable_inputs"]["v20_prereg"]["path"]
    )
    v20_runtime = read_yaml(v20_runtime_path)
    parent_design = read_json(repo_root / v20_prereg["immutable_inputs"]["parent_design"]["path"])
    design = read_json(repo_root / v20_prereg["immutable_inputs"]["design"]["path"])
    baseline_path = repo_root / prereg["immutable_inputs"]["baseline_raw_scores"]["path"]
    baseline_rows = read_jsonl(baseline_path)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    bundle = _direction_bundle(
        repo_root / v20_prereg["immutable_inputs"]["v19_directions"]["path"],
        repo_root / v20_prereg["immutable_inputs"]["null_signs"]["path"],
    )
    runtime = LoadedV17GLM53(
        model_path=model_path,
        config=_loader_compatible_runtime_config(v20_runtime),
    )
    try:
        fp8 = runtime.fp8_scale_report()
        atomic_json(output_root / "fp8_runtime_check.json", fp8)
        if not fp8["passed"]:
            return {"stage": "runtime", "decision": "stop_exact_fp8_runtime_failed"}

        downstream, all_proxy_rows, _ = validate_downstream_assets(
            repo_root=repo_root, manifest_path=downstream_manifest_path
        )
        selected = _selected_proxy_rows(all_proxy_rows, parent_design)
        _validate_baseline(baseline_rows, selected)
        atomic_json(
            output_root / "baseline_binding.json",
            {
                "schema_version": "glm53_v21_baseline_binding_v1",
                "rows": len(baseline_rows),
                "sha256": sha256_file(baseline_path),
                "v20_decision": "stop_local_parity_failed",
                "reused_without_rescoring": True,
            },
        )

        downstream_binding = read_json(
            repo_root / v20_prereg["immutable_inputs"]["downstream_manifest"]["path"]
        )
        codebook_payload = read_json(
            repo_root / downstream_binding["assets"]["proxy_codebooks"]["path"]
        )
        token_contract = read_json(
            repo_root / downstream_binding["assets"]["proxy_contract"]["path"]
        )
        token_check = validate_runtime_proxy_token_contract(
            runtime.processor,
            proxy_rows=all_proxy_rows,
            codebook_payload=codebook_payload,
            contract=token_contract,
        )
        atomic_json(output_root / "runtime_proxy_token_validation.json", token_check)
        label_ids = [int(value) for value in downstream["label_ids"]]
        smoke = runtime_smoke(
            runtime,
            selected,
            baseline_rows,
            label_ids=label_ids,
            all16_direction=bundle["all16"],
            config=config,
        )
        atomic_json(output_root / "runtime_smoke.json", smoke)
        if not smoke["passed"]:
            return {"stage": "runtime", "decision": "runtime_or_projection_failed"}
        _require_time(float(smoke["projected_new_science_seconds_with_headroom"]))

        binding = {
            "v21_prereg": sha256_file(prereg_path),
            "v20_baseline": sha256_file(baseline_path),
            "v20_design": sha256_file(
                repo_root / v20_prereg["immutable_inputs"]["design"]["path"]
            ),
            "v19_directions": v20_prereg["immutable_inputs"]["v19_directions"]["sha256"],
            "runtime_smoke": sha256_file(output_root / "runtime_smoke.json"),
        }
        causal_root = output_root / "causal"
        baseline = [dict(row, arm="baseline", scope="user_content") for row in baseline_rows]
        full_rows = list(baseline)
        full_specs = {
            "all16_deploy": ("all16", 0.6),
            "all16_eval": ("all16", -0.6),
            "status_deploy": ("status", 0.6),
            "actor_deploy": ("actor", 0.6),
            "formality": ("formality", 0.6),
        }
        for arm, (direction_name, alpha) in full_specs.items():
            full_rows.extend(
                _score_arm(
                    runtime,
                    selected,
                    arm=arm,
                    label_ids=label_ids,
                    batch_size=1,
                    output_root=causal_root / "parts",
                    binding=binding,
                    direction=bundle[direction_name],
                    alpha=alpha,
                )
            )
            atomic_jsonl(causal_root / "full_rows.partial.jsonl", full_rows)
        atomic_jsonl(causal_root / "full_rows.jsonl", full_rows)

        pilot_task_ids = [
            parent_design["tasks"][index]
            for index in design["signflip_control_surface"]["task_indices"]
        ]
        pilot_rows = [
            row
            for row in selected
            if int(row["stage_index"])
            in set(design["signflip_control_surface"]["identity_indices"])
            and str(row["stimulus_id"]) in set(pilot_task_ids)
        ]
        if len(pilot_rows) != 80:
            raise ValueError("V21 sign-flip subset does not contain 80 rows")
        null_rows: list[dict[str, Any]] = []
        for index in range(20):
            arm = f"signflip_{index:02d}"
            null_rows.extend(
                _score_arm(
                    runtime,
                    pilot_rows,
                    arm=arm,
                    label_ids=label_ids,
                    batch_size=1,
                    output_root=causal_root / "parts",
                    binding=binding,
                    direction=bundle[arm],
                    alpha=0.6,
                )
            )
            atomic_jsonl(causal_root / "null_rows.partial.jsonl", null_rows)
        atomic_jsonl(causal_root / "null_rows.jsonl", null_rows)

        analysis = analyze_causal_rows(
            full_rows,
            null_rows,
            pilot_task_ids=pilot_task_ids,
            reps=int(prereg["statistics"]["bootstrap_reps"]),
            seed=int(prereg["statistics"]["primary_seed"]),
        )
        atomic_json(causal_root / "analysis.json", analysis)
        verification = verify_causal(
            causal_root,
            pilot_task_ids=pilot_task_ids,
            primary_bootstrap_ci=analysis["arms"]["all16_deploy"]["delta_ci95_pp"],
            reps=int(prereg["statistics"]["bootstrap_reps"]),
            seed=int(prereg["statistics"]["independent_seed"]),
        )
        atomic_json(causal_root / "verification.json", verification)
        state = str(verification["recomputed_decision"])
        positive = bool(
            verification["passed"]
            and state == "fixed_hua_intervention_selectively_attenuates_user_conditioning"
        )
        decision = {
            "schema_version": "glm53_v21_exploratory_decision_v1",
            "completed": True,
            "positive_under_v20_causal_rules": positive,
            "decision": state,
            "scope": "exploratory_post_failed_local_parity_gate",
            "v20_local_parity_remains_failed": True,
            "checks": verification["recomputed_decision_checks"],
            "analysis_sha256": sha256_file(causal_root / "analysis.json"),
            "verification_sha256": sha256_file(causal_root / "verification.json"),
            "baseline_sha256": sha256_file(baseline_path),
        }
        atomic_json(output_root / "decision.json", decision)
        return {"stage": "causal", "decision": state, "exploratory": True, "completed": True}
    finally:
        atomic_json(
            output_root / "runtime_summary.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "deadline_utc": _deadline().isoformat(),
            },
        )
        runtime.close()


__all__ = ["run_exploratory_continuation", "runtime_smoke"]
