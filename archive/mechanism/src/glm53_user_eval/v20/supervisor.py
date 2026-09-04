"""Fail-closed paid execution for the V20 direct Hua experiment."""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from src.glm53_user_eval.v11.downstream import (
    analyze_local_proxy,
    proxy_from_compact_logits,
    validate_downstream_assets,
    validate_runtime_proxy_token_contract,
)
from src.glm53_user_eval.v17.runtime import LoadedV17GLM53, raw_layer_deltas
from src.glm53_user_eval.v20.analysis import GROUPS, analyze_causal_rows
from src.glm53_user_eval.v20.contract import (
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    read_json,
    read_yaml,
    sha256_file,
)
from src.glm53_user_eval.v20.verification import verify_causal, verify_local_parity


def _loader_compatible_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Supply the proven V11 loader with V20's already-frozen runtime facts.

    V20 froze the B300 topology and exact GLM architecture, but its compact
    runtime document omitted the compatibility keys used by the inherited
    V11 loader. Keep the immutable runtime file unchanged and construct the
    loader-facing view explicitly.
    """

    compatible = copy.deepcopy(config)
    runpod = compatible["runpod"]
    architecture = compatible["architecture"]
    if int(runpod["gpu_count"]) != 2:
        raise ValueError("V20 loader compatibility requires the frozen two-GPU topology")
    if str(runpod["gpu_id"]) != "NVIDIA B300 SXM6 AC":
        raise ValueError("V20 loader compatibility requires the frozen B300 GPU type")
    if int(architecture["text_layers"]) != 45:
        raise ValueError("V20 loader compatibility requires 45 text layers")
    if int(architecture["forget_gate_scale_inv_tensors"]) != 68:
        raise ValueError("V20 loader compatibility requires 68 forget-gate scale tensors")

    # These counts are properties of the hash-bound official checkpoint and
    # are the same constants used by the previously passing V17 runtime.
    architecture["linear_attention_layers"] = 34
    architecture["sparse_attention_layers"] = 11
    if (
        int(architecture["linear_attention_layers"])
        + int(architecture["sparse_attention_layers"])
        != int(architecture["text_layers"])
    ):
        raise ValueError("V20 attention-layer counts do not sum to the frozen layer count")
    if 2 * int(architecture["linear_attention_layers"]) != int(
        architecture["forget_gate_scale_inv_tensors"]
    ):
        raise ValueError("V20 forget-gate scale count is inconsistent with the architecture")
    compatible["runtime_checks"] = {
        "expected_cuda_devices": int(runpod["gpu_count"]),
        "expected_gpu_name": str(runpod["gpu_id"]),
    }
    return compatible


def _deadline() -> dt.datetime:
    raw = os.environ.get("GLM53_V20_DEADLINE_UTC", "")
    if not raw:
        raise ValueError("GLM53_V20_DEADLINE_UTC is required")
    value = dt.datetime.fromisoformat(raw).astimezone(dt.UTC)
    if value <= dt.datetime.now(dt.UTC):
        raise ValueError("V20 deadline has passed")
    return value


def _remaining_seconds() -> float:
    return (_deadline() - dt.datetime.now(dt.UTC)).total_seconds()


def _require_time(seconds: float, reserve: float = 600.0) -> None:
    if float(seconds) + reserve > _remaining_seconds():
        raise RuntimeError("next V20 stage does not fit before the backup reserve")


def _selected_proxy_rows(
    rows: list[dict[str, Any]], parent_design: dict[str, Any]
) -> list[dict[str, Any]]:
    people = {group: set(parent_design["identities"][group]) for group in GROUPS}
    tasks = set(parent_design["tasks"])
    selected = [
        dict(row)
        for row in rows
        if row["persona_key"] in people[row["group"]] and row["stimulus_id"] in tasks
    ]
    if len(selected) != int(parent_design["reconstructable_base_rows"]):
        raise ValueError("V20 reconstructable row count differs from the frozen design")
    counts = {group: sum(row["group"] == group for row in selected) for group in GROUPS}
    if counts != parent_design["reconstructable_rows_by_group"]:
        raise ValueError("V20 group row counts differ from the frozen design")
    indices = {
        group: {person: index for index, person in enumerate(parent_design["identities"][group])}
        for group in GROUPS
    }
    for row in selected:
        row["stage_index"] = indices[row["group"]][row["persona_key"]]
    order = {
        (group, person, task): ordinal
        for group in GROUPS
        for person in parent_design["identities"][group]
        for task in parent_design["tasks"]
        for ordinal in [
            GROUPS.index(group) * 16 * len(parent_design["tasks"])
            + parent_design["identities"][group].index(person) * len(parent_design["tasks"])
            + parent_design["tasks"].index(task)
        ]
    }
    selected.sort(key=lambda row: order[(row["group"], row["persona_key"], row["stimulus_id"])])
    return selected


def _calibration_rows(
    runtime: LoadedV17GLM53, rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    lengths = runtime.downstream_token_lengths([row["messages"] for row in rows], continuation=True)
    ordered = sorted(
        zip(rows, lengths, strict=True),
        key=lambda item: (item[1], item[0]["sample_id"]),
    )
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[index][0] for index in positions]


def _direction_bundle(directions_path: Path, nulls_path: Path) -> dict[str, np.ndarray]:
    with np.load(directions_path) as arrays:
        pairs = np.asarray(arrays["hua_pair_differences"], dtype=np.float32)
        all16 = np.asarray(arrays["all16"], dtype=np.float32)
        frozen_status = np.asarray(arrays["status"], dtype=np.float32)
        status = pairs[np.asarray([9, 10, 12, 13, 14, 15])].mean(axis=0).astype(np.float32)
        actor = pairs[np.asarray([0, 2, 3, 4, 5, 7, 11])].mean(axis=0).astype(np.float32)
        formality = np.asarray(arrays["formality"], dtype=np.float32)
    if pairs.shape != (16, 45, 4096):
        raise ValueError("V20 Hua pair differences have the wrong shape")
    if not np.array_equal(status, frozen_status):
        raise ValueError("V20 status direction differs from the frozen V19 direction")
    bundle = {
        "all16": all16,
        "status": status,
        "actor": actor,
        "formality": formality,
    }
    nulls = read_json(nulls_path)["controls"]
    target_norm = np.linalg.norm(all16.astype(np.float64), axis=1)
    if np.any(target_norm == 0):
        raise ValueError("V20 all-16 direction contains a zero-norm layer")
    for record in nulls:
        signs = np.asarray(record["signs"], dtype=np.float64)[:, None, None]
        raw = np.mean(pairs.astype(np.float64) * signs, axis=0)
        raw_norm = np.linalg.norm(raw, axis=1)
        if np.any(raw_norm == 0):
            raise ValueError("V20 sign-flip direction contains a zero-norm layer")
        matched = raw * (target_norm / raw_norm)[:, None]
        bundle[str(record["control_id"])] = matched.astype(np.float32)
    return bundle


def _full_logit_error(left: Any, right: Any) -> float:
    if left.full_logits is None or right.full_logits is None:
        raise ValueError("V20 batch calibration omitted full logits")
    return float(np.max(np.abs(left.full_logits - right.full_logits)))


def calibrate_batches(
    runtime: LoadedV17GLM53,
    rows: list[dict[str, Any]],
    *,
    label_ids: list[int],
    all16_direction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    forward_config = config["forward"]
    throughput = config["throughput_gate"]
    candidates = [int(value) for value in forward_config["candidate_batch_sizes"]]
    representatives = _calibration_rows(runtime, rows, int(throughput["benchmark_rows"]))
    runtime.forward_intervened_batch(
        [representatives[0]["messages"]],
        allowed_token_ids=label_ids,
        continuation=True,
    )
    reference_started = time.perf_counter()
    references = [
        runtime.forward_intervened_batch(
            [row["messages"]],
            allowed_token_ids=label_ids,
            continuation=True,
            return_full_logits=True,
        )[0]
        for row in representatives
    ]
    reference_seconds = time.perf_counter() - reference_started
    layers = [10, 14, 18, 22, 26, 30]
    deployment_deltas = raw_layer_deltas(all16_direction, layers, 0.6)
    intervention_references = [
        runtime.forward_intervened_batch(
            [row["messages"]],
            allowed_token_ids=label_ids,
            layer_deltas=deployment_deltas,
            scope="user_content",
            continuation=True,
            return_full_logits=True,
        )[0]
        for row in representatives[:4]
    ]
    zero = {layer: np.zeros(4096, dtype=np.float32) for layer in layers}
    alpha_zero = runtime.forward_intervened_batch(
        [representatives[0]["messages"]],
        allowed_token_ids=label_ids,
        layer_deltas=zero,
        scope="user_content",
        continuation=True,
        return_full_logits=True,
    )[0]
    alpha_zero_error = _full_logit_error(alpha_zero, references[0])
    records: list[dict[str, Any]] = []
    for batch_size in candidates:
        started = time.perf_counter()
        observed: list[Any] = []
        error: str | None = None
        try:
            for start in range(0, len(representatives), batch_size):
                batch = representatives[start : start + batch_size]
                observed.extend(
                    runtime.forward_intervened_batch(
                        [row["messages"] for row in batch],
                        allowed_token_ids=label_ids,
                        continuation=True,
                        return_full_logits=True,
                    )
                )
        except RuntimeError as caught:
            if "out of memory" not in str(caught).lower():
                raise
            error = "cuda_out_of_memory"
            with suppress(Exception):
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        if error is not None:
            records.append(
                {
                    "batch_size": batch_size,
                    "passed": False,
                    "error": error,
                    "seconds": elapsed,
                }
            )
            continue
        intervention_observed: list[Any] = []
        try:
            for start in range(0, 4, batch_size):
                batch = representatives[start : min(start + batch_size, 4)]
                intervention_observed.extend(
                    runtime.forward_intervened_batch(
                        [row["messages"] for row in batch],
                        allowed_token_ids=label_ids,
                        layer_deltas=deployment_deltas,
                        scope="user_content",
                        continuation=True,
                        return_full_logits=True,
                    )
                )
        except RuntimeError as caught:
            if "out of memory" not in str(caught).lower():
                raise
            records.append(
                {
                    "batch_size": batch_size,
                    "passed": False,
                    "error": "intervention_cuda_out_of_memory",
                    "seconds": elapsed,
                }
            )
            with suppress(Exception):
                torch.cuda.empty_cache()
            continue
        allowed_error = max(
            float(np.max(np.abs(left.allowed_logits - right.allowed_logits)))
            for left, right in zip(references, observed, strict=True)
        )
        full_error = max(
            _full_logit_error(left, right) for left, right in zip(references, observed, strict=True)
        )
        logsum_error = max(
            abs(left.full_logsumexp - right.full_logsumexp)
            for left, right in zip(references, observed, strict=True)
        )
        intervention_error = max(
            _full_logit_error(left, right)
            for left, right in zip(intervention_references, intervention_observed, strict=True)
        )
        exact_metadata = all(
            left.prompt_sha256 == right.prompt_sha256
            and left.prompt_tokens == right.prompt_tokens
            and left.full_argmax_token_id == right.full_argmax_token_id
            for left, right in zip(references, observed, strict=True)
        )
        free_fractions = [
            float(torch.cuda.mem_get_info(device)[0] / torch.cuda.mem_get_info(device)[1])
            for device in range(torch.cuda.device_count())
        ]
        passed = bool(
            allowed_error <= float(forward_config["batch_allowed_logit_tolerance"])
            and full_error <= float(forward_config["batch_full_logit_tolerance"])
            and logsum_error <= float(forward_config["batch_full_logsumexp_tolerance"])
            and intervention_error <= float(forward_config["batch_full_logit_tolerance"])
            and exact_metadata
            and min(free_fractions) >= float(forward_config["minimum_free_vram_fraction"])
        )
        records.append(
            {
                "batch_size": batch_size,
                "passed": passed,
                "seconds": elapsed,
                "prompt_evaluations": len(representatives),
                "prompts_per_second": len(representatives) / elapsed,
                "max_allowed_logit_error": allowed_error,
                "max_full_logit_error": full_error,
                "max_full_logsumexp_error": logsum_error,
                "max_intervention_full_logit_error": intervention_error,
                "exact_metadata": exact_metadata,
                "free_vram_fractions": free_fractions,
            }
        )
    passing = [record for record in records if record["passed"]]
    if not passing:
        selected = None
        projected = float("inf")
    else:
        selected = max(
            passing,
            key=lambda record: (
                float(record["prompts_per_second"]),
                -int(record["batch_size"]),
            ),
        )
        projected = (
            float(throughput["projection_headroom_multiplier"])
            * int(throughput["planned_scientific_prompt_evaluations"])
            / float(selected["prompts_per_second"])
        )
    reserve = float(throughput["backup_reserve_seconds"])
    passed = bool(
        selected is not None
        and alpha_zero_error == 0.0
        and alpha_zero.full_argmax_token_id == references[0].full_argmax_token_id
        and projected + reserve <= _remaining_seconds()
    )
    return {
        "schema_version": "glm53_v20_batch_calibration_v1",
        "passed": passed,
        "reference_batch_one_seconds": reference_seconds,
        "reference_batch_one_prompts_per_second": len(representatives) / reference_seconds,
        "alpha_zero_max_full_logit_error": alpha_zero_error,
        "alpha_zero_argmax_equal": alpha_zero.full_argmax_token_id
        == references[0].full_argmax_token_id,
        "candidate_results": records,
        "selected_batch_size": (int(selected["batch_size"]) if selected is not None else None),
        "selected_prompts_per_second": (
            float(selected["prompts_per_second"]) if selected is not None else None
        ),
        "projected_scientific_seconds_with_headroom": projected,
        "fits_deadline_with_backup_reserve": projected + reserve <= _remaining_seconds(),
        "representative_sample_ids": [row["sample_id"] for row in representatives],
    }


def _part_valid(path: Path, manifest_path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file() or not manifest_path.is_file():
        return False
    record = read_json(manifest_path)
    return all(record.get(key) == value for key, value in expected.items()) and record.get(
        "part_sha256"
    ) == sha256_file(path)


def _score_arm(
    runtime: LoadedV17GLM53,
    rows: list[dict[str, Any]],
    *,
    arm: str,
    label_ids: list[int],
    batch_size: int,
    output_root: Path,
    binding: dict[str, str],
    direction: np.ndarray | None = None,
    alpha: float = 0.0,
) -> list[dict[str, Any]]:
    lengths = runtime.downstream_token_lengths([row["messages"] for row in rows], continuation=True)
    execution = [
        row
        for row, _ in sorted(
            zip(rows, lengths, strict=True),
            key=lambda item: (item[1], item[0]["sample_id"]),
        )
    ]
    deltas = (
        None if direction is None else raw_layer_deltas(direction, [10, 14, 18, 22, 26, 30], alpha)
    )
    paths: list[Path] = []
    for part_index, start in enumerate(range(0, len(execution), 32)):
        chunk = execution[start : start + 32]
        path = output_root / arm / f"part-{part_index:04d}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        expected = {
            "schema_version": "glm53_v20_scored_part_v1",
            "binding_sha256": canonical_sha256(binding | {"arm": arm}),
            "sample_ids_sha256": canonical_sha256([row["sample_id"] for row in chunk]),
            "row_count": len(chunk),
            "arm": arm,
            "batch_size": batch_size,
        }
        if not _part_valid(path, manifest_path, expected):
            output: list[dict[str, Any]] = []
            for batch_start in range(0, len(chunk), batch_size):
                batch = chunk[batch_start : batch_start + batch_size]
                try:
                    forwards = runtime.forward_intervened_batch(
                        [row["messages"] for row in batch],
                        allowed_token_ids=label_ids,
                        layer_deltas=deltas,
                        scope="user_content",
                        continuation=True,
                    )
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        raise RuntimeError(
                            "V20 production batch OOM; stopped without selective retry"
                        ) from error
                    raise
                for row, forward in zip(batch, forwards, strict=True):
                    proxy = proxy_from_compact_logits(
                        forward.allowed_logits,
                        full_logsumexp=forward.full_logsumexp,
                        full_argmax_token_id=forward.full_argmax_token_id,
                        label_ids=label_ids,
                        codebook_values=row["codebook_values"],
                    )
                    output.append(
                        {key: value for key, value in row.items() if key != "messages"}
                        | proxy
                        | {
                            "arm": arm,
                            "scope": "user_content",
                            "allowed_logits": forward.allowed_logits.astype(float).tolist(),
                            "full_logsumexp": forward.full_logsumexp,
                            "full_argmax_token_id": forward.full_argmax_token_id,
                            "prompt_sha256": forward.prompt_sha256,
                            "prompt_tokens": forward.prompt_tokens,
                        }
                    )
            atomic_jsonl(path, output)
            atomic_json(manifest_path, expected | {"part_sha256": sha256_file(path)})
        if not _part_valid(path, manifest_path, expected):
            raise ValueError("V20 scored part failed post-write validation")
        paths.append(path)
    merged = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    lookup = {str(row["sample_id"]): row for row in merged}
    if len(lookup) != len(rows):
        raise ValueError("V20 scorer duplicated or lost rows")
    return [lookup[str(row["sample_id"])] for row in rows]


def _local_manifest(parent: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent_valid = sum(row["original_folded_confidence"] is not None for row in rows)
    if parent_valid != int(parent["api_parent_valid_rows"]):
        raise ValueError("V20 API-matched parent row count changed")
    return {
        "local_proxy": {
            "identities_per_group": 16,
            "expected_pre_missing_rows_per_group": int(parent["planned_base_rows"]) // 4,
            "parent_interaction_pp": float(parent["parent_api_interaction_pp"]),
            "bootstrap_reps": 20000,
            "bootstrap_seed": 20261020,
        }
    }


def _write_direction_bundle(path: Path, bundle: dict[str, np.ndarray]) -> None:
    from src.glm53_user_eval.v17.contract import atomic_npz

    atomic_npz(path, **bundle)


def run_paid_ladder(
    *,
    repo_root: Path,
    prereg_path: Path,
    runtime_path: Path,
    downstream_manifest_path: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    prereg = read_yaml(prereg_path)
    runtime_config = read_yaml(runtime_path)
    parent_design = read_json(repo_root / prereg["immutable_inputs"]["parent_design"]["path"])
    design = read_json(repo_root / prereg["immutable_inputs"]["design"]["path"])
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    direction_bundle = _direction_bundle(
        repo_root / prereg["immutable_inputs"]["v19_directions"]["path"],
        repo_root / prereg["immutable_inputs"]["null_signs"]["path"],
    )
    direction_path = output_root / "direction_bundle.npz"
    _write_direction_bundle(direction_path, direction_bundle)
    direction_record = {
        "schema_version": "glm53_v20_direction_bundle_v1",
        "source_sha256": prereg["immutable_inputs"]["v19_directions"]["sha256"],
        "bundle_sha256": sha256_file(direction_path),
        "directions": sorted(direction_bundle),
        "literal_layers": prereg["intervention"]["layers"],
        "deployment_alpha": prereg["intervention"]["deployment_alpha"],
    }
    atomic_json(output_root / "direction_bundle_manifest.json", direction_record)
    loader_config = _loader_compatible_runtime_config(runtime_config)
    runtime = LoadedV17GLM53(model_path=model_path, config=loader_config)
    try:
        fp8 = runtime.fp8_scale_report()
        atomic_json(output_root / "fp8_runtime_check.json", fp8)
        if not fp8["passed"]:
            return {"stage": "runtime", "decision": "stop_exact_fp8_runtime_failed"}

        downstream, all_proxy_rows, _ = validate_downstream_assets(
            repo_root=repo_root, manifest_path=downstream_manifest_path
        )
        selected = _selected_proxy_rows(all_proxy_rows, parent_design)
        atomic_json(output_root / "downstream_preflight.json", downstream)
        downstream_binding = read_json(
            repo_root / prereg["immutable_inputs"]["downstream_manifest"]["path"]
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
        calibration = calibrate_batches(
            runtime,
            selected,
            label_ids=label_ids,
            all16_direction=direction_bundle["all16"],
            config=runtime_config,
        )
        atomic_json(output_root / "runtime_calibration.json", calibration)
        if not calibration["passed"]:
            return {
                "stage": "runtime",
                "decision": "runtime_or_batch_projection_failed_before_local_parity",
            }
        _require_time(float(calibration["projected_scientific_seconds_with_headroom"]))
        batch_size = int(calibration["selected_batch_size"])
        binding = {
            "prereg": sha256_file(prereg_path),
            "design": sha256_file(repo_root / prereg["immutable_inputs"]["design"]["path"]),
            "direction_bundle": sha256_file(direction_path),
            "batch_calibration": sha256_file(output_root / "runtime_calibration.json"),
        }

        proxy_root = output_root / "local_proxy"
        scored = _score_arm(
            runtime,
            selected,
            arm="baseline",
            label_ids=label_ids,
            batch_size=batch_size,
            output_root=proxy_root / "parts",
            binding=binding,
        )
        baseline_scores = [
            {key: value for key, value in row.items() if key not in {"arm", "scope"}}
            for row in scored
        ]
        atomic_jsonl(proxy_root / "raw_scores.jsonl", baseline_scores)
        parity = analyze_local_proxy(
            baseline_scores, _local_manifest(parent_design, baseline_scores)
        )
        atomic_json(proxy_root / "analysis.json", parity)
        parity_verification = verify_local_parity(
            proxy_root,
            parent_interaction_pp=float(parent_design["parent_api_interaction_pp"]),
            expected_rows_per_group=int(parent_design["planned_base_rows"]) // 4,
        )
        atomic_json(proxy_root / "verification.json", parity_verification)
        parity_passed = bool(parity["passed"] and parity_verification["passed"])
        parity_decision = {
            "schema_version": "glm53_v20_local_parity_decision_v1",
            "passed": parity_passed,
            "decision": ("local_parity_passed" if parity_passed else "stop_local_parity_failed"),
            "authorization": {"direct_user_intervention": parity_passed},
            "analysis_sha256": sha256_file(proxy_root / "analysis.json"),
            "verification_sha256": sha256_file(proxy_root / "verification.json"),
        }
        atomic_json(proxy_root / "decision.json", parity_decision)
        if not parity_passed:
            return {"stage": "local_parity", "decision": parity_decision["decision"]}

        remaining_evaluations = 5 * len(selected) + 20 * 80
        remaining_seconds = (
            float(runtime_config["throughput_gate"]["projection_headroom_multiplier"])
            * remaining_evaluations
            / float(calibration["selected_prompts_per_second"])
        )
        _require_time(remaining_seconds)
        causal_root = output_root / "causal"
        baseline = [dict(row, arm="baseline", scope="user_content") for row in baseline_scores]
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
                    batch_size=batch_size,
                    output_root=causal_root / "parts",
                    binding=binding,
                    direction=direction_bundle[direction_name],
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
            raise ValueError("V20 sign-flip control subset does not contain 80 rows")
        null_rows: list[dict[str, Any]] = []
        for index in range(20):
            arm = f"signflip_{index:02d}"
            null_rows.extend(
                _score_arm(
                    runtime,
                    pilot_rows,
                    arm=arm,
                    label_ids=label_ids,
                    batch_size=batch_size,
                    output_root=causal_root / "parts",
                    binding=binding,
                    direction=direction_bundle[arm],
                    alpha=0.6,
                )
            )
            atomic_jsonl(causal_root / "null_rows.partial.jsonl", null_rows)
        atomic_jsonl(causal_root / "null_rows.jsonl", null_rows)

        causal = analyze_causal_rows(
            full_rows,
            null_rows,
            pilot_task_ids=pilot_task_ids,
            reps=int(prereg["statistics"]["bootstrap_reps"]),
            seed=int(prereg["seeds"]["causal_bootstrap"]),
        )
        atomic_json(causal_root / "analysis.json", causal)
        verification = verify_causal(
            causal_root,
            pilot_task_ids=pilot_task_ids,
            primary_bootstrap_ci=causal["arms"]["all16_deploy"]["delta_ci95_pp"],
            reps=int(prereg["statistics"]["bootstrap_reps"]),
            seed=int(prereg["seeds"]["independent_verifier_bootstrap"]),
        )
        atomic_json(causal_root / "verification.json", verification)
        state = str(verification["recomputed_decision"])
        passed = bool(
            verification["passed"]
            and state == "fixed_hua_intervention_selectively_attenuates_user_conditioning"
        )
        decision = {
            "schema_version": "glm53_v20_final_decision_v1",
            "passed": passed,
            "decision": state,
            "checks": verification["recomputed_decision_checks"],
            "causal_analysis_sha256": sha256_file(causal_root / "analysis.json"),
            "independent_verification_sha256": sha256_file(causal_root / "verification.json"),
            "local_parity_sha256": sha256_file(proxy_root / "decision.json"),
            "v19_direction_sha256": prereg["immutable_inputs"]["v19_directions"]["sha256"],
        }
        atomic_json(output_root / "decision.json", decision)
        return {"stage": "causal", "decision": state, "passed": passed}
    finally:
        atomic_json(
            output_root / "runtime_summary.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "deadline_utc": _deadline().isoformat(),
            },
        )
        runtime.close()


__all__ = [
    "_direction_bundle",
    "_score_arm",
    "_selected_proxy_rows",
    "calibrate_batches",
    "run_paid_ladder",
]
