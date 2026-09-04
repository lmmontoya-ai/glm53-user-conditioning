"""Fail-closed conditional V11 local-proxy and recruitment branch.

This module deliberately contains no source-gate fitting or selection logic.  It
accepts only a completed, independently verified source decision and a frozen
readout, then scores the two preregistered downstream surfaces.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from src.glm53_user_eval.v8.datasets import load_proxy_surface, load_user_surface
from src.glm53_user_eval.v8.proxy import proxy_messages, validate_label_tokens

GROUP_ORDER = ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
SHORT_GROUPS = {"F": "famous_ai", "U": "unknown_ai", "FN": "famous_nonai", "G": "genpop"}
EXPECTED_LABELS = list("ABCDEFGHIJK")
EXPECTED_CODEBOOKS = {
    "0": dict(zip(EXPECTED_LABELS, range(0, 101, 10), strict=True)),
    "1": dict(
        zip(
            EXPECTED_LABELS,
            (50, 40, 30, 20, 10, 0, 100, 80, 90, 70, 60),
            strict=True,
        )
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "glm53_v11_downstream_manifest_v1":
        raise ValueError("unexpected V11 downstream manifest schema")
    return value


def validate_v11_codebooks(payload: dict[str, Any]) -> None:
    """Validate the folded-antithetic two-codebook contract."""

    labels = payload.get("labels")
    codebooks = payload.get("codebooks")
    if labels != EXPECTED_LABELS or not isinstance(codebooks, dict):
        raise ValueError("V11 proxy labels or codebook container differ")
    if codebooks != EXPECTED_CODEBOOKS:
        raise ValueError("V11 proxy codebook mappings differ")
    values = [set(codebook.values()) for codebook in codebooks.values()]
    expected_values = set(range(0, 101, 10))
    if any(value != expected_values for value in values):
        raise ValueError("each V11 codebook must use every frozen raw value once")
    folded = {
        key: [max(value, 100 - value) for value in codebook.values()]
        for key, codebook in codebooks.items()
    }
    if sorted(folded["0"]) != sorted(folded["1"]):
        raise ValueError("V11 codebooks do not preserve the folded-value multiset")
    if any(left == right for left, right in zip(folded["0"], folded["1"], strict=True)):
        raise ValueError("a V11 label retains the same folded value across codebooks")
    assignment = payload.get("assignment", {})
    if assignment != {
        "rule": "(analysis_index + frozen_task_rank) modulo 2",
        "group_shared": True,
        "balanced_per_identity_over_frozen_tasks": True,
        "uses_cached_codebook_id": False,
    }:
        raise ValueError("V11 codebook assignment contract differs")


def build_proxy_token_contract(
    processor: Any,
    *,
    row: dict[str, Any],
    codebook_payload: dict[str, Any],
    codebook_sha256: str,
    tokenizer_json_sha256: str,
    chat_template_sha256: str,
) -> dict[str, Any]:
    """Build the exact two-context V11 one-token contract."""

    validate_v11_codebooks(codebook_payload)
    expected_sample = codebook_payload["token_contract_validation_sample_id"]
    if row.get("sample_id") != expected_sample or row.get("proxy_eligible") is not True:
        raise ValueError("V11 token-contract row differs or is ineligible")
    contexts: dict[str, Any] = {}
    for codebook_id in ("0", "1"):
        messages = proxy_messages(
            system_text=row["system_text"],
            dilemma_text=row["dilemma_text"],
            first_assistant_text=row["first_assistant_text"],
            codebook=codebook_payload["codebooks"][codebook_id],
            followup=codebook_payload["followup"],
            assistant_prefix=codebook_payload["assistant_prefix"],
        )
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            reasoning_effort="high",
            clear_thinking=True,
        )
        token_result = validate_label_tokens(processor, messages, codebook_payload["labels"])
        contexts[codebook_id] = {
            "rendered_context": rendered,
            "rendered_context_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "base_token_count": token_result["base_token_count"],
            "label_ids": token_result["label_ids"],
            "one_token_extension_checks": {
                label: True for label in codebook_payload["labels"]
            },
        }
    if contexts["0"]["label_ids"] != contexts["1"]["label_ids"]:
        raise ValueError("V11 codebook contexts yield different label token IDs")
    return {
        "schema_version": "glm53_v11_proxy_token_contract_v2",
        "passed": True,
        "model_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "validation_sample_id": expected_sample,
        "codebook_sha256": codebook_sha256,
        "tokenizer_json_sha256": tokenizer_json_sha256,
        "chat_template_sha256": chat_template_sha256,
        "contexts": contexts,
    }


def validate_runtime_proxy_token_contract(
    processor: Any,
    *,
    proxy_rows: list[dict[str, Any]],
    codebook_payload: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Re-tokenize both V11 codebook prompts with the loaded exact processor."""

    validate_v11_codebooks(codebook_payload)
    if contract.get("schema_version") != "glm53_v11_proxy_token_contract_v2":
        raise ValueError("unexpected V11 proxy token contract schema")
    sample_id = str(contract.get("validation_sample_id", ""))
    candidates = [row for row in proxy_rows if row["sample_id"] == sample_id]
    if len(candidates) != 1:
        raise ValueError("V11 token-contract sample is absent or duplicated")
    row = candidates[0]
    observed: dict[str, Any] = {}
    for codebook_id in ("0", "1"):
        messages = proxy_messages(
            system_text=row["system_text"],
            dilemma_text=row["dilemma_text"],
            first_assistant_text=row["first_assistant_text"],
            codebook=codebook_payload["codebooks"][codebook_id],
            followup=codebook_payload["followup"],
            assistant_prefix=codebook_payload["assistant_prefix"],
        )
        result = validate_label_tokens(processor, messages, codebook_payload["labels"])
        expected = contract["contexts"][codebook_id]
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            reasoning_effort="high",
            clear_thinking=True,
        )
        if rendered != expected["rendered_context"]:
            raise ValueError(f"V11 codebook {codebook_id} rendered context differs")
        if result != {
            "label_ids": expected["label_ids"],
            "base_token_count": expected["base_token_count"],
        }:
            raise ValueError(f"V11 codebook {codebook_id} token contract differs")
        observed[codebook_id] = result
    if observed["0"]["label_ids"] != observed["1"]["label_ids"]:
        raise ValueError("V11 codebooks do not share the same label-token set")
    return {
        "schema_version": "glm53_v11_proxy_runtime_token_validation_v1",
        "passed": True,
        "validation_sample_id": sample_id,
        "contexts": observed,
    }


def _asset_path(repo_root: Path, record: dict[str, Any]) -> Path:
    return repo_root / str(record.get("path", record.get("target_path")))


def validate_downstream_assets(
    *, repo_root: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify every frozen input and reconstruct both exact downstream surfaces."""

    manifest = load_manifest(manifest_path)
    checks: dict[str, bool] = {}
    for name, record in manifest["assets"].items():
        path = _asset_path(repo_root, record)
        checks[f"asset_{name}"] = path.is_file() and sha256_file(path) == record["sha256"]
    if not all(checks.values()):
        raise ValueError(f"downstream asset lock failed: {checks}")

    assets = manifest["assets"]
    cache = _asset_path(repo_root, assets["transcript_cache"])
    codebooks = _asset_path(repo_root, assets["proxy_codebooks"])
    personas = _asset_path(repo_root, assets["personas"])
    schedule_path = _asset_path(repo_root, assets["causal_schedule"])
    templates = _asset_path(repo_root, assets["user_templates"])
    proxy_contract = json.loads(
        _asset_path(repo_root, assets["proxy_contract"]).read_text(encoding="utf-8")
    )
    if proxy_contract.get("passed") is not True:
        raise ValueError("frozen V11 proxy token contract did not pass")
    codebook_payload = json.loads(codebooks.read_text(encoding="utf-8"))
    validate_v11_codebooks(codebook_payload)
    labels = codebook_payload["labels"]
    label_ids = [
        int(proxy_contract["contexts"]["0"]["label_ids"][label]) for label in labels
    ]
    if proxy_contract["contexts"]["0"]["label_ids"] != proxy_contract["contexts"]["1"][
        "label_ids"
    ]:
        raise ValueError("V11 codebook token contexts use different label IDs")
    if len(label_ids) != 11 or len(set(label_ids)) != 11:
        raise ValueError("frozen proxy labels are not eleven distinct token IDs")

    parent_surface = json.loads(
        _asset_path(repo_root, assets["parent_proxy_surface"]).read_text(encoding="utf-8")
    )
    task_ids = list(parent_surface["scope"]["task_ids"])
    proxy_rows = load_proxy_surface(
        cache,
        codebooks,
        set(task_ids),
        personas,
    )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    selected = parent_surface["scope"]["identities"]
    expected_selected = {
        "famous_ai": [row["famous_ai"] for row in schedule["pairs"]],
        "unknown_ai": [row["unknown_ai"] for row in schedule["pairs"]],
        "famous_nonai": list(schedule["famous_nonai_controls"]),
        "genpop": list(schedule["genpop_controls"]),
    }
    if selected != expected_selected:
        raise ValueError("V11 proxy selected-persona schedule differs from causal_schedule_v1")
    selected_sets = {group: set(values) for group, values in selected.items()}
    cache_rows = [
        json.loads(line)
        for line in cache.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_cache = [
        row
        for row in cache_rows
        if row["persona_key"] in selected_sets[row["group"]]
        and row["stimulus_id"] in set(task_ids)
    ]
    final_evidence = json.loads(
        _asset_path(repo_root, assets["v7_final_evidence"]).read_text(encoding="utf-8")
    )
    allowed_source_logs = {
        str(record["sha256"]) for record in final_evidence["successful_eval_logs"]
    }
    selected_source_logs = {str(row["source_eval_sha256"]) for row in selected_cache}
    ineligible = [row for row in selected_cache if not row["proxy_eligible"]]
    eligible_cache = [row for row in selected_cache if row["proxy_eligible"]]
    parent_valid_cache = [
        row for row in eligible_cache if row["original_folded_confidence"] is not None
    ]
    def scientific_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (row["group"], row["persona_key"], row["stimulus_id"])

    eligible_key_hash = canonical_sha256(sorted(scientific_key(row) for row in eligible_cache))
    parent_valid_key_hash = canonical_sha256(
        sorted(scientific_key(row) for row in parent_valid_cache)
    )
    ineligible_key_hash = canonical_sha256(
        sorted(scientific_key(row) for row in ineligible)
    )
    analysis_index = {
        group: {persona: index for index, persona in enumerate(personas)}
        for group, personas in selected.items()
    }
    proxy_rows = [
        row
        | {
            "roster_index": int(row["pair_index"]),
            "analysis_index": analysis_index[row["group"]][row["persona_key"]],
        }
        for row in proxy_rows
        if row["persona_key"] in analysis_index[row["group"]]
    ]
    task_rank = {
        task_id: index for index, task_id in enumerate(task_ids)
    }
    for row in proxy_rows:
        assignment = str((int(row["analysis_index"]) + task_rank[row["stimulus_id"]]) % 2)
        codebook = codebook_payload["codebooks"][assignment]
        row["codebook_id"] = assignment
        row["codebook_values"] = [codebook[label] for label in labels]
        row["messages"] = proxy_messages(
            system_text=row["system_text"],
            dilemma_text=row["dilemma_text"],
            first_assistant_text=row["first_assistant_text"],
            codebook=codebook,
            followup=codebook_payload["followup"],
            assistant_prefix=codebook_payload["assistant_prefix"],
        )
    user_rows = load_user_surface(
        personas_path=personas,
        cache_path=cache,
        templates_path=templates,
        schedule_path=schedule_path,
    )
    proxy_counts = dict(sorted(_counts(proxy_rows, "group").items()))
    expected_proxy = dict(sorted(manifest["local_proxy"]["expected_eligible_by_group"].items()))
    user_counts = _counts(user_rows, "group")
    personas_payload = json.loads(personas.read_text(encoding="utf-8"))
    proxy_keys = [
        (row["group"], int(row["analysis_index"]), row["stimulus_id"])
        for row in proxy_rows
    ]
    user_keys = [
        (row["group"], int(row["pair_index"]), row["template_id"]) for row in user_rows
    ]
    assignments: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in proxy_rows:
        assignments[(int(row["analysis_index"]), row["stimulus_id"])].add(
            row["codebook_id"]
        )
    task_count = len(task_ids)
    checks |= {
        "parent_proxy_surface": (
            parent_surface.get("passed") is True
            and parent_surface["scope"]["schedule_sha256"]
            == manifest["assets"]["causal_schedule"]["sha256"]
            and parent_surface["estimate"]["interaction_pp"]
            == manifest["local_proxy"]["parent_interaction_pp"]
            and len(task_ids) == 100
            and len(set(task_ids)) == 100
            and parent_surface["counts"]["local_reconstructable"]
            == manifest["local_proxy"]["expected_eligible_rows"]
            and parent_surface["counts"]["api_parent_valid"]
            == manifest["local_proxy"]["expected_parent_valid_rows"]
            and parent_surface["counts"]["ineligible_empty_first_assistant"]
            == manifest["local_proxy"]["expected_ineligible_rows"]
            and parent_surface["independent_recomputation"]["passed"] is True
            and parent_surface["independent_recomputation"][
                "maximum_endpoint_difference_pp"
            ]
            <= parent_surface["independent_recomputation"][
                "bootstrap_endpoint_tolerance_pp"
            ]
        ),
        "proxy_selected_pre_missing_total": len(selected_cache)
        == int(manifest["local_proxy"]["expected_pre_missing_rows"]),
        "proxy_total": len(proxy_rows) == int(manifest["local_proxy"]["expected_eligible_rows"]),
        "proxy_groups": proxy_counts == expected_proxy,
        "proxy_tasks": {row["stimulus_id"] for row in proxy_rows}
        == set(task_ids),
        "proxy_scientific_keys_unique": len(proxy_keys) == len(set(proxy_keys)),
        "proxy_codebook_group_shared": all(len(values) == 1 for values in assignments.values()),
        "proxy_codebook_planned_balance": all(
            sum(
                (pair_index + task_index) % 2 == codebook
                for task_index in range(task_count)
            )
            == task_count // 2
            for pair_index in range(int(manifest["local_proxy"]["identities_per_group"]))
            for codebook in (0, 1)
        ),
        "proxy_ineligible_count": len(ineligible)
        == int(manifest["local_proxy"]["expected_ineligible_rows"]),
        "proxy_parent_valid_count": len(parent_valid_cache)
        == int(manifest["local_proxy"]["expected_parent_valid_rows"]),
        "proxy_key_hashes": (
            eligible_key_hash == parent_surface["key_sets"]["local_reconstructable_sha256"]
            and parent_valid_key_hash
            == parent_surface["key_sets"]["api_parent_valid_sha256"]
            and ineligible_key_hash == parent_surface["key_sets"]["ineligible_sha256"]
        ),
        "v7_source_log_provenance": (
            selected_source_logs <= allowed_source_logs
            and len(selected_source_logs)
            == int(manifest["local_proxy"]["expected_source_log_count"])
            and all(
                isinstance(row.get("source_eval_sha256"), str)
                and len(row["source_eval_sha256"]) == 64
                for row in selected_cache
            )
        ),
        "recruitment_total": len(user_rows) == int(manifest["recruitment"]["expected_rows"]),
        "recruitment_groups": set(user_counts) == set(manifest["recruitment"]["groups"])
        and all(
            count == int(manifest["recruitment"]["expected_rows_per_group"])
            for count in user_counts.values()
        ),
        "recruitment_templates": {row["template_id"] for row in user_rows}
        == set(manifest["recruitment"]["template_ids"]),
        "recruitment_scientific_keys_unique": len(user_keys) == len(set(user_keys)),
        "f_u_roster_org_match": all(
            personas_payload["famous_ai"][index]["org"]
            == personas_payload["unknown_ai"][index]["org"]
            for index in range(70)
        ),
        "u_g_roster_name_match": all(
            personas_payload["unknown_ai"][index]["name"]
            == personas_payload["genpop"][index]["name"]
            for index in range(70)
        ),
        "early_cot_forbidden": manifest["execution"]["early_cot_forbidden"] is True,
        "steering_forbidden": manifest["execution"]["steering_forbidden"] is True,
        "same_process": manifest["source_gate"]["same_loaded_model_process_required"] is True
        and manifest["source_gate"]["model_reload_allowed"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"downstream surface contract failed: {checks}")
    report = {
        "schema_version": "glm53_v11_downstream_preflight_v1",
        "passed": True,
        "manifest_sha256": sha256_file(manifest_path),
        "checks": checks,
        "proxy_row_count": len(proxy_rows),
        "proxy_row_ids_sha256": canonical_sha256([row["sample_id"] for row in proxy_rows]),
        "recruitment_row_count": len(user_rows),
        "recruitment_row_ids_sha256": canonical_sha256([row["sample_id"] for row in user_rows]),
        "label_ids": label_ids,
        "selected_source_log_count": len(selected_source_logs),
        "selected_source_log_hashes_sha256": canonical_sha256(sorted(selected_source_logs)),
        "technical_errors": [
            {
                "surface": "proxy_source_transcript",
                "sample_id": row["sample_id"],
                "group": row["group"],
                "persona_key": row["persona_key"],
                "stimulus_id": row["stimulus_id"],
                "system_text": row["system_text"],
                "dilemma_text": row["dilemma_text"],
                "first_assistant_text": row["first_assistant_text"],
                "source_error": row["source_error"],
                "source_eval_sha256": row["source_eval_sha256"],
                "original_folded_confidence": row["original_folded_confidence"],
                "reviewed": "",
                "notes": "",
            }
            for row in sorted(ineligible, key=scientific_key)
        ],
    }
    return report, proxy_rows, user_rows


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for row in rows:
        result[str(row[field])] += 1
    return dict(result)


def proxy_from_compact_logits(
    allowed_logits: np.ndarray,
    *,
    full_logsumexp: float,
    full_argmax_token_id: int,
    label_ids: list[int],
    codebook_values: list[float],
) -> dict[str, Any]:
    allowed = np.asarray(allowed_logits, dtype=np.float64)
    values = np.asarray(codebook_values, dtype=np.float64)
    if allowed.shape != (11,) or values.shape != (11,) or len(set(label_ids)) != 11:
        raise ValueError("compact proxy requires eleven logits, values, and distinct token IDs")
    shifted = allowed - np.max(allowed)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    log_allowed = float(np.max(allowed) + math.log(float(np.exp(shifted).sum())))
    folded = np.maximum(values, 100.0 - values)
    winner = int(np.argmax(allowed))
    entropy = -float(np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities), 0)))
    return {
        "conditional_probabilities": probabilities.tolist(),
        "expected_raw_confidence": float(probabilities @ values),
        "expected_folded_confidence": float(probabilities @ folded),
        "allowed_mass": float(math.exp(log_allowed - float(full_logsumexp))),
        "conditional_entropy": entropy,
        "argmax_label_position": winner,
        "argmax_raw_confidence": float(values[winner]),
        "argmax_folded_confidence": float(folded[winner]),
        "full_vocab_argmax_allowed": int(full_argmax_token_id) in set(label_ids),
    }


def calibrate_downstream_batch(
    runtime: Any,
    rows: list[dict[str, Any]],
    *,
    selected_layer: int,
    continuation: bool,
    allowed_token_ids: list[int] | None,
    candidate_batch_sizes: list[int],
    logits_tolerance: float,
    activation_tolerance: float,
    selected_span: bool,
) -> dict[str, Any]:
    """Select the largest batch that matches single-row outputs on frozen rows."""

    candidates = sorted({int(value) for value in candidate_batch_sizes})
    if not candidates or candidates[0] != 1 or any(value <= 0 for value in candidates):
        raise ValueError("batch calibration candidates must start with one")
    required = max(candidates)
    if len(rows) < required:
        raise ValueError("batch calibration has too few frozen rows")
    lengths = runtime.downstream_token_lengths(
        [row["messages"] for row in rows],
        continuation=continuation,
    )
    ordered = sorted(
        zip(rows, lengths, strict=True),
        key=lambda item: (item[1], item[0]["sample_id"]),
    )
    positions = np.linspace(0, len(ordered) - 1, required).round().astype(int)
    representative_rows = [ordered[index][0] for index in positions]
    representative_lengths = [int(ordered[index][1]) for index in positions]

    def spans(batch: list[dict[str, Any]]) -> list[str] | None:
        if not selected_span:
            return None
        return [str(row["messages"][-1]["content"]) for row in batch]

    references: list[Any] = []
    single_started = time.perf_counter()
    for row in representative_rows:
        references.append(
            runtime.forward_downstream(
                row["messages"],
                selected_layer=selected_layer,
                continuation=continuation,
                allowed_token_ids=allowed_token_ids,
                selected_span_text=(
                    str(row["messages"][-1]["content"]) if selected_span else None
                ),
            )
        )
    single_seconds = time.perf_counter() - single_started
    records: list[dict[str, Any]] = [
        {
            "batch_size": 1,
            "passed": True,
            "seconds": single_seconds / required,
            "rows": 1,
            "max_logit_error": 0.0,
            "max_activation_error": 0.0,
            "exact_metadata": True,
            "sample_ids": [representative_rows[-1]["sample_id"]],
            "token_lengths": [representative_lengths[-1]],
        }
    ]
    total_seconds = single_seconds
    selected_record = records[0]
    for batch_size in candidates[1:]:
        batch_indices = np.linspace(0, required - 1, batch_size).round().astype(int)
        batch = [representative_rows[index] for index in batch_indices]
        expected_rows = [references[index] for index in batch_indices]
        started = time.perf_counter()
        try:
            observed = runtime.forward_downstream_batch(
                [row["messages"] for row in batch],
                selected_layer=selected_layer,
                continuation=continuation,
                allowed_token_ids=allowed_token_ids,
                selected_span_texts=spans(batch),
            )
        except RuntimeError as error:
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            if "out of memory" not in str(error).lower():
                raise
            records.append(
                {
                    "batch_size": batch_size,
                    "passed": False,
                    "seconds": elapsed,
                    "rows": batch_size,
                    "error": "cuda_out_of_memory",
                }
            )
            break
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        logit_error = max(
            max(
                float(np.max(np.abs(left.allowed_logits - right.allowed_logits))),
                abs(left.full_logsumexp - right.full_logsumexp),
            )
            for left, right in zip(expected_rows, observed, strict=True)
        )
        activation_error = max(
            float(np.max(np.abs(left.prompt_final - right.prompt_final)))
            for left, right in zip(expected_rows, observed, strict=True)
        )
        if selected_span:
            activation_error = max(
                activation_error,
                max(
                    float(
                        np.max(
                            np.abs(left.selected_span_mean - right.selected_span_mean)
                        )
                    )
                    for left, right in zip(expected_rows, observed, strict=True)
                ),
            )
        exact_metadata = all(
            left.prompt_sha256 == right.prompt_sha256
            and left.prompt_tokens == right.prompt_tokens
            and left.full_argmax_token_id == right.full_argmax_token_id
            for left, right in zip(expected_rows, observed, strict=True)
        )
        passed = (
            logit_error <= logits_tolerance
            and activation_error <= activation_tolerance
            and exact_metadata
        )
        record = {
            "batch_size": batch_size,
            "passed": passed,
            "seconds": elapsed,
            "rows": batch_size,
            "rows_per_second": batch_size / elapsed,
            "max_logit_error": logit_error,
            "max_activation_error": activation_error,
            "exact_metadata": exact_metadata,
            "sample_ids": [row["sample_id"] for row in batch],
            "token_lengths": [representative_lengths[index] for index in batch_indices],
        }
        records.append(record)
        if not passed:
            break
        selected_record = record
    return {
        "schema_version": "glm53_v11_downstream_batch_calibration_v1",
        "passed": int(selected_record["batch_size"]) > 1,
        "selected_batch_size": int(selected_record["batch_size"]),
        "selected_batch_seconds": float(selected_record["seconds"]),
        "selected_batch_rows": int(selected_record["rows"]),
        "total_calibration_seconds": total_seconds,
        "representative_sample_ids": [row["sample_id"] for row in representative_rows],
        "representative_token_lengths": representative_lengths,
        "candidate_results": records,
    }


def _part_is_valid(path: Path, manifest_path: Path, binding: dict[str, Any]) -> bool:
    if not path.is_file() or not manifest_path.is_file():
        return False
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    return all(record.get(key) == value for key, value in binding.items()) and record.get(
        "part_sha256"
    ) == sha256_file(path)


def _bucketed_execution_order(
    runtime: Any,
    rows: list[dict[str, Any]],
    *,
    continuation: bool,
    batch_size: int,
    output_root: Path,
    binding: dict[str, str],
) -> list[dict[str, Any]]:
    if batch_size <= 1:
        raise ValueError("V11 production scoring requires a calibrated batch larger than one")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("downstream production rows contain duplicate sample IDs")
    lengths = runtime.downstream_token_lengths(
        [row["messages"] for row in rows], continuation=continuation
    )
    records = sorted(
        [
            {
                "sample_id": sample_id,
                "token_length": int(length),
                "original_index": index,
            }
            for index, (sample_id, length) in enumerate(zip(sample_ids, lengths, strict=True))
        ],
        key=lambda record: (record["token_length"], record["sample_id"]),
    )
    order = {
        "schema_version": "glm53_v11_downstream_execution_order_v1",
        "binding_sha256": canonical_sha256(binding),
        "batch_size": batch_size,
        "continuation": continuation,
        "oom_policy": "stop_without_selective_row_retry",
        "rows": records,
    }
    order_path = output_root / "execution_order.json"
    if order_path.exists() and json.loads(order_path.read_text(encoding="utf-8")) != order:
        raise ValueError("downstream execution order differs from its existing manifest")
    atomic_json(order_path, order)
    return [rows[int(record["original_index"])] for record in records]


def score_local_proxy(
    runtime: Any,
    rows: list[dict[str, Any]],
    *,
    selected_layer: int,
    label_ids: list[int],
    output_root: Path,
    binding: dict[str, str],
    checkpoint_rows: int = 32,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Score the frozen V8 two-codebook surface with resumable atomic parts."""

    execution_rows = _bucketed_execution_order(
        runtime,
        rows,
        continuation=True,
        batch_size=batch_size,
        output_root=output_root,
        binding=binding,
    )
    parts = output_root / "parts"
    paths: list[Path] = []
    for part_index, start in enumerate(range(0, len(execution_rows), checkpoint_rows)):
        chunk = execution_rows[start : start + checkpoint_rows]
        path = parts / f"part-{part_index:04d}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        expected = {
            "schema_version": "glm53_v11_proxy_part_v1",
            "binding_sha256": canonical_sha256(binding),
            "sample_ids_sha256": canonical_sha256([row["sample_id"] for row in chunk]),
            "row_count": len(chunk),
        }
        if not _part_is_valid(path, manifest_path, expected):
            output: list[dict[str, Any]] = []
            for batch_start in range(0, len(chunk), batch_size):
                batch = chunk[batch_start : batch_start + batch_size]
                try:
                    forwards = runtime.forward_downstream_batch(
                        [row["messages"] for row in batch],
                        selected_layer=selected_layer,
                        continuation=True,
                        allowed_token_ids=label_ids,
                    )
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        raise RuntimeError(
                            "proxy production batch OOM; stopped without selective retry"
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
                            "prompt_sha256": forward.prompt_sha256,
                            "prompt_tokens": forward.prompt_tokens,
                            "allowed_logits": forward.allowed_logits.astype(float).tolist(),
                            "full_logsumexp": forward.full_logsumexp,
                            "full_argmax_token_id": forward.full_argmax_token_id,
                        }
                    )
            atomic_jsonl(path, output)
            atomic_json(manifest_path, expected | {"part_sha256": sha256_file(path)})
        if not _part_is_valid(path, manifest_path, expected):
            raise ValueError(f"completed proxy part does not validate: {path}")
        paths.append(path)
    merged = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(merged) != len(rows):
        raise ValueError("proxy scorer lost rows")
    by_sample = {row["sample_id"]: row for row in merged}
    if len(by_sample) != len(rows):
        raise ValueError("proxy scorer duplicated rows")
    merged = [by_sample[row["sample_id"]] for row in rows]
    atomic_jsonl(output_root / "raw_scores.jsonl", merged)
    return merged


def _matrix(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    task_key: str,
    group_map: dict[str, str] | None = None,
    identity_key: str = "pair_index",
    identity_count: int = 70,
) -> dict[str, np.ndarray]:
    mapping = group_map or {key: key for key in GROUP_ORDER}
    tasks = sorted({str(row[task_key]) for row in rows})
    result: dict[str, np.ndarray] = {}
    for source_group, target_group in mapping.items():
        group_rows = [row for row in rows if row["group"] == source_group]
        observed = {int(row[identity_key]) for row in group_rows}
        if not observed <= set(range(identity_count)):
            raise ValueError(f"{source_group} contains an out-of-range identity index")
        identities = list(range(identity_count))
        lookup = {
            (int(row[identity_key]), str(row[task_key])): float(row[value_key])
            for row in group_rows
        }
        if len(lookup) != len(group_rows):
            raise ValueError(f"{source_group} contains duplicate identity/task keys")
        result[target_group] = np.asarray(
            [[lookup.get((identity, task), np.nan) for task in tasks] for identity in identities],
            dtype=np.float64,
        )
    return result


def four_group_bootstrap(
    matrices: dict[str, np.ndarray], *, reps: int, seed: int
) -> tuple[float, tuple[float, float], np.ndarray]:
    if matrices["famous_ai"].shape != matrices["unknown_ai"].shape:
        raise ValueError("F/U matrices are not paired")

    def person_weighted(value: np.ndarray) -> float:
        counts = np.sum(np.isfinite(value), axis=1)
        if not np.all(counts > 0):
            raise ValueError("one or more identities have no valid downstream rows")
        return float(np.mean(np.nansum(value, axis=1) / counts))

    def centered_means(values: dict[str, np.ndarray]) -> dict[str, float]:
        center = np.nanmean(values["genpop"], axis=0)
        if not np.isfinite(center).all():
            raise ValueError("a sampled dilemma lacks a valid GenPop center")
        return {
            group: person_weighted(values[group] - center[None, :]) for group in GROUP_ORDER
        }

    means = centered_means(matrices)
    point = means["famous_ai"] - means["unknown_ai"] - means["famous_nonai"] + means["genpop"]
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    n_pairs, n_tasks = matrices["famous_ai"].shape
    for rep in range(reps):
        pair = rng.integers(0, n_pairs, n_pairs)
        task = rng.integers(0, n_tasks, n_tasks)
        fn = rng.integers(0, matrices["famous_nonai"].shape[0], matrices["famous_nonai"].shape[0])
        gp = rng.integers(0, matrices["genpop"].shape[0], matrices["genpop"].shape[0])
        values = {
            "famous_ai": matrices["famous_ai"][pair][:, task],
            "unknown_ai": matrices["unknown_ai"][pair][:, task],
            "famous_nonai": matrices["famous_nonai"][fn][:, task],
            "genpop": matrices["genpop"][gp][:, task],
        }
        sampled = centered_means(values)
        draws[rep] = (
            sampled["famous_ai"]
            - sampled["unknown_ai"]
            - sampled["famous_nonai"]
            + sampled["genpop"]
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(point), (float(low), float(high)), draws


def _equal_person_mean(matrix: np.ndarray, *, require_all: bool) -> float:
    counts = np.sum(np.isfinite(matrix), axis=1)
    person_means = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    eligible = counts > 0
    person_means[eligible] = np.nansum(matrix[eligible], axis=1) / counts[eligible]
    if require_all and not eligible.all():
        raise ValueError("one or more identities have no valid downstream rows")
    if not eligible.any():
        raise ValueError("no identity has a valid downstream row")
    return float(np.mean(person_means[eligible]))


def _centered_group_means(matrices: dict[str, np.ndarray]) -> dict[str, float]:
    center = np.nanmean(matrices["genpop"], axis=0)
    if not np.isfinite(center).all():
        raise ValueError("a dilemma lacks a valid GenPop center")
    return {
        group: _equal_person_mean(matrix - center[None, :], require_all=True)
        for group, matrix in matrices.items()
    }


def analyze_local_proxy(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest["local_proxy"]
    matrices = _matrix(
        rows,
        value_key="expected_folded_confidence",
        task_key="stimulus_id",
        identity_key="analysis_index",
        identity_count=int(config["identities_per_group"]),
    )
    point, _, draws = four_group_bootstrap(
        matrices, reps=int(config["bootstrap_reps"]), seed=int(config["bootstrap_seed"])
    )
    ci90 = np.percentile(draws, [5, 95]).tolist()
    means = _centered_group_means(matrices)
    raw_means = {
        group: _equal_person_mean(matrix, require_all=True) for group, matrix in matrices.items()
    }
    codebooks: dict[str, float] = {}
    for codebook in sorted({str(row["codebook_id"]) for row in rows}):
        subset = [row for row in rows if str(row["codebook_id"]) == codebook]
        codebook_matrices = _matrix(
            subset,
            value_key="expected_folded_confidence",
            task_key="stimulus_id",
            identity_key="analysis_index",
            identity_count=int(config["identities_per_group"]),
        )
        codebook_center = np.nanmean(codebook_matrices["genpop"], axis=0)
        group_means = {
            group: _equal_person_mean(
                codebook_matrices[group] - codebook_center[None, :],
                require_all=False,
            )
            for group in GROUP_ORDER
        }
        codebooks[codebook] = (
            group_means["famous_ai"]
            - group_means["unknown_ai"]
            - group_means["famous_nonai"]
            + group_means["genpop"]
        )
    local = np.asarray([row["expected_folded_confidence"] for row in rows])
    original = np.asarray(
        [
            np.nan if row["original_folded_confidence"] is None else row["original_folded_confidence"]
            for row in rows
        ],
        dtype=np.float64,
    )
    parent_valid = np.isfinite(original)
    matched_rows = [
        row for row, valid in zip(rows, parent_valid, strict=True) if bool(valid)
    ]
    matched_matrices = _matrix(
        matched_rows,
        value_key="expected_folded_confidence",
        task_key="stimulus_id",
        identity_key="analysis_index",
        identity_count=int(config["identities_per_group"]),
    )
    matched_means = {
        group: _equal_person_mean(
            matrix - np.nanmean(matched_matrices["genpop"], axis=0)[None, :],
            require_all=True,
        )
        for group, matrix in matched_matrices.items()
    }
    matched_point = (
        matched_means["famous_ai"]
        - matched_means["unknown_ai"]
        - matched_means["famous_nonai"]
        + matched_means["genpop"]
    )
    allowed = np.asarray([row["allowed_mass"] for row in rows])
    eligible = _counts(rows, "group")
    expected_per_group = int(config["expected_pre_missing_rows_per_group"])
    rates = [eligible[group] / expected_per_group for group in GROUP_ORDER]
    report = {
        "schema_version": "glm53_v11_local_proxy_analysis_v1",
        "interaction_pp": point,
        "ci90_pp": ci90,
        "group_means_pp": means,
        "group_raw_means_pp": raw_means,
        "famous_ai_minus_unknown_ai_pp": means["famous_ai"] - means["unknown_ai"],
        "famous_nonai_minus_genpop_pp": means["famous_nonai"] - means["genpop"],
        "codebook_interactions_pp": codebooks,
        "codebook_interaction_range_pp": float(max(codebooks.values()) - min(codebooks.values())),
        "api_matched_interaction_pp": matched_point,
        "api_matched_group_means_pp": matched_means,
        "retained_fraction": abs(matched_point)
        / abs(float(config["parent_interaction_pp"])),
        "row_spearman": float(spearmanr(local, original, nan_policy="omit").statistic),
        "uncalibrated_mean_absolute_error_pp": float(
            np.mean(np.abs(local[parent_valid] - original[parent_valid]))
        ),
        "parent_confidence_comparison_n": int(parent_valid.sum()),
        "local_reconstructable_key_sha256": canonical_sha256(
            sorted((row["group"], row["persona_key"], row["stimulus_id"]) for row in rows)
        ),
        "api_matched_key_sha256": canonical_sha256(
            sorted(
                (row["group"], row["persona_key"], row["stimulus_id"])
                for row in matched_rows
            )
        ),
        "allowed_mass_median": float(np.median(allowed)),
        "allowed_mass_p05": float(np.percentile(allowed, 5)),
        "conditional_entropy_median": float(
            np.median([row["conditional_entropy"] for row in rows])
        ),
        "full_vocab_argmax_allowed_rate": float(
            np.mean([row["full_vocab_argmax_allowed"] for row in rows])
        ),
        "condition_missingness_spread": float(max(rates) - min(rates)),
        "codebook_explains_result": max(codebooks.values()) >= 0,
    }
    checks = {
        "negative": point < 0,
        "codebooks": all(value < 0 for value in codebooks.values()),
        "api_matched_negative": matched_point < 0,
        "magnitude_or_ci": report["retained_fraction"] >= 0.40 or ci90[1] < 0,
        "components": report["famous_ai_minus_unknown_ai_pp"] <= 0
        and report["famous_nonai_minus_genpop_pp"] >= 0,
        "mass_median": report["allowed_mass_median"] >= 0.80,
        "mass_p05": report["allowed_mass_p05"] >= 0.50,
        "argmax": report["full_vocab_argmax_allowed_rate"] >= 0.95,
        "codebook_artifact": not report["codebook_explains_result"],
        "missingness": report["condition_missingness_spread"] <= 0.005,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def extract_recruitment_features(
    runtime: Any,
    rows: list[dict[str, Any]],
    *,
    selected_layer: int,
    output_root: Path,
    binding: dict[str, str],
    checkpoint_rows: int = 32,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract only neutral-task-token means and prompt-final vectors."""

    execution_rows = _bucketed_execution_order(
        runtime,
        rows,
        continuation=False,
        batch_size=batch_size,
        output_root=output_root,
        binding=binding,
    )
    parts = output_root / "parts"
    paths: list[Path] = []
    for part_index, start in enumerate(range(0, len(execution_rows), checkpoint_rows)):
        chunk = execution_rows[start : start + checkpoint_rows]
        path = parts / f"part-{part_index:04d}.npz"
        metadata_path = path.with_suffix(".jsonl")
        manifest_path = path.with_suffix(".manifest.json")
        expected = {
            "schema_version": "glm53_v11_recruitment_part_v1",
            "binding_sha256": canonical_sha256(binding),
            "sample_ids_sha256": canonical_sha256([row["sample_id"] for row in chunk]),
            "row_count": len(chunk),
        }
        valid = _part_is_valid(path, manifest_path, expected)
        if valid:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = record.get("metadata_sha256") == sha256_file(metadata_path)
        if not valid:
            task_features: list[np.ndarray] = []
            prompt_features: list[np.ndarray] = []
            metadata: list[dict[str, Any]] = []
            for batch_start in range(0, len(chunk), batch_size):
                batch = chunk[batch_start : batch_start + batch_size]
                try:
                    forwards = runtime.forward_downstream_batch(
                        [row["messages"] for row in batch],
                        selected_layer=selected_layer,
                        continuation=False,
                        selected_span_texts=[
                            str(row["messages"][-1]["content"]) for row in batch
                        ],
                    )
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        raise RuntimeError(
                            "recruitment production batch OOM; stopped without selective retry"
                        ) from error
                    raise
                for row, forward in zip(batch, forwards, strict=True):
                    if forward.selected_span_mean is None:
                        raise ValueError("recruitment forward lacks neutral-task feature")
                    task_features.append(forward.selected_span_mean.astype(np.float16))
                    prompt_features.append(forward.prompt_final.astype(np.float16))
                    metadata.append(
                        {key: value for key, value in row.items() if key != "messages"}
                        | {
                            "prompt_sha256": forward.prompt_sha256,
                            "prompt_tokens": forward.prompt_tokens,
                        }
                    )
            atomic_npz(
                path,
                neutral_task_mean=np.stack(task_features),
                prompt_final=np.stack(prompt_features),
            )
            atomic_jsonl(metadata_path, metadata)
            atomic_json(
                manifest_path,
                expected
                | {
                    "part_sha256": sha256_file(path),
                    "metadata_sha256": sha256_file(metadata_path),
                },
            )
        if not _part_is_valid(path, manifest_path, expected):
            raise ValueError(f"completed recruitment part does not validate: {path}")
        paths.append(path)
    task_execution = np.concatenate(
        [np.load(path)["neutral_task_mean"] for path in paths]
    ).astype(np.float32)
    prompt_execution = np.concatenate(
        [np.load(path)["prompt_final"] for path in paths]
    ).astype(np.float32)
    metadata_execution = [
        json.loads(line)
        for path in paths
        for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    execution_index = {
        row["sample_id"]: index for index, row in enumerate(metadata_execution)
    }
    if len(execution_index) != len(rows):
        raise ValueError("recruitment extraction duplicated or lost rows")
    order = [execution_index[row["sample_id"]] for row in rows]
    task = task_execution[order]
    prompt = prompt_execution[order]
    metadata = [metadata_execution[index] for index in order]
    if (
        task.shape != (len(rows), 4096)
        or prompt.shape != (len(rows), 4096)
        or len(metadata) != len(rows)
    ):
        raise ValueError("recruitment extraction shape differs")
    atomic_npz(output_root / "features.npz", neutral_task_mean=task, prompt_final=prompt)
    atomic_jsonl(output_root / "metadata.jsonl", metadata)
    return task, prompt, metadata


def load_frozen_source_probe(
    *, source_root: Path, feature_root: Path
) -> tuple[int, dict[str, Any]]:
    lock_path = source_root / "source_readout_lock.json"
    arrays_path = source_root / "source_readout_arrays.npz"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(arrays_path) != lock["arrays_sha256"]:
        raise ValueError("frozen source readout arrays differ")
    with np.load(arrays_path) as arrays:
        probe = {
            "mean": arrays["logistic_mean"].astype(np.float64),
            "scale": arrays["logistic_scale"].astype(np.float64),
            "weight": arrays["logistic_weight"].astype(np.float64),
            "bias": float(lock["logistic"]["bias"]),
        }
    from src.glm53_user_eval.v11.probes import load_partition

    development, metadata = load_partition(feature_root, "development")
    selected_layer = int(lock["selected_layer"])
    train = np.asarray([row["split"] == "train" for row in metadata])
    x = development[train, selected_layer].astype(np.float64)
    scores = ((x - probe["mean"]) / probe["scale"]) @ probe["weight"] + probe["bias"]
    probe["source_train_sd"] = float(np.std(scores, ddof=1))
    if not np.isfinite(probe["source_train_sd"]) or probe["source_train_sd"] <= 0:
        raise ValueError("source training score SD is invalid")
    return selected_layer, probe


def _probe_scores(features: np.ndarray, probe: dict[str, Any]) -> np.ndarray:
    raw = ((features.astype(np.float64) - probe["mean"]) / probe["scale"]) @ probe[
        "weight"
    ] + probe["bias"]
    return raw / probe["source_train_sd"]


def analyze_recruitment(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    *,
    probe: dict[str, Any],
    schedule: dict[str, Any],
    manifest: dict[str, Any],
    view: str,
) -> dict[str, Any]:
    scores = _probe_scores(features, probe)
    scored = [
        row | {"eval_score_sd": float(value)} for row, value in zip(rows, scores, strict=True)
    ]
    matrices = _matrix(
        scored,
        value_key="eval_score_sd",
        task_key="template_id",
        group_map=SHORT_GROUPS,
    )
    config = manifest["recruitment"]
    point, interval, _ = four_group_bootstrap(
        matrices, reps=int(config["bootstrap_reps"]), seed=int(config["bootstrap_seed"])
    )
    means = {
        group: _equal_person_mean(value, require_all=True) for group, value in matrices.items()
    }
    target_sets: dict[str, list[int]] = {"enriched": [], "prospective": []}
    for pair in schedule["pairs"]:
        target_sets[str(pair["set"])].append(int(pair["twin_index"]))
    subsets = {
        name: float(
            np.nanmean(matrices["famous_ai"][indices] - matrices["unknown_ai"][indices])
        )
        for name, indices in target_sets.items()
    }
    loo = []
    for index in range(70):
        keep = np.arange(70) != index
        loo.append(
            float(
                np.nanmean(matrices["famous_ai"][keep])
                - np.nanmean(matrices["unknown_ai"][keep])
                - np.nanmean(matrices["famous_nonai"])
                + np.nanmean(matrices["genpop"])
            )
        )
    report = {
        "schema_version": "glm53_v11_recruitment_analysis_v1",
        "view": view,
        "interaction_sd": point,
        "ci95_sd": list(interval),
        "group_means_sd": means,
        "famous_ai_minus_unknown_ai_sd": means["famous_ai"] - means["unknown_ai"],
        "famous_nonai_minus_genpop_sd": means["famous_nonai"] - means["genpop"],
        "enriched_sd": subsets["enriched"],
        "prospective_sd": subsets["prospective"],
        "loo_min_sd": min(loo),
        "loo_max_sd": max(loo),
        "source_train_sd": probe["source_train_sd"],
    }
    checks = {
        "point": point >= 0.20,
        "interval": interval[0] > 0,
        "f_u_positive": report["famous_ai_minus_unknown_ai_sd"] > 0,
        "fame_control": report["famous_nonai_minus_genpop_sd"] <= 0,
        "subset_signs": subsets["enriched"] > 0 and subsets["prospective"] > 0,
        "loo": min(loo) > 0,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def downstream_resource_decision(
    *,
    proxy_seconds: float,
    proxy_benchmark_rows: int,
    proxy_total_rows: int,
    recruitment_seconds: float,
    recruitment_benchmark_rows: int,
    recruitment_total_rows: int,
    deadline_utc_seconds: float,
    hourly_rate_usd: float,
    manifest: dict[str, Any],
    benchmark_seconds_spent: float = 0.0,
    prior_conditional_compute_cost_usd: float = 0.0,
) -> dict[str, Any]:
    execution = manifest["execution"]
    if (
        hourly_rate_usd <= 0
        or proxy_total_rows < 0
        or recruitment_total_rows < 0
        or benchmark_seconds_spent < 0
        or prior_conditional_compute_cost_usd < 0
    ):
        raise ValueError("downstream resource inputs are invalid")
    if proxy_total_rows and (proxy_seconds <= 0 or proxy_benchmark_rows <= 0):
        raise ValueError("proxy projection inputs must be positive")
    if recruitment_total_rows and (
        recruitment_seconds <= 0 or recruitment_benchmark_rows <= 0
    ):
        raise ValueError("recruitment projection inputs must be positive")
    multiplier = float(execution["projection_headroom_multiplier"])
    projected_seconds = multiplier * (
        (
            proxy_total_rows * proxy_seconds / proxy_benchmark_rows
            if proxy_total_rows
            else 0.0
        )
        + (
            recruitment_total_rows * recruitment_seconds / recruitment_benchmark_rows
            if recruitment_total_rows
            else 0.0
        )
    )
    allowance = int(execution["analysis_and_upload_allowance_seconds"])
    reserve = int(execution["backup_reserve_seconds"])
    projected_all_in_seconds = projected_seconds + allowance + reserve
    projected_forward_cost = projected_seconds / 3600 * hourly_rate_usd
    benchmark_cost = benchmark_seconds_spent / 3600 * hourly_rate_usd
    projected_compute_cost = (
        prior_conditional_compute_cost_usd
        + benchmark_cost
        + projected_forward_cost
    )
    projected_cost = projected_compute_cost + (allowance + reserve) / 3600 * hourly_rate_usd
    remaining = deadline_utc_seconds - time.time()
    checks = {
        "deadline_headroom": projected_all_in_seconds <= remaining,
        "conditional_cost": projected_cost
        <= float(execution["maximum_conditional_downstream_cost_usd"]),
    }
    return {
        "schema_version": "glm53_v11_downstream_resource_decision_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "projected_seconds_with_headroom": projected_seconds,
        "analysis_and_upload_allowance_seconds": allowance,
        "projected_all_in_seconds": projected_all_in_seconds,
        "projected_cost_usd": projected_cost,
        "projected_compute_cost_usd": projected_compute_cost,
        "projected_forward_cost_usd": projected_forward_cost,
        "benchmark_cost_usd": benchmark_cost,
        "prior_conditional_compute_cost_usd": prior_conditional_compute_cost_usd,
        "remaining_seconds": remaining,
        "backup_reserve_seconds": reserve,
    }


def build_manual_audit_packet(
    *,
    proxy_rows: list[dict[str, Any]],
    recruitment_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a deterministic score-blind packet; never mark it reviewed."""

    config = manifest["manual_audit"]
    seed = int(config["seed"])

    def ordered(candidates: list[dict[str, Any]], surface: str) -> list[dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{seed}|{surface}|{row['sample_id']}".encode()
            ).hexdigest(),
        )

    proxy = []
    tasks = sorted({row["stimulus_id"] for row in proxy_rows})
    audit_task_count = int(config["proxy_task_coverage_per_group"])
    audit_tasks = sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            f"{seed}|proxy-task|{task}".encode()
        ).hexdigest(),
    )[:audit_task_count]
    for group in GROUP_ORDER:
        for task_index, task in enumerate(audit_tasks):
            desired_codebook = str(task_index % 2)
            candidates = [
                row
                for row in proxy_rows
                if row["group"] == group
                and row["stimulus_id"] == task
                and row["codebook_id"] == desired_codebook
            ]
            if not candidates:
                raise ValueError("proxy manual quota has no eligible balanced candidate")
            proxy.append(ordered(candidates, "proxy")[0])
    recruitment = []
    templates = sorted({row["template_id"] for row in recruitment_rows})
    for group in ("F", "U", "FN", "G"):
        for template in templates:
            candidates = [
                row
                for row in recruitment_rows
                if row["group"] == group and row["template_id"] == template
            ]
            if not candidates:
                raise ValueError("recruitment manual quota has no balanced candidate")
            recruitment.append(ordered(candidates, "recruitment")[0])
    if len(proxy) != int(config["proxy_random_rows"]) or len(recruitment) != int(
        config["recruitment_random_rows"]
    ):
        raise ValueError("manual packet quotas differ from the frozen counts")
    proxy_group_counts = _counts(proxy, "group")
    proxy_codebook_counts = _counts(proxy, "codebook_id")
    if any(
        proxy_group_counts.get(group) != int(config["proxy_rows_per_group"])
        for group in GROUP_ORDER
    ):
        raise ValueError("proxy manual group quotas differ")
    if any(
        proxy_codebook_counts.get(codebook) != int(config["proxy_rows_per_codebook"])
        for codebook in ("0", "1")
    ):
        raise ValueError("proxy manual codebook quotas differ")
    if any(
        len({row["stimulus_id"] for row in proxy if row["group"] == group})
        != audit_task_count
        for group in GROUP_ORDER
    ):
        raise ValueError("proxy manual task coverage differs")
    packet = [
        {
            "surface": "proxy",
            "sample_id": row["sample_id"],
            "group": row["group"],
            "codebook_id": row["codebook_id"],
            "stimulus_id": row["stimulus_id"],
            "messages": row["messages"],
            "reviewed": "",
            "notes": "",
        }
        for row in proxy
    ] + [
        {
            "surface": "recruitment",
            "sample_id": row["sample_id"],
            "group": row["group"],
            "template_id": row["template_id"],
            "messages": row["messages"],
            "reviewed": "",
            "notes": "",
        }
        for row in recruitment
    ]
    report = {
        "schema_version": "glm53_v11_downstream_manual_packet_v1",
        "passed": False,
        "status": config["pending_review_state"],
        "human_review_required": True,
        "proxy_rows": len(proxy),
        "recruitment_rows": len(recruitment),
        "technical_error_rows_required": True,
        "packet_sha256": canonical_sha256(packet),
    }
    return packet, report


__all__ = [
    "analyze_local_proxy",
    "analyze_recruitment",
    "atomic_json",
    "atomic_jsonl",
    "build_manual_audit_packet",
    "build_proxy_token_contract",
    "calibrate_downstream_batch",
    "canonical_sha256",
    "downstream_resource_decision",
    "extract_recruitment_features",
    "four_group_bootstrap",
    "load_frozen_source_probe",
    "load_manifest",
    "proxy_from_compact_logits",
    "score_local_proxy",
    "sha256_file",
    "validate_downstream_assets",
    "validate_runtime_proxy_token_contract",
    "validate_v11_codebooks",
]
