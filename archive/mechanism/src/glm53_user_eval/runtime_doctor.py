"""Paid G2 runtime checks for the official GLM-5.3 FP8 checkpoint."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from .mhc import (
    add_collapsed_delta,
    collapse_streams,
    replace_streams,
    resolve_glm53_text_layers,
    select_prompt_vectors,
    streams_from_output,
)
from .runtime import prompt_final_indices, validate_glm53_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_vcs_commit(distribution_name: str) -> str | None:
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return str(commit) if commit else None


def stage_model_snapshot(
    *,
    model_id: str,
    revision: str,
    output_root: Path,
    expected_shards: int,
    expected_bytes: int,
    max_workers: int,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    destination = output_root / revision
    started = time.perf_counter()
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
        max_workers=max_workers,
    )
    shards = sorted(destination.glob("*.safetensors"))
    total_bytes = sum(path.stat().st_size for path in shards)
    if len(shards) != expected_shards:
        raise ValueError(f"expected {expected_shards} safetensor shards, found {len(shards)}")
    if total_bytes != expected_bytes:
        raise ValueError(f"expected {expected_bytes} safetensor bytes, found {total_bytes}")
    checksums = {path.name: sha256_file(path) for path in shards}
    return {
        "schema_version": "glm53_model_stage_v1",
        "model_id": model_id,
        "revision": revision,
        "model_path": str(destination),
        "safetensor_shards": len(shards),
        "safetensor_bytes": total_bytes,
        "safetensor_sha256": checksums,
        "elapsed_seconds": time.perf_counter() - started,
    }


@contextmanager
def registered_hook(module: Any, hook: Any) -> Iterator[None]:
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def zero_delta_hook(_module: Any, _inputs: Any, output: Any) -> Any:
    streams = streams_from_output(output)
    delta = torch.zeros(streams.shape[-1], dtype=streams.dtype, device=streams.device)
    return replace_streams(output, add_collapsed_delta(streams, delta))


def additive_probe_hook(delta: torch.Tensor, trace: dict[str, Any]):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        streams = streams_from_output(output)
        before = collapse_streams(streams).detach()
        changed = add_collapsed_delta(streams, delta.to(streams.device, streams.dtype))
        observed = collapse_streams(changed).detach() - before
        trace["max_delta_error"] = float(
            (observed - delta.to(observed.device, observed.dtype)).abs().max().item()
        )
        trace["shape"] = list(streams.shape)
        return replace_streams(output, changed)

    return hook


def extraction_hook(
    layer_index: int,
    token_indices: list[int],
    store: dict[int, torch.Tensor],
):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        streams = streams_from_output(output)
        store[layer_index] = select_prompt_vectors(streams, token_indices)
        return output

    return hook


def _move_inputs_to_embedding_device(inputs: dict[str, Any], text_model: Any) -> dict[str, Any]:
    device = text_model.embed_tokens.weight.device
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _gpu_memory() -> list[dict[str, Any]]:
    values = []
    for index in range(torch.cuda.device_count()):
        values.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "total_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
        )
    return values


def run_runtime_doctor(
    *,
    model_path: Path,
    revision: str,
    prompts: list[list[dict[str, str]]],
    reasoning_effort: str,
    clear_thinking: bool,
    deadline_minutes: int,
    expected_transformers_commit: str,
) -> dict[str, Any]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    started = time.perf_counter()

    def enforce_deadline() -> None:
        if time.perf_counter() - started > deadline_minutes * 60:
            raise TimeoutError("G2 runtime doctor exceeded its paid deadline")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("G2 requires a paid multi-GPU CUDA Pod")
    transformers_commit = installed_vcs_commit("transformers")
    if transformers_commit != expected_transformers_commit:
        raise RuntimeError(
            f"installed Transformers commit {transformers_commit!r} differs from "
            f"{expected_transformers_commit}"
        )
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    static_contract = validate_glm53_config(config)
    processor = AutoProcessor.from_pretrained(model_path, revision=revision, trust_remote_code=False)
    load_started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="balanced",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        trust_remote_code=False,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    enforce_deadline()
    text_model, layers = resolve_glm53_text_layers(model)
    if len(layers) != static_contract["text_layers"]:
        raise ValueError("loaded decoder layer count differs from the pinned config")
    if int(text_model.config.hidden_size) != static_contract["hidden_size"]:
        raise ValueError("loaded hidden size differs from the pinned config")
    if int(text_model.config.hc_mult) != static_contract["hc_mult"]:
        raise ValueError("loaded hc_mult differs from the pinned config")

    rendered_hashes: list[str] = []
    forward_seconds: list[float] = []
    extraction_shapes: dict[str, list[int]] = {}
    sample_inputs: dict[str, Any] | None = None
    for prompt_index, messages in enumerate(prompts):
        enforce_deadline()
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
        )
        rendered_hashes.append(
            hashlib.sha256(inputs["input_ids"].cpu().numpy().tobytes()).hexdigest()
        )
        inputs = _move_inputs_to_embedding_device(dict(inputs), text_model)
        token_indices = prompt_final_indices(inputs["attention_mask"])
        store: dict[int, torch.Tensor] = {}
        target_layers = [0, len(layers) // 2, len(layers) - 1]
        handles = [
            layers[index].register_forward_hook(extraction_hook(index, token_indices, store))
            for index in target_layers
        ]
        forward_started = time.perf_counter()
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        forward_seconds.append(time.perf_counter() - forward_started)
        if set(store) != set(target_layers):
            raise RuntimeError("prompt-vector extraction missed a target layer")
        for index, value in store.items():
            extraction_shapes[f"prompt_{prompt_index}_layer_{index}"] = list(value.shape)
        if sample_inputs is None:
            sample_inputs = inputs

    if sample_inputs is None:
        raise ValueError("runtime doctor requires at least one prompt")
    enforce_deadline()
    test_streams = torch.randn(
        1,
        2,
        static_contract["hc_mult"],
        static_contract["hidden_size"],
        device=next(text_model.parameters()).device,
        dtype=torch.float16,
    )
    observed_mean = text_model.hc_head(test_streams)
    expected_mean = test_streams.mean(dim=2).to(observed_mean.device)
    hyper_head_error = float((expected_mean - observed_mean).abs().max().item())

    with torch.inference_mode():
        baseline_logits = model(**sample_inputs, use_cache=False).logits.detach().cpu()
    hook_count_before = len(layers[len(layers) // 2]._forward_hooks)
    with registered_hook(layers[len(layers) // 2], zero_delta_hook):
        with torch.inference_mode():
            zero_logits = model(**sample_inputs, use_cache=False).logits.detach().cpu()
    hook_count_after = len(layers[len(layers) // 2]._forward_hooks)
    zero_logit_error = float((baseline_logits - zero_logits).abs().max().item())

    generation_kwargs = {"do_sample": False, "max_new_tokens": 8}
    with torch.inference_mode():
        baseline_generation = model.generate(**sample_inputs, **generation_kwargs).detach().cpu()
    with registered_hook(layers[len(layers) // 2], zero_delta_hook):
        with torch.inference_mode():
            zero_generation = model.generate(**sample_inputs, **generation_kwargs).detach().cpu()
    greedy_equal = bool(torch.equal(baseline_generation, zero_generation))

    # The live residual stream is BF16. A 1e-3 diagnostic delta can round to
    # zero at ordinary activation magnitudes, which tests quantization rather
    # than whether the hook changes the requested coordinate. Use a small but
    # BF16-resolvable delta and judge it at half-delta absolute tolerance.
    requested_delta = 0.25
    delta = torch.zeros(static_contract["hidden_size"], dtype=torch.float32)
    delta[0] = requested_delta
    trace: dict[str, Any] = {}
    with registered_hook(layers[len(layers) // 2], additive_probe_hook(delta, trace)):
        with torch.inference_mode():
            model(**sample_inputs, use_cache=False)
    if len(layers[len(layers) // 2]._forward_hooks) != hook_count_before:
        raise RuntimeError("additive test hook leaked after removal")

    report = {
        "schema_version": "glm53_runtime_doctor_v2",
        "model_path": str(model_path),
        "revision": revision,
        "transformers_commit": transformers_commit,
        "deadline_minutes": deadline_minutes,
        "static_contract": static_contract,
        "loaded_layer_count": len(layers),
        "rendered_prompt_hashes": rendered_hashes,
        "prompt_count": len(prompts),
        "extraction_shapes": extraction_shapes,
        "hyper_head_max_error": hyper_head_error,
        "zero_hook_logit_max_error": zero_logit_error,
        "zero_hook_greedy_equal": greedy_equal,
        "hook_count_before": hook_count_before,
        "hook_count_after": hook_count_after,
        "additive_test": trace,
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "memory": _gpu_memory(),
        },
        "device_map": getattr(model, "hf_device_map", None),
        "checks": {
            "twenty_prompts": len(prompts) == 20,
            "layer_shape_contract": all(
                shape == [1, static_contract["hidden_size"]]
                for shape in extraction_shapes.values()
            ),
            "hyper_head_exact": hyper_head_error == 0.0,
            "zero_logits_exact": zero_logit_error == 0.0,
            "zero_greedy_exact": greedy_equal,
            "hooks_removed": hook_count_before == hook_count_after,
            "additive_delta_within_bf16_tolerance": trace.get(
                "max_delta_error", 1.0
            )
            <= requested_delta / 2,
        },
    }
    report["passed"] = all(report["checks"].values())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report
