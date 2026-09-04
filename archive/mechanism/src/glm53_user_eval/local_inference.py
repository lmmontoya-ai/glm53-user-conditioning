"""Resumable local generation with the exact official GLM-5.3 FP8 checkpoint."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mhc import resolve_glm53_text_layers
from .runtime_doctor import installed_vcs_commit
from .schemas import LocalSubjectResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _token_hash(token_ids: torch.Tensor) -> str:
    contiguous = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_inputs(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    text_model, _ = resolve_glm53_text_layers(model)
    device = text_model.embed_tokens.weight.device
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _render(processor: Any, messages: list[dict[str, str]], config: dict[str, Any]) -> Any:
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        reasoning_effort=str(config["reasoning_effort"]),
        clear_thinking=bool(config["clear_thinking"]),
    )


def _generate_turn(
    *,
    model: Any,
    processor: Any,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    seed: int,
) -> tuple[str, int, torch.Tensor, float]:
    _seed_everything(seed)
    encoded = _move_inputs(dict(_render(processor, messages, config)), model)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=bool(config["do_sample"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            max_new_tokens=int(config["max_new_tokens"]),
        )
    latency = time.perf_counter() - started
    new_tokens = output[0, prompt_tokens:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return text, prompt_tokens, new_tokens, latency


def _validate_g2_decision(path: Path, *, revision: str) -> dict[str, Any]:
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("gate") != "G2" or decision.get("passed") is not True:
        raise ValueError("local behavior requires a passing G2 decision")
    if decision.get("model_revision") != revision:
        raise ValueError("G2 decision model revision differs from local behavior subject")
    runtime_hash = str(decision.get("runtime_hash") or "")
    if len(runtime_hash) != 64:
        raise ValueError("G2 decision lacks a valid runtime hash")
    return decision


def run_local_subject_schedule(
    *,
    schedule_rows: list[dict[str, Any]],
    prompt_rows: dict[str, dict[str, Any]],
    model_path: Path,
    model_id: str,
    revision: str,
    transformers_commit: str,
    g2_decision_path: Path,
    generation_config: dict[str, Any],
    output_root: Path,
    run_id: str,
    deadline_minutes: int,
    max_samples: int,
) -> dict[str, Any]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    decision = _validate_g2_decision(g2_decision_path, revision=revision)
    runtime_hash = str(decision["runtime_hash"])
    observed_commit = installed_vcs_commit("transformers")
    if observed_commit != transformers_commit:
        raise RuntimeError(
            f"installed Transformers commit {observed_commit!r} differs from {transformers_commit}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("local behavior requires the preregistered multi-GPU CUDA runtime")

    output_root.mkdir(parents=True, exist_ok=True)
    calls_root = output_root / "calls"
    calls_root.mkdir(exist_ok=True)
    completed = {
        path.parent.name
        for path in calls_root.glob("*/subject.json")
        if path.is_file()
    }
    pending = [row for row in schedule_rows if row["sample_id"] not in completed]
    if max_samples > 0:
        pending = pending[:max_samples]

    started = time.perf_counter()
    deadline_seconds = deadline_minutes * 60
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=False,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="balanced",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        trust_remote_code=False,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    new_completed = 0
    stopped_for_deadline = False
    failed = 0

    try:
        for schedule in pending:
            if time.perf_counter() - started >= deadline_seconds:
                stopped_for_deadline = True
                break
            sample_id = str(schedule["sample_id"])
            prompt = prompt_rows[sample_id]
            if prompt["prompt_hash"] != schedule["prompt_hash"]:
                raise ValueError(f"prompt hash mismatch for {sample_id}")
            seed = int(schedule["generation_seed"])
            main_messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["main_prompt"]},
            ]
            try:
                main_text, main_prompt_tokens, main_tokens, main_latency = _generate_turn(
                    model=model,
                    processor=processor,
                    messages=main_messages,
                    config=generation_config,
                    seed=seed,
                )
                followup_messages = [
                    *main_messages,
                    {"role": "assistant", "content": main_text},
                    {"role": "user", "content": prompt["followup_prompt"]},
                ]
                followup_text, followup_prompt_tokens, followup_tokens, followup_latency = (
                    _generate_turn(
                        model=model,
                        processor=processor,
                        messages=followup_messages,
                        config=generation_config,
                        seed=seed,
                    )
                )
                result = LocalSubjectResult(
                    run_id=run_id,
                    sample_id=sample_id,
                    scenario_id=str(schedule["scenario_id"]),
                    persona_key=str(schedule["persona_key"]),
                    condition=str(schedule["condition"]),
                    model_id=model_id,
                    model_revision=revision,
                    runtime_hash=runtime_hash,
                    prompt_hash=str(schedule["prompt_hash"]),
                    main_text=main_text,
                    followup_text=followup_text,
                    main_prompt_tokens=main_prompt_tokens,
                    main_generated_tokens=int(main_tokens.numel()),
                    followup_prompt_tokens=followup_prompt_tokens,
                    followup_generated_tokens=int(followup_tokens.numel()),
                    main_output_token_sha256=_token_hash(main_tokens),
                    followup_output_token_sha256=_token_hash(followup_tokens),
                    generation_seed=seed,
                    generation_config=generation_config,
                    latency_seconds={"main": main_latency, "followup": followup_latency},
                    completed_at=_now_iso(),
                )
                _atomic_json(
                    calls_root / sample_id / "subject.json",
                    {
                        "schedule": schedule,
                        "prompt": prompt,
                        "subject": result.model_dump(mode="json"),
                    },
                )
                new_completed += 1
            except Exception as exc:
                _atomic_json(
                    calls_root / sample_id / "subject.error.json",
                    {
                        "sample_id": sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "created_at": _now_iso(),
                    },
                )
                failed += 1
                raise
    finally:
        device_map = getattr(model, "hf_device_map", None)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    total_completed = len(completed) + new_completed
    summary = {
        "schema_version": "glm53_local_generation_summary_v1",
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "runtime_hash": runtime_hash,
        "transformers_commit": observed_commit,
        "scheduled": len(schedule_rows),
        "completed_before": len(completed),
        "new_completed": new_completed,
        "total_completed": total_completed,
        "remaining": len(schedule_rows) - total_completed,
        "failed": failed,
        "stopped_for_deadline": stopped_for_deadline,
        "deadline_minutes": deadline_minutes,
        "load_seconds": load_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "device_map": device_map,
        "completed_at": _now_iso(),
    }
    _atomic_json(output_root / "generation_summary.json", summary)
    return summary
