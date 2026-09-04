"""Tensor-parallel throughput smoke for the exact local GLM-5.3 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_tp_smoke(
    *,
    model_path: Path,
    revision: str,
    output: Path,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    started = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=False,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            revision=revision,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=False,
            distributed_config=DistributedConfig(tp_size=world_size),
        )
        model.eval()
        load_seconds = time.perf_counter() - started
        messages = [
            {"role": "system", "content": "You are a helpful assistant named GLM-5.3-Flash."},
            {"role": "user", "content": "Reply with one short sentence about testing."},
        ]
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            reasoning_effort="max",
            clear_thinking=True,
        )
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in dict(encoded).items()
        }
        dist.barrier()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        dist.barrier()
        generation_seconds = time.perf_counter() - generation_started
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        output_tokens = generated[0, prompt_tokens:].detach().cpu()
        if rank != 0:
            return None
        report = {
            "schema_version": "glm53_tp_smoke_v1",
            "model_path": str(model_path),
            "revision": revision,
            "world_size": world_size,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(output_tokens.numel()),
            "tokens_per_second": (
                float(output_tokens.numel()) / generation_seconds
                if generation_seconds > 0
                else None
            ),
            "text": processor.decode(output_tokens, skip_special_tokens=True),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(output, report)
        return report
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    report = run_tp_smoke(
        model_path=args.model_path,
        revision=args.revision,
        output=args.output,
        max_new_tokens=args.max_new_tokens,
    )
    if report is not None:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
