"""Load-once exact-checkpoint prompt-forward runtime for v8."""

from __future__ import annotations

import gc
import hashlib
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from src.glm53_user_eval.mhc import resolve_glm53_text_layers, streams_from_output
from src.glm53_user_eval.runtime_doctor import installed_vcs_commit, sha256_file

from .interventions import batched_add_hook
from .token_positions import final_nonpadding_index, one_hot_final_mask


@dataclass(frozen=True)
class Intervention:
    layer_deltas: dict[int, np.ndarray]
    scope: str = "all_nonpadding_prompt_positions"


def _subsequence_end(sequence: list[int], subsequence: list[int]) -> int:
    matches = [
        start + len(subsequence) - 1
        for start in range(len(sequence) - len(subsequence) + 1)
        if sequence[start : start + len(subsequence)] == subsequence
    ]
    if len(matches) != 1:
        raise ValueError(f"identity span mapping is ambiguous ({len(matches)} matches)")
    return matches[0]


class LoadedGLM53:
    def __init__(self, *, model_path: Path, config: dict[str, Any]):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        expected_devices = int(config.get("expected_cuda_devices", 0))
        if expected_devices and torch.cuda.device_count() != expected_devices:
            raise RuntimeError(
                f"expected {expected_devices} CUDA devices, found "
                f"{torch.cuda.device_count()}"
            )
        expected_commit = str(config["transformers_commit"])
        observed = installed_vcs_commit("transformers")
        if observed != expected_commit:
            raise RuntimeError(f"Transformers commit {observed!r} != {expected_commit}")
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=config["revision"], trust_remote_code=False
        )
        started = time.perf_counter()
        max_memory_setting = config["max_memory_gib_per_gpu"]
        max_memory = None
        if isinstance(max_memory_setting, list):
            if len(max_memory_setting) != torch.cuda.device_count():
                raise ValueError("max-memory vector must match CUDA device count")
            max_memory = {
                index: f"{int(value)}GiB"
                for index, value in enumerate(max_memory_setting)
            }
        elif max_memory_setting != "auto_detected":
            max_memory_gib = int(max_memory_setting)
            max_memory = {
                index: f"{max_memory_gib}GiB"
                for index in range(torch.cuda.device_count())
            }
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            revision=config["revision"],
            device_map=str(config["device_map"]),
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=False,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.text_model, self.layers = resolve_glm53_text_layers(self.model)
        if len(self.layers) != int(config["text_layers"]):
            raise ValueError("loaded layer count differs from runtime contract")
        self.embedding_device = self.text_model.embed_tokens.weight.device

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def _render(
        self, conversations: list[list[dict[str, str]]], *, continuation: bool
    ) -> dict[str, torch.Tensor]:
        kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
            "reasoning_effort": self.config["reasoning_effort"],
            "clear_thinking": bool(self.config["clear_thinking"]),
        }
        if continuation:
            kwargs |= {"add_generation_prompt": False, "continue_final_message": True}
        else:
            kwargs |= {"add_generation_prompt": True}
        encoded = self.processor.apply_chat_template(conversations, **kwargs)
        return {
            key: value.to(self.embedding_device) if isinstance(value, torch.Tensor) else value
            for key, value in dict(encoded).items()
        }

    def _identity_indices(
        self,
        conversations: list[list[dict[str, str]]],
        attention_mask: torch.Tensor,
        *,
        continuation: bool,
    ) -> torch.Tensor:
        indices: list[int] = []
        tokenizer = self.processor.tokenizer
        for row_index, messages in enumerate(conversations):
            system = next(message["content"] for message in messages if message["role"] == "system")
            render_kwargs = {
                "tokenize": False,
                "reasoning_effort": self.config["reasoning_effort"],
                "clear_thinking": bool(self.config["clear_thinking"]),
            }
            if continuation:
                render_kwargs |= {
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                }
            else:
                render_kwargs["add_generation_prompt"] = True
            rendered = self.processor.apply_chat_template(messages, **render_kwargs)
            if rendered.count(system) != 1:
                raise ValueError("system-text offset is ambiguous in rendered prompt")
            end_character = rendered.index(system) + len(system)
            unpadded_end = (
                len(tokenizer(rendered[:end_character], add_special_tokens=False)["input_ids"]) - 1
            )
            padding = int(attention_mask.shape[1] - attention_mask[row_index].sum().item())
            indices.append(unpadded_end + padding)
        return torch.tensor(indices, dtype=torch.long, device=attention_mask.device)

    def forward(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        layers: list[int],
        views: tuple[str, ...] = ("prompt_final",),
        continuation: bool = False,
        intervention: Intervention | None = None,
    ) -> dict[str, Any]:
        inputs = self._render(conversations, continuation=continuation)
        attention_mask = inputs["attention_mask"]
        positions = {"prompt_final": final_nonpadding_index(attention_mask)}
        if "identity_line_final" in views:
            positions["identity_line_final"] = self._identity_indices(
                conversations, attention_mask, continuation=continuation
            )
        features: dict[tuple[int, str], torch.Tensor] = {}

        def collect(layer_index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                streams = streams_from_output(output)
                collapsed = streams.mean(dim=2)
                batch = torch.arange(collapsed.shape[0], device=collapsed.device)
                for view in views:
                    index = positions[view].to(collapsed.device)
                    features[(layer_index, view)] = collapsed[batch, index].detach().float().cpu()
                return output

            return hook

        with ExitStack() as stack:
            if intervention is not None:
                for layer_index, delta in intervention.layer_deltas.items():
                    per_example = torch.as_tensor(delta, dtype=torch.float32)
                    if per_example.ndim == 1:
                        per_example = per_example.expand(len(conversations), -1)
                    scope_mask = (
                        attention_mask
                        if intervention.scope == "all_nonpadding_prompt_positions"
                        else one_hot_final_mask(attention_mask)
                    )
                    handle = self.layers[layer_index].register_forward_hook(
                        batched_add_hook(per_example, scope_mask)
                    )
                    stack.callback(handle.remove)
            for layer_index in layers:
                handle = self.layers[layer_index].register_forward_hook(collect(layer_index))
                stack.callback(handle.remove)
            with torch.inference_mode():
                output = self.model(
                    **inputs,
                    use_cache=False,
                    logits_to_keep=int(self.config["logits_to_keep"]),
                )
        if output.logits.shape[:2] != (len(conversations), 1):
            raise ValueError(f"unexpected final-logit shape {tuple(output.logits.shape)}")
        return {
            "logits": output.logits[:, 0].detach().float().cpu(),
            "features": features,
            "input_tokens": attention_mask.sum(1).detach().cpu(),
            "prompt_hashes": [
                hashlib.sha256(
                    ids[mask.to(torch.bool)].detach().cpu().numpy().tobytes()
                ).hexdigest()
                for ids, mask in zip(inputs["input_ids"], attention_mask, strict=True)
            ],
        }


def verify_model_snapshot(
    model_path: Path, stage_manifest: dict[str, Any], *, full_rehash: bool
) -> dict[str, Any]:
    shards = sorted(model_path.glob("*.safetensors"))
    expected = stage_manifest["safetensor_sha256"]
    checks: dict[str, bool] = {}
    for shard in shards:
        checks[shard.name] = (
            sha256_file(shard) == expected[shard.name]
            if full_rehash
            else shard.name in expected and shard.stat().st_size > 0
        )
    return {
        "shard_count": len(shards),
        "total_bytes": sum(path.stat().st_size for path in shards),
        "full_rehash": full_rehash,
        "all_shards_match": len(shards) == len(expected) and all(checks.values()),
        "checks": checks,
    }
