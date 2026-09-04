"""Exact-checkpoint extraction runtime for the v9 prompt representations."""

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
from src.glm53_user_eval.runtime_doctor import installed_vcs_commit

from .datasets import EvalRow
from .masking import TokenMasks, build_token_masks


@dataclass(frozen=True)
class PromptFeatures:
    sample_id: str
    rendered_sha256: str
    input_ids_sha256: str
    prompt_tokens: int
    mask: TokenMasks
    masked_prompt_mean: np.ndarray
    prompt_final: np.ndarray
    last_unmasked_prompt_token: np.ndarray
    cue_token_mean: np.ndarray
    token_bags: dict[int, np.ndarray]


class LoadedV9GLM53:
    """Load once and extract only prompt activations needed by v9."""

    def __init__(self, *, model_path: Path, config: dict[str, Any]):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        expected_devices = int(config["expected_cuda_devices"])
        if torch.cuda.device_count() != expected_devices:
            raise RuntimeError(
                f"expected {expected_devices} CUDA devices, found {torch.cuda.device_count()}"
            )
        observed_commit = installed_vcs_commit("transformers")
        if observed_commit != config["transformers_commit"]:
            raise RuntimeError(
                f"Transformers commit {observed_commit!r} != {config['transformers_commit']}"
            )
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=config["revision"], trust_remote_code=False
        )
        torch.manual_seed(int(config.get("model_initialization_seed", 0)))
        torch.cuda.manual_seed_all(int(config.get("model_initialization_seed", 0)))
        started = time.perf_counter()
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            revision=config["revision"],
            device_map=config["device_map"],
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=False,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.text_model, self.layers = resolve_glm53_text_layers(self.model)
        if len(self.layers) != int(config["text_layers"]):
            raise ValueError("decoder-layer count differs from contract")
        self.embedding_device = self.text_model.embed_tokens.weight.device

    def fp8_scale_report(self) -> dict[str, Any]:
        layer_types = list(self.model.config.text_config.layer_types)
        linear_layers = [
            layer_index
            for layer_index, layer_type in enumerate(layer_types)
            if layer_type == "linear_attention"
        ]
        sparse_layers = [
            layer_index
            for layer_index, layer_type in enumerate(layer_types)
            if layer_type == "deepseek_sparse_attention"
        ]
        expected_names = {
            (
                f"model.language_model.layers.{layer_index}.self_attn.forget_gate."
                f"{projection}.weight_scale_inv"
            )
            for layer_index in linear_layers
            for projection in ("f_a_proj", "f_b_proj")
        }
        tensors = dict(self.model.named_parameters()) | dict(self.model.named_buffers())
        selected = {
            name: tensor
            for name, tensor in tensors.items()
            if "forget_gate" in name
            and ("f_a_proj.weight_scale_inv" in name or "f_b_proj.weight_scale_inv" in name)
        }
        records = []
        for name, tensor in sorted(selected.items()):
            value = tensor.detach().float().cpu().numpy()
            records.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "finite": bool(np.isfinite(value).all()),
                    "minimum": float(value.min()),
                    "maximum": float(value.max()),
                    "mean": float(value.mean()),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
            )
        actual_names = set(selected)
        expected_count = int(self.config["expected_forget_gate_scale_inv_tensors"])
        architecture_matches = (
            len(linear_layers) == int(self.config["expected_linear_attention_layers"])
            and len(sparse_layers) == int(self.config["expected_sparse_attention_layers"])
            and len(layer_types) == int(self.config["text_layers"])
        )
        names_match = actual_names == expected_names
        all_finite = all(record["finite"] for record in records)
        return {
            "initialization_seed": int(self.config.get("model_initialization_seed", 0)),
            "tensor_count": len(records),
            "expected_tensor_count": expected_count,
            "linear_attention_layers": linear_layers,
            "sparse_attention_layers": sparse_layers,
            "architecture_matches": architecture_matches,
            "names_match": names_match,
            "missing_names": sorted(expected_names - actual_names),
            "unexpected_names": sorted(actual_names - expected_names),
            "all_finite": all_finite,
            "passed": (
                architecture_matches
                and names_match
                and len(records) == expected_count
                and all_finite
            ),
            "records": records,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def render_and_encode(self, row: EvalRow) -> tuple[str, dict[str, torch.Tensor], TokenMasks]:
        template_kwargs = {
            "add_generation_prompt": True,
            "reasoning_effort": self.config["reasoning_effort"],
            "clear_thinking": bool(self.config["clear_thinking"]),
        }
        rendered = self.processor.apply_chat_template(
            row.messages, tokenize=False, **template_kwargs
        )
        tokenizer = self.processor.tokenizer
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_tensor = encoded.pop("offset_mapping")
        offsets = [(int(start), int(end)) for start, end in offsets_tensor[0].tolist()]
        direct = self.processor.apply_chat_template(
            row.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        if not torch.equal(encoded["input_ids"], direct["input_ids"]):
            raise ValueError("render-then-tokenize IDs differ from direct chat-template IDs")
        if not torch.equal(encoded["attention_mask"], direct["attention_mask"]):
            raise ValueError("render-then-tokenize attention mask differs from direct template")
        masks = build_token_masks(
            rendered=rendered,
            offsets=offsets,
            attention_mask=encoded["attention_mask"][0].tolist(),
            cue_spans=row.cue_spans,
        )
        if masks.status not in {"masked", "not_available"}:
            raise ValueError(f"invalid cue mask for {row.sample_id}: {masks.status}")
        return rendered, encoded, masks

    def extract(self, row: EvalRow) -> PromptFeatures:
        rendered, encoded, masks = self.render_and_encode(row)
        inputs = {
            key: value.to(self.embedding_device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        retained_index = torch.as_tensor(np.flatnonzero(masks.retained), dtype=torch.long)
        valid_index = torch.as_tensor(np.flatnonzero(masks.valid), dtype=torch.long)
        cue_index = torch.as_tensor(np.flatnonzero(masks.cue), dtype=torch.long)
        last_valid = int(valid_index[-1])
        last_unmasked = int(retained_index[-1])
        masked_means: list[np.ndarray | None] = [None] * len(self.layers)
        prompt_final: list[np.ndarray | None] = [None] * len(self.layers)
        last_unmasked_values: list[np.ndarray | None] = [None] * len(self.layers)
        cue_means: list[np.ndarray | None] = [None] * len(self.layers)
        bags: dict[int, np.ndarray] = {}

        def collect(layer_index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                streams = streams_from_output(output)
                if streams.ndim != 4 or tuple(streams.shape[-2:]) != (4, 4096):
                    raise ValueError(f"unexpected mHC shape at layer {layer_index}: {streams.shape}")
                collapsed = streams.mean(dim=2)[0]
                kept = collapsed.index_select(0, retained_index.to(collapsed.device))
                masked_means[layer_index] = kept.mean(0).detach().float().cpu().numpy()
                prompt_final[layer_index] = collapsed[last_valid].detach().float().cpu().numpy()
                last_unmasked_values[layer_index] = (
                    collapsed[last_unmasked].detach().float().cpu().numpy()
                )
                if cue_index.numel():
                    cue_means[layer_index] = (
                        collapsed.index_select(0, cue_index.to(collapsed.device))
                        .mean(0)
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                else:
                    cue_means[layer_index] = np.full(4096, np.nan, dtype=np.float32)
                bags[layer_index] = kept.detach().to(torch.float16).cpu().numpy()
                return output

            return hook

        with ExitStack() as stack:
            for layer_index, layer in enumerate(self.layers):
                handle = layer.register_forward_hook(collect(layer_index))
                stack.callback(handle.remove)
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        if tuple(output.logits.shape[:2]) != (1, 1):
            raise ValueError(f"unexpected logits shape: {tuple(output.logits.shape)}")
        if any(value is None for value in masked_means + prompt_final + last_unmasked_values):
            raise ValueError("one or more layer hooks did not run")
        return PromptFeatures(
            sample_id=row.sample_id,
            rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            input_ids_sha256=hashlib.sha256(
                encoded["input_ids"].numpy().tobytes()
            ).hexdigest(),
            prompt_tokens=int(encoded["attention_mask"].sum()),
            mask=masks,
            masked_prompt_mean=np.stack(masked_means).astype(np.float16),
            prompt_final=np.stack(prompt_final).astype(np.float16),
            last_unmasked_prompt_token=np.stack(last_unmasked_values).astype(np.float16),
            cue_token_mean=np.stack(cue_means).astype(np.float16),
            token_bags=bags,
        )
