"""Exact-checkpoint runtime for v11 latent-source feature extraction."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from src.glm53_user_eval.mhc import resolve_glm53_text_layers, streams_from_output
from src.glm53_user_eval.runtime_doctor import installed_vcs_commit
from src.glm53_user_eval.v8.token_positions import final_nonpadding_index

from .tokenizer_audit import messages_from_row


def verify_transformers_source(config: dict[str, Any]) -> str:
    """Verify either VCS metadata or the exact transported source archive."""

    expected_commit = str(config["software"]["transformers_commit"])
    observed_commit = installed_vcs_commit("transformers")
    if observed_commit == expected_commit:
        return expected_commit
    expected_hash = str(config["software"].get("transformers_source_sha256", ""))
    expected_name = str(config["software"].get("transformers_source_filename", ""))
    if len(expected_hash) != 64 or not expected_name:
        raise RuntimeError(f"Transformers commit {observed_commit!r} != {expected_commit}")
    distribution = importlib.metadata.distribution("transformers")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    parsed = urlparse(str(direct_url.get("url", "")))
    if parsed.scheme != "file":
        raise RuntimeError("Transformers source archive is not a local file URL")
    source_value = unquote(parsed.path)
    if len(source_value) >= 3 and source_value[0] == "/" and source_value[2] == ":":
        source_value = source_value[1:]
    source_path = Path(source_value)
    if source_path.name != expected_name or not source_path.is_file():
        raise RuntimeError("Transformers source archive name or path differs")
    observed_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise RuntimeError("Transformers source archive SHA-256 differs")
    return expected_commit


@dataclass(frozen=True)
class SourceFeatures:
    sample_id: str
    rendered_sha256: str
    input_ids_sha256: str
    prompt_tokens: int
    shared_task_suffix_mean: np.ndarray
    prompt_final: np.ndarray
    masked_prompt_mean: np.ndarray
    decisive_fact_token_mean: np.ndarray


@dataclass(frozen=True)
class DownstreamForward:
    prompt_sha256: str
    prompt_tokens: int
    allowed_logits: np.ndarray
    full_logsumexp: float
    full_argmax_token_id: int
    prompt_final: np.ndarray
    selected_span_mean: np.ndarray | None


def pool_layer_streams(
    streams: torch.Tensor,
    token_record: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if streams.ndim != 4 or streams.shape[0] != 1:
        raise ValueError(f"expected one [B,S,streams,H] tensor, got {tuple(streams.shape)}")
    if tuple(streams.shape[-2:]) != (4, 4096):
        raise ValueError(f"unexpected mHC shape {tuple(streams.shape)}")
    collapsed = streams.mean(dim=2)[0]

    def mean_at(name: str) -> torch.Tensor:
        indices = token_record[name]
        if not indices:
            return torch.full(
                (collapsed.shape[-1],),
                torch.nan,
                device=collapsed.device,
                dtype=torch.float32,
            )
        index = torch.as_tensor(indices, device=collapsed.device, dtype=torch.long)
        return collapsed.index_select(0, index).mean(0).float()

    final_index = int(token_record["prompt_final_index"])
    if not 0 <= final_index < collapsed.shape[0]:
        raise ValueError("prompt-final token index is out of range")
    return {
        "shared_task_suffix_mean": mean_at("shared_suffix_token_indices"),
        "prompt_final": collapsed[final_index].float(),
        "masked_prompt_mean": mean_at("masked_prompt_token_indices"),
        "decisive_fact_token_mean": mean_at("decisive_token_indices"),
    }


class LoadedV11GLM53:
    """Load the official FP8 model once and retain only four pooled views."""

    def __init__(self, *, model_path: Path, config: dict[str, Any]):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        expected_devices = int(config["runtime_checks"]["expected_cuda_devices"])
        if torch.cuda.device_count() != expected_devices:
            raise RuntimeError(
                f"expected {expected_devices} CUDA devices, found {torch.cuda.device_count()}"
            )
        expected_gpu = str(config["runtime_checks"]["expected_gpu_name"])
        observed_gpus = [torch.cuda.get_device_name(index) for index in range(expected_devices)]
        if any(name != expected_gpu for name in observed_gpus):
            raise RuntimeError(f"GPU names {observed_gpus!r} != {expected_gpu!r}")
        expected_torch = str(config["software"]["torch"])
        expected_cuda = str(config["software"]["cuda"])
        if str(torch.__version__) != expected_torch:
            raise RuntimeError(f"Torch {torch.__version__!s} != {expected_torch}")
        if str(torch.version.cuda) != expected_cuda:
            raise RuntimeError(f"CUDA {torch.version.cuda!s} != {expected_cuda}")
        verify_transformers_source(config)
        self.config = config
        model_config = config["model"]
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            revision=model_config["revision"],
            trust_remote_code=False,
        )
        seed = int(model_config["initialization_seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            revision=model_config["revision"],
            device_map=model_config["device_map"],
            low_cpu_mem_usage=bool(model_config["low_cpu_mem_usage"]),
            torch_dtype="auto",
            trust_remote_code=False,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.text_model, self.layers = resolve_glm53_text_layers(self.model)
        if len(self.layers) != int(config["architecture"]["text_layers"]):
            raise ValueError("decoder layer count differs from v11 contract")
        self.embedding_device = self.text_model.embed_tokens.weight.device
        self.observed_gpu_names = observed_gpus

    def fp8_scale_report(self) -> dict[str, Any]:
        layer_types = list(self.model.config.text_config.layer_types)
        linear_layers = [
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type == "linear_attention"
        ]
        sparse_layers = [
            index
            for index, layer_type in enumerate(layer_types)
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
        finite = {
            name: bool(torch.isfinite(tensor.detach().float()).all().item())
            for name, tensor in selected.items()
        }
        architecture = self.config["architecture"]
        return {
            "passed": (
                len(layer_types) == int(architecture["text_layers"])
                and len(linear_layers) == int(architecture["linear_attention_layers"])
                and len(sparse_layers) == int(architecture["sparse_attention_layers"])
                and len(selected) == int(architecture["forget_gate_scale_inv_tensors"])
                and set(selected) == expected_names
                and all(finite.values())
            ),
            "tensor_count": len(selected),
            "linear_attention_layers": linear_layers,
            "sparse_attention_layers": sparse_layers,
            "missing_names": sorted(expected_names - set(selected)),
            "unexpected_names": sorted(set(selected) - expected_names),
            "all_finite": all(finite.values()),
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def render_and_encode(
        self,
        row: dict[str, Any],
        token_record: dict[str, Any],
    ) -> tuple[str, dict[str, torch.Tensor]]:
        rendering = self.config["rendering"]
        template_kwargs = {
            "add_generation_prompt": True,
            "reasoning_effort": rendering["reasoning_effort"],
            "clear_thinking": bool(rendering["clear_thinking"]),
        }
        messages = messages_from_row(row)
        rendered = self.processor.apply_chat_template(messages, tokenize=False, **template_kwargs)
        encoded = self.processor.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )
        observed_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        if observed_ids != [int(value) for value in token_record["token_ids"]]:
            raise ValueError(f"runtime token IDs differ for {row['sample_id']}")
        rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if rendered_hash != token_record["rendered_sha256"]:
            raise ValueError(f"runtime rendered prompt differs for {row['sample_id']}")
        return rendered, encoded

    def extract(
        self,
        row: dict[str, Any],
        token_record: dict[str, Any],
    ) -> SourceFeatures:
        rendered, encoded = self.render_and_encode(row, token_record)
        inputs = {
            key: value.to(self.embedding_device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        collected: dict[str, list[np.ndarray | None]] = {
            view: [None] * len(self.layers)
            for view in (
                "shared_task_suffix_mean",
                "prompt_final",
                "masked_prompt_mean",
                "decisive_fact_token_mean",
            )
        }

        def collect(layer_index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                pooled = pool_layer_streams(streams_from_output(output), token_record)
                for view, value in pooled.items():
                    collected[view][layer_index] = value.detach().cpu().numpy()
                return output

            return hook

        with ExitStack() as stack:
            for layer_index, layer in enumerate(self.layers):
                stack.callback(layer.register_forward_hook(collect(layer_index)).remove)
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        if tuple(output.logits.shape[:2]) != (1, 1):
            raise ValueError(f"unexpected logits shape: {tuple(output.logits.shape)}")
        if any(value is None for values in collected.values() for value in values):
            raise ValueError("one or more v11 layer hooks did not run")

        def stack(view: str) -> np.ndarray:
            return np.stack(collected[view]).astype(np.float16)

        input_ids = encoded["input_ids"].detach().cpu().numpy()
        return SourceFeatures(
            sample_id=str(row["sample_id"]),
            rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            input_ids_sha256=hashlib.sha256(input_ids.tobytes()).hexdigest(),
            prompt_tokens=int(encoded["attention_mask"].sum()),
            shared_task_suffix_mean=stack("shared_task_suffix_mean"),
            prompt_final=stack("prompt_final"),
            masked_prompt_mean=stack("masked_prompt_mean"),
            decisive_fact_token_mean=stack("decisive_fact_token_mean"),
        )

    def forward_downstream(
        self,
        messages: list[dict[str, str]],
        *,
        selected_layer: int,
        continuation: bool,
        allowed_token_ids: list[int] | None = None,
        selected_span_text: str | None = None,
    ) -> DownstreamForward:
        """Run one prompt through the padding-safe batch implementation."""

        return self.forward_downstream_batch(
            [messages],
            selected_layer=selected_layer,
            continuation=continuation,
            allowed_token_ids=allowed_token_ids,
            selected_span_texts=(
                [selected_span_text] if selected_span_text is not None else None
            ),
        )[0]

    def forward_downstream_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        selected_layer: int,
        continuation: bool,
        allowed_token_ids: list[int] | None = None,
        selected_span_texts: list[str] | None = None,
    ) -> list[DownstreamForward]:
        """Run a padded prompt batch and retain only the selected layer and final logits."""

        if not 0 <= selected_layer < len(self.layers):
            raise ValueError("downstream selected layer is out of range")
        if not messages_batch:
            raise ValueError("downstream batch is empty")
        if selected_span_texts is not None and len(selected_span_texts) != len(messages_batch):
            raise ValueError("downstream selected-span batch does not align")
        rendering = self.config["rendering"]
        template_kwargs = {
            "reasoning_effort": rendering["reasoning_effort"],
            "clear_thinking": bool(rendering["clear_thinking"]),
        }
        if continuation:
            template_kwargs |= {
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
        else:
            template_kwargs["add_generation_prompt"] = True
        rendered = [
            self.processor.apply_chat_template(messages, tokenize=False, **template_kwargs)
            for messages in messages_batch
        ]
        tokenizer = self.processor.tokenizer
        original_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                padding=True,
                return_offsets_mapping=selected_span_texts is not None,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = original_padding_side
        if not torch.all(encoded["attention_mask"][:, -1] == 1):
            raise ValueError("downstream left-padded batch ends on padding")
        offsets = encoded.pop("offset_mapping", None)
        span_indices: list[list[int]] = []
        if selected_span_texts is not None:
            if offsets is None:
                raise ValueError("downstream tokenizer omitted requested offsets")
            for row_index, selected_span_text in enumerate(selected_span_texts):
                text = rendered[row_index]
                if text.count(selected_span_text) != 1:
                    raise ValueError("downstream selected span is ambiguous")
                start = text.index(selected_span_text)
                end = start + len(selected_span_text)
                indices = [
                    index
                    for index, (token_start, token_end) in enumerate(offsets[row_index].tolist())
                    if token_end > token_start and token_start < end and token_end > start
                ]
                if not indices:
                    raise ValueError("downstream selected span has no tokens")
                span_indices.append(indices)
        inputs = {
            key: value.to(self.embedding_device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        attention_mask = inputs["attention_mask"]
        final_indices = final_nonpadding_index(attention_mask)
        collected: dict[str, list[np.ndarray]] = {}

        def collect(_module: Any, _inputs: Any, output: Any) -> Any:
            streams = streams_from_output(output)
            if streams.ndim != 4 or tuple(streams.shape[-2:]) != (4, 4096):
                raise ValueError("downstream mHC output shape differs")
            if streams.shape[0] != len(messages_batch):
                raise ValueError("downstream mHC batch dimension differs")
            collapsed = streams.mean(dim=2)
            batch = torch.arange(collapsed.shape[0], device=collapsed.device)
            final = collapsed[
                batch,
                final_indices.to(collapsed.device),
            ]
            collected["prompt_final"] = [
                row for row in final.detach().float().cpu().numpy()
            ]
            if span_indices:
                collected["selected_span_mean"] = [
                    collapsed[row_index]
                    .index_select(
                        0,
                        torch.as_tensor(
                            indices,
                            device=collapsed.device,
                            dtype=torch.long,
                        ),
                    )
                    .mean(0)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for row_index, indices in enumerate(span_indices)
                ]
            return output

        before = len(self.layers[selected_layer]._forward_hooks)
        handle = self.layers[selected_layer].register_forward_hook(collect)
        try:
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
        if len(self.layers[selected_layer]._forward_hooks) != before:
            raise ValueError("downstream hook leaked")
        if tuple(output.logits.shape[:2]) != (len(messages_batch), 1):
            raise ValueError(f"unexpected downstream logits shape {tuple(output.logits.shape)}")
        logits = output.logits[:, 0].detach().float()
        token_ids = list(allowed_token_ids or [])
        if len(token_ids) != len(set(token_ids)) or any(
            token_id < 0 or token_id >= logits.shape[1] for token_id in token_ids
        ):
            raise ValueError("downstream allowed-token IDs are invalid")
        allowed = (
            logits.index_select(
                1,
                torch.as_tensor(token_ids, device=logits.device, dtype=torch.long),
            )
            .cpu()
            .numpy()
            if token_ids
            else np.empty((len(messages_batch), 0), dtype=np.float32)
        )
        input_ids = encoded["input_ids"].detach().cpu()
        mask = encoded["attention_mask"].detach().cpu().to(torch.bool)
        full_logsumexp = torch.logsumexp(logits, dim=1).cpu().numpy()
        full_argmax = torch.argmax(logits, dim=1).cpu().numpy()
        selected = collected.get("selected_span_mean")
        return [
            DownstreamForward(
                prompt_sha256=hashlib.sha256(input_ids[index][mask[index]].numpy().tobytes()).hexdigest(),
                prompt_tokens=int(mask[index].sum()),
                allowed_logits=allowed[index],
                full_logsumexp=float(full_logsumexp[index]),
                full_argmax_token_id=int(full_argmax[index]),
                prompt_final=collected["prompt_final"][index],
                selected_span_mean=None if selected is None else selected[index],
            )
            for index in range(len(messages_batch))
        ]

    def downstream_token_lengths(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        continuation: bool,
    ) -> list[int]:
        """Return exact rendered-token lengths without executing the model."""

        rendering = self.config["rendering"]
        template_kwargs = {
            "reasoning_effort": rendering["reasoning_effort"],
            "clear_thinking": bool(rendering["clear_thinking"]),
        }
        if continuation:
            template_kwargs |= {
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
        else:
            template_kwargs["add_generation_prompt"] = True
        rendered = [
            self.processor.apply_chat_template(messages, tokenize=False, **template_kwargs)
            for messages in messages_batch
        ]
        return [
            len(
                self.processor.tokenizer(
                    text,
                    add_special_tokens=False,
                )["input_ids"]
            )
            for text in rendered
        ]

    def no_op_equivalence(
        self,
        row: dict[str, Any],
        token_record: dict[str, Any],
    ) -> dict[str, Any]:
        """Check that an identity hook changes neither logits nor pooled features."""

        hook_counts_before = [len(layer._forward_hooks) for layer in self.layers]

        def run(identity_hook: bool) -> tuple[np.ndarray, SourceFeatures]:
            handle = None
            if identity_hook:
                handle = self.layers[len(self.layers) // 2].register_forward_hook(
                    lambda _module, _inputs, output: output
                )
            try:
                rendered, encoded = self.render_and_encode(row, token_record)
                inputs = {
                    key: value.to(self.embedding_device)
                    for key, value in encoded.items()
                    if isinstance(value, torch.Tensor)
                }
                collected: dict[str, list[np.ndarray | None]] = {
                    view: [None] * len(self.layers)
                    for view in (
                        "shared_task_suffix_mean",
                        "prompt_final",
                        "masked_prompt_mean",
                        "decisive_fact_token_mean",
                    )
                }

                def collect(layer_index: int):
                    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                        pooled = pool_layer_streams(streams_from_output(output), token_record)
                        for view, value in pooled.items():
                            collected[view][layer_index] = value.detach().cpu().numpy()
                        return output

                    return hook

                with ExitStack() as stack:
                    for layer_index, layer in enumerate(self.layers):
                        stack.callback(layer.register_forward_hook(collect(layer_index)).remove)
                    with torch.inference_mode():
                        output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                if any(value is None for values in collected.values() for value in values):
                    raise ValueError("diagnostic layer hook did not run")

                def stacked(view: str) -> np.ndarray:
                    return np.stack(collected[view]).astype(np.float16)

                input_ids = encoded["input_ids"].detach().cpu().numpy()
                features = SourceFeatures(
                    sample_id=str(row["sample_id"]),
                    rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                    input_ids_sha256=hashlib.sha256(input_ids.tobytes()).hexdigest(),
                    prompt_tokens=int(encoded["attention_mask"].sum()),
                    shared_task_suffix_mean=stacked("shared_task_suffix_mean"),
                    prompt_final=stacked("prompt_final"),
                    masked_prompt_mean=stacked("masked_prompt_mean"),
                    decisive_fact_token_mean=stacked("decisive_fact_token_mean"),
                )
                return output.logits.detach().float().cpu().numpy(), features
            finally:
                if handle is not None:
                    handle.remove()

        baseline_logits, baseline = run(False)
        identity_logits, identity = run(True)
        hook_counts_after = [len(layer._forward_hooks) for layer in self.layers]

        synthetic_before = len(self.layers[0]._forward_hooks)
        try:
            with ExitStack() as stack:
                stack.callback(
                    self.layers[0]
                    .register_forward_hook(lambda _module, _inputs, output: output)
                    .remove
                )
                raise RuntimeError("synthetic hook-cleanup check")
        except RuntimeError as exc:
            if str(exc) != "synthetic hook-cleanup check":
                raise
        synthetic_after = len(self.layers[0]._forward_hooks)

        views = (
            "shared_task_suffix_mean",
            "prompt_final",
            "masked_prompt_mean",
            "decisive_fact_token_mean",
        )
        feature_exact = all(
            np.array_equal(getattr(baseline, view), getattr(identity, view), equal_nan=True)
            for view in views
        )
        logits_exact = bool(np.array_equal(baseline_logits, identity_logits))
        hooks_removed = (
            hook_counts_before == hook_counts_after and synthetic_before == synthetic_after
        )
        return {
            "passed": logits_exact and feature_exact and hooks_removed,
            "logits_exact": logits_exact,
            "features_exact": feature_exact,
            "hooks_removed_after_forward": hook_counts_before == hook_counts_after,
            "hooks_removed_after_exception": synthetic_before == synthetic_after,
            "gpu_names": self.observed_gpu_names,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        }


__all__ = [
    "DownstreamForward",
    "LoadedV11GLM53",
    "SourceFeatures",
    "pool_layer_streams",
]
