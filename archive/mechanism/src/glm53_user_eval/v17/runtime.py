"""Exact-checkpoint Hua direction extraction and token-scoped interventions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from src.glm53_user_eval.mhc import replace_streams, streams_from_output
from src.glm53_user_eval.v11.runtime import LoadedV11GLM53


@dataclass(frozen=True)
class CompactForward:
    prompt_sha256: str
    prompt_tokens: int
    allowed_logits: np.ndarray
    full_logsumexp: float
    full_argmax_token_id: int
    full_logits: np.ndarray | None = None


def token_indices_for_span(offsets: torch.Tensor, start: int, end: int) -> list[int]:
    if not 0 <= start < end:
        raise ValueError("invalid character span")
    indices = [
        index
        for index, (left, right) in enumerate(offsets.tolist())
        if right > left and left < end and right > start
    ]
    if not indices:
        raise ValueError("character span maps to no content token")
    return indices


def content_character_spans(
    rendered: str,
    messages: list[dict[str, str]],
    *,
    scope: str,
) -> list[tuple[int, int]]:
    """Locate message content in rendered order, excluding template and system tokens."""

    if scope not in {"user_content", "user_plus_replay_assistant"}:
        raise ValueError("unknown intervention scope")
    spans: list[tuple[int, int]] = []
    cursor = 0
    final_message = len(messages) - 1
    for index, message in enumerate(messages):
        role = message["role"]
        include_user = role == "user"
        include_replayed_assistant = (
            scope == "user_plus_replay_assistant"
            and role == "assistant"
            and index < final_message
        )
        if not include_user and not include_replayed_assistant:
            continue
        content = str(message["content"])
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError(f"message {index} content is absent from rendered prompt")
        end = start + len(content)
        cursor = end
        if include_user:
            spans.append((start, end))
        else:
            match = re.search(r"[.!?](?:\s+|$)", content)
            if match is not None and match.end() < len(content):
                spans.append((start + match.end(), end))
    return spans


def intervention_mask(
    rendered: str,
    messages: list[dict[str, str]],
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scope: str,
) -> torch.Tensor:
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for start, end in content_character_spans(rendered, messages, scope=scope):
        mask[token_indices_for_span(offsets, start, end)] = True
    mask &= attention_mask.to(torch.bool)
    if not mask.any():
        raise ValueError("intervention scope contains no tokens")
    return mask


class LoadedV17GLM53(LoadedV11GLM53):
    """V11 exact runtime with Hua extraction and masked activation addition."""

    def extract_prompt_directions(
        self,
        prompts: Iterable[dict[str, str]],
        *,
        system_text: str = "detailed thinking on",
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        pair_differences: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for pair in prompts:
            activations: dict[str, np.ndarray] = {}
            for variant in ("deployment", "evaluation"):
                prompt = str(pair[f"{variant}_prompt"])
                if not prompt.endswith(":"):
                    raise ValueError("Hua source prompt must end in a colon")
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt},
                ]
                rendered = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort=self.config["rendering"]["reasoning_effort"],
                    clear_thinking=bool(self.config["rendering"]["clear_thinking"]),
                )
                tokenizer = self.processor.tokenizer
                encoded = tokenizer(
                    rendered,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets = encoded.pop("offset_mapping")[0]
                prompt_start = rendered.rfind(prompt)
                if prompt_start < 0 or rendered.count(prompt) != 1:
                    raise ValueError("Hua prompt does not occur exactly once after rendering")
                colon_start = prompt_start + len(prompt) - 1
                colon_indices = token_indices_for_span(offsets, colon_start, colon_start + 1)
                if len(colon_indices) != 1:
                    raise ValueError("final source colon must map to exactly one token")
                colon_index = colon_indices[0]
                inputs = {
                    key: value.to(self.embedding_device)
                    for key, value in encoded.items()
                    if isinstance(value, torch.Tensor)
                }
                collected: list[np.ndarray | None] = [None] * len(self.layers)

                def collect(
                    layer_index: int,
                    *,
                    destination: list[np.ndarray | None] = collected,
                    selected_token: int = colon_index,
                ):
                    def hook(_module: Any, _inputs: Any, layer_output: Any) -> Any:
                        streams = streams_from_output(layer_output)
                        if streams.ndim != 4 or tuple(streams.shape[-2:]) != (4, 4096):
                            raise ValueError("unexpected GLM mHC layer output")
                        destination[layer_index] = (
                            streams[0, selected_token].mean(0).detach().float().cpu().numpy()
                        )
                        return layer_output

                    return hook

                with ExitStack() as stack:
                    for layer_index, layer in enumerate(self.layers):
                        stack.callback(layer.register_forward_hook(collect(layer_index)).remove)
                    with torch.inference_mode():
                        model_output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                if tuple(model_output.logits.shape[:2]) != (1, 1) or any(x is None for x in collected):
                    raise ValueError("Hua extraction did not collect every layer")
                activations[variant] = np.stack(collected).astype(np.float32)
                metadata.append(
                    {
                        "pair_id": pair["pair_id"],
                        "variant": variant,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                        "colon_token_index": colon_index,
                        "colon_token_id": int(encoded["input_ids"][0, colon_index]),
                        "prompt_tokens": int(encoded["attention_mask"].sum()),
                    }
                )
            pair_differences.append(activations["deployment"] - activations["evaluation"])
        return np.stack(pair_differences), metadata

    def forward_intervened_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        allowed_token_ids: list[int],
        layer_deltas: dict[int, np.ndarray] | None = None,
        scope: str = "user_content",
        continuation: bool = True,
        return_full_logits: bool = False,
    ) -> list[CompactForward]:
        if not messages_batch:
            raise ValueError("empty intervention batch")
        rendering = self.config["rendering"]
        kwargs: dict[str, Any] = {
            "reasoning_effort": rendering["reasoning_effort"],
            "clear_thinking": bool(rendering["clear_thinking"]),
        }
        if continuation:
            kwargs.update(add_generation_prompt=False, continue_final_message=True)
        else:
            kwargs["add_generation_prompt"] = True
        rendered = [self.processor.apply_chat_template(x, tokenize=False, **kwargs) for x in messages_batch]
        tokenizer = self.processor.tokenizer
        old_padding = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = old_padding
        offsets = encoded.pop("offset_mapping")
        attention = encoded["attention_mask"]
        masks = torch.stack(
            [
                intervention_mask(text, messages, row_offsets, row_attention, scope=scope)
                for text, messages, row_offsets, row_attention in zip(
                    rendered, messages_batch, offsets, attention, strict=True
                )
            ]
        )
        inputs = {
            key: value.to(self.embedding_device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        deltas = layer_deltas or {}
        if any(not 0 <= layer < len(self.layers) for layer in deltas):
            raise ValueError("intervention layer is out of range")
        with ExitStack() as stack:
            for layer_index, delta_array in sorted(deltas.items()):
                delta = torch.as_tensor(delta_array, dtype=torch.float32)
                if delta.shape == (4096,):
                    delta = delta[None, :].expand(len(messages_batch), -1)
                if delta.shape != (len(messages_batch), 4096):
                    raise ValueError("intervention delta does not align with batch")

                def hook(_module: Any, _inputs: Any, output: Any, *, delta=delta) -> Any:
                    streams = streams_from_output(output)
                    local_mask = masks.to(streams.device, streams.dtype)[:, :, None, None]
                    local_delta = delta.to(streams.device, streams.dtype)[:, None, None, :]
                    return replace_streams(output, streams + local_mask * local_delta)

                stack.callback(self.layers[layer_index].register_forward_hook(hook).remove)
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits[:, 0].detach().float()
        if tuple(logits.shape[:1]) != (len(messages_batch),):
            raise ValueError("intervention output batch differs")
        ids = torch.as_tensor(allowed_token_ids, device=logits.device, dtype=torch.long)
        if len(set(allowed_token_ids)) != len(allowed_token_ids):
            raise ValueError("allowed token IDs are not unique")
        allowed = logits.index_select(1, ids).cpu().numpy()
        logsumexp = torch.logsumexp(logits, dim=1).cpu().numpy()
        argmax = torch.argmax(logits, dim=1).cpu().numpy()
        input_ids = encoded["input_ids"].detach().cpu()
        attention_bool = attention.to(torch.bool)
        return [
            CompactForward(
                prompt_sha256=hashlib.sha256(input_ids[i][attention_bool[i]].numpy().tobytes()).hexdigest(),
                prompt_tokens=int(attention_bool[i].sum()),
                allowed_logits=allowed[i],
                full_logsumexp=float(logsumexp[i]),
                full_argmax_token_id=int(argmax[i]),
                full_logits=(logits[i].cpu().numpy() if return_full_logits else None),
            )
            for i in range(len(messages_batch))
        ]


def raw_layer_deltas(
    directions: np.ndarray,
    layers: list[int],
    alpha: float,
) -> dict[int, np.ndarray]:
    values = np.asarray(directions, dtype=np.float32)
    if values.shape != (45, 4096):
        raise ValueError("directions must have shape [45,4096]")
    if len(layers) != len(set(layers)):
        raise ValueError("intervention layers are duplicated")
    return {layer: float(alpha) * values[layer] for layer in layers}


__all__ = [
    "CompactForward",
    "LoadedV17GLM53",
    "content_character_spans",
    "intervention_mask",
    "raw_layer_deltas",
    "token_indices_for_span",
]
