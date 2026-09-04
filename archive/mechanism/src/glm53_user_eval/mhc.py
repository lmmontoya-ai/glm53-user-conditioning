"""Manifold-Constrained Hyper-Connection extraction and intervention algebra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def resolve_glm53_text_layers(model: Any) -> tuple[Any, list[Any]]:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(
            getattr(getattr(model, "base_model", None), "model", None),
            "language_model",
            None,
        ),
    ]
    for text_model in candidates:
        layers = getattr(text_model, "layers", None)
        if layers is not None:
            return text_model, list(layers)
    raise RuntimeError("could not resolve GLM-5.3 text decoder layers")


def streams_from_output(output: Any) -> torch.Tensor:
    streams = output[0] if isinstance(output, tuple) else output
    if not isinstance(streams, torch.Tensor) or streams.ndim != 4:
        raise ValueError("GLM-5.3 layer output must be [batch, sequence, hc_mult, hidden]")
    return streams


def replace_streams(output: Any, streams: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (streams, *output[1:])
    return streams


def collapse_streams(streams: torch.Tensor) -> torch.Tensor:
    if streams.ndim != 4:
        raise ValueError("stream tensor must have four dimensions")
    return streams.mean(dim=2)


def add_collapsed_delta(streams: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim != 1 or delta.shape[0] != streams.shape[-1]:
        raise ValueError("delta must match hidden size")
    return streams + delta.view(1, 1, 1, -1)


def project_out_collapsed(streams: torch.Tensor, unit_direction: torch.Tensor) -> torch.Tensor:
    if unit_direction.ndim != 1 or unit_direction.shape[0] != streams.shape[-1]:
        raise ValueError("direction must match hidden size")
    norm = torch.linalg.vector_norm(unit_direction)
    if not torch.isclose(norm, torch.tensor(1.0, device=norm.device, dtype=norm.dtype), atol=1e-5):
        raise ValueError("projection direction must be unit norm")
    mean = collapse_streams(streams)
    coefficients = torch.einsum("bsd,d->bs", mean, unit_direction)
    component = coefficients[..., None] * unit_direction
    return streams - component[:, :, None, :]


def select_prompt_vectors(streams: torch.Tensor, token_indices: Sequence[int]) -> torch.Tensor:
    collapsed = collapse_streams(streams)
    if len(token_indices) != collapsed.shape[0]:
        raise ValueError("one token index is required per batch row")
    index = torch.tensor(token_indices, device=collapsed.device, dtype=torch.long)
    if torch.any(index < 0) or torch.any(index >= collapsed.shape[1]):
        raise IndexError("token index lies outside sequence")
    batch = torch.arange(collapsed.shape[0], device=collapsed.device)
    return collapsed[batch, index].detach().float().cpu()
