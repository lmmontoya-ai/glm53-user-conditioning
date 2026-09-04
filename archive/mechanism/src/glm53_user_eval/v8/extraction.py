"""Prompt feature collection and feature-shard validation."""

from __future__ import annotations

from typing import Any

import torch
from src.glm53_user_eval.mhc import streams_from_output


class PromptFeatureCollector:
    def __init__(self, token_indices: torch.Tensor):
        self.token_indices = token_indices
        self.features: dict[int, torch.Tensor] = {}

    def hook(self, layer_index: int):
        def inner(_module: Any, _inputs: Any, output: Any) -> Any:
            streams = streams_from_output(output)
            collapsed = streams.mean(dim=2)
            batch = torch.arange(collapsed.shape[0], device=collapsed.device)
            index = self.token_indices.to(collapsed.device)
            self.features[layer_index] = collapsed[batch, index].detach().float().cpu()
            return output

        return inner


def validate_feature_matrix(value: torch.Tensor, rows: int, hidden_size: int = 4096) -> None:
    if value.shape != (rows, hidden_size):
        raise ValueError(f"feature matrix has shape {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError("feature matrix contains NaN or Inf")


def compare_batched(single: torch.Tensor, batched: torch.Tensor, tolerance: float) -> float:
    if single.shape != batched.shape:
        raise ValueError("batch equivalence shapes differ")
    error = float((single.float() - batched.float()).abs().max().item())
    if error > tolerance:
        raise ValueError(f"batch equivalence error {error} exceeds {tolerance}")
    return error
