"""Batched mHC intervention algebra and null directions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from src.glm53_user_eval.mhc import collapse_streams, replace_streams, streams_from_output


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("direction must have finite nonzero norm")
    return value / norm


def gaussian_controls(candidate: np.ndarray, count: int, seed: int) -> np.ndarray:
    unit = normalize(candidate)
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < count:
        draw = rng.normal(size=unit.shape)
        draw = draw - float(draw @ unit) * unit
        values.append(normalize(draw))
    return np.asarray(values)


def signflip_nulls(paired_differences: np.ndarray, count: int, seed: int) -> np.ndarray:
    values = np.asarray(paired_differences, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("paired differences must be [pairs, hidden]")
    rng = np.random.default_rng(seed)
    return np.asarray(
        [
            normalize((rng.choice([-1.0, 1.0], size=len(values))[:, None] * values).mean(0))
            for _ in range(count)
        ]
    )


def batched_add_hook(per_example_delta: torch.Tensor, attention_mask: torch.Tensor):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        streams = streams_from_output(output)
        if per_example_delta.shape != (streams.shape[0], streams.shape[-1]):
            raise ValueError("delta batch does not align")
        if attention_mask.shape != streams.shape[:2]:
            raise ValueError("attention mask does not align")
        mask = attention_mask.to(streams.device, streams.dtype)[:, :, None, None]
        delta = per_example_delta.to(streams.device, streams.dtype)[:, None, None, :]
        return replace_streams(output, streams + mask * delta)

    return hook


def project_out_hook(unit_direction: torch.Tensor, attention_mask: torch.Tensor):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        streams = streams_from_output(output)
        unit = unit_direction.to(streams.device, streams.dtype)
        if not torch.isclose(
            torch.linalg.vector_norm(unit.float()), torch.tensor(1.0, device=unit.device), atol=2e-3
        ):
            raise ValueError("projection direction must be unit norm")
        mean = collapse_streams(streams)
        coefficient = torch.einsum("bsh,h->bs", mean, unit)
        component = coefficient[..., None] * unit
        mask = attention_mask.to(streams.device, streams.dtype)[:, :, None, None]
        return replace_streams(output, streams - mask * component[:, :, None, :])

    return hook


def natural_gap_delta(unit: np.ndarray, gap: float, alpha: float, layer_count: int) -> np.ndarray:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    return normalize(unit) * (float(alpha) * float(gap) / layer_count)
