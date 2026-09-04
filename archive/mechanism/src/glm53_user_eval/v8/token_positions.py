"""Padding-safe token-position helpers."""

from __future__ import annotations

import torch


def final_nonpadding_index(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError("attention mask must be [B,S]")
    active = mask.to(torch.bool)
    if not torch.all(active.sum(dim=1) > 0):
        raise ValueError("every row needs a non-padding token")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~active, -1).max(dim=1).values


def one_hot_final_mask(mask: torch.Tensor) -> torch.Tensor:
    indices = final_nonpadding_index(mask)
    result = torch.zeros_like(mask)
    result.scatter_(1, indices[:, None], 1)
    return result
