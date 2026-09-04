from __future__ import annotations

import numpy as np
import pytest
import torch
from src.glm53_user_eval.v8.interventions import (
    batched_add_hook,
    gaussian_controls,
    natural_gap_delta,
    project_out_hook,
    signflip_nulls,
)


def test_batched_delta_changes_mean() -> None:
    streams = torch.zeros(2, 3, 4, 5)
    delta = torch.ones(2, 5)
    changed = batched_add_hook(delta, torch.tensor([[1, 1, 0], [1, 0, 0]]))(None, None, streams)
    assert torch.all(changed[0, :2] == 1) and torch.all(changed[0, 2] == 0)


def test_batched_delta_preserves_stream_differences() -> None:
    streams = torch.randn(2, 3, 4, 5)
    changed = batched_add_hook(torch.ones(2, 5), torch.ones(2, 3))(None, None, streams)
    assert torch.allclose(changed[:, :, 1] - changed[:, :, 0], streams[:, :, 1] - streams[:, :, 0])


def test_bad_delta_shape() -> None:
    with pytest.raises(ValueError):
        batched_add_hook(torch.ones(1, 5), torch.ones(2, 3))(None, None, torch.zeros(2, 3, 4, 5))


def test_project_out_zeroes_direction() -> None:
    streams = torch.randn(2, 3, 4, 5)
    direction = torch.tensor([1.0, 0, 0, 0, 0])
    changed = project_out_hook(direction, torch.ones(2, 3))(None, None, streams)
    assert torch.max(torch.abs(changed.mean(2)[..., 0])) < 1e-6


def test_natural_gap_sum() -> None:
    delta = natural_gap_delta(np.array([1.0, 0.0]), 3.0, -1.0, 3)
    assert np.allclose(delta, [-1.0, 0.0])


def test_gaussian_controls_orthogonal() -> None:
    controls = gaussian_controls(np.array([1.0, 0, 0, 0]), 10, 4)
    assert controls.shape == (10, 4)
    assert np.max(np.abs(controls[:, 0])) < 1e-12


def test_signflip_nulls_reproducible() -> None:
    values = np.eye(4)
    assert np.allclose(signflip_nulls(values, 3, 1), signflip_nulls(values, 3, 1))
