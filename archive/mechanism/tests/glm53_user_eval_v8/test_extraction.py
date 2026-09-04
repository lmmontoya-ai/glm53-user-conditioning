from __future__ import annotations

import pytest
import torch
from src.glm53_user_eval.v8.extraction import (
    PromptFeatureCollector,
    compare_batched,
    validate_feature_matrix,
)


def test_collector_selects_rows() -> None:
    collector = PromptFeatureCollector(torch.tensor([1, 0]))
    streams = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    collector.hook(2)(None, None, streams)
    assert collector.features[2].shape == (2, 5)


def test_feature_shape() -> None:
    validate_feature_matrix(torch.ones(3, 4096), 3)


def test_bad_feature_shape() -> None:
    with pytest.raises(ValueError):
        validate_feature_matrix(torch.ones(3, 4), 3)


def test_nan_feature_rejected() -> None:
    value = torch.ones(1, 4096)
    value[0, 0] = torch.nan
    with pytest.raises(ValueError):
        validate_feature_matrix(value, 1)


def test_batch_equivalence() -> None:
    assert compare_batched(torch.ones(2, 3), torch.ones(2, 3), 1e-5) == 0


def test_batch_difference_rejected() -> None:
    with pytest.raises(ValueError):
        compare_batched(torch.zeros(1), torch.ones(1), 0.1)
