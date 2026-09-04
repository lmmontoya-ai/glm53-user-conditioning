from __future__ import annotations

import pytest
import torch
from src.glm53_user_eval.v8.token_positions import final_nonpadding_index, one_hot_final_mask


@pytest.mark.parametrize(
    "mask,expected",
    [([[1, 1, 0], [1, 0, 0]], [1, 0]), ([[0, 1, 1], [0, 0, 1]], [2, 2]), ([[1], [1]], [0, 0])],
)
def test_final_indices(mask: list[list[int]], expected: list[int]) -> None:
    assert final_nonpadding_index(torch.tensor(mask)).tolist() == expected


def test_all_padding_fails() -> None:
    with pytest.raises(ValueError):
        final_nonpadding_index(torch.zeros(1, 3))


def test_wrong_dimension_fails() -> None:
    with pytest.raises(ValueError):
        final_nonpadding_index(torch.ones(3))


def test_one_hot_final_mask() -> None:
    result = one_hot_final_mask(torch.tensor([[0, 1, 1], [1, 0, 0]]))
    assert result.tolist() == [[0, 0, 1], [1, 0, 0]]
