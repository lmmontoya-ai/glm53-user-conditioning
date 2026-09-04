from __future__ import annotations

import numpy as np
import pytest
import torch
from src.glm53_user_eval.v11.extraction import feature_partition
from src.glm53_user_eval.v11.runtime import pool_layer_streams


def token_record() -> dict:
    return {
        "shared_suffix_token_indices": [3, 4],
        "prompt_final_index": 5,
        "masked_prompt_token_indices": [0, 1, 3, 4, 5],
        "decisive_token_indices": [2],
    }


def test_four_stream_collapse_and_pooled_views_are_exact() -> None:
    base = torch.arange(6 * 4096, dtype=torch.float32).reshape(1, 6, 1, 4096)
    offsets = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 1, 4, 1)
    streams = base + offsets
    output = pool_layer_streams(streams, token_record())
    collapsed = streams.mean(dim=2)[0]
    assert torch.equal(output["prompt_final"], collapsed[5])
    assert torch.equal(output["shared_task_suffix_mean"], collapsed[[3, 4]].mean(0))
    assert torch.equal(output["decisive_fact_token_mean"], collapsed[2])
    assert torch.equal(output["masked_prompt_mean"], collapsed[[0, 1, 3, 4, 5]].mean(0))


def test_empty_decisive_span_produces_only_the_expected_nan_control() -> None:
    record = token_record() | {"decisive_token_indices": []}
    output = pool_layer_streams(torch.zeros(1, 6, 4, 4096), record)
    assert torch.isnan(output["decisive_fact_token_mean"]).all()
    assert torch.isfinite(output["shared_task_suffix_mean"]).all()


def test_pooling_rejects_wrong_mhc_shape() -> None:
    with pytest.raises(ValueError, match="mHC shape"):
        pool_layer_streams(torch.zeros(1, 6, 3, 4096), token_record())


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"split": "train"}, "development"),
        ({"split": "validation"}, "development"),
        ({"split": "development_counterfactual"}, "development"),
        ({"split": "ordinary_test"}, "final"),
        ({"split": "final_counterfactual"}, "final"),
        ({"split": "neutral_controls", "control_partition": "development"}, "development"),
        ({"split": "neutral_controls", "control_partition": "final"}, "final"),
        ({"split": "factorial_calibration"}, "calibration"),
    ],
)
def test_feature_partition_keeps_selection_and_final_rows_separate(row, expected) -> None:
    assert feature_partition(row) == expected


def test_feature_partition_fails_closed_on_unassigned_neutral() -> None:
    with pytest.raises(ValueError, match="lacks a valid"):
        feature_partition({"split": "neutral_controls", "control_partition": None})


def test_stream_pooling_does_not_depend_on_numpy_state() -> None:
    np.random.seed(123)
    first = pool_layer_streams(torch.ones(1, 6, 4, 4096), token_record())
    np.random.seed(999)
    second = pool_layer_streams(torch.ones(1, 6, 4, 4096), token_record())
    assert all(torch.equal(first[key], second[key]) for key in first)
