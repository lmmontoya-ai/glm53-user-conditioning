import pytest
import torch

from src.glm53_user_eval.mhc import (
    add_collapsed_delta,
    collapse_streams,
    project_out_collapsed,
    replace_streams,
    streams_from_output,
)


def _streams() -> torch.Tensor:
    return torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)


def test_collapse_is_stream_mean() -> None:
    streams = _streams()
    assert torch.equal(collapse_streams(streams), streams.mean(dim=2))


def test_collapse_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError):
        collapse_streams(torch.zeros(2, 3, 4))


def test_tuple_metadata_is_preserved() -> None:
    streams = _streams()
    metadata = torch.tensor([7])
    replaced = replace_streams((streams, metadata), streams + 1)
    assert torch.equal(replaced[1], metadata)


def test_stream_extractor_accepts_tuple() -> None:
    streams = _streams()
    assert streams_from_output((streams, "routing")) is streams


def test_equal_stream_addition_changes_mean_by_delta() -> None:
    streams = _streams()
    delta = torch.tensor([1.0, -2.0, 3.0])
    changed = add_collapsed_delta(streams, delta)
    observed = collapse_streams(changed) - collapse_streams(streams)
    assert torch.allclose(observed, delta.view(1, 1, -1))


def test_equal_stream_addition_preserves_relative_streams() -> None:
    streams = _streams()
    changed = add_collapsed_delta(streams, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(changed[:, :, 1] - changed[:, :, 0], streams[:, :, 1] - streams[:, :, 0])


def test_projection_removes_collapsed_component() -> None:
    streams = _streams()
    direction = torch.tensor([1.0, 0.0, 0.0])
    projected = project_out_collapsed(streams, direction)
    coefficients = collapse_streams(projected) @ direction
    assert torch.allclose(coefficients, torch.zeros_like(coefficients), atol=1e-6)


def test_projection_requires_unit_direction() -> None:
    with pytest.raises(ValueError, match="unit norm"):
        project_out_collapsed(_streams(), torch.tensor([2.0, 0.0, 0.0]))


def test_alpha_zero_is_exact() -> None:
    streams = _streams()
    changed = add_collapsed_delta(streams, torch.zeros(3))
    assert torch.equal(changed, streams)
