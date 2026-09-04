import numpy as np
import pytest

from src.glm53_user_eval.directions import (
    natural_gap_scale,
    paired_mean_direction,
    random_directions,
    raw_logistic_weights,
    residualize_direction,
)


def test_paired_mean_direction_orientation() -> None:
    positive = np.array([[2.0, 0.0], [4.0, 0.0]])
    negative = np.array([[1.0, 0.0], [1.0, 0.0]])
    raw, unit = paired_mean_direction(positive, negative)
    assert np.allclose(raw, [2.0, 0.0])
    assert np.allclose(unit, [1.0, 0.0])


def test_zero_direction_fails() -> None:
    with pytest.raises(ValueError, match="zero"):
        paired_mean_direction(np.ones((2, 3)), np.ones((2, 3)))


def test_raw_probe_weight_mapping() -> None:
    assert np.allclose(raw_logistic_weights(np.array([2.0, 3.0]), np.array([2.0, 1.5])), [1.0, 2.0])


def test_random_directions_are_deterministic_and_unit() -> None:
    first = random_directions(8, 4, seed=7)
    second = random_directions(8, 4, seed=7)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_random_directions_are_orthogonalized() -> None:
    target = np.array([1.0, 0.0, 0.0, 0.0])
    randoms = random_directions(4, 5, seed=8, orthogonal_to=target)
    assert np.allclose(randoms @ target, 0.0, atol=1e-12)


def test_residualization_removes_span() -> None:
    result = residualize_direction(np.array([1.0, 1.0, 1.0]), np.array([[1.0, 0.0, 0.0]]))
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_natural_gap_scaling_divides_across_layers() -> None:
    assert np.allclose(natural_gap_scale(np.array([3.0, 6.0, 9.0]), -1.0, 3), [-1.0, -2.0, -3.0])
