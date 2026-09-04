from __future__ import annotations

import numpy as np
import pytest
from src.glm53_user_eval.v8.statistics import (
    causal_delta,
    control_rank,
    empirical_p,
    four_group_bootstrap,
    fraction_removed,
    interaction,
)


def test_known_interaction() -> None:
    assert interaction({"famous_ai": 1, "unknown_ai": 2, "famous_nonai": 4, "genpop": 3}) == -2


def test_causal_delta() -> None:
    assert causal_delta(-1.0, -0.2) == pytest.approx(0.8)


def test_fraction_removed() -> None:
    assert fraction_removed(-1.0, -0.4) == pytest.approx(0.6)


def test_fraction_zero_fails() -> None:
    with pytest.raises(ValueError):
        fraction_removed(0, 1)


def test_empirical_p_beats_twenty() -> None:
    assert empirical_p(1.0, np.zeros(20)) == pytest.approx(1 / 21)


def test_control_rank() -> None:
    assert control_rank(0.5, np.array([0.1, 0.2, 0.6])) == 2


def test_bootstrap_reproducible() -> None:
    values = {
        "famous_ai": np.zeros((4, 3)),
        "unknown_ai": np.ones((4, 3)),
        "famous_nonai": np.ones((4, 3)),
        "genpop": np.zeros((4, 3)),
    }
    left = four_group_bootstrap(values, reps=50, seed=2)
    right = four_group_bootstrap(values, reps=50, seed=2)
    assert np.array_equal(left[2], right[2]) and left[0] == -2
