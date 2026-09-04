from __future__ import annotations

import numpy as np
import pytest
from src.glm53_user_eval.v8.direction_geometry import cosine, residualize
from src.glm53_user_eval.v8.probes import fit_probe, paired_mean_direction, raw_logit, select_layer


def test_paired_mean_sign() -> None:
    direction = paired_mean_direction(np.ones((3, 2)), np.zeros((3, 2)))
    assert np.all(direction > 0)


def test_degenerate_direction_rejected() -> None:
    with pytest.raises(ValueError):
        paired_mean_direction(np.ones((2, 2)), np.ones((2, 2)))


def test_raw_probe_mapping() -> None:
    x = np.array([[0.0, 0], [0.1, 0], [1.0, 1], [1.1, 1]])
    y = np.array([0, 0, 1, 1])
    artifact, model = fit_probe(x, y, 1.0)
    assert np.allclose(
        raw_logit(x, artifact),
        model.decision_function((x - artifact.scaler_mean) / artifact.scaler_scale),
    )


def test_layer_selection_lexicographic() -> None:
    rows = [
        {
            "layer": 3,
            "validation_auroc": 0.8,
            "realism_auroc": 0.7,
            "hard_negative_fpr_at_80_tpr": 0.2,
        },
        {
            "layer": 2,
            "validation_auroc": 0.8,
            "realism_auroc": 0.8,
            "hard_negative_fpr_at_80_tpr": 0.3,
        },
    ]
    assert select_layer(rows)["layer"] == 2


def test_cosine() -> None:
    assert cosine(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1)


def test_residualize() -> None:
    value = residualize(np.array([1.0, 2.0]), np.array([[1.0, 0.0]]))
    assert value[0] == pytest.approx(0)
