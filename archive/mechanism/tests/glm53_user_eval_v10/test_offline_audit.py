from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from src.glm53_user_eval.v10.analysis import atomic_json
from src.glm53_user_eval.v10.data import (
    deterministic_task_folds,
    metadata_matrix,
    validate_core_grid,
)
from src.glm53_user_eval.v10.models import Model, fit_lda_layers, metrics, score


def test_task_folds_are_deterministic_disjoint_and_complete() -> None:
    folds = deterministic_task_folds()
    assert folds == deterministic_task_folds()
    assert len(folds) == 8
    assert all(len(fold) == 2 for fold in folds)
    assert sorted(value for fold in folds for value in fold) == list(range(1, 17))


def test_core_grid_validation() -> None:
    rows = []
    for family in range(8):
        for task in range(1, 17):
            for label in (0, 1):
                rows.append(
                    {
                        "slice_id": "core_context_pairs",
                        "family_id": f"f{family}",
                        "task_number": task,
                        "label": label,
                    }
                )
    validate_core_grid(rows)


def test_metadata_matrix_has_no_label_or_text() -> None:
    rows = [
        {
            "pair_id": "p",
            "prompt_tokens": 12,
            "valid_token_count": 12,
            "retained_token_count": 9,
            "masked_token_count": 3,
            "cue_span_count": 1,
            "masked_span_count": 1,
            "prompt_role": "user_prefix",
            "cue_location": "user",
            "label": 1,
        },
        {
            "pair_id": "p",
            "prompt_tokens": 10,
            "valid_token_count": 10,
            "retained_token_count": 7,
            "masked_token_count": 3,
            "cue_span_count": 1,
            "masked_span_count": 1,
            "prompt_role": "user_prefix",
            "cue_location": "user",
            "label": 0,
        },
    ]
    matrix, names = metadata_matrix(rows, np.asarray([0, 1]))
    assert matrix.shape == (2, 13)
    assert not any("label" in name or "prompt_text" in name for name in names)
    assert matrix[:, names.index("within_pair_prompt_length_gap")].tolist() == [2.0, 2.0]


def test_metrics_and_raw_direction() -> None:
    y = np.asarray([0, 0, 1, 1])
    values = np.asarray([-2.0, -1.0, 1.0, 2.0])
    assert metrics(y, values)["auroc"] == 1.0
    model = Model(
        "toy", 0, np.zeros(2), np.asarray([2.0, 1.0]), np.asarray([2.0, 0.0]), 0.0, 1.0, 0.0, {}
    )
    assert np.allclose(model.raw_direction, [1.0, 0.0])
    features = np.asarray([[[2.0, 0.0]], [[-2.0, 0.0]]])
    assert score(model, features).tolist() == [2.0, -2.0]


def test_atomic_json(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"
    atomic_json(destination, {"value": 3})
    assert json.loads(destination.read_text()) == {"value": 3}
    assert not destination.with_suffix(".json.tmp").exists()


def test_dual_shrinkage_lda_returns_finite_direction() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(16, 2, 20)).astype(np.float32)
    labels = np.asarray([0, 1] * 8)
    train = np.zeros(16, dtype=bool)
    val = np.zeros(16, dtype=bool)
    train[:12] = True
    val[12:] = True
    model = fit_lda_layers(features, labels, train, val)
    assert model.layer in (0, 1)
    assert np.isfinite(model.weight).all()
    assert np.isclose(np.linalg.norm(model.raw_direction), 1.0)


def test_independent_verifier_does_not_import_primary_modules() -> None:
    source = Path("src/glm53_user_eval/v10/verification.py").read_text(encoding="utf-8")
    assert "from .analysis" not in source
    assert "from .models" not in source
    assert "from .data" not in source
