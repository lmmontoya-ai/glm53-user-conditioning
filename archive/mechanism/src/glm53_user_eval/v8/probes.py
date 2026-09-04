"""Grouped probe fitting, direction construction, and layer selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeArtifact:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficient_standardized: np.ndarray
    coefficient_raw: np.ndarray
    intercept: float
    c_value: float


def paired_mean_direction(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("paired classes must have equal [pairs, hidden] shape")
    direction = (positive.astype(np.float64) - negative.astype(np.float64)).mean(0)
    norm = float(np.linalg.norm(direction))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("paired mean direction is degenerate")
    return direction / norm


def fit_probe(
    train_x: np.ndarray, train_y: np.ndarray, c_value: float
) -> tuple[ProbeArtifact, LogisticRegression]:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=c_value, penalty="l2", max_iter=5000).fit(
        scaler.transform(train_x), train_y
    )
    raw = model.coef_[0] / scaler.scale_
    artifact = ProbeArtifact(
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        coefficient_standardized=model.coef_[0].copy(),
        coefficient_raw=raw,
        intercept=float(model.intercept_[0]),
        c_value=float(c_value),
    )
    return artifact, model


def raw_logit(features: np.ndarray, artifact: ProbeArtifact) -> np.ndarray:
    standardized = (features - artifact.scaler_mean) / artifact.scaler_scale
    return standardized @ artifact.coefficient_standardized + artifact.intercept


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(y, scores)
    eligible = fpr[tpr >= 0.8]
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "fpr_at_80_tpr": float(eligible.min()) if len(eligible) else 1.0,
    }


def select_layer(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("no layer metrics")
    return min(
        rows,
        key=lambda row: (
            -row["validation_auroc"],
            -row["realism_auroc"],
            row["hard_negative_fpr_at_80_tpr"],
            row["layer"],
        ),
    )


def grouped_permutation(labels: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    mapping = dict(
        zip(unique, rng.permutation([labels[groups == group][0] for group in unique]), strict=True)
    )
    return np.asarray([mapping[group] for group in groups])
