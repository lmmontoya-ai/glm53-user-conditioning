"""Linear audit models with a shared scoring contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


@dataclass(frozen=True)
class Model:
    name: str
    layer: int
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    val_auroc: float
    val_brier: float
    detail: dict[str, float | int | str]

    @property
    def raw_direction(self) -> np.ndarray:
        raw = self.weight / self.scale
        norm = np.linalg.norm(raw)
        return raw / norm if norm else raw


def sigmoid(score: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(score, -60, 60)))


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) != 2:
        raise ValueError("metrics require two classes")
    probability = sigmoid(score)
    fpr, tpr, _ = roc_curve(y, score)
    eligible = fpr[tpr >= 0.8]
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, probability)),
        "fpr_at_80_tpr": float(eligible.min()) if len(eligible) else 1.0,
    }


def threshold_balanced(y: np.ndarray, score: np.ndarray) -> float:
    best = (-1.0, 0.0)
    for threshold in np.concatenate(([-np.inf], np.unique(score), [np.inf])):
        predicted = score >= threshold
        value = 0.5 * (float(predicted[y == 1].mean()) + float((~predicted[y == 0]).mean()))
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def score(model: Model, features: np.ndarray) -> np.ndarray:
    return ((features[:, model.layer] - model.mean) / model.scale) @ model.weight + model.bias


def _standardize(
    train_x: np.ndarray, other_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale == 0] = 1.0
    return (train_x - mean) / scale, (other_x - mean) / scale, mean, scale


def fit_paired_mean_layers(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
) -> Model:
    candidates: list[Model] = []
    for layer in range(features.shape[1]):
        train_x, val_x, mean, scale = _standardize(features[train, layer], features[val, layer])
        weight = train_x[labels[train] == 1].mean(0) - train_x[labels[train] == 0].mean(0)
        bias = -0.5 * float(
            (train_x[labels[train] == 1].mean(0) + train_x[labels[train] == 0].mean(0)) @ weight
        )
        val_score = val_x @ weight + bias
        result = metrics(labels[val], val_score)
        candidates.append(
            Model(
                "paired_mean",
                layer,
                mean,
                scale,
                weight,
                bias,
                result["auroc"],
                result["brier"],
                {},
            )
        )
    return min(candidates, key=lambda item: (-item.val_auroc, item.val_brier, item.layer))


def fit_ridge_layers(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    c_grid: list[float],
) -> Model:
    candidates: list[Model] = []
    for layer in range(features.shape[1]):
        train_x, val_x, mean, scale = _standardize(features[train, layer], features[val, layer])
        for c in c_grid:
            classifier = LogisticRegression(
                C=c, penalty="l2", solver="liblinear", max_iter=2000, random_state=0
            )
            classifier.fit(train_x, labels[train])
            val_score = classifier.decision_function(val_x)
            result = metrics(labels[val], val_score)
            candidates.append(
                Model(
                    "ridge_logistic",
                    layer,
                    mean,
                    scale,
                    classifier.coef_[0].copy(),
                    float(classifier.intercept_[0]),
                    result["auroc"],
                    result["brier"],
                    {"C": c},
                )
            )
    return min(
        candidates,
        key=lambda item: (-item.val_auroc, item.val_brier, item.layer, float(item.detail["C"])),
    )


def fit_lda_layers(
    features: np.ndarray, labels: np.ndarray, train: np.ndarray, val: np.ndarray
) -> Model:
    candidates: list[Model] = []
    for layer in range(features.shape[1]):
        train_x, val_x, mean, scale = _standardize(features[train, layer], features[val, layer])
        centered = train_x - train_x.mean(0)
        shrinkage = float(ledoit_wolf_shrinkage(train_x, assume_centered=False))
        sample_count, feature_count = centered.shape
        variance = float(np.sum(centered * centered) / (sample_count * feature_count))
        ridge = max(shrinkage * variance, 1e-12)
        beta = (1.0 - shrinkage) / sample_count
        class0 = train_x[labels[train] == 0].mean(0)
        class1 = train_x[labels[train] == 1].mean(0)
        difference = class1 - class0
        gram = np.eye(sample_count) + (beta / ridge) * (centered @ centered.T)
        correction = centered.T @ np.linalg.solve(gram, centered @ difference)
        weight = difference / ridge - (beta / (ridge * ridge)) * correction
        bias = -0.5 * float((class1 + class0) @ weight)
        val_score = val_x @ weight + bias
        result = metrics(labels[val], val_score)
        candidates.append(
            Model(
                "shrinkage_lda",
                layer,
                mean,
                scale,
                weight,
                bias,
                result["auroc"],
                result["brier"],
                {},
            )
        )
    return min(candidates, key=lambda item: (-item.val_auroc, item.val_brier, item.layer))


def fit_pca_ridge_layers(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    c_grid: list[float],
    dimensions: list[int],
) -> Model:
    candidates: list[Model] = []
    max_components = min(int(train.sum()) - 1, features.shape[2])
    for layer in range(features.shape[1]):
        train_x, val_x, mean, scale = _standardize(features[train, layer], features[val, layer])
        eligible_dimensions = [value for value in dimensions if value <= max_components]
        pca = PCA(n_components=max(eligible_dimensions), svd_solver="randomized", random_state=0)
        train_full = pca.fit_transform(train_x)
        val_full = pca.transform(val_x)
        for dimension in eligible_dimensions:
            train_pca = train_full[:, :dimension]
            val_pca = val_full[:, :dimension]
            for c in c_grid:
                classifier = LogisticRegression(
                    C=c, penalty="l2", solver="liblinear", max_iter=2000, random_state=0
                )
                classifier.fit(train_pca, labels[train])
                val_score = classifier.decision_function(val_pca)
                result = metrics(labels[val], val_score)
                raw_standardized_weight = pca.components_[:dimension].T @ classifier.coef_[0]
                adjusted_bias = float(
                    classifier.intercept_[0] - pca.mean_ @ raw_standardized_weight
                )
                candidates.append(
                    Model(
                        "pca_ridge",
                        layer,
                        mean,
                        scale,
                        raw_standardized_weight,
                        adjusted_bias,
                        result["auroc"],
                        result["brier"],
                        {"C": c, "dimensions": dimension},
                    )
                )
    return min(
        candidates,
        key=lambda item: (
            -item.val_auroc,
            item.val_brier,
            item.layer,
            int(item.detail["dimensions"]),
            float(item.detail["C"]),
        ),
    )
