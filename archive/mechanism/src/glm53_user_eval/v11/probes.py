"""Robust source readouts for the latent v11 eval/deploy instrument."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

DEFAULT_C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
MAX_PREPARED_STANDARDIZED_BYTES = 512 * 1024**2
PERMUTATION_BINDING_KEYS = {
    "config_sha256",
    "feature_manifest_sha256",
    "readout_lock_sha256",
}
FACTORIAL_COLUMNS = (
    "intercept",
    "real_external_recipient",
    "operational_consequence",
    "model_scoring",
    "comparison_among_models",
    "formal_register",
)
FACTORIAL_DECISIVE_COLUMNS = FACTORIAL_COLUMNS[1:5]


@dataclass(frozen=True)
class FrozenLinear:
    name: str
    layer: int
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    threshold_80_tpr: float

    @property
    def raw_direction(self) -> np.ndarray:
        value = self.weight / self.scale
        norm = float(np.linalg.norm(value))
        if norm == 0:
            raise ValueError(f"{self.name} direction has zero norm")
        return value / norm


@dataclass(frozen=True)
class DevelopmentFit:
    logistic: FrozenLinear
    paired_mean: FrozenLinear
    selected_c: float
    objective: float
    report: dict[str, Any]


@dataclass(frozen=True)
class _PreparedLayer:
    mean: np.ndarray
    scale: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    development_counterfactual: np.ndarray
    neutral_controls: np.ndarray


@dataclass(frozen=True)
class _PreparedDevelopment:
    layers: tuple[_PreparedLayer, ...]
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    development_counterfactual_indices: np.ndarray
    neutral_control_indices: np.ndarray
    pair_indices: tuple[tuple[int, int], ...]
    standardized_bytes: int

    def permuted_labels(self, *, seed: int) -> np.ndarray:
        labels = self.labels.copy()
        rng = np.random.default_rng(seed)
        for first, second in self.pair_indices:
            if int(rng.integers(0, 2)):
                labels[first] = 1 - labels[first]
                labels[second] = 1 - labels[second]
        return labels


def load_partition(
    feature_root: Path,
    partition: str,
    *,
    view: str = "shared_task_suffix_mean",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads((feature_root / "feature_manifest.json").read_text(encoding="utf-8"))
    record = manifest["partitions"][partition]
    feature_path = feature_root / record["features"]
    metadata_path = feature_root / record["metadata"]
    if hashlib.sha256(feature_path.read_bytes()).hexdigest() != record["features_sha256"]:
        raise ValueError(f"{partition} feature hash differs")
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != record["metadata_sha256"]:
        raise ValueError(f"{partition} metadata hash differs")
    with np.load(feature_path) as archive:
        features = archive[view].astype(np.float32)
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if features.shape != (len(metadata), 45, 4096):
        raise ValueError(f"{partition} feature shape differs: {features.shape}")
    return features, metadata


def _standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def _scores(model: FrozenLinear, features: np.ndarray) -> np.ndarray:
    return ((features[:, model.layer] - model.mean) / model.scale) @ model.weight + model.bias


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("binary metrics require both labels")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= 0.80)
    index = int(eligible[0]) if len(eligible) else len(thresholds) - 1
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "fpr_at_80_tpr": float(fpr[index]),
        "threshold_at_80_tpr": float(thresholds[index]),
        "score_gap": float(scores[labels == 1].mean() - scores[labels == 0].mean()),
    }


def _fpr(scores: np.ndarray, threshold: float) -> float:
    if not len(scores):
        raise ValueError("neutral FPR requires at least one row")
    return float(np.mean(scores >= threshold))


def _paired_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    layer: int,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold_features: np.ndarray,
    threshold_labels: np.ndarray,
) -> FrozenLinear:
    standardized = (train_x - mean) / scale
    class_zero = standardized[train_y == 0].mean(axis=0)
    class_one = standardized[train_y == 1].mean(axis=0)
    weight = class_one - class_zero
    bias = -0.5 * float((class_one + class_zero) @ weight)
    temporary = FrozenLinear("paired_mean", layer, mean, scale, weight, bias, 0.0)
    threshold = _binary_metrics(threshold_labels, _scores(temporary, threshold_features))[
        "threshold_at_80_tpr"
    ]
    return FrozenLinear("paired_mean", layer, mean, scale, weight, bias, threshold)


def _logistic_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    layer: int,
    c_value: float,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold_features: np.ndarray,
    threshold_labels: np.ndarray,
) -> FrozenLinear:
    classifier = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=0,
    ).fit((train_x - mean) / scale, train_y)
    temporary = FrozenLinear(
        "ridge_logistic",
        layer,
        mean,
        scale,
        classifier.coef_[0].astype(np.float32),
        float(classifier.intercept_[0]),
        0.0,
    )
    threshold = _binary_metrics(threshold_labels, _scores(temporary, threshold_features))[
        "threshold_at_80_tpr"
    ]
    return FrozenLinear(
        temporary.name,
        layer,
        mean,
        scale,
        temporary.weight,
        temporary.bias,
        threshold,
    )


def _prepare_development(
    features: np.ndarray,
    metadata: list[dict[str, Any]],
) -> _PreparedDevelopment:
    """Precompute every label-independent matrix used by the permutation null."""

    if features.shape != (len(metadata), 45, 4096):
        raise ValueError("development feature shape differs")
    splits = np.asarray([row["split"] for row in metadata])
    forbidden = set(splits.tolist()) - {
        "train",
        "validation",
        "development_counterfactual",
        "neutral_controls",
    }
    if forbidden:
        raise ValueError(f"development fitter received forbidden splits: {sorted(forbidden)}")
    labels = np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in metadata],
        dtype=np.int64,
    )
    indices = {
        split: np.flatnonzero(splits == split)
        for split in (
            "train",
            "validation",
            "development_counterfactual",
            "neutral_controls",
        )
    }
    if not all(
        set(labels[indices[split]].tolist()) == {0, 1}
        for split in ("train", "validation", "development_counterfactual")
    ) or not np.all(labels[indices["neutral_controls"]] == -1):
        raise ValueError("development labels violate the binary/control contract")

    pairs: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        if row["label"] in {0, 1}:
            pairs.setdefault(str(row["pair_id"]), []).append(index)
    pair_indices: list[tuple[int, int]] = []
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or sorted(labels[pair].tolist()) != [0, 1]:
            raise ValueError(f"pair-preserving null found incomplete pair {pair_id}")
        pair_indices.append((pair[0], pair[1]))

    expected_bytes = int(features.size * np.dtype(np.float32).itemsize)
    expected_bytes += 2 * 45 * 4096 * np.dtype(np.float32).itemsize
    if expected_bytes > MAX_PREPARED_STANDARDIZED_BYTES:
        raise MemoryError(
            "prepared development matrices exceed the fixed memory bound: "
            f"{expected_bytes} > {MAX_PREPARED_STANDARDIZED_BYTES} bytes"
        )

    layers: list[_PreparedLayer] = []
    standardized_bytes = 0
    for layer in range(45):
        mean, scale = _standardizer(features[indices["train"], layer])
        matrices = {
            split: np.ascontiguousarray(
                (features[split_indices, layer] - mean) / scale,
                dtype=np.float32,
            )
            for split, split_indices in indices.items()
        }
        for matrix in matrices.values():
            matrix.setflags(write=False)
            standardized_bytes += matrix.nbytes
        mean.setflags(write=False)
        scale.setflags(write=False)
        standardized_bytes += mean.nbytes + scale.nbytes
        layers.append(
            _PreparedLayer(
                mean=mean,
                scale=scale,
                train=matrices["train"],
                validation=matrices["validation"],
                development_counterfactual=matrices["development_counterfactual"],
                neutral_controls=matrices["neutral_controls"],
            )
        )
    if standardized_bytes != expected_bytes:
        raise AssertionError(
            f"prepared development byte count differs: {standardized_bytes} != {expected_bytes}"
        )
    labels.setflags(write=False)
    for value in indices.values():
        value.setflags(write=False)
    return _PreparedDevelopment(
        layers=tuple(layers),
        labels=labels,
        train_indices=indices["train"],
        validation_indices=indices["validation"],
        development_counterfactual_indices=indices["development_counterfactual"],
        neutral_control_indices=indices["neutral_controls"],
        pair_indices=tuple(pair_indices),
        standardized_bytes=standardized_bytes,
    )


def _scores_prepared(model: FrozenLinear, standardized: np.ndarray) -> np.ndarray:
    return standardized @ model.weight + model.bias


def _paired_model_prepared(
    prepared: _PreparedLayer,
    train_y: np.ndarray,
    *,
    layer: int,
) -> FrozenLinear:
    class_zero = prepared.train[train_y == 0].mean(axis=0)
    class_one = prepared.train[train_y == 1].mean(axis=0)
    weight = class_one - class_zero
    bias = -0.5 * float((class_one + class_zero) @ weight)
    return FrozenLinear(
        "paired_mean",
        layer,
        prepared.mean,
        prepared.scale,
        weight,
        bias,
        0.0,
    )


def _logistic_model_prepared(
    prepared: _PreparedLayer,
    train_y: np.ndarray,
    *,
    layer: int,
    c_value: float,
) -> FrozenLinear:
    classifier = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=0,
    ).fit(prepared.train, train_y)
    return FrozenLinear(
        "ridge_logistic",
        layer,
        prepared.mean,
        prepared.scale,
        classifier.coef_[0].astype(np.float32),
        float(classifier.intercept_[0]),
        0.0,
    )


def _fit_source_development_prepared(
    prepared: _PreparedDevelopment,
    labels: np.ndarray,
    *,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    neutral_penalty: float = 0.5,
    neutral_target: float = 0.20,
    compact_report: bool = False,
) -> DevelopmentFit:
    """Run the unchanged full selection rule on cached standardized matrices."""

    if not c_grid:
        raise ValueError("development C grid is empty")
    if labels.shape != prepared.labels.shape:
        raise ValueError("label override shape differs")
    train_y = labels[prepared.train_indices]
    validation_y = labels[prepared.validation_indices]
    development_y = labels[prepared.development_counterfactual_indices]
    neutral_y = labels[prepared.neutral_control_indices]
    if not all(
        set(values.tolist()) == {0, 1}
        for values in (train_y, validation_y, development_y)
    ) or not np.all(neutral_y == -1):
        raise ValueError("development labels violate the binary/control contract")

    best: tuple[
        tuple[float, float, float, int, float],
        FrozenLinear,
        FrozenLinear,
        dict[str, Any],
    ] | None = None
    layer_reports: list[dict[str, Any]] = []
    paired_layers: list[
        tuple[
            float,
            int,
            _PreparedLayer,
            FrozenLinear,
            dict[str, float],
            dict[str, float],
            float,
        ]
    ] = []
    for layer, matrices in enumerate(prepared.layers):
        paired = _paired_model_prepared(
            matrices,
            train_y,
            layer=layer,
        )
        paired_val = _binary_metrics(
            validation_y,
            _scores_prepared(paired, matrices.validation),
        )
        paired_dev = _binary_metrics(
            development_y,
            _scores_prepared(paired, matrices.development_counterfactual),
        )
        paired = FrozenLinear(
            paired.name,
            paired.layer,
            paired.mean,
            paired.scale,
            paired.weight,
            paired.bias,
            paired_dev["threshold_at_80_tpr"],
        )
        paired_neutral_fpr = _fpr(
            _scores_prepared(paired, matrices.neutral_controls),
            paired.threshold_80_tpr,
        )
        paired_upper_bound = min(paired_val["auroc"], paired_dev["auroc"])
        paired_layers.append(
            (
                paired_upper_bound,
                layer,
                matrices,
                paired,
                paired_val,
                paired_dev,
                paired_neutral_fpr,
            )
        )

    if compact_report:
        paired_layers.sort(key=lambda item: (-item[0], item[1]))

    evaluated_candidate_count = 0
    pruned_layers = 0
    for (
        paired_upper_bound,
        layer,
        matrices,
        paired,
        paired_val,
        paired_dev,
        paired_neutral_fpr,
    ) in paired_layers:
        if (
            compact_report
            and best is not None
            and paired_upper_bound < -best[0][0]
        ):
            pruned_layers += 1
            continue
        for c_value in c_grid:
            logistic = _logistic_model_prepared(
                matrices,
                train_y,
                layer=layer,
                c_value=float(c_value),
            )
            logistic_val = _binary_metrics(
                validation_y,
                _scores_prepared(logistic, matrices.validation),
            )
            logistic_dev = _binary_metrics(
                development_y,
                _scores_prepared(logistic, matrices.development_counterfactual),
            )
            logistic = FrozenLinear(
                logistic.name,
                logistic.layer,
                logistic.mean,
                logistic.scale,
                logistic.weight,
                logistic.bias,
                logistic_dev["threshold_at_80_tpr"],
            )
            logistic_neutral_fpr = _fpr(
                _scores_prepared(logistic, matrices.neutral_controls),
                logistic.threshold_80_tpr,
            )
            robust_auc = min(
                paired_val["auroc"],
                paired_dev["auroc"],
                logistic_val["auroc"],
                logistic_dev["auroc"],
            )
            worst_neutral_fpr = max(paired_neutral_fpr, logistic_neutral_fpr)
            objective = robust_auc - neutral_penalty * max(
                0.0, worst_neutral_fpr - neutral_target
            )
            record = {
                "layer": layer,
                "C": float(c_value),
                "objective": objective,
                "robust_development_auroc": robust_auc,
                "worst_development_neutral_fpr": worst_neutral_fpr,
                "logistic": {
                    "validation": logistic_val,
                    "development_counterfactual": logistic_dev,
                    "development_neutral_fpr": logistic_neutral_fpr,
                },
                "paired_mean": {
                    "validation": paired_val,
                    "development_counterfactual": paired_dev,
                    "development_neutral_fpr": paired_neutral_fpr,
                },
            }
            key = (
                -objective,
                -logistic_val["auroc"],
                worst_neutral_fpr,
                layer,
                float(c_value),
            )
            candidate = (key, logistic, paired, record)
            if best is None or key < best[0]:
                best = candidate
            if not compact_report:
                layer_reports.append(record)
            evaluated_candidate_count += 1
    if best is None:
        raise AssertionError("prepared development search produced no candidate")
    _, logistic, paired, selected = best
    report_without_hash = {
        "schema_version": "glm53_v11_source_development_fit_v1",
        "primary_view": "shared_task_suffix_mean",
        "selection_used_final_rows": False,
        "selected_layer": logistic.layer,
        "selected_C": selected["C"],
        "objective": selected["objective"],
        "selected": selected,
        "candidate_count": len(prepared.layers) * len(c_grid),
        "candidates": layer_reports,
    }
    if compact_report:
        report_without_hash["exact_search_optimization"] = {
            "method": "paired_auroc_upper_bound_branch_and_bound_v1",
            "evaluated_candidate_count": evaluated_candidate_count,
            "pruned_layer_count": pruned_layers,
            "total_candidate_count": len(prepared.layers) * len(c_grid),
        }
    report = report_without_hash | {
        "readout_lock_sha256": hashlib.sha256(
            json.dumps(report_without_hash, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    }
    return DevelopmentFit(
        logistic,
        paired,
        float(selected["C"]),
        selected["objective"],
        report,
    )


def fit_source_development(
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    neutral_penalty: float = 0.5,
    neutral_target: float = 0.20,
    labels_override: np.ndarray | None = None,
) -> DevelopmentFit:
    if features.shape != (len(metadata), 45, 4096):
        raise ValueError("development feature shape differs")
    forbidden = {row["split"] for row in metadata} - {
        "train",
        "validation",
        "development_counterfactual",
        "neutral_controls",
    }
    if forbidden:
        raise ValueError(f"development fitter received forbidden splits: {sorted(forbidden)}")
    splits = np.asarray([row["split"] for row in metadata])
    labels = np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in metadata],
        dtype=np.int64,
    )
    if labels_override is not None:
        if labels_override.shape != labels.shape:
            raise ValueError("label override shape differs")
        labels = labels_override.astype(np.int64)
    train = splits == "train"
    validation = splits == "validation"
    devcf = splits == "development_counterfactual"
    neutral = splits == "neutral_controls"
    if not all(
        set(labels[mask].tolist()) == {0, 1} for mask in (train, validation, devcf)
    ) or not np.all(labels[neutral] == -1):
        raise ValueError("development labels violate the binary/control contract")
    candidates: list[tuple[tuple[float, float, float, int, float], Any]] = []
    layer_reports: list[dict[str, Any]] = []
    for layer in range(45):
        mean, scale = _standardizer(features[train, layer])
        paired = _paired_model(
            features[train, layer],
            labels[train],
            layer=layer,
            mean=mean,
            scale=scale,
            threshold_features=features[devcf],
            threshold_labels=labels[devcf],
        )
        paired_val = _binary_metrics(labels[validation], _scores(paired, features[validation]))
        paired_dev = _binary_metrics(labels[devcf], _scores(paired, features[devcf]))
        paired_neutral_fpr = _fpr(_scores(paired, features[neutral]), paired.threshold_80_tpr)
        for c_value in c_grid:
            logistic = _logistic_model(
                features[train, layer],
                labels[train],
                layer=layer,
                c_value=float(c_value),
                mean=mean,
                scale=scale,
                threshold_features=features[devcf],
                threshold_labels=labels[devcf],
            )
            logistic_val = _binary_metrics(
                labels[validation], _scores(logistic, features[validation])
            )
            logistic_dev = _binary_metrics(labels[devcf], _scores(logistic, features[devcf]))
            logistic_neutral_fpr = _fpr(
                _scores(logistic, features[neutral]), logistic.threshold_80_tpr
            )
            robust_auc = min(
                paired_val["auroc"],
                paired_dev["auroc"],
                logistic_val["auroc"],
                logistic_dev["auroc"],
            )
            worst_neutral_fpr = max(paired_neutral_fpr, logistic_neutral_fpr)
            objective = robust_auc - neutral_penalty * max(
                0.0, worst_neutral_fpr - neutral_target
            )
            record = {
                "layer": layer,
                "C": float(c_value),
                "objective": objective,
                "robust_development_auroc": robust_auc,
                "worst_development_neutral_fpr": worst_neutral_fpr,
                "logistic": {
                    "validation": logistic_val,
                    "development_counterfactual": logistic_dev,
                    "development_neutral_fpr": logistic_neutral_fpr,
                },
                "paired_mean": {
                    "validation": paired_val,
                    "development_counterfactual": paired_dev,
                    "development_neutral_fpr": paired_neutral_fpr,
                },
            }
            key = (
                -objective,
                -logistic_val["auroc"],
                worst_neutral_fpr,
                layer,
                float(c_value),
            )
            candidates.append((key, (logistic, paired, record)))
            layer_reports.append(record)
    _, (logistic, paired, selected) = min(candidates, key=lambda item: item[0])
    report_without_hash = {
        "schema_version": "glm53_v11_source_development_fit_v1",
        "primary_view": "shared_task_suffix_mean",
        "selection_used_final_rows": False,
        "selected_layer": logistic.layer,
        "selected_C": selected["C"],
        "objective": selected["objective"],
        "selected": selected,
        "candidate_count": len(layer_reports),
        "candidates": layer_reports,
    }
    report = report_without_hash | {
        "readout_lock_sha256": hashlib.sha256(
            json.dumps(report_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    }
    return DevelopmentFit(logistic, paired, float(selected["C"]), selected["objective"], report)


def save_development_fit(fit: DevelopmentFit, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    arrays_path = output_root / "source_readout_arrays.npz"
    temporary = arrays_path.with_suffix(arrays_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            logistic_mean=fit.logistic.mean,
            logistic_scale=fit.logistic.scale,
            logistic_weight=fit.logistic.weight,
            paired_mean=fit.paired_mean.mean,
            paired_scale=fit.paired_mean.scale,
            paired_weight=fit.paired_mean.weight,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, arrays_path)
    report = fit.report | {
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "logistic": {
            "bias": fit.logistic.bias,
            "threshold_80_tpr": fit.logistic.threshold_80_tpr,
        },
        "paired_mean": {
            "bias": fit.paired_mean.bias,
            "threshold_80_tpr": fit.paired_mean.threshold_80_tpr,
        },
    }
    report_path = output_root / "source_readout_lock.json"
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    return report


def load_development_fit(output_root: Path) -> DevelopmentFit:
    report = json.loads((output_root / "source_readout_lock.json").read_text(encoding="utf-8"))
    arrays_path = output_root / "source_readout_arrays.npz"
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != report["arrays_sha256"]:
        raise ValueError("source readout array hash differs")
    with np.load(arrays_path) as arrays:
        logistic = FrozenLinear(
            "ridge_logistic",
            int(report["selected_layer"]),
            arrays["logistic_mean"].copy(),
            arrays["logistic_scale"].copy(),
            arrays["logistic_weight"].copy(),
            float(report["logistic"]["bias"]),
            float(report["logistic"]["threshold_80_tpr"]),
        )
        paired = FrozenLinear(
            "paired_mean",
            int(report["selected_layer"]),
            arrays["paired_mean"].copy(),
            arrays["paired_scale"].copy(),
            arrays["paired_weight"].copy(),
            float(report["paired_mean"]["bias"]),
            float(report["paired_mean"]["threshold_80_tpr"]),
        )
    return DevelopmentFit(
        logistic,
        paired,
        float(report["selected_C"]),
        float(report["objective"]),
        report,
    )


def pair_preserving_labels(metadata: list[dict[str, Any]], *, seed: int) -> np.ndarray:
    labels = np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in metadata],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    pairs: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        if row["label"] in {0, 1}:
            pairs.setdefault(str(row["pair_id"]), []).append(index)
    for pair_id, indices in pairs.items():
        if len(indices) != 2 or sorted(labels[indices].tolist()) != [0, 1]:
            raise ValueError(f"pair-preserving null found incomplete pair {pair_id}")
        if int(rng.integers(0, 2)):
            labels[indices] = 1 - labels[indices]
    return labels


def run_full_selection_permutations(
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    observed_objective: float,
    reps: int = 1000,
    seed: int = 20260912,
    checkpoint_path: Path | None = None,
    checkpoint_binding: dict[str, str] | None = None,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    neutral_penalty: float = 0.5,
    neutral_target: float = 0.20,
    workers: int = 1,
    max_new_repetitions: int | None = None,
) -> dict[str, Any]:
    if reps <= 0:
        raise ValueError("permutation repetitions must be positive")
    if not 1 <= workers <= 64:
        raise ValueError("permutation workers must be between 1 and 64")
    if max_new_repetitions is not None and max_new_repetitions <= 0:
        raise ValueError("maximum new permutation repetitions must be positive")
    if max_new_repetitions is not None and checkpoint_path is None:
        raise ValueError("bounded permutation progress requires a checkpoint path")

    checkpoint_contract: dict[str, Any] | None = None
    checkpoint_contract_sha256: str | None = None
    checkpoint_manifest_path: Path | None = None
    checkpoint_parts: Path | None = None
    if checkpoint_path is not None:
        if checkpoint_binding is None or set(checkpoint_binding) != PERMUTATION_BINDING_KEYS:
            raise ValueError(
                "permutation checkpoint binding must contain exactly "
                f"{sorted(PERMUTATION_BINDING_KEYS)}"
            )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in checkpoint_binding.values()
        ):
            raise ValueError("permutation checkpoint bindings must be lowercase SHA-256 values")
        checkpoint_contract = {
            "schema_version": "glm53_v11_permutation_checkpoint_contract_v1",
            "reps": reps,
            "seed": seed,
            "observed_objective": float(observed_objective),
            "c_grid": [float(value) for value in c_grid],
            "neutral_penalty": float(neutral_penalty),
            "neutral_target": float(neutral_target),
            "workers_do_not_affect_results": True,
            "artifact_binding": dict(sorted(checkpoint_binding.items())),
        }
        checkpoint_contract_sha256 = hashlib.sha256(
            json.dumps(checkpoint_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        checkpoint_manifest_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".manifest.json"
        )
        checkpoint_parts = checkpoint_path.with_suffix(checkpoint_path.suffix + ".parts")
        manifest = checkpoint_contract | {"contract_sha256": checkpoint_contract_sha256}
        if checkpoint_manifest_path.exists():
            observed_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
            if observed_manifest != manifest:
                raise ValueError("permutation checkpoint contract differs from the current run")
        else:
            if checkpoint_path.exists() or checkpoint_parts.exists():
                raise ValueError("unbound permutation checkpoint artifacts already exist")
            checkpoint_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_manifest = checkpoint_manifest_path.with_suffix(
                checkpoint_manifest_path.suffix + ".tmp"
            )
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary_manifest, checkpoint_manifest_path)
        checkpoint_parts.mkdir(parents=True, exist_ok=True)

    completed: dict[int, dict[str, Any]] = {}
    if checkpoint_parts is not None:
        for part_path in sorted(checkpoint_parts.glob("rep-*.json")):
            record = json.loads(part_path.read_text(encoding="utf-8"))
            repetition = int(record["repetition"])
            if part_path.name != f"rep-{repetition:04d}.json":
                raise ValueError(f"unexpected permutation checkpoint part name: {part_path.name}")
            if not 0 <= repetition < reps:
                raise ValueError("permutation checkpoint repetition is out of range")
            if record.get("contract_sha256") != checkpoint_contract_sha256:
                raise ValueError("permutation checkpoint part has the wrong contract")
            if int(record.get("seed", -1)) != seed + repetition:
                raise ValueError("permutation checkpoint part has the wrong seed")
            if repetition in completed:
                raise ValueError("duplicate permutation checkpoint repetition")
            completed[repetition] = record
    prepared: _PreparedDevelopment | None = None

    def fit_repetition(repetition: int) -> dict[str, Any]:
        if prepared is None:
            labels = pair_preserving_labels(metadata, seed=seed + repetition)
            fit = fit_source_development(
                features,
                metadata,
                labels_override=labels,
                c_grid=c_grid,
                neutral_penalty=neutral_penalty,
                neutral_target=neutral_target,
            )
        else:
            labels = prepared.permuted_labels(seed=seed + repetition)
            fit = _fit_source_development_prepared(
                prepared,
                labels,
                c_grid=c_grid,
                neutral_penalty=neutral_penalty,
                neutral_target=neutral_target,
                compact_report=True,
            )
        result = {
            "repetition": repetition,
            "seed": seed + repetition,
            "selected_layer": fit.logistic.layer,
            "selected_C": fit.selected_c,
            "objective": fit.objective,
        }
        search = fit.report.get("exact_search_optimization")
        if search is not None:
            result["evaluated_candidate_count"] = int(
                search["evaluated_candidate_count"]
            )
            result["pruned_layer_count"] = int(search["pruned_layer_count"])
        if checkpoint_contract_sha256 is not None:
            result["contract_sha256"] = checkpoint_contract_sha256
        return result

    def checkpoint_record(record: dict[str, Any]) -> None:
        repetition = int(record["repetition"])
        completed[repetition] = record
        if checkpoint_parts is not None:
            part_path = checkpoint_parts / f"rep-{repetition:04d}.json"
            temporary_part = part_path.with_suffix(part_path.suffix + ".tmp")
            with temporary_part.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_part, part_path)

    pending_all = [repetition for repetition in range(reps) if repetition not in completed]
    completed_before = len(completed)
    pending = (
        pending_all[:max_new_repetitions]
        if max_new_repetitions is not None
        else pending_all
    )
    started_at = time.perf_counter()
    if pending and features.ndim == 3:
        prepared = _prepare_development(features, metadata)
    with threadpool_limits(limits=1):
        if workers == 1:
            for repetition in pending:
                checkpoint_record(fit_repetition(repetition))
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="v11-permutation",
            ) as pool:
                futures = {
                    pool.submit(fit_repetition, repetition): repetition
                    for repetition in pending
                }
                for future in as_completed(futures):
                    checkpoint_record(future.result())
    elapsed_seconds = time.perf_counter() - started_at
    complete = set(completed) == set(range(reps))
    fitted_rate = len(pending) / elapsed_seconds if pending and elapsed_seconds > 0 else None
    optimization = {
        "engine": (
            "prestandardized_full_selection_v1"
            if prepared is not None
            else "checkpoint_or_test_fallback"
        ),
        "standardized_bytes": prepared.standardized_bytes if prepared is not None else 0,
        "standardized_bytes_cap": MAX_PREPARED_STANDARDIZED_BYTES,
        "numerical_library_threads_per_worker": 1,
        "outer_worker_count": workers,
        "completed_repetitions_loaded": completed_before,
        "repetitions_fitted": len(pending),
        "elapsed_seconds": elapsed_seconds,
        "fitted_repetitions_per_second": fitted_rate,
        "projected_remaining_seconds_at_measured_rate": (
            (reps - len(completed)) / fitted_rate if fitted_rate else None
        ),
    }
    evaluated_candidates = [
        int(completed[index]["evaluated_candidate_count"])
        for index in sorted(completed)
        if "evaluated_candidate_count" in completed[index]
    ]
    optimization["exact_search_candidate_counts"] = (
        {
            "reported_repetitions": len(evaluated_candidates),
            "minimum": min(evaluated_candidates),
            "maximum": max(evaluated_candidates),
            "mean": float(np.mean(evaluated_candidates)),
            "full_grid": 45 * len(c_grid),
        }
        if evaluated_candidates
        else None
    )
    if not complete:
        return {
            "schema_version": "glm53_v11_full_selection_permutation_progress_v1",
            "complete": False,
            "reps": reps,
            "seed": seed,
            "workers": workers,
            "permutations_completed": len(completed),
            "permutations_remaining": reps - len(completed),
            "checkpoint_contract_sha256": checkpoint_contract_sha256,
            "checkpoint_manifest_sha256": (
                hashlib.sha256(checkpoint_manifest_path.read_bytes()).hexdigest()
                if checkpoint_manifest_path is not None
                else None
            ),
            "checkpoint_sha256": None,
            "optimization": optimization,
        }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index in range(reps):
                handle.write(
                    json.dumps(completed[index], sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint_path)
    objectives = np.asarray(
        [float(completed[index]["objective"]) for index in range(reps)], dtype=np.float64
    )
    exceedances = int(np.sum(objectives >= observed_objective))
    return {
        "schema_version": "glm53_v11_full_selection_permutation_v1",
        "complete": True,
        "reps": reps,
        "seed": seed,
        "workers": workers,
        "observed_objective": observed_objective,
        "null_mean": float(objectives.mean()),
        "null_median": float(np.median(objectives)),
        "null_99th_percentile": float(np.percentile(objectives, 99)),
        "exceedances": exceedances,
        "add_one_empirical_p": (1 + exceedances) / (1 + reps),
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "checkpoint_manifest_sha256": (
            hashlib.sha256(checkpoint_manifest_path.read_bytes()).hexdigest()
            if checkpoint_manifest_path is not None
            else None
        ),
        "checkpoint_sha256": (
            hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if checkpoint_path is not None
            else None
        ),
        "optimization": optimization,
    }


def leave_one_generator_score_gaps(
    fit: DevelopmentFit,
    development_features: np.ndarray,
    development_metadata: list[dict[str, Any]],
    final_features: np.ndarray,
    final_metadata: list[dict[str, Any]],
) -> dict[str, float]:
    layer = fit.logistic.layer
    train_indices = [
        index for index, row in enumerate(development_metadata) if row["split"] == "train"
    ]
    generators = sorted({development_metadata[index]["generator_family"] for index in train_indices})
    heldout_indices = [
        index for index, row in enumerate(final_metadata) if row["split"] == "final_counterfactual"
    ]
    heldout_y = np.asarray([int(final_metadata[index]["label"]) for index in heldout_indices])
    output: dict[str, float] = {}
    for generator in generators:
        retained = [
            index
            for index in train_indices
            if development_metadata[index]["generator_family"] != generator
        ]
        train_x = development_features[retained, layer]
        train_y = np.asarray([int(development_metadata[index]["label"]) for index in retained])
        mean, scale = _standardizer(train_x)
        classifier = LogisticRegression(
            C=fit.selected_c,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=0,
        ).fit((train_x - mean) / scale, train_y)
        score = classifier.decision_function(
            (final_features[heldout_indices, layer] - mean) / scale
        )
        output[generator] = float(
            score[heldout_y == 1].mean() - score[heldout_y == 0].mean()
        )
    return output


def factorial_calibration_analysis(
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit the frozen post-selection 2^4 factorial score model.

    The four decisive factors use 0/1 coding.  The balanced design therefore
    puts every decisive coefficient on the same mean-score-difference scale.
    Register is a fifth 0/1 covariate and is never used for readout selection.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (32,) or len(metadata) != 32:
        raise ValueError("factorial calibration requires exactly 32 scored rows")
    if any(row.get("split") != "factorial_calibration" for row in metadata):
        raise ValueError("factorial calibration received a non-calibration row")
    if any(row.get("label") is not None for row in metadata):
        raise ValueError("factorial calibration rows must remain unlabeled")
    if not np.isfinite(values).all():
        raise ValueError("factorial calibration scores contain NaN or Inf")

    design_rows: list[list[float]] = []
    cells: list[tuple[bool, bool, bool, bool]] = []
    registers: list[bool] = []
    for row in metadata:
        factors = row.get("factors")
        if not isinstance(factors, dict):
            raise TypeError("factorial calibration row lacks factors")
        try:
            cell = tuple(bool(factors[column]) for column in FACTORIAL_DECISIVE_COLUMNS)
        except KeyError as error:
            raise ValueError("factorial calibration row lacks a decisive factor") from error
        register = row.get("register")
        if register not in {"formal", "casual"}:
            raise ValueError("factorial calibration register is invalid")
        formal = register == "formal"
        cells.append(cell)
        registers.append(formal)
        design_rows.append([1.0, *(float(value) for value in cell), float(formal)])

    cell_counts = {cell: cells.count(cell) for cell in set(cells)}
    if len(cell_counts) != 16 or set(cell_counts.values()) != {2}:
        raise ValueError("factorial calibration does not cover two complete 2^4 replicates")
    for cell in cell_counts:
        observed = {
            registers[index]
            for index, candidate in enumerate(cells)
            if candidate == cell
        }
        if observed != {False, True}:
            raise ValueError("factorial calibration register is not counterrotated within cell")

    design = np.asarray(design_rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) != len(FACTORIAL_COLUMNS):
        raise ValueError("factorial calibration design matrix is rank deficient")
    coefficient, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficient
    degrees_of_freedom = int(len(values) - design.shape[1])
    residual_variance = float((residual @ residual) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    critical = float(student_t.ppf(0.975, degrees_of_freedom))
    coefficient_report = {
        column: {
            "estimate": float(coefficient[index]),
            "standard_error": float(standard_error[index]),
            "ci95": [
                float(coefficient[index] - critical * standard_error[index]),
                float(coefficient[index] + critical * standard_error[index]),
            ],
        }
        for index, column in enumerate(FACTORIAL_COLUMNS)
    }

    recipient = np.asarray([cell[0] for cell in cells], dtype=bool)
    consequence = np.asarray([cell[1] for cell in cells], dtype=bool)
    scoring = np.asarray([cell[2] for cell in cells], dtype=bool)
    comparison = np.asarray([cell[3] for cell in cells], dtype=bool)
    clean_evaluation = ~recipient & ~consequence & (scoring | comparison)
    clean_deployment = (recipient | consequence) & ~scoring & ~comparison
    if int(clean_evaluation.sum()) != 6 or int(clean_deployment.sum()) != 6:
        raise ValueError("factorial clean-cell counts differ from the frozen design")
    contrast = np.asarray(
        [0.0, -2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]
    )
    ols_clean_difference = float(contrast @ coefficient)
    ols_clean_standard_error = float(
        np.sqrt(max(float(contrast @ covariance @ contrast), 0.0))
    )
    evaluation_values = values[clean_evaluation]
    deployment_values = values[clean_deployment]
    clean_difference = float(evaluation_values.mean() - deployment_values.mean())
    evaluation_variance_term = float(evaluation_values.var(ddof=1) / len(evaluation_values))
    deployment_variance_term = float(deployment_values.var(ddof=1) / len(deployment_values))
    clean_standard_error = float(
        np.sqrt(evaluation_variance_term + deployment_variance_term)
    )
    if clean_standard_error == 0:
        clean_degrees_of_freedom = float(len(evaluation_values) + len(deployment_values) - 2)
        clean_critical = float(student_t.ppf(0.975, clean_degrees_of_freedom))
    else:
        clean_degrees_of_freedom = float(
            (evaluation_variance_term + deployment_variance_term) ** 2
            / (
                evaluation_variance_term**2 / (len(evaluation_values) - 1)
                + deployment_variance_term**2 / (len(deployment_values) - 1)
            )
        )
        clean_critical = float(student_t.ppf(0.975, clean_degrees_of_freedom))

    decisive = {
        column: coefficient_report[column]["estimate"] for column in FACTORIAL_DECISIVE_COLUMNS
    }
    register_estimate = coefficient_report["formal_register"]["estimate"]
    sign_checks = {
        "real_external_recipient_negative": decisive["real_external_recipient"] < 0,
        "operational_consequence_negative": decisive["operational_consequence"] < 0,
        "model_scoring_positive": decisive["model_scoring"] > 0,
        "comparison_among_models_positive": decisive["comparison_among_models"] > 0,
        "clean_evaluation_mean_above_clean_deployment": clean_difference > 0,
        "register_smaller_than_every_decisive_factor": abs(register_estimate)
        < min(abs(value) for value in decisive.values()),
    }
    return {
        "row_count": len(metadata),
        "design_columns": list(FACTORIAL_COLUMNS),
        "coding": "zero_one_with_intercept",
        "degrees_of_freedom": degrees_of_freedom,
        "residual_standard_deviation": float(np.sqrt(residual_variance)),
        "coefficients": coefficient_report,
        "clean_cell_contrast": {
            "clean_evaluation_count": int(clean_evaluation.sum()),
            "clean_deployment_count": int(clean_deployment.sum()),
            "clean_evaluation_mean": float(evaluation_values.mean()),
            "clean_deployment_mean": float(deployment_values.mean()),
            "difference": clean_difference,
            "standard_error": clean_standard_error,
            "degrees_of_freedom": clean_degrees_of_freedom,
            "ci95": [
                float(clean_difference - clean_critical * clean_standard_error),
                float(clean_difference + clean_critical * clean_standard_error),
            ],
            "ols_main_effect_contrast": {
                "difference": ols_clean_difference,
                "standard_error": ols_clean_standard_error,
                "ci95": [
                    float(ols_clean_difference - critical * ols_clean_standard_error),
                    float(ols_clean_difference + critical * ols_clean_standard_error),
                ],
            },
        },
        "checks": sign_checks,
        "passed": all(sign_checks.values()),
        "score_rows": [
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": row["label"],
                "register": row["register"],
                "factors": row["factors"],
                "score": float(score),
            }
            for row, score in zip(metadata, values, strict=True)
        ],
    }


def evaluate_source_final(
    fit: DevelopmentFit,
    development_features: np.ndarray,
    development_metadata: list[dict[str, Any]],
    final_features: np.ndarray,
    final_metadata: list[dict[str, Any]],
    calibration_features: np.ndarray,
    calibration_metadata: list[dict[str, Any]],
    *,
    bootstrap_reps: int = 500,
    bootstrap_seed: int = 20260913,
) -> dict[str, Any]:
    final_splits = {row["split"] for row in final_metadata}
    if final_splits != {"ordinary_test", "final_counterfactual", "neutral_controls"}:
        raise ValueError(f"final source splits differ: {final_splits}")
    split = np.asarray([row["split"] for row in final_metadata])
    labels = np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in final_metadata],
        dtype=np.int64,
    )
    scores = {
        "logistic": _scores(fit.logistic, final_features),
        "paired_mean": _scores(fit.paired_mean, final_features),
    }
    calibration_scores = {
        "logistic": _scores(fit.logistic, calibration_features),
        "paired_mean": _scores(fit.paired_mean, calibration_features),
    }
    model_reports: dict[str, Any] = {}
    for name, model in (("logistic", fit.logistic), ("paired_mean", fit.paired_mean)):
        ordinary = split == "ordinary_test"
        heldout = split == "final_counterfactual"
        neutral = split == "neutral_controls"
        neutral_by_family = {
            family: _fpr(
                scores[name][
                    neutral
                    & np.asarray(
                        [row["generator_family"] == family for row in final_metadata]
                    )
                ],
                model.threshold_80_tpr,
            )
            for family in sorted(
                {row["generator_family"] for row in final_metadata if row["split"] == "neutral_controls"}
            )
        }
        model_reports[name] = {
            "ordinary_test": _binary_metrics(labels[ordinary], scores[name][ordinary]),
            "final_counterfactual": _binary_metrics(labels[heldout], scores[name][heldout]),
            "final_neutral_fpr_by_family": neutral_by_family,
            "factorial_calibration": factorial_calibration_analysis(
                calibration_scores[name], calibration_metadata
            ),
            "score_rows": [
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "generator_family": row["generator_family"],
                    "score": float(score),
                }
                for row, score in zip(final_metadata, scores[name], strict=True)
            ],
        }

    train_indices = [
        index for index, row in enumerate(development_metadata) if row["split"] == "train"
    ]
    by_pair: dict[str, list[int]] = {}
    for index in train_indices:
        by_pair.setdefault(str(development_metadata[index]["pair_id"]), []).append(index)
    pair_ids = sorted(by_pair)
    rng = np.random.default_rng(bootstrap_seed)
    full_direction = fit.paired_mean.raw_direction
    cosines = np.empty(bootstrap_reps, dtype=np.float64)
    layer = fit.paired_mean.layer
    for repetition in range(bootstrap_reps):
        sampled_ids = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sampled_indices = [index for pair_id in sampled_ids for index in by_pair[str(pair_id)]]
        sampled_x = development_features[sampled_indices, layer]
        sampled_y = np.asarray(
            [int(development_metadata[index]["label"]) for index in sampled_indices]
        )
        mean, scale = _standardizer(sampled_x)
        standardized = (sampled_x - mean) / scale
        weight = standardized[sampled_y == 1].mean(0) - standardized[sampled_y == 0].mean(0)
        raw = weight / scale
        raw /= np.linalg.norm(raw)
        cosines[repetition] = float(raw @ full_direction)

    logistic_heldout = scores["logistic"][split == "final_counterfactual"]
    paired_heldout = scores["paired_mean"][split == "final_counterfactual"]
    heldout_labels = labels[split == "final_counterfactual"]
    direction_cosine = float(fit.logistic.raw_direction @ fit.paired_mean.raw_direction)
    rank = float(spearmanr(logistic_heldout, paired_heldout).statistic)
    return {
        "schema_version": "glm53_v11_source_final_analysis_v2",
        "selection_used_final_rows": False,
        "selection_used_calibration_rows": False,
        "selected_layer": fit.logistic.layer,
        "selected_C": fit.selected_c,
        "models": model_reports,
        "direction_agreement": {
            "raw_cosine": direction_cosine,
            "final_score_spearman": rank,
            "logistic_score_gap": float(
                logistic_heldout[heldout_labels == 1].mean()
                - logistic_heldout[heldout_labels == 0].mean()
            ),
            "paired_mean_score_gap": float(
                paired_heldout[heldout_labels == 1].mean()
                - paired_heldout[heldout_labels == 0].mean()
            ),
        },
        "paired_direction_stability": {
            "reps": bootstrap_reps,
            "seed": bootstrap_seed,
            "median_cosine": float(np.median(cosines)),
            "fifth_percentile_cosine": float(np.percentile(cosines, 5)),
            "minimum_cosine": float(cosines.min()),
            "cosines": cosines.tolist(),
        },
    }


__all__ = [
    "DevelopmentFit",
    "FrozenLinear",
    "evaluate_source_final",
    "factorial_calibration_analysis",
    "fit_source_development",
    "leave_one_generator_score_gaps",
    "load_development_fit",
    "load_partition",
    "pair_preserving_labels",
    "run_full_selection_permutations",
    "save_development_fit",
]
