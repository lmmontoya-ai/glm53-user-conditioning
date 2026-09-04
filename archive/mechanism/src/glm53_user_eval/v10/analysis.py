"""Offline construct-validity diagnostics for the preserved v9 archive."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.glm53_user_eval.v9.probes import fit_sequence_layers
from src.probe.sequence_linear import train_sequence_linear_models

from .data import AuditData, deterministic_task_folds, metadata_matrix
from .models import (
    Model,
    fit_lda_layers,
    fit_paired_mean_layers,
    fit_pca_ridge_layers,
    fit_ridge_layers,
    metrics,
    score,
    threshold_balanced,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _split_masks(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = np.asarray([row["split"] for row in rows])
    return split == "train", split == "val", split == "test"


def _model_summary(model: Model, data: AuditData) -> dict[str, Any]:
    labels = data.labels
    scores = score(model, data.features)
    split = np.asarray([row["split"] for row in data.rows])
    output: dict[str, Any] = {
        "selected_layer": model.layer,
        "validation_auroc": model.val_auroc,
        "validation_brier": model.val_brier,
        "detail": model.detail,
        "weight_norm_standardized": float(np.linalg.norm(model.weight)),
        "raw_direction_sha256": hashlib.sha256(
            model.raw_direction.astype(np.float64).tobytes()
        ).hexdigest(),
    }
    threshold = threshold_balanced(labels[split == "val"], scores[split == "val"])
    for name in ("test", "holdout_realism", "holdout_hardneg", "holdout_mats_aux"):
        binary = (split == name) & np.isin(labels, [0, 1])
        output[name] = metrics(labels[binary], scores[binary])
        neutral = (split == name) & (labels == -1)
        if neutral.any():
            output[name]["neutral_false_positive_rate"] = float(
                np.mean(scores[neutral] >= threshold)
            )
    return output | {"model": model}


def _fit_historical(data: AuditData, config: dict[str, Any]) -> Model:
    train, val, _ = _split_masks(data.rows)
    train_features = {
        layer: [
            (data.rows[i]["sample_id"], data.features[i, layer], int(data.labels[i]))
            for i in np.flatnonzero(train)
        ]
        for layer in range(45)
    }
    val_features = {
        layer: [
            (data.rows[i]["sample_id"], data.features[i, layer], int(data.labels[i]))
            for i in np.flatnonzero(val)
        ]
        for layer in range(45)
    }
    model, _ = train_sequence_linear_models(
        train_features=train_features,
        val_features=val_features,
        epochs=int(config["epochs"]),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
    )
    return Model(
        "historical_full_batch",
        model.layer_index,
        model.feature_mean,
        model.feature_std,
        model.weights,
        model.bias,
        float(model.val_auroc),
        float(model.val_brier),
        {"epochs": config["epochs"], "lr": config["lr"]},
    )


def fit_fixed_split_models(
    data: AuditData, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Model]]:
    train, val, _ = _split_masks(data.rows)
    methods: dict[str, Model] = {
        "v9_adamw": _fit_v9_adamw(data, config["v9_adamw"]),
        "historical_full_batch": _fit_historical(data, config["historical"]),
        "paired_mean": fit_paired_mean_layers(data.features, data.labels, train, val),
        "ridge_logistic": fit_ridge_layers(
            data.features, data.labels, train, val, config["ridge_c_grid"]
        ),
        "shrinkage_lda": fit_lda_layers(data.features, data.labels, train, val),
        "pca_ridge": fit_pca_ridge_layers(
            data.features, data.labels, train, val, config["ridge_c_grid"], config["pca_dimensions"]
        ),
    }
    report: dict[str, Any] = {}
    for name, model in methods.items():
        summary = _model_summary(model, data)
        summary.pop("model")
        report[name] = summary
    report["cross_method_direction_geometry"] = {
        "status": "not_computed",
        "reason": "Validation-selected methods chose different layers; cross-layer coordinate cosines are not interpretable.",
    }
    return report, methods


def _fit_v9_adamw(data: AuditData, config: dict[str, Any]) -> Model:
    probe, _ = fit_sequence_layers(
        data.features,
        data.rows,
        config=config,
        seed=int(config["seed"]),
    )
    return Model(
        "v9_adamw",
        probe.layer,
        probe.mean,
        probe.scale,
        probe.weight,
        probe.bias,
        probe.val_auroc,
        probe.val_brier,
        {"best_epoch": probe.best_epoch, "seed": config["seed"]},
    )


def leave_family_out(data: AuditData) -> dict[str, Any]:
    core = np.asarray([row["slice_id"] == "core_context_pairs" for row in data.rows])
    families = sorted(
        {row["family_id"] for row in data.rows if row["slice_id"] == "core_context_pairs"}
    )
    folds: list[dict[str, Any]] = []
    directions: list[np.ndarray] = []
    for test_family in families:
        for val_family in families:
            if val_family == test_family:
                continue
            train = core & ~np.isin(
                [row["family_id"] for row in data.rows], [test_family, val_family]
            )
            val = core & (np.asarray([row["family_id"] for row in data.rows]) == val_family)
            test = core & (np.asarray([row["family_id"] for row in data.rows]) == test_family)
            model = fit_paired_mean_layers(data.features, data.labels, train, val)
            test_metric = metrics(data.labels[test], score(model, data.features[test]))
            folds.append(
                {
                    "test_family": test_family,
                    "val_family": val_family,
                    "selected_layer": model.layer,
                    **test_metric,
                }
            )
            directions.append(model.raw_direction)
    aurocs = np.asarray([fold["auroc"] for fold in folds])
    direction_matrix = np.vstack(directions)
    reference = direction_matrix.mean(0)
    reference /= np.linalg.norm(reference)
    cosines = direction_matrix @ reference
    return {
        "method": "paired_mean_full_layer_selection",
        "fold_count": len(folds),
        "median_auroc": float(np.median(aurocs)),
        "minimum_auroc": float(np.min(aurocs)),
        "maximum_auroc": float(np.max(aurocs)),
        "mean_auroc": float(np.mean(aurocs)),
        "selected_layer_counts": {
            str(layer): int(sum(fold["selected_layer"] == layer for fold in folds))
            for layer in sorted({fold["selected_layer"] for fold in folds})
        },
        "direction_cosine_to_fold_mean": {
            "median": float(np.median(cosines)),
            "minimum": float(np.min(cosines)),
        },
        "folds": folds,
    }


def family_task_blocked(data: AuditData) -> dict[str, Any]:
    core = np.asarray([row["slice_id"] == "core_context_pairs" for row in data.rows])
    families = sorted(
        {row["family_id"] for row in data.rows if row["slice_id"] == "core_context_pairs"}
    )
    family_array = np.asarray([row["family_id"] for row in data.rows])
    task_array = np.asarray(
        [row["task_number"] if row["task_number"] is not None else -1 for row in data.rows]
    )
    task_folds = deterministic_task_folds()
    folds: list[dict[str, Any]] = []
    for test_family in families:
        for val_family in families:
            if test_family == val_family:
                continue
            digest = hashlib.sha256(
                f"glm53-v10-family-task|{test_family}|{val_family}".encode()
            ).digest()
            test_fold_index = digest[0] % len(task_folds)
            val_fold_index = digest[1] % (len(task_folds) - 1)
            if val_fold_index >= test_fold_index:
                val_fold_index += 1
            test_tasks = task_folds[test_fold_index]
            val_tasks = task_folds[val_fold_index]
            excluded_tasks = set(test_tasks + val_tasks)
            train = (
                core
                & ~np.isin(family_array, [test_family, val_family])
                & ~np.isin(task_array, list(excluded_tasks))
            )
            val = core & (family_array == val_family) & np.isin(task_array, val_tasks)
            test = core & (family_array == test_family) & np.isin(task_array, test_tasks)
            model = fit_paired_mean_layers(data.features, data.labels, train, val)
            test_metric = metrics(data.labels[test], score(model, data.features[test]))
            folds.append(
                {
                    "test_family": test_family,
                    "val_family": val_family,
                    "test_tasks": test_tasks,
                    "val_tasks": val_tasks,
                    "selected_layer": model.layer,
                    **test_metric,
                }
            )
    aurocs = np.asarray([fold["auroc"] for fold in folds])
    return {
        "method": "paired_mean_full_layer_selection",
        "fold_count": len(folds),
        "median_auroc": float(np.median(aurocs)),
        "minimum_auroc": float(np.min(aurocs)),
        "mean_auroc": float(np.mean(aurocs)),
        "fraction_perfect": float(np.mean(aurocs == 1.0)),
        "folds": folds,
    }


def metadata_baseline(data: AuditData, c_grid: list[float]) -> dict[str, Any]:
    train, val, test = _split_masks(data.rows)
    eligible = train | val | test
    indices = np.flatnonzero(eligible)
    matrix, names = metadata_matrix(data.rows, indices)
    position = {int(index): offset for offset, index in enumerate(indices)}
    train_pos = np.asarray([position[int(index)] for index in np.flatnonzero(train)])
    val_pos = np.asarray([position[int(index)] for index in np.flatnonzero(val)])
    test_pos = np.asarray([position[int(index)] for index in np.flatnonzero(test)])
    scaler = StandardScaler().fit(matrix[train_pos])
    transformed = scaler.transform(matrix)
    candidates = []
    for c in c_grid:
        model = LogisticRegression(
            C=c, penalty="l2", solver="liblinear", max_iter=2000, random_state=0
        ).fit(transformed[train_pos], data.labels[train])
        val_score = model.decision_function(transformed[val_pos])
        result = metrics(data.labels[val], val_score)
        candidates.append((result["auroc"], result["brier"], c, model))
    _, _, selected_c, selected = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
    test_score = selected.decision_function(transformed[test_pos])
    coefficients = {
        name: float(value) for name, value in zip(names, selected.coef_[0], strict=True)
    }
    return {
        "selected_C": selected_c,
        "test": metrics(data.labels[test], test_score),
        "standardized_coefficients": coefficients,
        "unavailable_metadata": [
            "average_absolute_token_position",
            "retained_token_position_indices",
        ],
    }


def hard_negative_decomposition(data: AuditData, models: dict[str, Model]) -> dict[str, Any]:
    labels = data.labels
    split = np.asarray([row["split"] for row in data.rows])
    output: dict[str, Any] = {}
    for name, model in models.items():
        scores = score(model, data.features)
        threshold = threshold_balanced(labels[split == "val"], scores[split == "val"])
        method: dict[str, Any] = {}
        for family in sorted(
            {row["family_id"] for row in data.rows if row["split"] == "holdout_hardneg"}
        ):
            mask = (split == "holdout_hardneg") & (
                np.asarray([row["family_id"] for row in data.rows]) == family
            )
            if np.isin(labels[mask], [0, 1]).all() and len(np.unique(labels[mask])) == 2:
                method[family] = metrics(labels[mask], scores[mask]) | {"count": int(mask.sum())}
            else:
                method[family] = {
                    "count": int(mask.sum()),
                    "false_positive_rate": float(np.mean(scores[mask] >= threshold)),
                    "mean_score": float(np.mean(scores[mask])),
                }
        role_rows: dict[str, Any] = {}
        for role in sorted(
            {row["prompt_role"] for row in data.rows if row["split"] == "holdout_hardneg"}
        ):
            mask = (
                (split == "holdout_hardneg")
                & (np.asarray([row["prompt_role"] for row in data.rows]) == role)
                & np.isin(labels, [0, 1])
            )
            role_rows[role] = metrics(labels[mask], scores[mask]) | {"count": int(mask.sum())}
        method["by_prompt_role_binary"] = role_rows
        output[name] = method
    return output


def direction_stability(data: AuditData, *, reps: int, seed: int) -> dict[str, Any]:
    train, val, test = _split_masks(data.rows)
    pair_ids = sorted({data.rows[index]["pair_id"] for index in np.flatnonzero(train)})
    by_pair = {
        pair: [index for index in np.flatnonzero(train) if data.rows[index]["pair_id"] == pair]
        for pair in pair_ids
    }
    full = fit_paired_mean_layers(data.features, data.labels, train, val)
    rng = np.random.default_rng(seed)
    records = []
    for repetition in range(reps):
        chosen = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sampled_indices = np.asarray(
            [index for pair in chosen for index in by_pair[str(pair)]], dtype=np.int64
        )
        sampled_features = np.concatenate(
            (data.features[sampled_indices], data.features[val]), axis=0
        )
        sampled_labels = np.concatenate((data.labels[sampled_indices], data.labels[val]))
        sampled_train = np.arange(len(sampled_indices))
        sampled_val = np.arange(len(sampled_indices), len(sampled_labels))
        model = fit_paired_mean_layers(
            sampled_features,
            sampled_labels,
            np.isin(np.arange(len(sampled_labels)), sampled_train),
            np.isin(np.arange(len(sampled_labels)), sampled_val),
        )
        test_score = (
            (data.features[test, model.layer] - model.mean) / model.scale
        ) @ model.weight + model.bias
        hard = np.asarray(
            [row["split"] == "holdout_hardneg" and row["label"] in (0, 1) for row in data.rows]
        )
        hard_score = (
            (data.features[hard, model.layer] - model.mean) / model.scale
        ) @ model.weight + model.bias
        records.append(
            {
                "repetition": repetition,
                "selected_layer": model.layer,
                "cosine_to_full": float(model.raw_direction @ full.raw_direction),
                "test_auroc": metrics(data.labels[test], test_score)["auroc"],
                "hard_negative_auroc": metrics(data.labels[hard], hard_score)["auroc"],
            }
        )
    return {
        "method": "paired_mean",
        "reps": reps,
        "selected_layer_counts": {
            str(layer): int(sum(row["selected_layer"] == layer for row in records))
            for layer in sorted({row["selected_layer"] for row in records})
        },
        "cosine_to_full": {
            "median": float(np.median([row["cosine_to_full"] for row in records])),
            "minimum": float(np.min([row["cosine_to_full"] for row in records])),
        },
        "test_auroc": {
            "median": float(np.median([row["test_auroc"] for row in records])),
            "minimum": float(np.min([row["test_auroc"] for row in records])),
        },
        "hard_negative_auroc": {
            "median": float(np.median([row["hard_negative_auroc"] for row in records])),
            "minimum": float(np.min([row["hard_negative_auroc"] for row in records])),
        },
        "records": records,
    }


def run_audit(data: AuditData, config: dict[str, Any], output: Path) -> dict[str, Any]:
    fixed, models = fit_fixed_split_models(data, config["analysis"])
    report = {
        "schema_version": "glm53_v10_offline_diagnostic_report_v1",
        "project_id": config["project_id"],
        "paid_compute_used": False,
        "fixed_split_trainer_comparison": fixed,
        "leave_family_out": leave_family_out(data),
        "family_and_task_blocked": family_task_blocked(data),
        "metadata_baseline": metadata_baseline(data, config["analysis"]["ridge_c_grid"]),
        "hard_negative_decomposition": hard_negative_decomposition(data, models),
        "direction_stability": direction_stability(
            data,
            reps=int(config["analysis"]["stability_reps"]),
            seed=int(config["analysis"]["stability_seed"]),
        ),
        "unavailable_checks": {
            "historical_paper_feature_cross_regression": "No archived historical activation feature matrix was found in the frozen local cache.",
            "shared_suffix_only_pooling": "The v9 token bags omit token IDs and retained-position indices, so exact shared task-token rows cannot be identified without a new forward.",
            "input_ablation": "Cue deletion, neutral replacement, and cue swap require new model forwards and were not authorized in this offline audit.",
        },
    }
    serializable = json.loads(json.dumps(report, default=lambda value: None))
    atomic_json(output, serializable)
    return serializable
