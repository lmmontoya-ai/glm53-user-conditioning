"""Frozen linear source readouts and their selection null."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from .contract import sha256_file

C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class FrozenReadout:
    name: str
    layer: int
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    upper_80_tpr: float
    lower_80_tnr: float

    @property
    def raw_direction(self) -> np.ndarray:
        value = self.weight.astype(np.float64) / self.scale.astype(np.float64)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"{self.name} has no finite direction")
        return value / norm


@dataclass(frozen=True)
class DevelopmentFit:
    logistic: FrozenReadout
    paired_mean: FrozenReadout
    selected_c: float
    objective: float
    report: dict[str, Any]


def _standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1.0
    return mean, scale


def scores(model: FrozenReadout, features: np.ndarray) -> np.ndarray:
    return ((features[:, model.layer] - model.mean) / model.scale) @ model.weight + model.bias


def binary_metrics(labels: np.ndarray, values: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    score = np.asarray(values, dtype=np.float64)
    if set(y.tolist()) != {0, 1} or not np.isfinite(score).all():
        raise ValueError("binary metrics need two labels and finite scores")
    fpr, tpr, thresholds = roc_curve(y, score)
    eligible = np.flatnonzero(tpr >= 0.80)
    index = int(eligible[0]) if len(eligible) else len(thresholds) - 1
    probability = expit(score)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, probability)),
        "fpr_at_80_tpr": float(fpr[index]),
        "threshold_at_80_tpr": float(thresholds[index]),
        "score_gap": float(score[y == 1].mean() - score[y == 0].mean()),
    }


def _thresholds(labels: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    positive = np.sort(np.asarray(values)[np.asarray(labels) == 1])
    negative = np.sort(np.asarray(values)[np.asarray(labels) == 0])
    if not len(positive) or not len(negative):
        raise ValueError("threshold set lacks a class")
    upper = float(np.quantile(positive, 0.20, method="higher"))
    lower = float(np.quantile(negative, 0.80, method="lower"))
    return upper, lower


def _indices(metadata: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    split = np.asarray([row["split"] for row in metadata])
    if set(split.tolist()) != {"train", "validation", "development_counterfactual"}:
        raise ValueError("V16 development fitter received a final split")
    labels = np.asarray([int(row["label"]) for row in metadata], dtype=np.int64)
    result = {name: np.flatnonzero(split == name) for name in set(split.tolist())}
    if any(set(labels[value].tolist()) != {0, 1} for value in result.values()):
        raise ValueError("a development split lacks a class")
    return result | {"labels": labels}


def _candidate_key(report: dict[str, Any], c_value: float, layer: int) -> tuple[float, ...]:
    val = report["validation"]
    dev = report["development_counterfactual"]
    return (
        min(val["auroc"], dev["auroc"]),
        min(val["auprc"], dev["auprc"]),
        -max(val["brier"], dev["brier"]),
        -float(c_value),
        -float(layer),
    )


def _fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[np.ndarray, float]:
    with threadpool_limits(limits=1):
        classifier = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=0,
        ).fit(x, y)
    return classifier.coef_[0].astype(np.float32), float(classifier.intercept_[0])


def _select(
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    include_candidates: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if features.shape != (len(metadata), 45, 4096):
        raise ValueError("V16 development feature shape differs")
    index = _indices(metadata)
    train, validation, development = (
        index["train"],
        index["validation"],
        index["development_counterfactual"],
    )
    best: tuple[tuple[float, ...], dict[str, Any], dict[str, np.ndarray]] | None = None
    candidates: list[dict[str, Any]] = []
    for layer in range(45):
        mean, scale = _standardizer(features[train, layer])
        standardized = {
            "train": np.ascontiguousarray((features[train, layer] - mean) / scale),
            "validation": np.ascontiguousarray((features[validation, layer] - mean) / scale),
            "development_counterfactual": np.ascontiguousarray(
                (features[development, layer] - mean) / scale
            ),
        }
        for c_value in C_GRID:
            weight, bias = _fit_logistic(standardized["train"], labels[train], c_value)
            report = {
                "layer": layer,
                "C": c_value,
                "validation": binary_metrics(
                    labels[validation], standardized["validation"] @ weight + bias
                ),
                "development_counterfactual": binary_metrics(
                    labels[development], standardized["development_counterfactual"] @ weight + bias
                ),
            }
            key = _candidate_key(report, c_value, layer)
            report["selection_key"] = list(key)
            if include_candidates:
                candidates.append(report)
            arrays = {"mean": mean, "scale": scale, "weight": weight}
            if best is None or key > best[0]:
                best = (key, report, arrays | {"bias": np.asarray(bias)})
    if best is None:
        raise AssertionError("V16 selection evaluated no candidates")
    selected = dict(best[1])
    selected["objective"] = best[0][0]
    return {"selected": selected, "candidates": candidates}, best[2]


def pair_preserving_labels(metadata: list[dict[str, Any]], *, seed: int) -> np.ndarray:
    labels = np.asarray([int(row["label"]) for row in metadata], dtype=np.int64)
    pairs: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        pairs.setdefault(str(row["pair_id"]), []).append(index)
    rng = np.random.default_rng(seed)
    for pair_id, members in pairs.items():
        if len(members) != 2 or sorted(labels[members].tolist()) != [0, 1]:
            raise ValueError(f"incomplete development pair {pair_id}")
        if int(rng.integers(0, 2)):
            labels[members] = 1 - labels[members]
    return labels


def fit_source_development(features: np.ndarray, metadata: list[dict[str, Any]]) -> DevelopmentFit:
    labels = np.asarray([int(row["label"]) for row in metadata], dtype=np.int64)
    selection, arrays = _select(features, metadata, labels, include_candidates=True)
    selected = selection["selected"]
    layer = int(selected["layer"])
    index = _indices(metadata)
    dev = index["development_counterfactual"]
    temporary = FrozenReadout(
        "ridge_logistic",
        layer,
        arrays["mean"],
        arrays["scale"],
        arrays["weight"],
        float(arrays["bias"]),
        0.0,
        0.0,
    )
    upper, lower = _thresholds(labels[dev], scores(temporary, features[dev]))
    logistic = FrozenReadout(
        temporary.name,
        layer,
        temporary.mean,
        temporary.scale,
        temporary.weight,
        temporary.bias,
        upper,
        lower,
    )
    train = index["train"]
    train_x = (features[train, layer] - arrays["mean"]) / arrays["scale"]
    class_zero = train_x[labels[train] == 0].mean(axis=0)
    class_one = train_x[labels[train] == 1].mean(axis=0)
    paired_weight = (class_one - class_zero).astype(np.float32)
    paired_bias = -0.5 * float((class_one + class_zero) @ paired_weight)
    paired_temp = FrozenReadout(
        "paired_mean",
        layer,
        arrays["mean"],
        arrays["scale"],
        paired_weight,
        paired_bias,
        0.0,
        0.0,
    )
    paired_upper, paired_lower = _thresholds(labels[dev], scores(paired_temp, features[dev]))
    paired = FrozenReadout(
        paired_temp.name,
        layer,
        paired_temp.mean,
        paired_temp.scale,
        paired_temp.weight,
        paired_temp.bias,
        paired_upper,
        paired_lower,
    )
    report = {
        "schema_version": "glm53_v16_source_readout_lock_v1",
        "primary_view": "shared_task_suffix_mean",
        "selection_used_final_rows": False,
        "selection_used_factorial_rows": False,
        "selection_used_fresh_controls": False,
        "selected_layer": layer,
        "selected_C": float(selected["C"]),
        "objective": float(selected["objective"]),
        "selected": selected,
        "candidate_count": len(selection["candidates"]),
        "candidates": selection["candidates"],
    }
    return DevelopmentFit(logistic, paired, float(selected["C"]), float(selected["objective"]), report)


def save_development_fit(fit: DevelopmentFit, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    arrays_path = output_root / "source_readout_arrays.npz"
    temporary = arrays_path.with_suffix(".npz.tmp")
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
        "arrays_sha256": sha256_file(arrays_path),
        "logistic": {
            "bias": fit.logistic.bias,
            "upper_80_tpr": fit.logistic.upper_80_tpr,
            "lower_80_tnr": fit.logistic.lower_80_tnr,
        },
        "paired_mean": {
            "bias": fit.paired_mean.bias,
            "upper_80_tpr": fit.paired_mean.upper_80_tpr,
            "lower_80_tnr": fit.paired_mean.lower_80_tnr,
        },
    }
    path = output_root / "source_readout_lock.json"
    temporary_json = path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_json, path)
    return report


def load_development_fit(output_root: Path) -> DevelopmentFit:
    report = json.loads((output_root / "source_readout_lock.json").read_text(encoding="utf-8"))
    arrays_path = output_root / "source_readout_arrays.npz"
    if sha256_file(arrays_path) != report["arrays_sha256"]:
        raise ValueError("V16 source readout arrays differ")
    with np.load(arrays_path) as arrays:
        logistic = FrozenReadout(
            "ridge_logistic",
            int(report["selected_layer"]),
            arrays["logistic_mean"],
            arrays["logistic_scale"],
            arrays["logistic_weight"],
            float(report["logistic"]["bias"]),
            float(report["logistic"]["upper_80_tpr"]),
            float(report["logistic"]["lower_80_tnr"]),
        )
        paired = FrozenReadout(
            "paired_mean",
            int(report["selected_layer"]),
            arrays["paired_mean"],
            arrays["paired_scale"],
            arrays["paired_weight"],
            float(report["paired_mean"]["bias"]),
            float(report["paired_mean"]["upper_80_tpr"]),
            float(report["paired_mean"]["lower_80_tnr"]),
        )
    return DevelopmentFit(logistic, paired, float(report["selected_C"]), float(report["objective"]), report)


def run_full_selection_permutations(
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    observed_objective: float,
    reps: int,
    seed: int,
    workers: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if reps != 1000:
        raise ValueError("V16 requires exactly 1000 full-selection permutations")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[int, dict[str, Any]] = {}
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[int(row["repetition"])] = row

    def one(repetition: int) -> dict[str, Any]:
        labels = pair_preserving_labels(metadata, seed=seed + repetition)
        selection, _ = _select(features, metadata, labels, include_candidates=False)
        selected = selection["selected"]
        return {
            "repetition": repetition,
            "objective": float(selected["objective"]),
            "selected_layer": int(selected["layer"]),
            "selected_C": float(selected["C"]),
        }

    pending = [index for index in range(reps) if index not in completed]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, repetition): repetition for repetition in pending}
        for future in as_completed(futures):
            record = future.result()
            completed[int(record["repetition"])] = record
            ordered = [completed[index] for index in sorted(completed)]
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            temporary.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in ordered),
                encoding="utf-8",
            )
            os.replace(temporary, checkpoint_path)
    rows = [completed[index] for index in range(reps)]
    null = np.asarray([row["objective"] for row in rows], dtype=np.float64)
    exceedances = int(np.sum(null >= observed_objective))
    return {
        "schema_version": "glm53_v16_full_selection_permutation_v1",
        "reps": reps,
        "seed": seed,
        "observed_objective": observed_objective,
        "null_objectives": null.tolist(),
        "exceedances": exceedances,
        "add_one_empirical_p": (1 + exceedances) / (1 + reps),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def bootstrap_direction_stability(
    fit: DevelopmentFit,
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if reps != 1000:
        raise ValueError("V16 requires exactly 1000 direction bootstraps")
    train = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    pairs: dict[str, list[int]] = {}
    for index in train:
        pairs.setdefault(str(metadata[index]["pair_id"]), []).append(index)
    if any(len(value) != 2 for value in pairs.values()):
        raise ValueError("training bootstrap found an incomplete pair")
    pair_ids = sorted(pairs)
    rng = np.random.default_rng(seed)
    cosines = np.empty(reps, dtype=np.float64)
    full = fit.paired_mean.raw_direction
    layer = fit.paired_mean.layer
    for repetition in range(reps):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        indices = [index for pair_id in sampled for index in pairs[str(pair_id)]]
        x = features[indices, layer]
        y = np.asarray([int(metadata[index]["label"]) for index in indices])
        mean, scale = _standardizer(x)
        z = (x - mean) / scale
        weight = z[y == 1].mean(0) - z[y == 0].mean(0)
        raw = weight / scale
        raw /= np.linalg.norm(raw)
        cosines[repetition] = float(raw @ full)
    return {
        "reps": reps,
        "seed": seed,
        "median_cosine": float(np.median(cosines)),
        "fifth_percentile_cosine": float(np.percentile(cosines, 5)),
        "minimum_cosine": float(cosines.min()),
        "cosines": cosines.tolist(),
    }


def leave_one_generator_gaps(
    fit: DevelopmentFit,
    development_features: np.ndarray,
    development_metadata: list[dict[str, Any]],
    final_features: np.ndarray,
    final_metadata: list[dict[str, Any]],
) -> dict[str, float]:
    train = [index for index, row in enumerate(development_metadata) if row["split"] == "train"]
    heldout = [index for index, row in enumerate(final_metadata) if row["split"] == "final_counterfactual"]
    y_final = np.asarray([int(final_metadata[index]["label"]) for index in heldout])
    layer = fit.logistic.layer
    output: dict[str, float] = {}
    for generator in sorted({development_metadata[index]["generator_family"] for index in train}):
        retained = [index for index in train if development_metadata[index]["generator_family"] != generator]
        x = development_features[retained, layer]
        y = np.asarray([int(development_metadata[index]["label"]) for index in retained])
        mean, scale = _standardizer(x)
        weight, bias = _fit_logistic((x - mean) / scale, y, fit.selected_c)
        score = ((final_features[heldout, layer] - mean) / scale) @ weight + bias
        output[generator] = float(score[y_final == 1].mean() - score[y_final == 0].mean())
    return output


__all__ = [
    "DevelopmentFit",
    "FrozenReadout",
    "binary_metrics",
    "bootstrap_direction_stability",
    "fit_source_development",
    "leave_one_generator_gaps",
    "load_development_fit",
    "pair_preserving_labels",
    "run_full_selection_permutations",
    "save_development_fit",
    "scores",
]
