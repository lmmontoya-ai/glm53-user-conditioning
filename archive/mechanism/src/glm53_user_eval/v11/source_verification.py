"""Import-isolated recomputation of the v11 source-activation result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FACTORIAL_COLUMNS = (
    "intercept",
    "real_external_recipient",
    "operational_consequence",
    "model_scoring",
    "comparison_among_models",
    "formal_register",
)
FACTORIAL_DECISIVE_COLUMNS = FACTORIAL_COLUMNS[1:5]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_partition(
    feature_root: Path,
    manifest: dict[str, Any],
    partition: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    record = manifest["partitions"][partition]
    feature_path = feature_root / record["features"]
    metadata_path = feature_root / record["metadata"]
    if _sha256(feature_path) != record["features_sha256"]:
        raise ValueError(f"independent {partition} feature hash differs")
    if _sha256(metadata_path) != record["metadata_sha256"]:
        raise ValueError(f"independent {partition} metadata hash differs")
    with np.load(feature_path) as archive:
        features = archive["shared_task_suffix_mean"].astype(np.float32)
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if features.shape != (len(metadata), 45, 4096):
        raise ValueError(f"independent {partition} feature shape differs: {features.shape}")
    if len({str(row["sample_id"]) for row in metadata}) != len(metadata):
        raise ValueError(f"independent {partition} metadata IDs are not unique")
    return features, metadata


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if set(labels.tolist()) != {0, 1} or labels.shape != scores.shape:
        raise ValueError("independent binary metric inputs violate the contract")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= 0.80)
    index = int(eligible[0]) if len(eligible) else len(fpr) - 1
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "fpr_at_80_tpr": float(fpr[index]),
        "threshold_at_80_tpr": float(thresholds[index]),
        "score_gap": float(scores[labels == 1].mean() - scores[labels == 0].mean()),
    }


def _independent_factorial_analysis(
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (32,) or len(metadata) != 32:
        raise ValueError("independent factorial calibration row count differs")
    if any(row.get("split") != "factorial_calibration" for row in metadata):
        raise ValueError("independent factorial partition contains another split")
    if any(row.get("label") is not None for row in metadata):
        raise ValueError("independent factorial rows have binary labels")

    design_rows: list[list[float]] = []
    cells: list[tuple[bool, bool, bool, bool]] = []
    registers: list[bool] = []
    for row in metadata:
        factors = row.get("factors")
        if not isinstance(factors, dict):
            raise TypeError("independent factorial row lacks factors")
        try:
            cell = tuple(bool(factors[column]) for column in FACTORIAL_DECISIVE_COLUMNS)
        except KeyError as error:
            raise ValueError("independent factorial row lacks a decisive factor") from error
        register = row.get("register")
        if register not in {"formal", "casual"}:
            raise ValueError("independent factorial row has an invalid register")
        formal = register == "formal"
        cells.append(cell)
        registers.append(formal)
        design_rows.append([1.0, *(float(value) for value in cell), float(formal)])

    cell_counts = {cell: cells.count(cell) for cell in set(cells)}
    if len(cell_counts) != 16 or set(cell_counts.values()) != {2}:
        raise ValueError("independent factorial cell coverage differs")
    if any(
        {
            registers[index]
            for index, candidate in enumerate(cells)
            if candidate == cell
        }
        != {False, True}
        for cell in cell_counts
    ):
        raise ValueError("independent factorial register counterrotation differs")

    design = np.asarray(design_rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) != len(FACTORIAL_COLUMNS):
        raise ValueError("independent factorial design is rank deficient")
    coefficient, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficient
    dof = int(len(values) - design.shape[1])
    residual_variance = float((residual @ residual) / dof)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    critical = float(student_t.ppf(0.975, dof))
    coefficients = {
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
    clean_eval = ~recipient & ~consequence & (scoring | comparison)
    clean_deploy = (recipient | consequence) & ~scoring & ~comparison
    if int(clean_eval.sum()) != 6 or int(clean_deploy.sum()) != 6:
        raise ValueError("independent factorial clean-cell counts differ")
    contrast = np.asarray(
        [0.0, -2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]
    )
    ols_difference = float(contrast @ coefficient)
    ols_contrast_se = float(np.sqrt(max(float(contrast @ covariance @ contrast), 0.0)))
    evaluation_values = values[clean_eval]
    deployment_values = values[clean_deploy]
    difference = float(evaluation_values.mean() - deployment_values.mean())
    evaluation_variance_term = float(evaluation_values.var(ddof=1) / len(evaluation_values))
    deployment_variance_term = float(deployment_values.var(ddof=1) / len(deployment_values))
    contrast_se = float(np.sqrt(evaluation_variance_term + deployment_variance_term))
    if contrast_se == 0:
        contrast_dof = float(len(evaluation_values) + len(deployment_values) - 2)
        contrast_critical = float(student_t.ppf(0.975, contrast_dof))
    else:
        contrast_dof = float(
            (evaluation_variance_term + deployment_variance_term) ** 2
            / (
                evaluation_variance_term**2 / (len(evaluation_values) - 1)
                + deployment_variance_term**2 / (len(deployment_values) - 1)
            )
        )
        contrast_critical = float(student_t.ppf(0.975, contrast_dof))
    decisive = {column: coefficients[column]["estimate"] for column in FACTORIAL_DECISIVE_COLUMNS}
    register_estimate = coefficients["formal_register"]["estimate"]
    checks = {
        "real_external_recipient_negative": decisive["real_external_recipient"] < 0,
        "operational_consequence_negative": decisive["operational_consequence"] < 0,
        "model_scoring_positive": decisive["model_scoring"] > 0,
        "comparison_among_models_positive": decisive["comparison_among_models"] > 0,
        "clean_evaluation_mean_above_clean_deployment": difference > 0,
        "register_smaller_than_every_decisive_factor": abs(register_estimate)
        < min(abs(value) for value in decisive.values()),
    }
    return {
        "row_count": len(metadata),
        "design_columns": list(FACTORIAL_COLUMNS),
        "coding": "zero_one_with_intercept",
        "degrees_of_freedom": dof,
        "residual_standard_deviation": float(np.sqrt(residual_variance)),
        "coefficients": coefficients,
        "clean_cell_contrast": {
            "clean_evaluation_count": int(clean_eval.sum()),
            "clean_deployment_count": int(clean_deploy.sum()),
            "clean_evaluation_mean": float(evaluation_values.mean()),
            "clean_deployment_mean": float(deployment_values.mean()),
            "difference": difference,
            "standard_error": contrast_se,
            "degrees_of_freedom": contrast_dof,
            "ci95": [
                float(difference - contrast_critical * contrast_se),
                float(difference + contrast_critical * contrast_se),
            ],
            "ols_main_effect_contrast": {
                "difference": ols_difference,
                "standard_error": ols_contrast_se,
                "ci95": [
                    float(ols_difference - critical * ols_contrast_se),
                    float(ols_difference + critical * ols_contrast_se),
                ],
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _verify_factorial_report(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    model_name: str,
) -> None:
    for key in ("row_count", "design_columns", "coding", "degrees_of_freedom", "checks", "passed"):
        if actual[key] != expected[key]:
            raise ValueError(f"independent {model_name} factorial {key} differs")
    _require_close(
        float(actual["residual_standard_deviation"]),
        float(expected["residual_standard_deviation"]),
        field=f"{model_name} factorial residual standard deviation",
    )
    for column in FACTORIAL_COLUMNS:
        for metric in ("estimate", "standard_error"):
            _require_close(
                float(actual["coefficients"][column][metric]),
                float(expected["coefficients"][column][metric]),
                field=f"{model_name} factorial {column} {metric}",
            )
        for index in range(2):
            _require_close(
                float(actual["coefficients"][column]["ci95"][index]),
                float(expected["coefficients"][column]["ci95"][index]),
                field=f"{model_name} factorial {column} ci95[{index}]",
            )
    for metric in (
        "clean_evaluation_mean",
        "clean_deployment_mean",
        "difference",
        "standard_error",
        "degrees_of_freedom",
    ):
        _require_close(
            float(actual["clean_cell_contrast"][metric]),
            float(expected["clean_cell_contrast"][metric]),
            field=f"{model_name} factorial clean contrast {metric}",
        )
    for key in ("clean_evaluation_count", "clean_deployment_count"):
        if actual["clean_cell_contrast"][key] != expected["clean_cell_contrast"][key]:
            raise ValueError(f"independent {model_name} factorial {key} differs")
    for index in range(2):
        _require_close(
            float(actual["clean_cell_contrast"]["ci95"][index]),
            float(expected["clean_cell_contrast"]["ci95"][index]),
            field=f"{model_name} factorial clean contrast ci95[{index}]",
        )
    for metric in ("difference", "standard_error"):
        _require_close(
            float(actual["clean_cell_contrast"]["ols_main_effect_contrast"][metric]),
            float(expected["clean_cell_contrast"]["ols_main_effect_contrast"][metric]),
            field=f"{model_name} factorial OLS clean contrast {metric}",
        )
    for index in range(2):
        _require_close(
            float(actual["clean_cell_contrast"]["ols_main_effect_contrast"]["ci95"][index]),
            float(expected["clean_cell_contrast"]["ols_main_effect_contrast"]["ci95"][index]),
            field=f"{model_name} factorial OLS clean contrast ci95[{index}]",
        )


def _require_close(actual: float, expected: float, *, field: str, atol: float = 1e-10) -> None:
    if np.isnan(actual) and np.isnan(expected):
        return
    if np.isinf(actual) and actual == expected:
        return
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > atol:
        raise ValueError(f"independent {field} differs: {actual} != {expected}")


def _score_rows(
    *,
    model_name: str,
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    layer: int,
    mean: np.ndarray,
    scale: np.ndarray,
    weight: np.ndarray,
    bias: float,
    reported: list[dict[str, Any]],
) -> np.ndarray:
    if mean.shape != (4096,) or scale.shape != (4096,) or weight.shape != (4096,):
        raise ValueError(f"independent {model_name} frozen-array shape differs")
    if not 0 <= layer < 45 or np.any(scale <= 0):
        raise ValueError(f"independent {model_name} scaler/layer contract differs")
    scores = ((features[:, layer] - mean) / scale) @ weight + bias
    canonical = {str(row["sample_id"]): (index, row) for index, row in enumerate(metadata)}
    if len(reported) != len(metadata) or len({str(row["sample_id"]) for row in reported}) != len(
        reported
    ):
        raise ValueError(f"independent {model_name} score-row count or uniqueness differs")
    if {str(row["sample_id"]) for row in reported} != set(canonical):
        raise ValueError(f"independent {model_name} score-row IDs differ from immutable metadata")
    for row in reported:
        sample_id = str(row["sample_id"])
        index, source = canonical[sample_id]
        if row["split"] != source["split"] or row["label"] != source["label"]:
            raise ValueError(f"independent {model_name} score-row metadata differs for {sample_id}")
        _require_close(
            float(scores[index]),
            float(row["score"]),
            field=f"{model_name} score for {sample_id}",
            atol=1e-6,
        )
    return scores


def _recompute_stability(
    development_features: np.ndarray,
    development_metadata: list[dict[str, Any]],
    *,
    layer: int,
    full_direction: np.ndarray,
    reps: int,
    seed: int,
) -> np.ndarray:
    train_indices = [
        index for index, row in enumerate(development_metadata) if row["split"] == "train"
    ]
    by_pair: dict[str, list[int]] = {}
    for index in train_indices:
        by_pair.setdefault(str(development_metadata[index]["pair_id"]), []).append(index)
    for pair_id, indices in by_pair.items():
        labels = sorted(int(development_metadata[index]["label"]) for index in indices)
        if len(indices) != 2 or labels != [0, 1]:
            raise ValueError(f"independent stability found incomplete pair {pair_id}")
    pair_ids = sorted(by_pair)
    rng = np.random.default_rng(seed)
    cosines = np.empty(reps, dtype=np.float64)
    for repetition in range(reps):
        sampled_ids = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sampled_indices = [index for pair_id in sampled_ids for index in by_pair[str(pair_id)]]
        sampled_x = development_features[sampled_indices, layer]
        sampled_y = np.asarray(
            [int(development_metadata[index]["label"]) for index in sampled_indices]
        )
        mean = sampled_x.mean(0)
        scale = sampled_x.std(0)
        scale[scale == 0] = 1.0
        standardized = (sampled_x - mean) / scale
        weight = standardized[sampled_y == 1].mean(0) - standardized[sampled_y == 0].mean(0)
        raw = weight / scale
        norm = float(np.linalg.norm(raw))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("independent stability produced an invalid direction")
        cosines[repetition] = float((raw / norm) @ full_direction)
    return cosines


def _recompute_leave_one_generator(
    development_features: np.ndarray,
    development_metadata: list[dict[str, Any]],
    final_features: np.ndarray,
    final_metadata: list[dict[str, Any]],
    *,
    layer: int,
    c_value: float,
) -> dict[str, float]:
    train_indices = [
        index for index, row in enumerate(development_metadata) if row["split"] == "train"
    ]
    heldout_indices = [
        index for index, row in enumerate(final_metadata) if row["split"] == "final_counterfactual"
    ]
    heldout_y = np.asarray([int(final_metadata[index]["label"]) for index in heldout_indices])
    output: dict[str, float] = {}
    generators = sorted({development_metadata[index]["generator_family"] for index in train_indices})
    for generator in generators:
        retained = [
            index
            for index in train_indices
            if development_metadata[index]["generator_family"] != generator
        ]
        train_x = development_features[retained, layer]
        train_y = np.asarray([int(development_metadata[index]["label"]) for index in retained])
        mean = train_x.mean(0)
        scale = train_x.std(0)
        scale[scale == 0] = 1.0
        classifier = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=0,
        ).fit((train_x - mean) / scale, train_y)
        score = classifier.decision_function(
            (final_features[heldout_indices, layer] - mean) / scale
        )
        output[str(generator)] = float(
            score[heldout_y == 1].mean() - score[heldout_y == 0].mean()
        )
    return output


def _verify_permutations(
    *,
    source_root: Path,
    feature_manifest_sha256: str,
    readout_lock_sha256: str,
    config_sha256: str,
) -> tuple[dict[str, Any], float]:
    permutation_path = source_root / "permutation_analysis.json"
    rows_path = source_root / "permutation_rows.jsonl"
    manifest_path = rows_path.with_suffix(rows_path.suffix + ".manifest.json")
    permutation = json.loads(permutation_path.read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(rows_path) != permutation["checkpoint_sha256"]:
        raise ValueError("independent permutation checkpoint hash differs")
    if _sha256(manifest_path) != permutation["checkpoint_manifest_sha256"]:
        raise ValueError("independent permutation checkpoint-manifest hash differs")
    expected_binding = {
        "config_sha256": config_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "readout_lock_sha256": readout_lock_sha256,
    }
    if checkpoint_manifest.get("artifact_binding") != expected_binding:
        raise ValueError("independent permutation artifact binding differs")
    contract = {key: value for key, value in checkpoint_manifest.items() if key != "contract_sha256"}
    contract_sha = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if contract_sha != checkpoint_manifest.get("contract_sha256") or contract_sha != permutation.get(
        "checkpoint_contract_sha256"
    ):
        raise ValueError("independent permutation checkpoint contract differs")
    rows = [
        json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    reps = int(permutation["reps"])
    seed = int(permutation["seed"])
    if int(checkpoint_manifest["reps"]) != reps or int(checkpoint_manifest["seed"]) != seed:
        raise ValueError("independent permutation reps/seed differ")
    if len(rows) != reps or {int(row["repetition"]) for row in rows} != set(range(reps)):
        raise ValueError("independent permutation rows are incomplete")
    for row in rows:
        repetition = int(row["repetition"])
        if int(row["seed"]) != seed + repetition or row.get("contract_sha256") != contract_sha:
            raise ValueError("independent permutation row binding differs")
    exceedances = sum(
        float(row["objective"]) >= float(permutation["observed_objective"]) for row in rows
    )
    empirical_p = (1 + exceedances) / (1 + reps)
    _require_close(
        empirical_p,
        float(permutation["add_one_empirical_p"]),
        field="permutation p-value",
        atol=1e-12,
    )
    return permutation, empirical_p


def verify_source_result(
    *,
    source_root: Path,
    feature_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Recompute the frozen result; ``passed`` means integrity agreement only."""

    analysis_path = source_root / "source_final_analysis.json"
    readout_path = source_root / "source_readout_lock.json"
    arrays_path = source_root / "source_readout_arrays.npz"
    feature_manifest_path = feature_root / "feature_manifest.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if _sha256(arrays_path) != readout["arrays_sha256"]:
        raise ValueError("independent source array hash differs")
    development_features, development_metadata = _load_partition(
        feature_root, feature_manifest, "development"
    )
    final_features, final_metadata = _load_partition(feature_root, feature_manifest, "final")
    calibration_features, calibration_metadata = _load_partition(
        feature_root, feature_manifest, "calibration"
    )
    if {row["split"] for row in final_metadata} != {
        "ordinary_test",
        "final_counterfactual",
        "neutral_controls",
    }:
        raise ValueError("independent final partition split contract differs")

    with np.load(arrays_path) as arrays:
        frozen = {
            "logistic": {
                "mean": arrays["logistic_mean"].copy(),
                "scale": arrays["logistic_scale"].copy(),
                "weight": arrays["logistic_weight"].copy(),
            },
            "paired_mean": {
                "mean": arrays["paired_mean"].copy(),
                "scale": arrays["paired_scale"].copy(),
                "weight": arrays["paired_weight"].copy(),
            },
        }
    layer = int(readout["selected_layer"])
    labels = np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in final_metadata],
        dtype=np.int64,
    )
    splits = np.asarray([row["split"] for row in final_metadata])
    recomputed: dict[str, Any] = {}
    model_scores: dict[str, np.ndarray] = {}
    raw_directions: dict[str, np.ndarray] = {}
    for name in ("logistic", "paired_mean"):
        model = frozen[name]
        score = _score_rows(
            model_name=name,
            features=final_features,
            metadata=final_metadata,
            layer=layer,
            mean=model["mean"],
            scale=model["scale"],
            weight=model["weight"],
            bias=float(readout[name]["bias"]),
            reported=analysis["models"][name]["score_rows"],
        )
        model_scores[name] = score
        ordinary = splits == "ordinary_test"
        heldout = splits == "final_counterfactual"
        neutral = splits == "neutral_controls"
        ordinary_metrics = _binary_metrics(labels[ordinary], score[ordinary])
        heldout_metrics = _binary_metrics(labels[heldout], score[heldout])
        for metric, value in ordinary_metrics.items():
            _require_close(
                value,
                float(analysis["models"][name]["ordinary_test"][metric]),
                field=f"{name} ordinary {metric}",
            )
        for metric, value in heldout_metrics.items():
            _require_close(
                value,
                float(analysis["models"][name]["final_counterfactual"][metric]),
                field=f"{name} final {metric}",
            )
        threshold = float(readout[name]["threshold_80_tpr"])
        neutral_fpr = {
            family: float(
                np.mean(
                    score[
                        neutral
                        & np.asarray(
                            [row["generator_family"] == family for row in final_metadata]
                        )
                    ]
                    >= threshold
                )
            )
            for family in sorted(
                {
                    row["generator_family"]
                    for row in final_metadata
                    if row["split"] == "neutral_controls"
                }
            )
        }
        if neutral_fpr != analysis["models"][name]["final_neutral_fpr_by_family"]:
            raise ValueError(f"independent neutral FPR differs for {name}")
        calibration_score = _score_rows(
            model_name=f"{name} factorial",
            features=calibration_features,
            metadata=calibration_metadata,
            layer=layer,
            mean=model["mean"],
            scale=model["scale"],
            weight=model["weight"],
            bias=float(readout[name]["bias"]),
            reported=analysis["models"][name]["factorial_calibration"]["score_rows"],
        )
        factorial = _independent_factorial_analysis(calibration_score, calibration_metadata)
        reported_factorial = {
            key: value
            for key, value in analysis["models"][name]["factorial_calibration"].items()
            if key != "score_rows"
        }
        _verify_factorial_report(
            factorial,
            reported_factorial,
            model_name=name,
        )
        raw = model["weight"] / model["scale"]
        raw_directions[name] = raw / np.linalg.norm(raw)
        recomputed[name] = {
            "ordinary_test": ordinary_metrics,
            "final_counterfactual": heldout_metrics,
            "neutral_fpr": neutral_fpr,
            "factorial_calibration": factorial,
        }

    heldout = splits == "final_counterfactual"
    raw_cosine = float(raw_directions["logistic"] @ raw_directions["paired_mean"])
    rank = float(
        spearmanr(model_scores["logistic"][heldout], model_scores["paired_mean"][heldout]).statistic
    )
    _require_close(
        raw_cosine,
        float(analysis["direction_agreement"]["raw_cosine"]),
        field="raw direction cosine",
    )
    _require_close(
        rank,
        float(analysis["direction_agreement"]["final_score_spearman"]),
        field="final rank agreement",
    )

    stability = analysis["paired_direction_stability"]
    cosines = _recompute_stability(
        development_features,
        development_metadata,
        layer=layer,
        full_direction=raw_directions["paired_mean"],
        reps=int(stability["reps"]),
        seed=int(stability["seed"]),
    )
    reported_cosines = np.asarray(stability["cosines"], dtype=np.float64)
    if reported_cosines.shape != cosines.shape or not np.allclose(
        reported_cosines, cosines, rtol=0.0, atol=1e-10
    ):
        raise ValueError("independent stability draws differ")
    fifth = float(np.percentile(cosines, 5))
    _require_close(
        fifth,
        float(stability["fifth_percentile_cosine"]),
        field="stability fifth percentile",
    )

    leave_one = _recompute_leave_one_generator(
        development_features,
        development_metadata,
        final_features,
        final_metadata,
        layer=layer,
        c_value=float(readout["selected_C"]),
    )
    reported_leave_one = analysis["leave_one_training_generator_score_gaps"]
    if set(leave_one) != set(reported_leave_one):
        raise ValueError("independent leave-one-generator keys differ")
    for generator, value in leave_one.items():
        _require_close(
            value,
            float(reported_leave_one[generator]),
            field=f"leave-one-generator gap {generator}",
            atol=1e-8,
        )

    feature_manifest_sha256 = _sha256(feature_manifest_path)
    readout_lock_sha256 = _sha256(readout_path)
    config_sha256 = _sha256(config_path)
    _, empirical_p = _verify_permutations(
        source_root=source_root,
        feature_manifest_sha256=feature_manifest_sha256,
        readout_lock_sha256=readout_lock_sha256,
        config_sha256=config_sha256,
    )

    scientific_checks = {
        "ordinary_test_auroc_ge_080": recomputed["logistic"]["ordinary_test"]["auroc"]
        >= 0.80,
        "final_counterfactual_auroc_ge_075": recomputed["logistic"]["final_counterfactual"][
            "auroc"
        ]
        >= 0.75,
        "final_counterfactual_fpr80_le_025": recomputed["logistic"]["final_counterfactual"][
            "fpr_at_80_tpr"
        ]
        <= 0.25,
        "formal_casual_neutral_fpr_le_020": recomputed["logistic"]["neutral_fpr"][
            "formal_casual_neutral"
        ]
        <= 0.20,
        "ai_human_neutral_fpr_le_020": recomputed["logistic"]["neutral_fpr"][
            "ai_human_neutral"
        ]
        <= 0.20,
        "directions_cosine_ge_050": raw_cosine >= 0.50,
        "final_score_spearman_ge_060": rank >= 0.60,
        "logistic_gap_positive": recomputed["logistic"]["final_counterfactual"][
            "score_gap"
        ]
        > 0,
        "paired_mean_gap_positive": recomputed["paired_mean"]["final_counterfactual"][
            "score_gap"
        ]
        > 0,
        "stability_fifth_percentile_gt_050": fifth > 0.50,
        "full_selection_permutation_p_lt_001": empirical_p < 0.01,
        "every_leave_one_generator_gap_positive": bool(leave_one)
        and all(value > 0 for value in leave_one.values()),
        "final_rows_did_not_select": analysis.get("selection_used_final_rows") is False,
        "calibration_rows_did_not_select": analysis.get("selection_used_calibration_rows")
        is False,
    }
    for model_name in ("logistic", "paired_mean"):
        factorial = recomputed[model_name]["factorial_calibration"]
        coefficients = factorial["coefficients"]
        decisive = {
            key: float(coefficients[key]["estimate"])
            for key in FACTORIAL_DECISIVE_COLUMNS
        }
        prefix = f"{model_name}_calibration"
        scientific_checks.update(
            {
                f"{prefix}_recipient_negative": decisive["real_external_recipient"] < 0,
                f"{prefix}_consequence_negative": decisive["operational_consequence"] < 0,
                f"{prefix}_scoring_positive": decisive["model_scoring"] > 0,
                f"{prefix}_comparison_positive": decisive["comparison_among_models"] > 0,
                f"{prefix}_clean_eval_gt_deploy": float(
                    factorial["clean_cell_contrast"]["difference"]
                )
                > 0,
                f"{prefix}_register_subordinate": abs(
                    float(coefficients["formal_register"]["estimate"])
                )
                < min(abs(value) for value in decisive.values()),
            }
        )
    return {
        "schema_version": "glm53_v11_source_independent_verification_v3",
        "passed": True,
        "integrity_agreement": True,
        "scientific_gate_would_pass": all(scientific_checks.values()),
        "scientific_checks": scientific_checks,
        "recomputed": recomputed,
        "raw_direction_cosine": raw_cosine,
        "final_score_spearman": rank,
        "stability_fifth_percentile": fifth,
        "leave_one_training_generator_score_gaps": leave_one,
        "permutation_p": empirical_p,
        "inputs": {
            "analysis": _sha256(analysis_path),
            "readout": readout_lock_sha256,
            "readout_arrays": _sha256(arrays_path),
            "feature_manifest": feature_manifest_sha256,
            "config": config_sha256,
        },
    }


__all__ = ["verify_source_result"]
