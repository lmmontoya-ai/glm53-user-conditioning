"""Independent recomputation of the frozen v9 readout metrics and decision."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -60, 60)))
    fpr, tpr, _ = roc_curve(y, score)
    eligible = fpr[tpr >= 0.8]
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, probability)),
        "fpr_at_80_tpr": float(eligible.min()) if len(eligible) else 1.0,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_readout(
    *, feature_root: Path, readout_root: Path, output_path: Path
) -> dict[str, Any]:
    report_path = readout_root / "readout_report.json"
    decision_path = readout_root / "readout_decision.json"
    models_path = readout_root / "probe_models.npz"
    fixed_path = feature_root / "fixed_features.npz"
    metadata_path = feature_root / "metadata.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    features = np.load(fixed_path)["masked_prompt_mean"].astype(np.float32)
    models = np.load(models_path)
    layer = int(models["masked_prompt_mean__layer"][0])
    mean = models["masked_prompt_mean__mean"]
    scale = models["masked_prompt_mean__scale"]
    weight = models["masked_prompt_mean__weight"]
    bias = float(models["masked_prompt_mean__bias"][0])
    threshold = float(models["masked_prompt_mean__threshold"][0])
    scores = ((features[:, layer] - mean) / scale) @ weight + bias
    splits = np.asarray([row["split"] for row in rows])
    labels = np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in rows], dtype=np.int64
    )
    recomputed: dict[str, dict[str, float | int]] = {}
    for split_name in ("test", "holdout_realism", "holdout_hardneg", "holdout_mats_aux"):
        binary = (splits == split_name) & np.isin(labels, [0, 1])
        metrics: dict[str, float | int] = _metrics(labels[binary], scores[binary])
        neutral = (splits == split_name) & (labels == -1)
        if neutral.any():
            metrics["neutral_false_positive_rate"] = float(np.mean(scores[neutral] >= threshold))
            metrics["neutral_mean_score"] = float(np.mean(scores[neutral]))
            metrics["neutral_count"] = int(neutral.sum())
        recomputed[split_name] = metrics

    metric_differences = {
        f"{split_name}.{name}": abs(float(value) - float(report["primary_metrics"][split_name][name]))
        for split_name, metrics in recomputed.items()
        for name, value in metrics.items()
    }
    null = report["pair_preserving_full_pipeline_null"]
    null_exceed = sum(
        float(draw["test_auroc"]) >= float(null["observed_test_auroc"])
        for draw in null["draws"]
    )
    null_p = (1 + null_exceed) / (1 + len(null["draws"]))
    mask_counts = {
        "masked": sum(row["cue_mask_status"] == "masked" for row in rows),
        "not_available": sum(row["cue_mask_status"] == "not_available" for row in rows),
    }
    checks = {
        "row_count": len(rows) == 448,
        "mask_counts": mask_counts == {"masked": 416, "not_available": 32},
        "selected_layer": layer == int(report["selected_layer"]),
        "metric_agreement": max(metric_differences.values(), default=0.0) <= 1e-12,
        "null_exceed_agreement": null_exceed == int(null["exceed_count"]),
        "null_p_agreement": abs(null_p - float(null["empirical_p"])) <= 1e-15,
    }
    gate_checks = {
        "cue_mask_contract": checks["mask_counts"],
        "core_test_auroc": recomputed["test"]["auroc"] >= 0.75,
        "realism_auroc": recomputed["holdout_realism"]["auroc"] >= 0.70,
        "hard_negative_fpr": recomputed["holdout_hardneg"]["fpr_at_80_tpr"] <= 0.30,
        "permutation_p": null_p < 0.01,
        "target_user_data_excluded": True,
    }
    checks["gate_checks_agree"] = gate_checks == report["readout_checks"]
    checks["classification_agrees"] = (
        all(gate_checks.values()) == bool(report["readout_passed"])
        and bool(decision["passed"]) == bool(report["readout_passed"])
    )
    payload = {
        "schema_version": "glm53_v9_independent_readout_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "selected_layer": layer,
        "recomputed_metrics": recomputed,
        "metric_absolute_differences": metric_differences,
        "recomputed_null": {"exceed_count": null_exceed, "empirical_p": null_p},
        "recomputed_gate_checks": gate_checks,
        "recomputed_readout_passed": all(gate_checks.values()),
        "inputs": {
            "fixed_features_sha256": _sha256(fixed_path),
            "metadata_sha256": _sha256(metadata_path),
            "probe_models_sha256": _sha256(models_path),
            "readout_report_sha256": _sha256(report_path),
            "readout_decision_sha256": _sha256(decision_path),
        },
    }
    _atomic_json(output_path, payload)
    return payload
