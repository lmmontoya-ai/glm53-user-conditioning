"""Independent recomputation for the v10 offline audit.

This module deliberately does not import the primary v10 data, model, or
analysis modules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def _fpr80(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    eligible = fpr[tpr >= 0.8]
    return float(eligible.min()) if len(eligible) else 1.0


def _balanced_threshold(y: np.ndarray, score: np.ndarray) -> float:
    best = (-1.0, 0.0)
    for threshold in np.concatenate(([-np.inf], np.unique(score), [np.inf])):
        predicted = score >= threshold
        value = 0.5 * (float(predicted[y == 1].mean()) + float((~predicted[y == 0]).mean()))
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def _paired_model(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    candidates = []
    for layer in range(45):
        train_x = features[train, layer]
        val_x = features[val, layer]
        mean = train_x.mean(0)
        scale = train_x.std(0)
        scale[scale == 0] = 1.0
        train_z = (train_x - mean) / scale
        val_z = (val_x - mean) / scale
        class0 = train_z[labels[train] == 0].mean(0)
        class1 = train_z[labels[train] == 1].mean(0)
        weight = class1 - class0
        bias = -0.5 * float((class1 + class0) @ weight)
        val_score = val_z @ weight + bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(val_score, -60, 60)))
        brier = float(np.mean((probability - labels[val]) ** 2))
        candidates.append((-_auc(labels[val], val_score), brier, layer, mean, scale, weight, bias))
    _, _, layer, mean, scale, weight, bias = min(candidates, key=lambda item: item[:3])
    return layer, mean, scale, weight, bias


def _score_model(
    model: tuple[int, np.ndarray, np.ndarray, np.ndarray, float], features: np.ndarray
) -> np.ndarray:
    layer, mean, scale, weight, bias = model
    return ((features[:, layer] - mean) / scale) @ weight + bias


def _task_folds() -> list[list[int]]:
    ordered = sorted(
        range(1, 17),
        key=lambda task: hashlib.sha256(f"glm53-v10-task-fold|{task:02d}".encode()).hexdigest(),
    )
    return [ordered[start : start + 2] for start in range(0, 16, 2)]


def verify(*, repo_root: Path, analysis_path: Path, output_path: Path) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    feature_root = repo_root / "artifacts/glm53_user_eval/v9/features/eval_prompt"
    metadata = [
        json.loads(line)
        for line in (feature_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    with (repo_root / "artifacts/datasets/contrastive_prompts_v2/samples.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        source = {row["sample_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for row in metadata:
        governed = source[row["sample_id"]]
        match = re.search(r"__task_(\d+)$", governed["semantic_task_id"])
        rows.append(
            row
            | {
                "slice_id": governed["slice_id"],
                "prompt_role": governed["prompt_role"],
                "task_number": int(match.group(1)) if match else -1,
            }
        )
    with np.load(feature_root / "fixed_features.npz", allow_pickle=False) as archive:
        features = archive["masked_prompt_mean"].astype(np.float32)
    labels = np.asarray([row["label"] if row["label"] is not None else -1 for row in rows])
    split = np.asarray([row["split"] for row in rows])
    families = np.asarray([row["family_id"] for row in rows])
    tasks = np.asarray([row["task_number"] for row in rows])

    # Frozen v9 model recomputation.
    with np.load(
        repo_root / "artifacts/glm53_user_eval/v9/readout/probe_models.npz", allow_pickle=False
    ) as models:
        layer = int(models["masked_prompt_mean__layer"][0])
        v9_score = (
            (features[:, layer] - models["masked_prompt_mean__mean"])
            / models["masked_prompt_mean__scale"]
        ) @ models["masked_prompt_mean__weight"] + float(models["masked_prompt_mean__bias"][0])
        stored_threshold = float(models["masked_prompt_mean__threshold"][0])
    test = split == "test"
    hard = (split == "holdout_hardneg") & np.isin(labels, [0, 1])
    neutral = (split == "holdout_hardneg") & (labels == -1)
    v9 = {
        "selected_layer": layer,
        "test_auroc": _auc(labels[test], v9_score[test]),
        "hard_negative_auroc": _auc(labels[hard], v9_score[hard]),
        "hard_negative_fpr_at_80_tpr": _fpr80(labels[hard], v9_score[hard]),
        "neutral_false_positive_rate": float(np.mean(v9_score[neutral] >= stored_threshold)),
    }

    core = np.asarray([row["slice_id"] == "core_context_pairs" for row in rows])
    core_families = sorted(set(families[core]))
    lofo = []
    blocked = []
    task_folds = _task_folds()
    for test_family in core_families:
        for val_family in core_families:
            if test_family == val_family:
                continue
            train_mask = core & ~np.isin(families, [test_family, val_family])
            val_mask = core & (families == val_family)
            test_mask = core & (families == test_family)
            model = _paired_model(features, labels, train_mask, val_mask)
            lofo.append(_auc(labels[test_mask], _score_model(model, features[test_mask])))

            digest = hashlib.sha256(
                f"glm53-v10-family-task|{test_family}|{val_family}".encode()
            ).digest()
            test_fold = digest[0] % len(task_folds)
            val_fold = digest[1] % (len(task_folds) - 1)
            if val_fold >= test_fold:
                val_fold += 1
            test_tasks = task_folds[test_fold]
            val_tasks = task_folds[val_fold]
            excluded = test_tasks + val_tasks
            train_mask = (
                core & ~np.isin(families, [test_family, val_family]) & ~np.isin(tasks, excluded)
            )
            val_mask = core & (families == val_family) & np.isin(tasks, val_tasks)
            test_mask = core & (families == test_family) & np.isin(tasks, test_tasks)
            model = _paired_model(features, labels, train_mask, val_mask)
            blocked.append(_auc(labels[test_mask], _score_model(model, features[test_mask])))

    expected_v9 = analysis["fixed_split_trainer_comparison"]["v9_adamw"]
    checks = {
        "v9_test_auroc": abs(v9["test_auroc"] - expected_v9["test"]["auroc"]) <= 1e-12,
        "v9_hard_auroc": abs(v9["hard_negative_auroc"] - expected_v9["holdout_hardneg"]["auroc"])
        <= 1e-12,
        "v9_hard_fpr": abs(
            v9["hard_negative_fpr_at_80_tpr"] - expected_v9["holdout_hardneg"]["fpr_at_80_tpr"]
        )
        <= 1e-12,
        "lofo_median": abs(float(np.median(lofo)) - analysis["leave_family_out"]["median_auroc"])
        <= 1e-12,
        "lofo_minimum": abs(float(np.min(lofo)) - analysis["leave_family_out"]["minimum_auroc"])
        <= 1e-12,
        "blocked_median": abs(
            float(np.median(blocked)) - analysis["family_and_task_blocked"]["median_auroc"]
        )
        <= 1e-12,
        "blocked_minimum": abs(
            float(np.min(blocked)) - analysis["family_and_task_blocked"]["minimum_auroc"]
        )
        <= 1e-12,
    }
    report = {
        "schema_version": "glm53_v10_independent_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "independent_recomputation": {
            "v9": v9,
            "leave_family_out": {
                "fold_count": len(lofo),
                "median_auroc": float(np.median(lofo)),
                "minimum_auroc": float(np.min(lofo)),
            },
            "family_task_blocked": {
                "fold_count": len(blocked),
                "median_auroc": float(np.median(blocked)),
                "minimum_auroc": float(np.min(blocked)),
            },
        },
        "inputs": {
            "analysis": _sha256(analysis_path),
            "fixed_features": _sha256(feature_root / "fixed_features.npz"),
            "metadata": _sha256(feature_root / "metadata.jsonl"),
        },
    }
    _atomic_json(output_path, report)
    return report
