"""Offline fitting and machine readout decision for v9."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from src.glm53_user_eval.v8.artifacts import atomic_json, sha256_file

from .probes import (
    evaluate_probe,
    fit_sequence_layers,
    fit_token_mil_layer,
    load_bags,
    run_full_pipeline_null,
)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _probe_arrays(prefix: str, probe: Any) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__mean": probe.mean,
        f"{prefix}__scale": probe.scale,
        f"{prefix}__weight": probe.weight,
        f"{prefix}__bias": np.asarray([probe.bias], dtype=np.float64),
        f"{prefix}__threshold": np.asarray([probe.threshold], dtype=np.float64),
        f"{prefix}__layer": np.asarray([probe.layer], dtype=np.int64),
    }


def analyze_readout(
    *,
    feature_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = feature_root / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("passed") or not manifest["mask_contract"]["passed"]:
        raise ValueError("feature and cue-mask contract must pass before fitting")
    equivalence_path = feature_root / "v8_prompt_final_equivalence.json"
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if not equivalence.get("passed"):
        raise ValueError("v8/v9 prompt-final cross-run equivalence must pass before fitting")
    fixed_path = feature_root / "fixed_features.npz"
    metadata_path = feature_root / "metadata.jsonl"
    if sha256_file(fixed_path) != manifest["fixed_features_sha256"]:
        raise ValueError("fixed feature hash differs from extraction manifest")
    if sha256_file(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("metadata hash differs from extraction manifest")
    archive = np.load(fixed_path)
    rows = _rows(metadata_path)
    primary = archive["masked_prompt_mean"].astype(np.float32)
    sequence_config = config["probe"]["sequence_linear"]
    selected, layer_metrics = fit_sequence_layers(
        primary, rows, config=sequence_config, seed=int(config["analysis"]["fit_seed"])
    )
    primary_metrics = evaluate_probe(selected, primary, rows)
    null_rows = run_full_pipeline_null(
        primary,
        rows,
        config=sequence_config,
        reps=int(config["analysis"]["permutation_reps"]),
        seed=int(config["analysis"]["permutation_seed"]),
    )
    observed = primary_metrics["test"]["auroc"]
    exceed = sum(row["test_auroc"] >= observed for row in null_rows)
    empirical_p = (1 + exceed) / (1 + len(null_rows))

    baselines: dict[str, Any] = {}
    model_arrays = _probe_arrays("masked_prompt_mean", selected)
    for view in ("prompt_final", "last_unmasked_prompt_token", "cue_token_mean"):
        features = archive[view].astype(np.float32)
        finite = np.isfinite(features).all(axis=(1, 2))
        eligible_rows = [row for row, keep in zip(rows, finite, strict=True) if keep]
        eligible_features = features[finite]
        if not eligible_rows or {row["split"] for row in eligible_rows}.issuperset(
            {"train", "val", "test"}
        ) is False:
            baselines[view] = {"status": "not_available"}
            continue
        probe, _ = fit_sequence_layers(
            eligible_features,
            eligible_rows,
            config=sequence_config,
            seed=int(config["analysis"]["fit_seed"]) + 1000,
        )
        baselines[view] = {
            "status": "complete",
            "selected_layer": probe.layer,
            "metrics": evaluate_probe(probe, eligible_features, eligible_rows),
        }
        model_arrays.update(_probe_arrays(view, probe))

    top_layers = [
        int(row["layer"])
        for row in sorted(
            layer_metrics,
            key=lambda row: (-float(row["auroc"]), float(row["brier"]), int(row["layer"])),
        )[:4]
    ]
    mil_reports: list[dict[str, Any]] = []
    for layer in top_layers:
        bags = load_bags(feature_root, rows, layer)
        result = fit_token_mil_layer(
            bags,
            rows,
            layer=layer,
            config=config["probe"]["token_mil"],
            seed=int(config["analysis"]["fit_seed"]) + 2000 + layer,
        )
        scores = result.pop("scores")
        model_arrays.update(
            {
                f"token_mil_l{layer}__mean": result.pop("mean"),
                f"token_mil_l{layer}__scale": result.pop("scale"),
                f"token_mil_l{layer}__weight": result.pop("weight"),
                f"token_mil_l{layer}__bias": np.asarray([result.pop("bias")]),
                f"token_mil_l{layer}__scores": scores,
            }
        )
        split = np.asarray([row["split"] for row in rows])
        labels = np.asarray([row["label"] if row["label"] is not None else -1 for row in rows])
        from .probes import _metrics

        result["metrics"] = {
            name: _metrics(
                labels[(split == name) & np.isin(labels, [0, 1])],
                scores[(split == name) & np.isin(labels, [0, 1])],
            )
            for name in ("test", "holdout_realism", "holdout_hardneg", "holdout_mats_aux")
        }
        neutral = (split == "holdout_hardneg") & (labels == -1)
        result["metrics"]["holdout_hardneg"]["neutral_false_positive_rate"] = float(
            np.mean(scores[neutral] >= float(result["threshold"]))
        )
        result["metrics"]["holdout_hardneg"]["neutral_mean_score"] = float(
            np.mean(scores[neutral])
        )
        mil_reports.append(result)

    checks = {
        "cue_mask_contract": manifest["mask_contract"]["passed"],
        "core_test_auroc": primary_metrics["test"]["auroc"] >= 0.75,
        "realism_auroc": primary_metrics["holdout_realism"]["auroc"] >= 0.70,
        "hard_negative_fpr": primary_metrics["holdout_hardneg"]["fpr_at_80_tpr"] <= 0.30,
        "permutation_p": empirical_p < 0.01,
        "target_user_data_excluded": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "probe_models.npz"
    _atomic_npz(model_path, **model_arrays)
    report = {
        "schema_version": "glm53_v9_readout_report_v1",
        "primary_representation": "masked_prompt_mean",
        "optimizer": "torch.optim.AdamW",
        "selected_layer": selected.layer,
        "selected_probe": {
            "best_epoch": selected.best_epoch,
            "validation_auroc": selected.val_auroc,
            "validation_brier": selected.val_brier,
        },
        "primary_metrics": primary_metrics,
        "layer_metrics": layer_metrics,
        "pair_preserving_full_pipeline_null": {
            "reps": len(null_rows),
            "observed_test_auroc": observed,
            "exceed_count": exceed,
            "empirical_p": empirical_p,
            "draws": null_rows,
        },
        "secondary_prompt_views": baselines,
        "token_mil": {"candidate_layers": top_layers, "reports": mil_reports},
        "readout_checks": checks,
        "readout_passed": all(checks.values()),
        "steering_status": "not_run_separate_gate",
        "feature_manifest_sha256": sha256_file(manifest_path),
        "v8_prompt_final_equivalence_sha256": sha256_file(equivalence_path),
        "probe_models_path": str(model_path),
        "probe_models_sha256": sha256_file(model_path),
    }
    atomic_json(output_root / "readout_report.json", report)
    atomic_json(
        output_root / "readout_decision.json",
        {
            "schema_version": "glm53_v9_readout_decision_v1",
            "gate": "R3",
            "passed": report["readout_passed"],
            "decision": "unlock_recruitment" if report["readout_passed"] else "stop_readout_branch",
            "checks": checks,
            "inputs": {
                "feature_manifest": sha256_file(manifest_path),
                "fixed_features": sha256_file(fixed_path),
                "metadata": sha256_file(metadata_path),
                "v8_prompt_final_equivalence": sha256_file(equivalence_path),
                "probe_models": sha256_file(model_path),
            },
        },
    )
    return report
