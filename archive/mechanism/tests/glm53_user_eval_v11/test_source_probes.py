from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import src.glm53_user_eval.v11.probes as probes_module
from src.glm53_user_eval.v11.probes import (
    DevelopmentFit,
    FrozenLinear,
    _fit_source_development_prepared,
    _prepare_development,
    factorial_calibration_analysis,
    fit_source_development,
    load_development_fit,
    pair_preserving_labels,
    run_full_selection_permutations,
    save_development_fit,
)
from src.glm53_user_eval.v11.source_decision import decide_source_instrument
from src.glm53_user_eval.v11.source_verification import (
    _binary_metrics,
    _recompute_leave_one_generator,
    verify_source_result,
)


def small_metadata() -> list[dict]:
    rows = []
    for pair in range(4):
        for label in (0, 1):
            rows.append(
                {
                    "pair_id": f"pair-{pair}",
                    "label": label,
                    "split": "train" if pair < 2 else "validation",
                }
            )
    rows.append({"pair_id": "neutral", "label": None, "split": "neutral_controls"})
    return rows


def test_pair_preserving_null_keeps_one_label_of_each_kind_per_pair() -> None:
    metadata = small_metadata()
    shuffled = pair_preserving_labels(metadata, seed=42)
    for pair in range(4):
        indices = [index for index, row in enumerate(metadata) if row["pair_id"] == f"pair-{pair}"]
        assert sorted(shuffled[indices].tolist()) == [0, 1]
    assert shuffled[-1] == -1


def deterministic_development_fixture() -> tuple[np.ndarray, list[dict]]:
    metadata = []
    split_pairs = {
        "train": 2,
        "validation": 1,
        "development_counterfactual": 1,
    }
    pair_number = 0
    for split, count in split_pairs.items():
        for _ in range(count):
            for label in (0, 1):
                metadata.append(
                    {
                        "pair_id": f"pair-{pair_number}",
                        "label": label,
                        "split": split,
                    }
                )
            pair_number += 1
    for neutral in range(2):
        metadata.append(
            {
                "pair_id": f"neutral-{neutral}",
                "label": None,
                "split": "neutral_controls",
            }
        )
    rng = np.random.default_rng(20260912)
    features = rng.normal(size=(len(metadata), 45, 4096)).astype(np.float32)
    labels = np.asarray(
        [row["label"] if row["label"] is not None else -1 for row in metadata]
    )
    features[:, :, :8] += labels[:, None, None].clip(min=0) * 0.15
    return features, metadata


def test_prestandardized_full_selection_is_exactly_reference_equivalent() -> None:
    features, metadata = deterministic_development_fixture()
    labels = pair_preserving_labels(metadata, seed=91)
    c_grid = (0.01, 0.1)
    reference = fit_source_development(
        features,
        metadata,
        labels_override=labels,
        c_grid=c_grid,
    )
    prepared = _prepare_development(features, metadata)
    optimized = _fit_source_development_prepared(
        prepared,
        prepared.permuted_labels(seed=91),
        c_grid=c_grid,
    )
    compact = _fit_source_development_prepared(
        prepared,
        prepared.permuted_labels(seed=91),
        c_grid=c_grid,
        compact_report=True,
    )

    assert np.array_equal(labels, prepared.permuted_labels(seed=91))
    assert optimized.report == reference.report
    assert optimized.selected_c == reference.selected_c
    assert optimized.objective == reference.objective
    assert optimized.logistic.layer == reference.logistic.layer
    assert optimized.logistic.bias == reference.logistic.bias
    assert optimized.logistic.threshold_80_tpr == reference.logistic.threshold_80_tpr
    assert np.array_equal(optimized.logistic.mean, reference.logistic.mean)
    assert np.array_equal(optimized.logistic.scale, reference.logistic.scale)
    assert np.array_equal(optimized.logistic.weight, reference.logistic.weight)
    assert np.array_equal(optimized.paired_mean.weight, reference.paired_mean.weight)
    assert compact.selected_c == reference.selected_c
    assert compact.objective == reference.objective
    assert compact.logistic.layer == reference.logistic.layer
    assert compact.logistic.bias == reference.logistic.bias
    assert np.array_equal(compact.logistic.weight, reference.logistic.weight)
    assert np.array_equal(compact.paired_mean.weight, reference.paired_mean.weight)
    search = compact.report["exact_search_optimization"]
    assert search["evaluated_candidate_count"] < search["total_candidate_count"]
    assert search["pruned_layer_count"] > 0
    assert prepared.standardized_bytes <= probes_module.MAX_PREPARED_STANDARDIZED_BYTES
    assert prepared.layers[0].train.flags.writeable is False


def test_prestandardized_matrices_fail_before_exceeding_memory_cap(
    monkeypatch,
) -> None:
    features, metadata = deterministic_development_fixture()
    monkeypatch.setattr(probes_module, "MAX_PREPARED_STANDARDIZED_BYTES", 1)
    with pytest.raises(MemoryError, match="fixed memory bound"):
        _prepare_development(features, metadata)


def test_readout_lock_roundtrip_preserves_raw_directions(tmp_path) -> None:
    mean = np.arange(4, dtype=np.float32)
    scale = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    logistic = FrozenLinear(
        "ridge_logistic", 3, mean, scale, np.ones(4, dtype=np.float32), 0.2, -0.1
    )
    paired = FrozenLinear(
        "paired_mean", 3, mean, scale, np.asarray([1, -1, 2, -2], dtype=np.float32), 0.0, 0.0
    )
    fit = DevelopmentFit(
        logistic,
        paired,
        0.1,
        0.8,
        {
            "readout_lock_sha256": "locked",
            "selected_layer": 3,
            "selected_C": 0.1,
            "objective": 0.8,
        },
    )
    save_development_fit(fit, tmp_path)
    loaded = load_development_fit(tmp_path)
    assert loaded.logistic.layer == 3
    assert np.allclose(loaded.logistic.raw_direction, logistic.raw_direction)
    assert np.allclose(loaded.paired_mean.raw_direction, paired.raw_direction)


def passing_factorial_calibration() -> dict:
    return {
        "coefficients": {
            "real_external_recipient": {"estimate": -1.0},
            "operational_consequence": {"estimate": -1.2},
            "model_scoring": {"estimate": 1.4},
            "comparison_among_models": {"estimate": 1.6},
            "formal_register": {"estimate": 0.1},
        },
        "clean_cell_contrast": {"difference": 3.0},
    }


def passing_analysis() -> dict:
    logistic = {
        "ordinary_test": {"auroc": 0.9},
        "final_counterfactual": {"auroc": 0.8, "fpr_at_80_tpr": 0.2},
        "final_neutral_fpr_by_family": {
            "formal_casual_neutral": 0.1,
            "ai_human_neutral": 0.1,
        },
        "factorial_calibration": passing_factorial_calibration(),
    }
    return {
        "selection_used_final_rows": False,
        "selection_used_calibration_rows": False,
        "models": {
            "logistic": logistic,
            "paired_mean": {
                "factorial_calibration": passing_factorial_calibration(),
            },
        },
        "direction_agreement": {
            "raw_cosine": 0.7,
            "final_score_spearman": 0.8,
            "logistic_score_gap": 1.0,
            "paired_mean_score_gap": 1.0,
        },
        "paired_direction_stability": {"fifth_percentile_cosine": 0.6},
    }


def test_factorial_calibration_recovers_frozen_main_effects_and_specificity() -> None:
    metadata = []
    scores = []
    for replicate, register in enumerate(("formal", "casual")):
        for cell in range(16):
            recipient = bool(cell & 1)
            consequence = bool(cell & 2)
            scoring = bool(cell & 4)
            comparison = bool(cell & 8)
            interaction_noise = 0.05 * (1 if (cell.bit_count() % 2) else -1)
            score = (
                2.0
                - 1.0 * recipient
                - 2.0 * consequence
                + 3.0 * scoring
                + 4.0 * comparison
                + 0.1 * (register == "formal")
                + interaction_noise
            )
            metadata.append(
                {
                    "sample_id": f"cal-{replicate}-{cell}",
                    "split": "factorial_calibration",
                    "label": None,
                    "register": register,
                    "factors": {
                        "real_external_recipient": recipient,
                        "operational_consequence": consequence,
                        "model_scoring": scoring,
                        "comparison_among_models": comparison,
                    },
                }
            )
            scores.append(score)

    report = factorial_calibration_analysis(np.asarray(scores), metadata)
    assert report["passed"] is True
    assert report["coefficients"]["real_external_recipient"]["estimate"] == pytest.approx(
        -1.0
    )
    assert report["coefficients"]["operational_consequence"]["estimate"] == pytest.approx(
        -2.0
    )
    assert report["coefficients"]["model_scoring"]["estimate"] == pytest.approx(3.0)
    assert report["coefficients"]["comparison_among_models"]["estimate"] == pytest.approx(
        4.0
    )
    assert report["coefficients"]["formal_register"]["estimate"] == pytest.approx(0.1)
    assert report["clean_cell_contrast"]["clean_evaluation_count"] == 6
    assert report["clean_cell_contrast"]["clean_deployment_count"] == 6
    assert report["clean_cell_contrast"]["difference"] > 0
    assert report["clean_cell_contrast"]["ols_main_effect_contrast"][
        "difference"
    ] == pytest.approx(20.0 / 3.0)
    assert report["clean_cell_contrast"]["standard_error"] > 0


def test_source_decision_fails_when_factorial_register_dominates() -> None:
    analysis = passing_analysis()
    analysis["models"]["paired_mean"]["factorial_calibration"]["coefficients"][
        "formal_register"
    ]["estimate"] = 2.0
    decision = decide_source_instrument(
        analysis,
        {"add_one_empirical_p": 0.005},
        {"g1": 0.4},
    )
    assert decision["passed"] is False
    assert decision["checks"]["paired_mean_calibration_register_subordinate"] is False


def test_source_decision_passes_only_complete_robust_gate() -> None:
    decision = decide_source_instrument(
        passing_analysis(),
        {"add_one_empirical_p": 0.005},
        {"g1": 0.4, "g2": 0.2},
    )
    assert decision["passed"] is True
    assert decision["authorization"]["local_proxy_parity"] is True
    assert decision["authorization"]["user_recruitment"] is False


def test_source_decision_fails_on_formality_false_positives() -> None:
    analysis = passing_analysis()
    analysis["models"]["logistic"]["final_neutral_fpr_by_family"][
        "formal_casual_neutral"
    ] = 0.21
    decision = decide_source_instrument(
        analysis,
        {"add_one_empirical_p": 0.005},
        {"g1": 0.4},
    )
    assert decision["passed"] is False
    assert decision["decision"] == "stop_before_user_recruitment"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_permutation_checkpoint_is_bound_and_resumes_without_refitting(
    tmp_path, monkeypatch
) -> None:
    metadata = [
        {"pair_id": "p0", "label": 0, "split": "train"},
        {"pair_id": "p0", "label": 1, "split": "train"},
    ]
    features = np.zeros((2, 1), dtype=np.float32)
    calls = []

    def fake_fit(_features, _metadata, **kwargs):
        calls.append(kwargs["labels_override"].copy())
        model = FrozenLinear(
            "fake", 0, np.zeros(1), np.ones(1), np.ones(1), 0.0, 0.0
        )
        repetition = len(calls)
        return DevelopmentFit(model, model, 0.1, float(repetition), {})

    monkeypatch.setattr(probes_module, "fit_source_development", fake_fit)
    checkpoint = tmp_path / "permutation_rows.jsonl"
    binding = {
        "config_sha256": "a" * 64,
        "feature_manifest_sha256": "b" * 64,
        "readout_lock_sha256": "c" * 64,
    }
    first = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=0.5,
        reps=3,
        checkpoint_path=checkpoint,
        checkpoint_binding=binding,
    )
    assert len(calls) == 3
    assert first["checkpoint_contract_sha256"]
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 3

    def forbidden_refit(*_args, **_kwargs):
        raise AssertionError("a valid completed checkpoint must resume without refitting")

    monkeypatch.setattr(probes_module, "fit_source_development", forbidden_refit)
    second = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=0.5,
        reps=3,
        checkpoint_path=checkpoint,
        checkpoint_binding=binding,
    )
    assert second["checkpoint_sha256"] == first["checkpoint_sha256"]
    with pytest.raises(ValueError, match="contract differs"):
        run_full_selection_permutations(
            features,
            metadata,
            observed_objective=0.5,
            reps=3,
            checkpoint_path=checkpoint,
            checkpoint_binding=binding | {"config_sha256": "d" * 64},
        )


def test_permutation_checkpoint_rejects_unbound_legacy_file(tmp_path) -> None:
    checkpoint = tmp_path / "permutation_rows.jsonl"
    checkpoint.write_text('{"repetition":0}\n', encoding="utf-8")
    binding = {
        "config_sha256": "a" * 64,
        "feature_manifest_sha256": "b" * 64,
        "readout_lock_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="unbound"):
        run_full_selection_permutations(
            np.zeros((2, 1)),
            [
                {"pair_id": "p", "label": 0, "split": "train"},
                {"pair_id": "p", "label": 1, "split": "train"},
            ],
            observed_objective=0.0,
            reps=1,
            checkpoint_path=checkpoint,
            checkpoint_binding=binding,
        )


def test_parallel_permutations_are_deterministic_and_worker_bounded(
    tmp_path, monkeypatch
) -> None:
    metadata = [
        {"pair_id": f"p{pair}", "label": label, "split": "train"}
        for pair in range(4)
        for label in (0, 1)
    ]
    features = np.zeros((len(metadata), 1), dtype=np.float32)
    model = FrozenLinear("fake", 0, np.zeros(1), np.ones(1), np.ones(1), 0.0, 0.0)

    def deterministic_fit(_features, _metadata, **kwargs):
        labels = kwargs["labels_override"]
        objective = float(labels @ np.arange(1, len(labels) + 1))
        return DevelopmentFit(model, model, 0.1, objective, {})

    monkeypatch.setattr(probes_module, "fit_source_development", deterministic_fit)
    binding = {
        "config_sha256": "a" * 64,
        "feature_manifest_sha256": "b" * 64,
        "readout_lock_sha256": "c" * 64,
    }
    sequential = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=10.0,
        reps=12,
        seed=123,
        checkpoint_path=tmp_path / "sequential.jsonl",
        checkpoint_binding=binding,
        workers=1,
    )
    parallel = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=10.0,
        reps=12,
        seed=123,
        checkpoint_path=tmp_path / "parallel.jsonl",
        checkpoint_binding=binding,
        workers=4,
    )
    assert parallel["workers"] == 4
    assert parallel["exceedances"] == sequential["exceedances"]
    assert parallel["add_one_empirical_p"] == sequential["add_one_empirical_p"]
    sequential_rows = (tmp_path / "sequential.jsonl").read_text(encoding="utf-8")
    parallel_rows = (tmp_path / "parallel.jsonl").read_text(encoding="utf-8")
    assert parallel_rows == sequential_rows
    with pytest.raises(ValueError, match="between 1 and 64"):
        run_full_selection_permutations(
            features,
            metadata,
            observed_objective=10.0,
            reps=1,
            workers=65,
        )


def test_bounded_permutation_batches_are_real_work_and_resume_to_one_shot_result(
    tmp_path,
    monkeypatch,
) -> None:
    metadata = [
        {"pair_id": f"p{pair}", "label": label, "split": "train"}
        for pair in range(4)
        for label in (0, 1)
    ]
    features = np.zeros((len(metadata), 1), dtype=np.float32)
    model = FrozenLinear("fake", 0, np.zeros(1), np.ones(1), np.ones(1), 0.0, 0.0)

    def deterministic_fit(_features, _metadata, **kwargs):
        labels = kwargs["labels_override"]
        objective = float(labels @ np.arange(1, len(labels) + 1))
        return DevelopmentFit(model, model, 0.1, objective, {})

    monkeypatch.setattr(probes_module, "fit_source_development", deterministic_fit)
    binding = {
        "config_sha256": "a" * 64,
        "feature_manifest_sha256": "b" * 64,
        "readout_lock_sha256": "c" * 64,
    }
    staged_path = tmp_path / "staged.jsonl"
    first = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=10.0,
        reps=6,
        seed=123,
        checkpoint_path=staged_path,
        checkpoint_binding=binding,
        workers=16,
        max_new_repetitions=2,
    )
    assert first["complete"] is False
    assert first["permutations_completed"] == 2
    assert first["permutations_remaining"] == 4
    assert first["optimization"]["repetitions_fitted"] == 2
    assert first["optimization"]["numerical_library_threads_per_worker"] == 1
    assert first["optimization"]["outer_worker_count"] == 16
    assert first["optimization"]["elapsed_seconds"] >= 0
    assert first["optimization"]["fitted_repetitions_per_second"] > 0
    assert first["optimization"]["projected_remaining_seconds_at_measured_rate"] > 0
    assert not staged_path.exists()

    second = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=10.0,
        reps=6,
        seed=123,
        checkpoint_path=staged_path,
        checkpoint_binding=binding,
        workers=32,
        max_new_repetitions=2,
    )
    assert second["complete"] is False
    assert second["permutations_completed"] == 4
    assert second["optimization"]["completed_repetitions_loaded"] == 2
    assert second["optimization"]["outer_worker_count"] == 32

    final = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=10.0,
        reps=6,
        seed=123,
        checkpoint_path=staged_path,
        checkpoint_binding=binding,
        workers=2,
    )
    assert final["complete"] is True
    assert final["optimization"]["completed_repetitions_loaded"] == 4

    one_shot_path = tmp_path / "one-shot.jsonl"
    one_shot = run_full_selection_permutations(
        features,
        metadata,
        observed_objective=10.0,
        reps=6,
        seed=123,
        checkpoint_path=one_shot_path,
        checkpoint_binding=binding,
        workers=3,
    )
    assert staged_path.read_bytes() == one_shot_path.read_bytes()
    assert final["exceedances"] == one_shot["exceedances"]
    assert final["add_one_empirical_p"] == one_shot["add_one_empirical_p"]

    with pytest.raises(ValueError, match="requires a checkpoint path"):
        run_full_selection_permutations(
            features,
            metadata,
            observed_objective=10.0,
            reps=6,
            max_new_repetitions=2,
        )


def test_source_verifier_rescores_features_and_separates_integrity_from_science(
    tmp_path,
) -> None:
    feature_root = tmp_path / "features"
    source_root = tmp_path / "source"
    feature_root.mkdir()
    source_root.mkdir()
    config_path = tmp_path / "prereg.yaml"
    config_path.write_text("project: test\n", encoding="utf-8")

    development_metadata = []
    development = np.zeros((4, 45, 4096), dtype=np.float32)
    for index, (generator, label, value) in enumerate(
        (("g0", 0, -1.0), ("g0", 1, 1.0), ("g1", 0, -2.0), ("g1", 1, 2.0))
    ):
        development[index, 0, 0] = value
        development_metadata.append(
            {
                "sample_id": f"train-{index}",
                "pair_id": f"pair-{index // 2}",
                "split": "train",
                "label": label,
                "generator_family": generator,
            }
        )
    final_metadata = []
    final = np.zeros((8, 45, 4096), dtype=np.float32)
    final_specs = (
        ("ordinary-0", "ordinary_test", 0, "ordinary", -1.0),
        ("ordinary-1", "ordinary_test", 1, "ordinary", 1.0),
        ("final-0", "final_counterfactual", 0, "final", -1.0),
        ("final-1", "final_counterfactual", 1, "final", 1.0),
        ("formal-0", "neutral_controls", None, "formal_casual_neutral", -1.0),
        ("formal-1", "neutral_controls", None, "formal_casual_neutral", -1.0),
        ("ai-0", "neutral_controls", None, "ai_human_neutral", -1.0),
        ("ai-1", "neutral_controls", None, "ai_human_neutral", -1.0),
    )
    for index, (sample_id, split, label, generator, value) in enumerate(final_specs):
        final[index, 0, 0] = value
        final_metadata.append(
            {
                "sample_id": sample_id,
                "pair_id": sample_id,
                "split": split,
                "label": label,
                "generator_family": generator,
            }
        )
    calibration_metadata = []
    calibration = np.zeros((32, 45, 4096), dtype=np.float32)
    for replicate, register in enumerate(("formal", "casual")):
        for cell in range(16):
            index = replicate * 16 + cell
            recipient = bool(cell & 1)
            consequence = bool(cell & 2)
            scoring = bool(cell & 4)
            comparison = bool(cell & 8)
            calibration[index, 0, 0] = (
                -1.0 * recipient
                - 2.0 * consequence
                + 3.0 * scoring
                + 4.0 * comparison
                + 0.1 * (register == "formal")
                + 0.05 * (1 if (cell.bit_count() % 2) else -1)
            )
            calibration_metadata.append(
                {
                    "sample_id": f"calibration-{replicate}-{cell}",
                    "pair_id": f"calibration-{replicate}-{cell}",
                    "split": "factorial_calibration",
                    "label": None,
                    "generator_family": "factorial_record",
                    "register": register,
                    "factors": {
                        "real_external_recipient": recipient,
                        "operational_consequence": consequence,
                        "model_scoring": scoring,
                        "comparison_among_models": comparison,
                    },
                }
            )

    def write_partition(name, features, metadata):
        feature_path = feature_root / f"{name}_features.npz"
        metadata_path = feature_root / f"{name}_metadata.jsonl"
        np.savez(feature_path, shared_task_suffix_mean=features.astype(np.float16))
        metadata_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata),
            encoding="utf-8",
        )
        return {
            "features": feature_path.name,
            "features_sha256": _sha(feature_path),
            "metadata": metadata_path.name,
            "metadata_sha256": _sha(metadata_path),
        }

    feature_manifest = {
        "partitions": {
            "development": write_partition("development", development, development_metadata),
            "final": write_partition("final", final, final_metadata),
            "calibration": write_partition(
                "calibration", calibration, calibration_metadata
            ),
        }
    }
    feature_manifest_path = feature_root / "feature_manifest.json"
    feature_manifest_path.write_text(json.dumps(feature_manifest), encoding="utf-8")

    mean = np.zeros(4096, dtype=np.float32)
    scale = np.ones(4096, dtype=np.float32)
    weight = np.zeros(4096, dtype=np.float32)
    weight[0] = 1.0
    arrays_path = source_root / "source_readout_arrays.npz"
    np.savez(
        arrays_path,
        logistic_mean=mean,
        logistic_scale=scale,
        logistic_weight=weight,
        paired_mean=mean,
        paired_scale=scale,
        paired_weight=weight,
    )
    readout = {
        "selected_layer": 0,
        "selected_C": 0.1,
        "arrays_sha256": _sha(arrays_path),
        "logistic": {"bias": 0.0, "threshold_80_tpr": 0.0},
        "paired_mean": {"bias": 0.0, "threshold_80_tpr": 0.0},
    }
    readout_path = source_root / "source_readout_lock.json"
    readout_path.write_text(json.dumps(readout), encoding="utf-8")

    labels = np.asarray([row["label"] if row["label"] is not None else -1 for row in final_metadata])
    splits = np.asarray([row["split"] for row in final_metadata])
    scores = final[:, 0, 0]
    ordinary_metrics = _binary_metrics(labels[splits == "ordinary_test"], scores[splits == "ordinary_test"])
    final_metrics = _binary_metrics(
        labels[splits == "final_counterfactual"], scores[splits == "final_counterfactual"]
    )
    score_rows = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "label": row["label"],
            "score": float(score),
        }
        for row, score in zip(final_metadata, scores, strict=True)
    ]
    leave_one = _recompute_leave_one_generator(
        development,
        development_metadata,
        final,
        final_metadata,
        layer=0,
        c_value=0.1,
    )
    model_block = {
        "ordinary_test": ordinary_metrics,
        "final_counterfactual": final_metrics,
        "final_neutral_fpr_by_family": {
            "ai_human_neutral": 0.0,
            "formal_casual_neutral": 0.0,
        },
        "factorial_calibration": factorial_calibration_analysis(
            calibration.astype(np.float16).astype(np.float32)[:, 0, 0],
            calibration_metadata,
        ),
        "score_rows": score_rows,
    }
    analysis = {
        "selection_used_final_rows": False,
        "selection_used_calibration_rows": False,
        "models": {"logistic": model_block, "paired_mean": model_block},
        "direction_agreement": {
            "raw_cosine": 1.0,
            "final_score_spearman": 1.0,
            "logistic_score_gap": 2.0,
            "paired_mean_score_gap": 2.0,
        },
        "paired_direction_stability": {
            "reps": 2,
            "seed": 7,
            "median_cosine": 1.0,
            "fifth_percentile_cosine": 1.0,
            "minimum_cosine": 1.0,
            "cosines": [1.0, 1.0],
        },
        "leave_one_training_generator_score_gaps": leave_one,
    }
    (source_root / "source_final_analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )

    binding = {
        "config_sha256": _sha(config_path),
        "feature_manifest_sha256": _sha(feature_manifest_path),
        "readout_lock_sha256": _sha(readout_path),
    }
    contract = {
        "schema_version": "glm53_v11_permutation_checkpoint_contract_v1",
        "reps": 1,
        "seed": 9,
        "observed_objective": 0.0,
        "c_grid": [0.1],
        "neutral_penalty": 0.5,
        "neutral_target": 0.2,
        "artifact_binding": binding,
    }
    contract_sha = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint_manifest = contract | {"contract_sha256": contract_sha}
    checkpoint_manifest_path = source_root / "permutation_rows.jsonl.manifest.json"
    checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest), encoding="utf-8")
    rows_path = source_root / "permutation_rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "repetition": 0,
                "seed": 9,
                "selected_layer": 0,
                "selected_C": 0.1,
                "objective": 1.0,
                "contract_sha256": contract_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    permutation = {
        "reps": 1,
        "seed": 9,
        "observed_objective": 0.0,
        "add_one_empirical_p": 1.0,
        "checkpoint_sha256": _sha(rows_path),
        "checkpoint_manifest_sha256": _sha(checkpoint_manifest_path),
        "checkpoint_contract_sha256": contract_sha,
    }
    (source_root / "permutation_analysis.json").write_text(
        json.dumps(permutation), encoding="utf-8"
    )

    report = verify_source_result(
        source_root=source_root,
        feature_root=feature_root,
        config_path=config_path,
    )
    assert report["passed"] is True
    assert report["integrity_agreement"] is True
    assert report["scientific_gate_would_pass"] is False

    original_factorial_score = analysis["models"]["logistic"]["factorial_calibration"][
        "score_rows"
    ][0]["score"]
    analysis["models"]["logistic"]["factorial_calibration"]["score_rows"][0][
        "score"
    ] = 99.0
    (source_root / "source_final_analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="logistic factorial score"):
        verify_source_result(
            source_root=source_root,
            feature_root=feature_root,
            config_path=config_path,
        )
    analysis["models"]["logistic"]["factorial_calibration"]["score_rows"][0][
        "score"
    ] = original_factorial_score

    analysis["models"]["logistic"]["score_rows"][0]["score"] = 99.0
    (source_root / "source_final_analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="score for ordinary-0"):
        verify_source_result(
            source_root=source_root,
            feature_root=feature_root,
            config_path=config_path,
        )
