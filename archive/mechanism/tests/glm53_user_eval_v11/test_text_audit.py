from __future__ import annotations

import copy
import json
from argparse import Namespace

import numpy as np
import pytest
from pipelines.glm53_user_eval.v11 import run as supervisor
from src.glm53_user_eval.v11.builder import build_dataset
from src.glm53_user_eval.v11.text_audit import (
    FROZEN_KEYWORDS,
    DatasetStructureError,
    evaluate_final_holdout,
    fit_development_baselines,
    load_development_audit,
    load_rows,
    save_development_audit,
    select_development_rows,
    select_final_holdout_rows,
    validate_dataset_structure,
)


@pytest.fixture(scope="module")
def governed_rows(tmp_path_factory) -> list[dict]:
    root = tmp_path_factory.mktemp("contrastive-v3")
    manifest = build_dataset(root)
    assert manifest["row_count"] == 576
    return load_rows(root / "samples.jsonl")


@pytest.fixture(scope="module")
def development_audit(governed_rows):
    return fit_development_baselines(select_development_rows(governed_rows), c_grid=(0.1,))


def test_governed_dataset_passes_structural_audit(governed_rows) -> None:
    report = validate_dataset_structure(governed_rows)
    assert report["passed"] is True
    assert report["row_count"] == 576
    assert report["split_counts"]["train"] == 256
    assert report["split_counts"]["final_counterfactual"] == 64
    assert report["generator_task_domain_disjoint"] is True


def test_structural_audit_rejects_cross_split_task_leakage(governed_rows) -> None:
    corrupted = copy.deepcopy(governed_rows)
    train_task = next(row["task_id"] for row in corrupted if row["split"] == "train")
    target_pair = next(row["pair_id"] for row in corrupted if row["split"] == "validation")
    for row in corrupted:
        if row["pair_id"] == target_pair:
            row["task_id"] = train_task
            without_id = dict(row)
            without_id.pop("sample_id")
            from src.glm53_user_eval.v11.builder import canonical_json, sha256_text

            row["sample_id"] = sha256_text(canonical_json(without_id))[:20]
    with pytest.raises(DatasetStructureError, match="task set differs|task leakage"):
        validate_dataset_structure(corrupted)


def test_development_fitter_rejects_final_holdout(governed_rows) -> None:
    development = select_development_rows(governed_rows)
    final_pair = select_final_holdout_rows(governed_rows)[:2]
    with pytest.raises(DatasetStructureError, match="forbidden splits"):
        fit_development_baselines(development + final_pair, c_grid=(0.1,))


def test_all_baselines_fit_train_only_and_select_on_development(development_audit) -> None:
    assert set(development_audit.models) == {
        "structural_metadata",
        "frozen_keyword",
        "word_tfidf",
        "char_3_5gram",
        "decisive_deleted_word_tfidf",
        "decisive_deleted_char_3_5gram",
    }
    report = development_audit.report
    assert report["fit_splits"] == ["train"]
    assert report["selection_splits"] == ["validation", "development_counterfactual"]
    assert report["post_selection_report_splits"] == ["ordinary_test"]
    assert report["final_holdout_evaluated"] is False
    assert all(row["selection"]["C"] == 0.1 for row in report["baselines"].values())


def test_structural_features_exclude_latent_factors(development_audit) -> None:
    names = development_audit.models["structural_metadata"].feature_names
    assert not any("factor" in name for name in names)
    assert not any("label" in name for name in names)
    assert any(name.startswith("nuisance.") for name in names)


def test_keyword_vocabulary_is_literal_and_frozen(development_audit) -> None:
    model = development_audit.models["frozen_keyword"]
    assert model.feature_names == FROZEN_KEYWORDS
    assert model.lock_record()["feature_count"] == len(FROZEN_KEYWORDS)


def test_final_only_words_do_not_enter_train_vocabulary(development_audit) -> None:
    vocabulary = development_audit.models["word_tfidf"].vectorizer.vocabulary_
    assert "microfiche" not in vocabulary
    assert "postcard" not in vocabulary


def test_final_holdout_is_scored_without_refit(governed_rows, development_audit) -> None:
    coefficient_hashes = {
        name: model.lock_record()["coefficient_sha256"]
        for name, model in development_audit.models.items()
    }
    final_rows = select_final_holdout_rows(governed_rows)
    result = evaluate_final_holdout(development_audit, final_rows)
    assert result["selection_performed"] is False
    assert result["evaluated_split"] == "final_counterfactual"
    assert result["row_count"] == 64
    assert all(row["metrics"]["count"] == 64 for row in result["baselines"].values())
    assert coefficient_hashes == {
        name: model.lock_record()["coefficient_sha256"]
        for name, model in development_audit.models.items()
    }


def test_final_evaluator_rejects_nonfinal_rows(governed_rows, development_audit) -> None:
    with pytest.raises(DatasetStructureError, match="forbidden splits|analysis splits differ"):
        evaluate_final_holdout(
            development_audit,
            select_final_holdout_rows(governed_rows) + select_development_rows(governed_rows)[:2],
        )


def test_development_report_is_json_serializable(development_audit) -> None:
    encoded = json.dumps(development_audit.report, sort_keys=True)
    assert "development_lock_sha256" in encoded
    assert np.isfinite(
        development_audit.report["baselines"]["word_tfidf"]["ordinary_test"]["auroc"]
    )


def test_development_models_roundtrip_under_hash_lock(tmp_path, development_audit) -> None:
    model_path = tmp_path / "models.joblib"
    report_path = tmp_path / "development.json"
    saved = save_development_audit(
        development_audit,
        model_path=model_path,
        report_path=report_path,
    )
    loaded = load_development_audit(model_path=model_path, report_path=report_path)
    assert saved["development_lock_sha256"] == loaded.report["development_lock_sha256"]
    assert set(loaded.models) == set(development_audit.models)


def test_downstream_gates_reject_incomplete_final_text_marker(tmp_path) -> None:
    output = tmp_path / "final_text_analysis.json"
    output.write_text("{}\n", encoding="utf-8")
    marker = tmp_path / "FINAL_TEXT_HOLDOUT_OPENED.json"
    marker.write_text(
        json.dumps({"opened_once": True, "status": "opening"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not finish"):
        supervisor._require_completed_final_text(Namespace(audit_root=tmp_path))


def test_completed_final_text_marker_must_bind_output_hash(tmp_path) -> None:
    output = tmp_path / "final_text_analysis.json"
    output.write_text("{}\n", encoding="utf-8")
    marker = tmp_path / "FINAL_TEXT_HOLDOUT_OPENED.json"
    marker.write_text(
        json.dumps(
            {
                "opened_once": True,
                "status": "complete",
                "final_analysis_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not bind"):
        supervisor._require_completed_final_text(Namespace(audit_root=tmp_path))
