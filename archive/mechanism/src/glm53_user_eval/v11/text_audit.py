"""Text-only construct audits for ``contrastive_prompts_v3``.

The development fitter never accepts final-counterfactual rows.  Final-holdout
evaluation is a separate operation over already fitted models.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .builder import canonical_json, sha256_text
from .spec import GENERATOR_FAMILIES, SPLIT_PAIR_COUNTS, TASKS

DEVELOPMENT_SPLITS = (
    "train",
    "validation",
    "ordinary_test",
    "development_counterfactual",
)
FINAL_SPLIT = "final_counterfactual"
CONTROL_SPLITS = ("neutral_controls", "factorial_calibration")
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)

# This vocabulary is fixed in code before any text-baseline result is opened.
FROZEN_KEYWORDS = (
    "evaluation",
    "evaluate",
    "deployment",
    "deploy",
    "benchmark",
    "production",
    "score",
    "scoring",
    "grader",
    "grading",
    "compare",
    "comparison",
    "candidate",
    "model",
    "research",
    "review",
    "quality",
    "live",
    "customer",
    "user",
    "operational",
    "external",
    "recipient",
    "real",
    "ai",
    "assistant",
    "formal",
    "casual",
    "logged",
    "archived",
)

DIRECT_LABEL_PHRASES = (
    "this is an evaluation",
    "this is a deployment",
    "the response is being benchmarked",
    "the model is serving a live user",
)


class DatasetStructureError(ValueError):
    """Raised when the governed v3 rows violate the text-audit contract."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise DatasetStructureError(f"dataset is empty: {path}")
    return rows


def prompt_text(row: dict[str, Any]) -> str:
    system = str(row.get("system_prompt") or "")
    user = str(row.get("user_prompt") or "")
    return f"{system}\n{user}" if system else user


def decisive_deleted_prompt_text(row: dict[str, Any]) -> str:
    text = prompt_text(row)
    for fact in row.get("decisive_fact_texts") or []:
        fact_text = str(fact)
        if text.count(fact_text) != 1:
            raise DatasetStructureError(
                f"decisive fact cannot be deleted uniquely: {row.get('sample_id')}"
            )
        text = text.replace(fact_text, "[DECISIVE_FACT]")
    return text


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetStructureError(message)


def _recomputed_sample_id(row: dict[str, Any]) -> str:
    without_id = dict(row)
    without_id.pop("sample_id", None)
    return sha256_text(canonical_json(without_id))[:20]


def _balance_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["generator_family"],
        row["task_id"],
        row["task_domain"],
        row["prompt_role"],
        row["register"],
        canonical_json(row.get("nuisance") or {}),
    )


def validate_dataset_structure(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate counts, pairing, split isolation, and label-neutral metadata.

    This validation intentionally does not inspect activation results or fit a
    classifier.  It is safe to run before preregistration and paid compute.
    """

    required = {
        "schema_version",
        "dataset_id",
        "sample_id",
        "pair_id",
        "split",
        "label",
        "latent_class",
        "generator_family",
        "task_id",
        "task_domain",
        "prompt_role",
        "register",
        "system_prompt",
        "user_prompt",
        "scenario_text",
        "shared_suffix",
        "shared_suffix_sha256",
        "decisive_fact_texts",
        "factors",
        "nuisance",
        "acceptable_judge_labels",
        "holdout_locked",
        "control_partition",
    }
    expected_counts = {
        **{split: 2 * count for split, count in SPLIT_PAIR_COUNTS.items()},
        "neutral_controls": 64,
        "factorial_calibration": 32,
    }
    observed_counts = Counter(str(row.get("split")) for row in rows)
    _require(observed_counts == expected_counts, f"split counts differ: {dict(observed_counts)}")
    _require(len(rows) == sum(expected_counts.values()), "row count differs from v3 contract")

    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    _require(all(sample_ids), "one or more sample IDs are empty")
    _require(len(sample_ids) == len(set(sample_ids)), "sample IDs are not unique")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = sorted(required - set(row))
        _require(not missing, f"row {row.get('sample_id')} lacks fields {missing}")
        _require(row["schema_version"] == "contrastive_prompts_v3_row_v1", "row schema differs")
        _require(row["dataset_id"] == "contrastive_prompts_v3", "row dataset ID differs")
        _require(row["sample_id"] == _recomputed_sample_id(row), f"sample ID hash differs: {row['sample_id']}")
        text = prompt_text(row)
        suffix = str(row["shared_suffix"])
        _require(text.count(suffix) == 1, f"shared suffix is not unique: {row['sample_id']}")
        _require(
            row["shared_suffix_sha256"] == sha256_text(suffix),
            f"shared suffix hash differs: {row['sample_id']}",
        )
        lowered = text.lower()
        _require(
            not any(phrase in lowered for phrase in DIRECT_LABEL_PHRASES),
            f"direct class statement found: {row['sample_id']}",
        )
        by_pair[str(row["pair_id"])].append(row)

    binary_splits = tuple(SPLIT_PAIR_COUNTS)
    generator_sets: dict[str, set[str]] = {}
    task_sets: dict[str, set[str]] = {}
    domain_sets: dict[str, set[str]] = {}
    balance: dict[str, dict[str, Any]] = {}
    for split in binary_splits:
        split_rows = [row for row in rows if row["split"] == split]
        _require(
            Counter(row["label"] for row in split_rows)
            == {0: SPLIT_PAIR_COUNTS[split], 1: SPLIT_PAIR_COUNTS[split]},
            f"class counts differ in {split}",
        )
        generator_sets[split] = {str(row["generator_family"]) for row in split_rows}
        task_sets[split] = {str(row["task_id"]) for row in split_rows}
        domain_sets[split] = {str(row["task_domain"]) for row in split_rows}
        _require(
            generator_sets[split] == set(GENERATOR_FAMILIES[split]),
            f"generator set differs in {split}",
        )
        _require(
            task_sets[split] == {task[0] for task in TASKS[split]},
            f"task set differs in {split}",
        )
        signatures = {
            label: Counter(_balance_signature(row) for row in split_rows if row["label"] == label)
            for label in (0, 1)
        }
        _require(signatures[0] == signatures[1], f"metadata is not label-balanced in {split}")
        balance[split] = {
            "rows_per_label": SPLIT_PAIR_COUNTS[split],
            "generator_families": sorted(generator_sets[split]),
            "task_count": len(task_sets[split]),
            "task_domain_count": len(domain_sets[split]),
            "metadata_signature_balanced": True,
        }

    for left_index, left in enumerate(binary_splits):
        for right in binary_splits[left_index + 1 :]:
            _require(not generator_sets[left] & generator_sets[right], f"generator leakage: {left}/{right}")
            _require(not task_sets[left] & task_sets[right], f"task leakage: {left}/{right}")
            _require(not domain_sets[left] & domain_sets[right], f"task-domain leakage: {left}/{right}")

    for pair_id, pair_rows in by_pair.items():
        split_names = {str(row["split"]) for row in pair_rows}
        _require(len(split_names) == 1, f"pair crosses splits: {pair_id}")
        split = next(iter(split_names))
        if split in binary_splits:
            _require(len(pair_rows) == 2, f"binary pair does not have two rows: {pair_id}")
            _require({row["label"] for row in pair_rows} == {0, 1}, f"pair is not binary: {pair_id}")
            _require(
                {row["latent_class"] for row in pair_rows} == {"deploy", "eval"},
                f"latent classes differ: {pair_id}",
            )
            for row in pair_rows:
                expected_label = 1 if row["latent_class"] == "eval" else 0
                _require(type(row["label"]) is int and row["label"] == expected_label, f"label mismatch: {row['sample_id']}")
                _require(
                    row["acceptable_judge_labels"] == [row["latent_class"]],
                    f"judge label differs: {row['sample_id']}",
                )
                _require(bool(row["decisive_fact_texts"]), f"binary row has no decisive facts: {row['sample_id']}")
                for fact in row["decisive_fact_texts"]:
                    _require(
                        str(row["scenario_text"]).count(str(fact)) == 1,
                        f"decisive fact is absent or repeated: {row['sample_id']}",
                    )
            invariant_fields = (
                "generator_family",
                "task_id",
                "task_domain",
                "prompt_role",
                "register",
                "shared_suffix",
                "shared_suffix_sha256",
                "nuisance",
                "control_partition",
            )
            for field_name in invariant_fields:
                _require(
                    pair_rows[0][field_name] == pair_rows[1][field_name],
                    f"pair field {field_name} differs: {pair_id}",
                )
            expected_lock = split == FINAL_SPLIT
            _require(
                all(bool(row["holdout_locked"]) == expected_lock for row in pair_rows),
                f"holdout lock differs: {pair_id}",
            )
        else:
            _require(all(row["label"] is None for row in pair_rows), f"control has binary label: {pair_id}")
            _require(all(bool(row["holdout_locked"]) for row in pair_rows), f"control is not locked: {pair_id}")

    return {
        "schema_version": "contrastive_prompts_v3_structure_audit_v1",
        "passed": True,
        "row_count": len(rows),
        "split_counts": dict(sorted(observed_counts.items())),
        "binary_balance": balance,
        "generator_task_domain_disjoint": True,
        "sample_ids_recomputed": True,
        "pair_contract_passed": True,
    }


def select_development_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") in DEVELOPMENT_SPLITS]


def select_final_holdout_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == FINAL_SPLIT]


def _analysis_rows_valid(rows: Sequence[dict[str, Any]], allowed: set[str]) -> None:
    if not rows:
        raise DatasetStructureError("analysis rows are empty")
    observed = {str(row.get("split")) for row in rows}
    if not observed <= allowed:
        raise DatasetStructureError(f"analysis received forbidden splits: {sorted(observed - allowed)}")
    if observed != allowed:
        raise DatasetStructureError(f"analysis splits differ: {sorted(observed)} != {sorted(allowed)}")
    if any(type(row.get("label")) is not int or row["label"] not in (0, 1) for row in rows):
        raise DatasetStructureError("text baselines require binary integer labels")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise DatasetStructureError("analysis sample IDs are not unique")
    pair_splits: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        pair_splits[str(row["pair_id"])].add(str(row["split"]))
        pair_labels[str(row["pair_id"])].append(int(row["label"]))
    if any(len(value) != 1 for value in pair_splits.values()):
        raise DatasetStructureError("an analysis pair crosses splits")
    if any(sorted(value) != [0, 1] for value in pair_labels.values()):
        raise DatasetStructureError("an analysis pair is incomplete")


def _structural_record(row: dict[str, Any]) -> dict[str, float]:
    system = str(row.get("system_prompt") or "")
    user = str(row.get("user_prompt") or "")
    scenario = str(row.get("scenario_text") or "")
    suffix = str(row.get("shared_suffix") or "")
    facts = [str(value) for value in row.get("decisive_fact_texts") or []]
    record: dict[str, float] = {
        "system_chars": float(len(system)),
        "user_chars": float(len(user)),
        "scenario_chars": float(len(scenario)),
        "suffix_chars": float(len(suffix)),
        "system_words": float(len(system.split())),
        "user_words": float(len(user.split())),
        "scenario_words": float(len(scenario.split())),
        "suffix_words": float(len(suffix.split())),
        "system_newlines": float(system.count("\n")),
        "user_newlines": float(user.count("\n")),
        "decisive_fact_count": float(len(facts)),
        "decisive_fact_chars": float(sum(len(value) for value in facts)),
        f"prompt_role={row['prompt_role']}": 1.0,
        f"register={row['register']}": 1.0,
        f"generator_family={row['generator_family']}": 1.0,
        f"task_domain={row['task_domain']}": 1.0,
    }
    for key, value in sorted((row.get("nuisance") or {}).items()):
        record[f"nuisance.{key}={bool(value)}"] = 1.0
    return record


def _keyword_count(text: str, keyword: str) -> float:
    expression = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return float(len(re.findall(expression, text.lower())))


def _keyword_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[_keyword_count(prompt_text(row), keyword) for keyword in FROZEN_KEYWORDS] for row in rows],
        dtype=np.float64,
    )


@dataclass
class FittedBaseline:
    name: str
    selected_c: float
    vectorizer: Any = field(repr=False)
    scaler: Any = field(repr=False)
    classifier: LogisticRegression = field(repr=False)
    feature_names: tuple[str, ...]
    candidate_metrics: tuple[dict[str, Any], ...]

    def transform(self, rows: Sequence[dict[str, Any]]) -> Any:
        if self.name == "structural_metadata":
            matrix = self.vectorizer.transform([_structural_record(row) for row in rows])
        elif self.name == "frozen_keyword":
            matrix = _keyword_matrix(rows)
        else:
            texts = (
                [decisive_deleted_prompt_text(row) for row in rows]
                if self.name.startswith("decisive_deleted_")
                else [prompt_text(row) for row in rows]
            )
            matrix = self.vectorizer.transform(texts)
        return self.scaler.transform(matrix) if self.scaler is not None else matrix

    def scores(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray(self.classifier.decision_function(self.transform(rows)), dtype=np.float64)

    def lock_record(self) -> dict[str, Any]:
        names_hash = hashlib.sha256("\n".join(self.feature_names).encode("utf-8")).hexdigest()
        return {
            "name": self.name,
            "selected_C": self.selected_c,
            "feature_count": len(self.feature_names),
            "feature_names_sha256": names_hash,
            "coefficient_sha256": hashlib.sha256(
                np.asarray(self.classifier.coef_, dtype=np.float64).tobytes()
            ).hexdigest(),
            "intercept_sha256": hashlib.sha256(
                np.asarray(self.classifier.intercept_, dtype=np.float64).tobytes()
            ).hexdigest(),
        }


@dataclass
class DevelopmentAudit:
    models: dict[str, FittedBaseline]
    report: dict[str, Any]


def save_development_audit(
    audit: DevelopmentAudit,
    *,
    model_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    import joblib

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(audit.models, temporary_model, compress=3)
    os.replace(temporary_model, model_path)
    report = audit.report | {
        "model_bundle_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    return report


def load_development_audit(
    *,
    model_path: Path,
    report_path: Path,
) -> DevelopmentAudit:
    import joblib

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != report["model_bundle_sha256"]:
        raise DatasetStructureError("development model bundle hash differs")
    models = joblib.load(model_path)
    if not isinstance(models, dict):
        raise DatasetStructureError("development model bundle is not a model mapping")
    for name, model in models.items():
        expected = report["baselines"][name]["model_lock"]
        if model.lock_record() != expected:
            raise DatasetStructureError(f"development model lock differs: {name}")
    report_without_bundle = dict(report)
    report_without_bundle.pop("model_bundle_sha256", None)
    return DevelopmentAudit(models=models, report=report_without_bundle)


def _metrics(rows: Sequence[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise DatasetStructureError("metrics require both binary classes")
    probability = 1.0 / (1.0 + np.exp(-np.clip(scores, -60, 60)))
    return {
        "count": len(rows),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, probability)),
    }


def _fit_representation(
    name: str, train_rows: Sequence[dict[str, Any]]
) -> tuple[Any, Any, Any, tuple[str, ...]]:
    if name == "structural_metadata":
        vectorizer = DictVectorizer(sparse=True, sort=True)
        matrix = vectorizer.fit_transform([_structural_record(row) for row in train_rows])
        scaler = StandardScaler(with_mean=False).fit(matrix)
        return vectorizer, scaler, scaler.transform(matrix), tuple(vectorizer.get_feature_names_out())
    if name == "frozen_keyword":
        matrix = _keyword_matrix(train_rows)
        scaler = StandardScaler().fit(matrix)
        return None, scaler, scaler.transform(matrix), FROZEN_KEYWORDS
    if name in {"word_tfidf", "decisive_deleted_word_tfidf"}:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
        )
    elif name in {"char_3_5gram", "decisive_deleted_char_3_5gram"}:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=2,
            max_features=75_000,
            sublinear_tf=True,
        )
    else:
        raise ValueError(f"unknown text baseline {name}")
    texts = (
        [decisive_deleted_prompt_text(row) for row in train_rows]
        if name.startswith("decisive_deleted_")
        else [prompt_text(row) for row in train_rows]
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, None, matrix, tuple(vectorizer.get_feature_names_out())


def _transformed(
    name: str,
    rows: Sequence[dict[str, Any]],
    vectorizer: Any,
    scaler: Any,
) -> Any:
    if name == "structural_metadata":
        matrix = vectorizer.transform([_structural_record(row) for row in rows])
    elif name == "frozen_keyword":
        matrix = _keyword_matrix(rows)
    else:
        texts = (
            [decisive_deleted_prompt_text(row) for row in rows]
            if name.startswith("decisive_deleted_")
            else [prompt_text(row) for row in rows]
        )
        matrix = vectorizer.transform(texts)
    return scaler.transform(matrix) if scaler is not None else matrix


def fit_development_baselines(
    rows: Sequence[dict[str, Any]],
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
) -> DevelopmentAudit:
    """Fit on train and select only with validation plus development controls.

    ``rows`` must not contain final-counterfactual or neutral-control rows.  The
    ordinary test is scored only after model selection and never enters the
    selection objective.
    """

    _analysis_rows_valid(rows, set(DEVELOPMENT_SPLITS))
    grid = tuple(float(value) for value in c_grid)
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("C grid must contain positive finite values")
    by_split = {split: [row for row in rows if row["split"] == split] for split in DEVELOPMENT_SPLITS}
    train_rows = by_split["train"]
    train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    models: dict[str, FittedBaseline] = {}
    reports: dict[str, Any] = {}

    for name in (
        "structural_metadata",
        "frozen_keyword",
        "word_tfidf",
        "char_3_5gram",
        "decisive_deleted_word_tfidf",
        "decisive_deleted_char_3_5gram",
    ):
        vectorizer, scaler, train_x, feature_names = _fit_representation(name, train_rows)
        transformed = {
            split: _transformed(name, split_rows, vectorizer, scaler)
            for split, split_rows in by_split.items()
            if split != "train"
        }
        candidates: list[tuple[tuple[float, float, float], LogisticRegression, dict[str, Any]]] = []
        for c_value in grid:
            classifier = LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="liblinear",
                max_iter=5_000,
                random_state=0,
            ).fit(train_x, train_y)
            val_metrics = _metrics(
                by_split["validation"],
                np.asarray(classifier.decision_function(transformed["validation"])),
            )
            dev_metrics = _metrics(
                by_split["development_counterfactual"],
                np.asarray(classifier.decision_function(transformed["development_counterfactual"])),
            )
            robust_auroc = min(val_metrics["auroc"], dev_metrics["auroc"])
            mean_brier = 0.5 * (val_metrics["brier"] + dev_metrics["brier"])
            record = {
                "C": c_value,
                "robust_development_auroc": robust_auroc,
                "mean_development_brier": mean_brier,
                "validation": val_metrics,
                "development_counterfactual": dev_metrics,
            }
            candidates.append(((-robust_auroc, mean_brier, c_value), classifier, record))
        _, selected, selected_record = min(candidates, key=lambda item: item[0])
        fitted = FittedBaseline(
            name=name,
            selected_c=float(selected_record["C"]),
            vectorizer=vectorizer,
            scaler=scaler,
            classifier=selected,
            feature_names=feature_names,
            candidate_metrics=tuple(item[2] for item in candidates),
        )
        models[name] = fitted
        ordinary_scores = fitted.scores(by_split["ordinary_test"])
        reports[name] = {
            "selected_C": fitted.selected_c,
            "feature_count": len(feature_names),
            "selection": selected_record,
            "ordinary_test": _metrics(by_split["ordinary_test"], ordinary_scores),
            "ordinary_test_scores": [
                {
                    "sample_id": row["sample_id"],
                    "label": int(row["label"]),
                    "score": float(score),
                }
                for row, score in zip(by_split["ordinary_test"], ordinary_scores, strict=True)
            ],
            "candidates": list(fitted.candidate_metrics),
            "model_lock": fitted.lock_record(),
        }

    report_without_hash = {
        "schema_version": "contrastive_prompts_v3_development_text_audit_v1",
        "fit_splits": ["train"],
        "selection_splits": ["validation", "development_counterfactual"],
        "post_selection_report_splits": ["ordinary_test"],
        "final_holdout_evaluated": False,
        "c_grid": list(grid),
        "baselines": reports,
    }
    report = report_without_hash | {
        "development_lock_sha256": hashlib.sha256(
            canonical_json(report_without_hash).encode("utf-8")
        ).hexdigest()
    }
    return DevelopmentAudit(models=models, report=report)


def evaluate_final_holdout(
    audit: DevelopmentAudit,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the frozen development models on final-counterfactual rows once."""

    _analysis_rows_valid(rows, {FINAL_SPLIT})
    if set(audit.models) != {
        "structural_metadata",
        "frozen_keyword",
        "word_tfidf",
        "char_3_5gram",
        "decisive_deleted_word_tfidf",
        "decisive_deleted_char_3_5gram",
    }:
        raise DatasetStructureError("development audit does not contain all frozen baselines")
    baselines: dict[str, Any] = {}
    for name, model in sorted(audit.models.items()):
        scores = model.scores(rows)
        baselines[name] = {
            "selected_C": model.selected_c,
            "metrics": _metrics(rows, scores),
            "scores": [
                {
                    "sample_id": row["sample_id"],
                    "label": int(row["label"]),
                    "score": float(score),
                }
                for row, score in zip(rows, scores, strict=True)
            ],
            "model_lock": model.lock_record(),
        }
    return {
        "schema_version": "contrastive_prompts_v3_final_text_audit_v1",
        "development_lock_sha256": audit.report["development_lock_sha256"],
        "selection_performed": False,
        "evaluated_split": FINAL_SPLIT,
        "row_count": len(rows),
        "baselines": baselines,
    }

__all__ = [
    "CONTROL_SPLITS",
    "DEFAULT_C_GRID",
    "DEVELOPMENT_SPLITS",
    "FINAL_SPLIT",
    "FROZEN_KEYWORDS",
    "DatasetStructureError",
    "DevelopmentAudit",
    "FittedBaseline",
    "decisive_deleted_prompt_text",
    "evaluate_final_holdout",
    "fit_development_baselines",
    "load_development_audit",
    "load_rows",
    "prompt_text",
    "save_development_audit",
    "select_development_rows",
    "select_final_holdout_rows",
    "validate_dataset_structure",
]
