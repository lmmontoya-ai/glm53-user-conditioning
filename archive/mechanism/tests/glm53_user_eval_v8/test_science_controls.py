from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from src.glm53_user_eval.v8.science import (
    analyze_causal_arms,
    analyze_recruitment_baselines,
    evaluate_eval_positive_controls,
    evaluate_user_positive_controls,
    fit_user_directions,
)


class _PositiveControlRuntime:
    def forward(
        self,
        messages: list[list[dict[str, str]]],
        *,
        layers: list[int],
        continuation: bool,
        intervention: Any,
    ) -> dict[str, torch.Tensor]:
        del layers, continuation
        alpha = 0.0 if intervention is None else float(intervention)
        logits = torch.zeros((len(messages), 4), dtype=torch.float32)
        logits[:, 1] = alpha
        return {"logits": logits}


def test_eval_positive_control_uses_observed_logits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.glm53_user_eval.v8.science.intervention_from_direction",
        lambda *_args, alpha, **_kwargs: alpha,
    )
    vector_path = tmp_path / "vectors.npz"
    np.savez(vector_path, placeholder=np.zeros(1))
    rows = [
        {
            "sample_id": f"sample-{label}",
            "pair_id": f"pair-{label}",
            "family_id": "family",
            "label": label,
            "split": "test",
            "messages": [{"role": "user", "content": "context"}],
        }
        for label in (0, 1)
    ]
    report = evaluate_eval_positive_controls(
        _PositiveControlRuntime(),
        rows,
        direction_report={"selected_layer": 1},
        vector_path=vector_path,
        construction="paired_mean",
        label_ids={"A": 1, "B": 2},
        batch_size=2,
        alphas=(-1.0, -0.5, 0.0, 0.5, 1.0),
        output_root=tmp_path,
    )
    assert report["passed"] is True
    assert report["passed_negative_alphas"] == {"-1.0": True, "-0.5": True}
    assert (tmp_path / "eval_positive_control_rows.jsonl").is_file()


def test_user_positive_control_uses_heldout_target_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.glm53_user_eval.v8.science.user_direction_intervention",
        lambda *_args, alpha, **_kwargs: alpha,
    )
    vector_path = tmp_path / "user_vectors.npz"
    np.savez(vector_path, placeholder=np.zeros(1))
    rows = [
        {
            "sample_id": f"{group}-{index}",
            "pair_index": index,
            "template_id": "neutral",
            "group": group,
            "target_identity": True,
            "messages": [{"role": "user", "content": "neutral task"}],
        }
        for group in ("F", "U")
        for index in range(2)
    ]
    report = evaluate_user_positive_controls(
        _PositiveControlRuntime(),
        rows,
        direction_report={"selected_layer": 1},
        vector_path=vector_path,
        concept="name",
        label_ids={"A": 1, "B": 2},
        batch_size=2,
        alphas=(-1.0, -0.5, 0.0, 0.5, 1.0),
        output_root=tmp_path,
    )
    assert report["passed"] is True
    assert report["heldout_rows"] == 4
    assert report["positive_group"] == "F"
    assert report["negative_group"] == "U"


def _causal_rows() -> list[dict[str, Any]]:
    baseline = {"famous_ai": 60.0, "unknown_ai": 70.0, "famous_nonai": 65.0, "genpop": 65.0}
    arms = {
        "baseline": baseline,
        "candidate": baseline | {"famous_ai": 68.0},
        "opposite": baseline | {"famous_ai": 58.0},
    }
    for index in range(20):
        arms[f"random_{index:02d}"] = baseline | {"famous_ai": 60.1 + index * 0.05}
    rows: list[dict[str, Any]] = []
    for arm, values in arms.items():
        for group, value in values.items():
            for identity in range(2):
                for task in range(2):
                    rows.append(
                        {
                            "arm_id": arm,
                            "group": group,
                            "persona_key": f"{group}-{identity}",
                            "pair_index": identity,
                            "stimulus_id": f"task-{task}",
                            "codebook_id": str(task),
                            "expected_folded_confidence": value,
                            "original_folded_confidence": value + identity,
                            "allowed_mass": 0.90,
                            "conditional_entropy": 2.0,
                            "argmax_label_position": task,
                            "full_vocab_argmax_allowed": True,
                        }
                    )
    return rows


def _causal_config() -> dict[str, Any]:
    return {
        "parent_result": {"interaction_pp": -0.65},
        "recruitment": {"bootstrap_reps": 50, "bootstrap_seed": 11},
        "intervention": {"bootstrap_reps": 50, "bootstrap_seed": 12},
    }


def test_causal_analysis_propagates_failed_positive_control() -> None:
    report = analyze_causal_arms(
        _causal_rows(),
        baseline_arm="baseline",
        candidate_arm="candidate",
        opposite_arm="opposite",
        positive_control_passed=False,
        config=_causal_config(),
    )
    assert report["positive_control_passed"] is False
    assert report["checks"]["positive_control"] is False
    assert report["passed"] is False


def test_causal_analysis_can_pass_with_measured_positive_control() -> None:
    report = analyze_causal_arms(
        _causal_rows(),
        baseline_arm="baseline",
        candidate_arm="candidate",
        opposite_arm="opposite",
        positive_control_passed=True,
        config=_causal_config(),
    )
    assert report["control_count"] == 20
    assert report["candidate_control_rank"] == 1
    assert report["checks"]["positive_control"] is True
    assert report["passed"] is True


def test_recruitment_baselines_emit_all_frozen_controls() -> None:
    rows = []
    direct = []
    groups = ("F", "U", "FN", "G")
    for group_index, group in enumerate(groups):
        for identity in range(2):
            for template in range(2):
                row = {
                    "sample_id": f"{group}-{identity}-{template}",
                    "group": group,
                    "persona_key": f"{group}-{identity}",
                    "pair_index": identity,
                    "template_id": f"template-{template}",
                    "messages": [
                        {"role": "system", "content": f"User group {group}"},
                        {"role": "user", "content": "Summarize a neutral sentence."},
                    ],
                }
                rows.append(row)
                direct.append(
                    {key: value for key, value in row.items() if key != "messages"}
                    | {"eval_log_odds": float(group_index)}
                )
    features = np.arange(len(rows) * 3, dtype=np.float32).reshape(len(rows), 3)
    report = analyze_recruitment_baselines(
        rows,
        features,
        direct,
        probe={
            "mean": np.zeros(3),
            "scale": np.ones(3),
            "coef": np.asarray([1.0, 0.0, 0.0]),
            "intercept": 0.0,
            "source_sd": 1.0,
        },
        random_seed=7,
        random_count=4,
    )
    assert report["random_direction_count"] == 4
    assert len(report["random_direction_interactions_sd"]) == 4
    assert "direct_prompt_interaction_log_odds" in report
    assert "length_and_cue_adjusted_interaction_sd" in report


def test_user_direction_splits_are_auditable_and_fame_has_no_identity_overlap(
    tmp_path,
) -> None:
    rows = []
    features = []
    group_feature = {
        "F": [3.0, 1.0, 0.0],
        "U": [2.0, 1.0, 0.0],
        "G": [1.0, 0.0, 0.0],
        "FN": [3.0, 0.0, 1.0],
    }
    for group in ("F", "U", "G", "FN"):
        for pair_index in range(20):
            sample_id = f"{group}-{pair_index}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "group": group,
                    "pair_index": pair_index,
                    "template_id": "neutral",
                    "organization": f"org-{pair_index}",
                    "target_identity": group in {"F", "U", "G"} and pair_index < 4,
                    "target_organization": pair_index < 4,
                }
            )
            features.append(group_feature[group])
    feature_path = tmp_path / "features.npz"
    np.savez(feature_path, l2__prompt_final=np.asarray(features, dtype=np.float32))
    feature_path.with_suffix(".jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
    )
    report = fit_user_directions(feature_path, selected_layer=2, output_root=tmp_path / "out")
    for concept in ("name", "affiliation", "fame"):
        values = report["concepts"][concept]
        assert not set(values["training_sample_ids"]) & set(values["test_sample_ids"])
    fame = report["concepts"]["fame"]
    assert all(int(sample.rsplit("-", 1)[1]) >= 16 for sample in fame["training_sample_ids"])
    assert all(int(sample.rsplit("-", 1)[1]) < 16 for sample in fame["test_sample_ids"])
