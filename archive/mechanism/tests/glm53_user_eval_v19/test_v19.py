from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from src.glm53_user_eval.v8.proxy import validate_label_tokens
from src.glm53_user_eval.v19 import supervisor, verification
from src.glm53_user_eval.v19.analysis import (
    analyze_causal_rows,
    arm_matrices,
    causal_delta_bootstrap,
    interaction,
)
from src.glm53_user_eval.v19.contract import validate_v19_prereg
from src.glm53_user_eval.v19.positive_control import analyze_positive_control

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "pipelines/glm53_user_eval/v19/configs/prereg_v19_lean_hua.yaml"


def test_preregistration_and_all_immutable_hashes_pass() -> None:
    report = validate_v19_prereg(ROOT, PREREG)
    assert report["passed"] is True
    assert report["planned_model_forwards"] == 7500


def test_design_is_exact_lean_surface() -> None:
    design = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v19/configs/design_v19.json").read_text()
    )
    assert len(design["tasks"]) == 22
    assert all(len(values) == 16 for values in design["identities"].values())
    assert design["planned_base_rows"] == 1408
    assert design["reconstructable_base_rows"] == 1404
    assert design["total_user_model_forwards"] == 7020


def test_selected_rows_match_the_real_frozen_transcript_cache() -> None:
    from src.glm53_user_eval.v11.downstream import validate_downstream_assets

    design = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v19/configs/design_v19.json").read_text()
    )
    _, rows, _ = validate_downstream_assets(
        repo_root=ROOT,
        manifest_path=ROOT
        / "pipelines/glm53_user_eval/v17/configs/downstream_manifest_v17.json",
    )
    selected = supervisor._selected_proxy_rows(rows, design)
    assert len(selected) == 1404
    assert {group: sum(row["group"] == group for row in selected) for group in design["identities"]} == {
        group: 351 for group in design["identities"]
    }
    local_manifest = supervisor._local_manifest(design, selected)
    assert local_manifest["local_proxy"]["expected_pre_missing_rows_per_group"] == 352


def test_positive_controls_are_balanced_and_frozen() -> None:
    selection = json.loads(
        (
            ROOT
            / "pipelines/glm53_user_eval/v19/configs/positive_control_selection_v19.json"
        ).read_text()
    )
    assert len(selection["sample_ids"]) == len(set(selection["sample_ids"])) == 32
    assert selection["per_class_and_source"] == 8


def _pc_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    configs = ("literal_a0.6", "depth_mapped_a0.6")
    for index in range(32):
        expected = "eval" if index < 16 else "deploy"
        for codebook in ("0", "1"):
            base = 1.0 if expected == "eval" else -1.0
            rows.append(
                {
                    "sample_id": f"pc-{index:02d}",
                    "codebook_id": codebook,
                    "arm": "baseline",
                    "eval_logodds": base,
                    "correct": True,
                    "allowed_mass": 0.9,
                }
            )
            for config in configs:
                rows.append(
                    {
                        "sample_id": f"pc-{index:02d}",
                        "codebook_id": codebook,
                        "arm": f"{config}_deploy",
                        "eval_logodds": base - 0.5,
                        "correct": expected == "deploy",
                        "allowed_mass": 0.9,
                    }
                )
                rows.append(
                    {
                        "sample_id": f"pc-{index:02d}",
                        "codebook_id": codebook,
                        "arm": f"{config}_eval",
                        "eval_logodds": base + 0.5,
                        "correct": expected == "eval",
                        "allowed_mass": 0.9,
                    }
                )
    damage = [
        {"configuration": config, "symmetric_kl": 0.01 if config.startswith("literal") else 0.02}
        for config in configs
        for _ in range(32)
    ]
    return rows, damage


def test_positive_control_uses_scenario_level_bootstrap() -> None:
    rows, damage = _pc_rows()
    report = analyze_positive_control(
        rows,
        damage,
        configurations=["literal_a0.6", "depth_mapped_a0.6"],
        reps=200,
        seed=7,
        damage_kl_max=0.1,
    )
    assert report["passed"] is True
    assert report["chosen_configuration"] == "literal_a0.6"
    assert report["configurations"]["literal_a0.6"]["deploy_delta"] == -0.5


def test_positive_control_fails_wrong_direction() -> None:
    rows, damage = _pc_rows()
    for row in rows:
        if str(row["arm"]).endswith("_deploy"):
            row["eval_logodds"] = float(row["eval_logodds"]) + 1.0
    report = analyze_positive_control(
        rows,
        damage,
        configurations=["literal_a0.6", "depth_mapped_a0.6"],
        reps=100,
        seed=8,
        damage_kl_max=0.1,
    )
    assert report["passed"] is False


def _causal_rows() -> list[dict[str, object]]:
    offsets = {
        "baseline": {"famous_ai": -0.6, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "all16_deploy": {"famous_ai": -0.1, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "all16_eval": {"famous_ai": -0.9, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "status_deploy": {"famous_ai": -0.2, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
        "formality": {"famous_ai": -0.55, "unknown_ai": 0.0, "famous_nonai": 0.2, "genpop": 0.0},
    }
    rows: list[dict[str, object]] = []
    for arm, by_group in offsets.items():
        for group, offset in by_group.items():
            for identity in range(16):
                for task in range(22):
                    rows.append(
                        {
                            "sample_id": f"{group}-{identity}-{task}",
                            "arm": arm,
                            "group": group,
                            "stage_index": identity,
                            "stimulus_id": f"task-{task:02d}",
                            "expected_folded_confidence": 75.0 + offset,
                            "codebook_id": str((identity + task) % 2),
                            "allowed_mass": 0.9,
                            "full_vocab_argmax_allowed": True,
                            "conditional_entropy": 0.5,
                        }
                    )
    return rows


def test_interaction_known_value() -> None:
    matrices = arm_matrices(_causal_rows(), "baseline")
    point, _ = interaction(matrices)
    assert point == pytest.approx(-0.8)


def test_causal_delta_bootstrap_retains_arms() -> None:
    rows = _causal_rows()
    baseline = arm_matrices(rows, "baseline")
    candidate = arm_matrices(rows, "all16_deploy")
    point, interval, draws = causal_delta_bootstrap(
        baseline, candidate, reps=100, seed=9
    )
    assert point == pytest.approx(0.5)
    assert interval[0] == pytest.approx(0.5)
    assert draws.shape == (100,)


def test_causal_analysis_is_row_order_invariant() -> None:
    rows = _causal_rows()
    left = analyze_causal_rows(rows, reps=100, seed=10)
    right = analyze_causal_rows(list(reversed(rows)), reps=100, seed=10)
    assert left["baseline_interaction_pp"] == right["baseline_interaction_pp"]
    assert left["arms"]["all16_deploy"]["delta_pp"] == right["arms"]["all16_deploy"]["delta_pp"]


def test_causal_duplicate_key_fails_closed() -> None:
    rows = _causal_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        analyze_causal_rows(rows, reps=10, seed=11)


def test_causal_requires_all_five_arms() -> None:
    rows = [row for row in _causal_rows() if row["arm"] != "formality"]
    with pytest.raises(ValueError, match="arms"):
        analyze_causal_rows(rows, reps=10, seed=12)


def test_independent_verifier_does_not_import_primary_analysis() -> None:
    source = inspect.getsource(verification)
    assert "v19.analysis" not in source


def test_batch_size_and_hardware_are_frozen() -> None:
    import yaml

    runtime = yaml.safe_load(
        (ROOT / "pipelines/glm53_user_eval/v19/configs/runtime_v19.yaml").read_text()
    )
    assert runtime["forward"]["primary_batch_size"] == 1
    assert runtime["runpod"]["gpu_id"] == "NVIDIA B300 SXM6 AC"
    assert runtime["runpod"]["gpu_count"] == 2
    assert runtime["runpod"]["allow_gpu_type_fallback"] is False
    assert runtime["throughput_gate"]["projection_headroom_multiplier"] == 1.10


def test_post_control_projection_uses_the_frozen_headroom() -> None:
    source = inspect.getsource(supervisor.run_paid_ladder)
    assert 'float(throughput["projection_headroom_multiplier"])' in source
    assert "1.30 * int(design" not in source


def test_no_unsupported_specificity_claim_is_registered() -> None:
    import yaml

    prereg = yaml.safe_load(PREREG.read_text())
    assert prereg["scope"]["actor_control_claim"] is False
    assert prereg["scope"]["random_direction_claim"] is False
    assert prereg["scope"]["prompting_comparison_claim"] is False


def test_token_contract_uses_full_frozen_cache_not_scientific_subset() -> None:
    source = inspect.getsource(supervisor.run_paid_ladder)
    assert "proxy_rows=all_proxy_rows" in source
    assert "proxy_rows=selected" not in source


def test_multimodal_processor_token_contract_uses_text_tokenizer() -> None:
    class FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            assert add_special_tokens is False
            labels = {"A": 101, "B": 102}
            suffix = [labels[text[-1]]] if text[-1] in labels else []
            return {"input_ids": [1, 2, 3, *suffix]}

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            assert kwargs["tokenize"] is False
            return "|".join(message["content"] for message in messages)

    result = validate_label_tokens(
        FakeProcessor(), [{"role": "assistant", "content": "Final answer: "}], ["A", "B"]
    )
    assert result == {"label_ids": {"A": 101, "B": 102}, "base_token_count": 3}
