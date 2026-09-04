from __future__ import annotations

import ast
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pipelines.glm53_user_eval.v16 import run as v16_run
from src.glm53_user_eval.v16.contract import (
    DATASET_MANIFEST_SHA256,
    DATASET_SHA256,
    MODEL_REVISION,
    PARENT_COMMIT,
    PARENT_TAG,
    sha256_file,
    validate_dataset_counts,
    validate_parent,
    validate_prereg,
)
from src.glm53_user_eval.v16.dataset import (
    feature_metadata,
    feature_partition,
    load_rows,
    verify_reused_rows,
)
from src.glm53_user_eval.v16.probes import (
    binary_metrics,
    fit_source_development,
    pair_preserving_labels,
)
from src.glm53_user_eval.v16.source_analysis import _fresh_control_analysis
from src.glm53_user_eval.v16.source_decision import decide_source

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v5/samples.jsonl"
MANIFEST = ROOT / "artifacts/datasets/contrastive_prompts_v5/manifest.json"
AUDIT = ROOT / "artifacts/datasets/contrastive_prompts_v5/tokenizer_audit_v16.json"
PREREG = ROOT / "pipelines/glm53_user_eval/v16/configs/prereg_v16_source_activation.yaml"
RUNTIME = ROOT / "pipelines/glm53_user_eval/v16/configs/runtime_v16.yaml"
DOWNSTREAM = ROOT / "pipelines/glm53_user_eval/v16/configs/downstream_manifest_v16.json"


def test_paid_commit_guard_allows_only_signed_input_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40

    def fake_git(*args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return commit
        if args[:3] == ("rev-list", "-n", "1"):
            return commit
        if args[:2] == ("diff", "--name-only"):
            return "artifacts/datasets/contrastive_prompts_v5/samples.jsonl"
        if args[:2] == ("ls-files", "--others"):
            return "artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json"
        raise AssertionError(args)

    monkeypatch.setattr(v16_run, "_git", fake_git)
    assert v16_run._require_preregistered_commit() == commit


@pytest.mark.parametrize(
    ("changed", "untracked"),
    [
        ("src/glm53_user_eval/v16/runtime.py", ""),
        ("", "unexpected.py"),
    ],
)
def test_paid_commit_guard_rejects_non_input_changes(
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
    untracked: str,
) -> None:
    commit = "b" * 40

    def fake_git(*args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return commit
        if args[:3] == ("rev-list", "-n", "1"):
            return commit
        if args[:2] == ("diff", "--name-only"):
            return changed
        if args[:2] == ("ls-files", "--others"):
            return untracked
        raise AssertionError(args)

    monkeypatch.setattr(v16_run, "_git", fake_git)
    with pytest.raises(ValueError, match="clean preregistered commit"):
        v16_run._require_preregistered_commit()


def test_paid_environment_uses_nonsecret_s3_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-test")
    monkeypatch.setenv("GLM53_V16_AGGREGATE_RATE_USD", "15.78")
    monkeypatch.setenv("GLM53_V16_LAUNCH_BALANCE_USD", "41.50")
    monkeypatch.setenv("GLM53_V16_BALANCE_FLOOR_USD", "21.50")
    monkeypatch.setenv("GLM53_V16_S3_TRANSPORT_VERIFIED", "1")
    monkeypatch.setenv(
        "GLM53_V16_DEADLINE_UTC",
        (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=30)).isoformat(),
    )
    report = v16_run._paid_environment(
        {"runpod": {"aggregate_gpu_rate_cap_usd_per_hour": 16.5}}
    )
    assert report["passed"] is True
    assert report["effective_compute_cap_usd"] == pytest.approx(20.0)
    assert report["checks"]["science_process_has_no_s3_secret"] is True


def test_paid_environment_rejects_secret_in_science_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-test")
    monkeypatch.setenv("GLM53_V16_AGGREGATE_RATE_USD", "15.78")
    monkeypatch.setenv("GLM53_V16_LAUNCH_BALANCE_USD", "41.50")
    monkeypatch.setenv("GLM53_V16_BALANCE_FLOOR_USD", "21.50")
    monkeypatch.setenv("GLM53_V16_S3_TRANSPORT_VERIFIED", "1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-reach-science")
    monkeypatch.setenv(
        "GLM53_V16_DEADLINE_UTC",
        (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=30)).isoformat(),
    )
    with pytest.raises(ValueError, match="paid environment failed"):
        v16_run._paid_environment(
            {"runpod": {"aggregate_gpu_rate_cap_usd_per_hour": 16.5}}
        )


def test_parent_tag_and_commit() -> None:
    report = validate_parent(ROOT)
    assert report["tag_commit"] == PARENT_COMMIT
    assert PARENT_TAG == "glm53-user-eval-v15-final"


def test_parent_hashes() -> None:
    assert sha256_file(DATASET) == DATASET_SHA256
    assert sha256_file(MANIFEST) == DATASET_MANIFEST_SHA256


def test_preregistration_passes() -> None:
    assert validate_prereg(ROOT, PREREG)["passed"]


def test_exact_model_revision_locked() -> None:
    assert MODEL_REVISION == "04c4e9e95c5da8862dced7e5056455116f83a7e0"


def test_dataset_counts() -> None:
    rows = load_rows(DATASET)
    report = validate_dataset_counts(rows)
    assert report["row_count"] == 576
    assert report["split_counts"]["neutral_controls"] == 64


def test_binary_and_control_counts() -> None:
    rows = load_rows(DATASET)
    assert Counter(row["label"] for row in rows) == Counter({0: 240, 1: 240, None: 96})
    controls = Counter(
        row["control_expected_label"] for row in rows if row["split"] == "neutral_controls"
    )
    assert controls == Counter({"eval": 16, "deploy": 16, "ambiguous": 16, "no_status": 16})


def test_reused_rows_are_byte_identical() -> None:
    assert verify_reused_rows(
        DATASET, ROOT / "artifacts/datasets/contrastive_prompts_v4/samples.jsonl"
    )["passed"]


def test_feature_partition_counts() -> None:
    rows = load_rows(DATASET)
    assert Counter(feature_partition(row) for row in rows) == Counter(
        {"development": 368, "final_binary": 112, "factorial": 32, "fresh_controls": 64}
    )


@pytest.mark.parametrize(
    ("split", "expected"),
    [
        ("train", "development"),
        ("validation", "development"),
        ("development_counterfactual", "development"),
        ("ordinary_test", "final_binary"),
        ("final_counterfactual", "final_binary"),
        ("factorial_calibration", "factorial"),
        ("neutral_controls", "fresh_controls"),
    ],
)
def test_partition_map(split: str, expected: str) -> None:
    assert feature_partition({"split": split}) == expected


def test_unknown_partition_fails() -> None:
    with pytest.raises(ValueError):
        feature_partition({"split": "new_split"})


def test_control_label_absent_from_feature_metadata() -> None:
    row = next(row for row in load_rows(DATASET) if row["split"] == "neutral_controls")
    metadata = feature_metadata(row, row_index=0, part="x.npz", part_sha256="a" * 64)
    assert "control_expected_label" not in metadata
    assert "acceptable_judge_labels" not in metadata


def test_tokenizer_audit_is_fresh_and_complete() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "glm53_v16_tokenizer_audit_v1"
    assert audit["dataset_id"] == "contrastive_prompts_v5"
    assert audit["row_count"] == 576
    assert audit["pair_contract"]["checked_pair_count"] == 240
    assert audit["pair_contract"]["singleton_control_count"] == 96


def test_tokenizer_records_have_required_fields() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    required = {
        "sample_id",
        "rendered_sha256",
        "token_ids",
        "token_ids_sha256",
        "rendered_token_count",
        "attention_mask_count",
        "prompt_final_index",
        "shared_suffix_char_span",
        "shared_suffix_token_indices",
        "masked_prompt_token_indices",
        "decisive_token_indices",
    }
    assert all(required <= set(row) for row in audit["records"])


def test_tokenizer_pair_alignment() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    by_pair: dict[str, list[dict]] = {}
    for row in audit["records"]:
        if row["label"] in {0, 1}:
            by_pair.setdefault(row["pair_id"], []).append(row)
    assert len(by_pair) == 240
    for members in by_pair.values():
        assert len(members) == 2
        assert members[0]["rendered_token_count"] == members[1]["rendered_token_count"]
        assert members[0]["shared_suffix_token_ids"] == members[1]["shared_suffix_token_ids"]
        assert members[0]["shared_suffix_start_index"] == members[1]["shared_suffix_start_index"]


def test_binary_metrics_known_separation() -> None:
    report = binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([-2.0, -1.0, 1.0, 2.0]))
    assert report["auroc"] == 1.0
    assert report["score_gap"] == 3.0


def test_pair_preserving_permutation_keeps_one_label_each() -> None:
    metadata = []
    for split, pair in zip(
        ("train", "validation", "development_counterfactual"), ("a", "b", "c"), strict=True
    ):
        metadata.extend(
            [
                {"split": split, "pair_id": pair, "label": 0},
                {"split": split, "pair_id": pair, "label": 1},
            ]
        )
    labels = pair_preserving_labels(metadata, seed=42)
    assert all(sorted(labels[index : index + 2].tolist()) == [0, 1] for index in range(0, 6, 2))


def test_development_fit_never_needs_final_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.glm53_user_eval.v16 import probes

    monkeypatch.setattr(probes, "C_GRID", (0.1,))
    metadata = []
    features = np.zeros((6, 45, 4096), dtype=np.float32)
    for split_index, split in enumerate(("train", "validation", "development_counterfactual")):
        for label in (0, 1):
            row_index = split_index * 2 + label
            metadata.append({"split": split, "pair_id": f"p{split_index}", "label": label})
            features[row_index, :, 0] = -1.0 if label == 0 else 1.0
    fit = fit_source_development(features, metadata)
    assert fit.logistic.layer == 0
    assert fit.report["selection_used_final_rows"] is False
    assert fit.report["selection_used_fresh_controls"] is False


def _control_fixture() -> tuple[np.ndarray, list[dict], dict]:
    labels = ["deploy"] * 16 + ["ambiguous"] * 16 + ["eval"] * 16 + ["no_status"] * 16
    scores = np.asarray([-2.0] * 16 + [0.0] * 16 + [2.0] * 16 + [0.0] * 16)
    metadata = [
        {
            "sample_id": f"s{index}",
            "nuisance": {
                "formal_register": bool(index % 2),
                "ai_vocabulary": bool((index // 2) % 2),
            },
        }
        for index in range(64)
    ]
    return scores, metadata, {f"s{index}": label for index, label in enumerate(labels)}


def test_fresh_control_analysis_handles_mixed_as_mixed() -> None:
    values, metadata, contract = _control_fixture()
    model = SimpleNamespace(upper_80_tpr=0.5, lower_80_tnr=-0.5)
    report = _fresh_control_analysis("logistic", model, values, metadata, contract)
    assert report["passed"]
    assert report["mean_scores"]["deploy"] < report["mean_scores"]["ambiguous"]
    assert report["mean_scores"]["ambiguous"] < report["mean_scores"]["eval"]


def test_fresh_control_no_status_abstains() -> None:
    values, metadata, contract = _control_fixture()
    model = SimpleNamespace(upper_80_tpr=0.5, lower_80_tnr=-0.5)
    report = _fresh_control_analysis("logistic", model, values, metadata, contract)
    assert report["no_status_eval_fire_rate"] == 0.0
    assert report["no_status_abstention_rate"] == 1.0


def _passing_source_analysis() -> dict:
    factorial = {
        "checks": {
            "real_external_recipient_negative": True,
            "operational_consequence_negative": True,
            "model_scoring_positive": True,
            "comparison_among_models_positive": True,
            "clean_evaluation_mean_above_clean_deployment": True,
            "register_smaller_than_every_decisive_factor": True,
        }
    }
    controls = {"checks": {name: True for name in (
        "clean_auroc", "eval_tpr", "deploy_tnr", "clean_gap", "no_status_eval_fire",
        "no_status_abstention", "mixed_order", "operational_negative", "assessment_positive",
        "formal_subordinate", "ai_subordinate", "abstention_interval"
    )}}
    return {
        "models": {
            "logistic": {
                "ordinary_test": {"auroc": 0.9},
                "final_counterfactual": {"auroc": 0.8, "fpr_at_80_tpr": 0.2, "score_gap": 1.0},
                "factorial_calibration": factorial,
                "fresh_controls": controls,
            },
            "paired_mean": {"factorial_calibration": factorial, "fresh_controls": controls},
        },
        "direction_agreement": {
            "raw_cosine": 0.7,
            "final_score_spearman": 0.8,
            "logistic_score_gap": 1.0,
            "paired_mean_score_gap": 1.0,
        },
        "paired_direction_stability": {"fifth_percentile_cosine": 0.6},
        "leave_one_training_generator_score_gaps": {"a": 0.2},
        "selection_used_final_rows": False,
        "selection_used_factorial_rows": False,
        "selection_used_fresh_controls": False,
    }


def test_source_decision_passes_only_all_checks() -> None:
    decision = decide_source(_passing_source_analysis(), {"add_one_empirical_p": 0.009})
    assert decision["passed"]
    assert decision["authorization"]["local_proxy_parity"]
    assert not decision["authorization"]["user_recruitment"]


def test_source_decision_fails_closed() -> None:
    analysis = _passing_source_analysis()
    analysis["models"]["logistic"]["final_counterfactual"]["auroc"] = 0.74
    decision = decide_source(analysis, {"add_one_empirical_p": 0.009})
    assert not decision["passed"]
    assert decision["decision"] == "stop_before_local_parity"


def test_exact_b300_runtime_lock() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert "gpu_id: NVIDIA B300 SXM6 AC" in text
    assert "gpu_count: 2" in text
    assert "network_volume_attached: false" in text


def test_runtime_uses_exact_software() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert "torch: 2.13.0+cu130" in text
    assert "cuda: '13.0'" in text
    assert "805a9e939fa8c1bff8d8ffdf041c051b71a914aa" in text


def test_budget_and_deadline_are_bounded() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert "compute_hard_cap_usd: 25.00" in text
    assert "wall_clock_hard_cap_minutes: 94" in text
    assert "minimum_uncommitted_balance_usd: 15.00" in text


def test_downstream_surface_counts_and_gates() -> None:
    value = json.loads(DOWNSTREAM.read_text(encoding="utf-8"))
    assert value["local_proxy"]["expected_eligible_rows"] == 6387
    assert value["recruitment"]["expected_rows"] == 2240
    assert value["execution"]["early_cot_forbidden"]
    assert value["execution"]["steering_forbidden"]


def test_verifier_does_not_import_primary_modules() -> None:
    path = ROOT / "src/glm53_user_eval/v16/verification.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(name.endswith(("v16.probes", "v16.source_analysis", "v16.source_decision")) for name in imports)


def test_paid_cli_has_no_cot_or_steering_command() -> None:
    source = (ROOT / "pipelines/glm53_user_eval/v16/run.py").read_text(encoding="utf-8")
    command_line = next(line for line in source.splitlines() if "choices=(" in line)
    assert "cot" not in command_line.lower()
    assert "steer" not in command_line.lower()


@pytest.mark.parametrize(
    ("relative_path", "forbidden"),
    [
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "contrastive_prompts_v3"),
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "source_text_instrument_valid"),
        ("infra/runpod/bootstrap_glm53_v16.sh", "contrastive_prompts_v3"),
        ("infra/runpod/bootstrap_glm53_v16.sh", "paid-ladder"),
    ],
)
def test_paid_infrastructure_has_no_stale_v11_contract(relative_path: str, forbidden: str) -> None:
    assert forbidden not in (ROOT / relative_path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("relative_path", "required"),
    [
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "fresh_control_bank_validated_by_both_codex_judges"),
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "NVIDIA B300 SXM6 AC"),
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "$WallClockMinutes = 75"),
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "$ComputeHardCapUsd = [decimal]20.00"),
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "${S3Bucket}?list-type=2"),
        ("infra/runpod/new_glm53_v16_source_pod.ps1", "glm53-user-eval-v16-preregistered-a12"),
        ("infra/runpod/bootstrap_glm53_v16.sh", "paid-supervisor"),
        ("infra/runpod/bootstrap_glm53_v16.sh", "GLM53_V16_LAUNCH_BALANCE_USD"),
        ("infra/runpod/bootstrap_glm53_v16.sh", "tokenizer_audit_v16.json"),
        ("infra/runpod/bootstrap_glm53_v16.sh", "stage_glm53_v8_hf_local.py"),
    ],
)
def test_paid_infrastructure_contains_required_contract(relative_path: str, required: str) -> None:
    assert required in (ROOT / relative_path).read_text(encoding="utf-8")


@pytest.mark.parametrize("field", ["local_proxy_parity", "prompt_recruitment", "first_cot_transfer", "steering"])
def test_parent_does_not_pre_authorize_downstream(field: str) -> None:
    decision = json.loads(
        (ROOT / "artifacts/glm53_user_eval/v15/reports/codex_cohort/decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["authorization"][field] is False


def test_infrastructure_hashes_are_preregistered() -> None:
    import yaml

    config = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    infrastructure = config["infrastructure"]
    for name in ("launcher", "bootstrap", "watchdog"):
        path = ROOT / infrastructure[f"{name}_path"]
        assert sha256_file(path) == infrastructure[f"{name}_sha256"]


def test_downstream_assets_match_locked_hashes() -> None:
    value = json.loads(DOWNSTREAM.read_text(encoding="utf-8"))
    for record in value["assets"].values():
        path_value = record.get("path") or record.get("target_path")
        if path_value is None:
            continue
        path = ROOT / path_value
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]
