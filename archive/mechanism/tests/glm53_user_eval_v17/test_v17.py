from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src.glm53_user_eval.v17.analysis import (
    build_direction_bundle,
    causal_delta_bootstrap,
    direction_stability,
    paired_mean_ci,
    softmax,
    symmetric_kl,
)
from src.glm53_user_eval.v17.contract import canonical_sha256, validate_v17_prereg
from src.glm53_user_eval.v17.prepare import subset_proxy_rows
from src.glm53_user_eval.v17.runtime import (
    content_character_spans,
    intervention_mask,
    raw_layer_deltas,
    token_indices_for_span,
)

ROOT = Path(__file__).resolve().parents[2]


def test_preregistration_and_locked_inputs_pass() -> None:
    report = validate_v17_prereg(
        ROOT, ROOT / "pipelines/glm53_user_eval/v17/configs/prereg_v17_hua_causal.yaml"
    )
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_hua_pair_contract() -> None:
    import yaml

    payload = yaml.safe_load(
        (ROOT / "manifests/task_sources/contrastive_prompts_v2/hua16_exact_raw_v1.yaml").read_text()
    )
    assert len(payload["pairs"]) == 16
    assert [row["order_index"] for row in payload["pairs"]] == list(range(1, 17))
    assert all(row["deployment_prompt"].endswith(":") for row in payload["pairs"])
    assert all(row["evaluation_prompt"].endswith(":") for row in payload["pairs"])


def test_positive_control_is_balanced_and_final() -> None:
    payload = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v17/configs/positive_control_manifest_v1.json").read_text()
    )
    assert len(payload["rows"]) == 64
    assert sum(row["expected_class"] == "eval" for row in payload["rows"]) == 32
    assert sum(row["expected_class"] == "deploy" for row in payload["rows"]) == 32


def test_causal_schedule_is_disjoint_and_exact() -> None:
    payload = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v17/configs/causal_schedule_v17.json").read_text()
    )
    assert payload["identity_sets_disjoint"] is True
    assert payload["task_sets_disjoint"] is True
    assert len(payload["pilot"]["tasks"]) == 10
    assert len(payload["confirmation"]["tasks"]) == 25
    assert all(len(values) == 4 for values in payload["pilot"]["identities"].values())
    assert all(len(values) == 8 for values in payload["confirmation"]["identities"].values())


def test_content_spans_exclude_system_and_current_assistant() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "Sentence one. replay me"},
        {"role": "user", "content": "second user"},
        {"role": "assistant", "content": "Final answer: "},
    ]
    rendered = "<s>sys<u>first user<a>Sentence one. replay me<u>second user<a>Final answer: "
    primary = content_character_spans(rendered, messages, scope="user_content")
    assert [rendered[a:b] for a, b in primary] == ["first user", "second user"]
    replay = content_character_spans(rendered, messages, scope="user_plus_replay_assistant")
    assert [rendered[a:b] for a, b in replay] == ["first user", "replay me", "second user"]


def test_user_scope_does_not_require_template_trimmed_assistant_prefix() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "classify this"},
        {"role": "assistant", "content": "Final answer: "},
    ]
    rendered = "<s>sys<u>classify this<a>Final answer:"
    spans = content_character_spans(rendered, messages, scope="user_content")
    assert [rendered[a:b] for a, b in spans] == ["classify this"]


def test_token_span_overlap_and_mask() -> None:
    offsets = torch.tensor([[0, 0], [0, 3], [3, 6], [6, 9]])
    assert token_indices_for_span(offsets, 2, 5) == [1, 2]
    messages = [{"role": "user", "content": "abcdefghi"}]
    mask = intervention_mask(
        "abcdefghi", messages, offsets, torch.ones(4, dtype=torch.long), scope="user_content"
    )
    assert mask.tolist() == [False, True, True, True]


def test_invalid_scope_and_empty_span_fail_closed() -> None:
    with pytest.raises(ValueError):
        content_character_spans("x", [{"role": "user", "content": "x"}], scope="all")
    with pytest.raises(ValueError):
        token_indices_for_span(torch.tensor([[0, 0]]), 0, 1)


def test_raw_layer_delta_is_not_depth_normalized() -> None:
    direction = np.ones((45, 4096), dtype=np.float32)
    result = raw_layer_deltas(direction, [1, 3], 0.6)
    assert set(result) == {1, 3}
    assert np.all(result[1] == np.float32(0.6))


def test_direction_bundle_subsets_and_nulls() -> None:
    hua = np.zeros((16, 45, 4096), dtype=np.float32)
    for index in range(16):
        hua[index, :, index] = index + 1
    formality = np.zeros_like(hua)
    formality[:, :, 100] = 2
    bundle = build_direction_bundle(hua, formality, null_count=3, seed=7)
    assert bundle["all16"].shape == (45, 4096)
    assert bundle["status"].shape == (45, 4096)
    assert bundle["actor"].shape == (45, 4096)
    assert bundle["signflip"].shape == (3, 45, 4096)
    assert bundle["gaussian"].shape == (3, 45, 4096)
    for control in bundle["gaussian"]:
        assert np.max(np.abs(np.sum(control * bundle["all16"], axis=1))) < 1e-3


def test_direction_stability_is_one_for_identical_pairs() -> None:
    values = np.ones((16, 2, 3), dtype=np.float64)
    report = direction_stability(values, reps=10, seed=1)
    assert np.allclose(report["per_layer_p05"], 1)


def test_softmax_and_symmetric_kl() -> None:
    assert np.allclose(softmax(np.array([0.0, 0.0])), [0.5, 0.5])
    assert symmetric_kl(np.array([1000.0, -1000.0]), np.array([1000.0, -1000.0])) == 0
    assert symmetric_kl(np.array([1.0, 0.0]), np.array([0.0, 1.0])) > 0


def test_paired_mean_interval_is_deterministic() -> None:
    left = paired_mean_ci(np.array([-1.0, -2.0]), reps=100, seed=2)
    right = paired_mean_ci(np.array([-1.0, -2.0]), reps=100, seed=2)
    assert left == right


def _matrices(offset: float) -> dict[str, np.ndarray]:
    return {
        "famous_ai": np.full((2, 3), 70.0 + offset),
        "unknown_ai": np.full((2, 3), 70.0),
        "famous_nonai": np.full((2, 3), 70.0),
        "genpop": np.full((2, 3), 70.0),
    }


def test_causal_bootstrap_retains_arm_pairing() -> None:
    point, interval, draws = causal_delta_bootstrap(
        _matrices(-1.0), _matrices(-0.5), reps=100, seed=3
    )
    assert point == pytest.approx(0.5)
    assert interval == pytest.approx([0.5, 0.5])
    assert np.allclose(draws, 0.5)


def test_subset_proxy_rows_requires_exact_schedule_count() -> None:
    rows = [
        {"group": "famous_ai", "persona_key": "f", "stimulus_id": "d"},
    ]
    schedule = {"pilot": {"identities": {"famous_ai": ["f"]}, "tasks": ["d"], "expected_base_rows": 1}}
    assert subset_proxy_rows(rows, schedule, "pilot") == rows
    schedule["pilot"]["expected_base_rows"] = 2
    with pytest.raises(ValueError):
        subset_proxy_rows(rows, schedule, "pilot")


def test_canonical_hash_ignores_mapping_order() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_paid_bootstrap_uses_hash_bound_transformers_archive() -> None:
    archive = ROOT / (
        "artifacts/glm53_user_eval/v17/infrastructure/"
        "transformers_805a9e939fa8c1bff8d8ffdf041c051b71a914aa.tar.gz"
    )
    expected = "17890f68cae495a88b51db8105fd9bca43d5357f671fce925e3fe1f63c3cac0a"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    bootstrap = (ROOT / "infra/runpod/bootstrap_glm53_v17.sh").read_text(encoding="utf-8")
    assert "git+https://github.com/huggingface/transformers" not in bootstrap
    assert expected in bootstrap
    assert "immutable_transformers_source.tar.gz" in bootstrap


def test_v17_runtime_is_preserved_while_v18_launcher_uses_two_b300s() -> None:
    runtime = (ROOT / "pipelines/glm53_user_eval/v17/configs/runtime_v17.yaml").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "infra/runpod/new_glm53_v17_hua_pod.ps1").read_text(
        encoding="utf-8"
    )
    assert "gpu_count: 4" in runtime
    assert "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" in runtime
    assert '$WallClockMinutes = 110' in launcher
    assert '$ComputeHardCapUsd = [decimal]30.00' in launcher
    assert '$ExpectedGpuCount = 2' in launcher
    assert 'NVIDIA B300 SXM6 AC' in launcher
