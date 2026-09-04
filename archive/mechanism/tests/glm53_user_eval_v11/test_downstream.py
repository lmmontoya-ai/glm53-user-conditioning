from __future__ import annotations

import ast
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pipelines.glm53_user_eval.v11 import run as supervisor
from src.glm53_user_eval.v11.downstream import (
    _matrix,
    analyze_local_proxy,
    build_manual_audit_packet,
    calibrate_downstream_batch,
    downstream_resource_decision,
    four_group_bootstrap,
    load_manifest,
    proxy_from_compact_logits,
    validate_downstream_assets,
)
from src.glm53_user_eval.v11.runtime import LoadedV11GLM53

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "pipelines/glm53_user_eval/v11/configs/downstream_manifest_v1.json"


def test_frozen_downstream_surfaces_and_hashes() -> None:
    report, proxy, recruitment = validate_downstream_assets(
        repo_root=ROOT,
        manifest_path=MANIFEST,
    )
    assert report["passed"] is True
    assert len(proxy) == 6387
    assert len(recruitment) == 2240
    assert len(report["technical_errors"]) == 13
    assert report["selected_source_log_count"] == 80
    assert report["label_ids"] == [362, 425, 356, 422, 468, 434, 479, 472, 358, 619, 730]


def test_proxy_compact_logits_match_closed_form() -> None:
    logits = np.arange(11, dtype=np.float64) / 3
    result = proxy_from_compact_logits(
        logits,
        full_logsumexp=float(np.log(np.exp(logits).sum()) + 0.2),
        full_argmax_token_id=730,
        label_ids=[362, 425, 356, 422, 468, 434, 479, 472, 358, 619, 730],
        codebook_values=list(range(0, 101, 10)),
    )
    assert 50 <= result["expected_folded_confidence"] <= 100
    assert result["allowed_mass"] == pytest.approx(np.exp(-0.2))
    assert result["argmax_label_position"] == 10
    assert result["full_vocab_argmax_allowed"] is True


def test_four_group_point_is_equal_person_not_raw_row_weighted() -> None:
    base = np.zeros((70, 10), dtype=np.float64)
    famous = base.copy()
    famous[0] = [10.0] * 9 + [np.nan]
    point, _, _ = four_group_bootstrap(
        {
            "famous_ai": famous,
            "unknown_ai": base.copy(),
            "famous_nonai": base.copy(),
            "genpop": base.copy(),
        },
        reps=8,
        seed=7,
    )
    assert point == pytest.approx(10.0 / 70.0)
    assert point != pytest.approx(np.nanmean(famous))


def test_matrix_preserves_all_70_identity_slots_and_fu_alignment() -> None:
    rows = [
        {
            "group": group,
            "pair_index": index,
            "stimulus_id": "task",
            "score": float(index),
        }
        for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop")
        for index in range(70)
        if not (group == "famous_ai" and index == 17)
    ]
    matrices = _matrix(rows, value_key="score", task_key="stimulus_id")
    assert all(value.shape == (70, 1) for value in matrices.values())
    assert np.isnan(matrices["famous_ai"][17, 0])
    assert matrices["unknown_ai"][17, 0] == 17


def test_missing_entire_identity_fails_main_proxy_estimand() -> None:
    manifest = load_manifest(MANIFEST)
    manifest["local_proxy"]["bootstrap_reps"] = 4
    rows = []
    for group in ("famous_ai", "unknown_ai", "famous_nonai", "genpop"):
        for index in range(16):
            if group == "famous_ai" and index == 9:
                continue
            for task in ("a", "b"):
                rows.append(
                    {
                        "group": group,
                        "analysis_index": index,
                        "pair_index": index,
                        "persona_key": f"{group}-{index}",
                        "stimulus_id": task,
                        "codebook_id": "0" if task == "a" else "1",
                        "expected_folded_confidence": 75.0,
                        "original_folded_confidence": 75.0,
                        "allowed_mass": 0.99,
                        "conditional_entropy": 1.0,
                        "full_vocab_argmax_allowed": True,
                    }
                )
    with pytest.raises(ValueError, match="identities have no valid"):
        analyze_local_proxy(rows, manifest)


def test_manual_packet_is_score_blind_and_freezes_quotas() -> None:
    manifest = load_manifest(MANIFEST)
    _, proxy, recruitment = validate_downstream_assets(repo_root=ROOT, manifest_path=MANIFEST)
    packet, report = build_manual_audit_packet(
        proxy_rows=proxy,
        recruitment_rows=recruitment,
        manifest=manifest,
    )
    assert len([row for row in packet if row["surface"] == "proxy"]) == 40
    assert len([row for row in packet if row["surface"] == "recruitment"]) == 32
    assert all("expected_folded_confidence" not in row for row in packet)
    assert report["passed"] is False
    assert report["human_review_required"] is True


def test_resource_gate_requires_deadline_and_cost_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_manifest(MANIFEST)
    monkeypatch.setattr("src.glm53_user_eval.v11.downstream.time.time", lambda: 1000.0)
    passed = downstream_resource_decision(
        proxy_seconds=1,
        proxy_benchmark_rows=4,
        proxy_total_rows=40,
        recruitment_seconds=1,
        recruitment_benchmark_rows=4,
        recruitment_total_rows=40,
        deadline_utc_seconds=2000,
        hourly_rate_usd=10,
        manifest=manifest,
    )
    assert passed["passed"] is True
    failed = downstream_resource_decision(
        proxy_seconds=100,
        proxy_benchmark_rows=1,
        proxy_total_rows=6387,
        recruitment_seconds=100,
        recruitment_benchmark_rows=1,
        recruitment_total_rows=2240,
        deadline_utc_seconds=2000,
        hourly_rate_usd=10,
        manifest=manifest,
    )
    assert failed["passed"] is False


def test_paid_ladder_is_in_process_and_closes_runtime() -> None:
    source = Path(supervisor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    paid = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "command_paid_ladder"
    )
    calls = {
        node.func.id
        for node in ast.walk(paid)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "command_extract_source" in calls
    assert "_run_source_gate_in_process" in calls
    assert "subprocess" not in source[source.index("def command_paid_ladder") :]
    assert "runtime.close()" in source[source.index("def command_paid_ladder") :]
    paid_source = source[source.index("def command_paid_ladder") : source.index("COMMANDS =")]
    assert paid_source.index('if not proxy_decision["authorization"]["user_recruitment"]') < (
        paid_source.index("recruitment_calibration =")
    )


def test_independent_verifier_does_not_import_primary_analysis() -> None:
    source = (ROOT / "src/glm53_user_eval/v11/downstream_verification.py").read_text(
        encoding="utf-8"
    )
    assert "glm53_user_eval.v11.downstream import" not in source
    assert "glm53_user_eval.v8.science" not in source
    assert "glm53_user_eval.v8.decisions" not in source
    assert 'scientific_gate_would_pass": bool(primary' not in source


def test_downstream_manifest_forbids_cot_and_steering() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest["source_gate"]["same_loaded_model_process_required"] is True
    assert manifest["source_gate"]["model_reload_allowed"] is False
    assert manifest["execution"]["early_cot_forbidden"] is True
    assert manifest["execution"]["steering_forbidden"] is True
    assert manifest["manual_audit"]["proxy_random_rows"] >= 40
    assert manifest["manual_audit"]["recruitment_random_rows"] >= 32


def test_runtime_keeps_only_last_logits_and_one_selected_layer() -> None:
    source = (ROOT / "src/glm53_user_eval/v11/runtime.py").read_text(encoding="utf-8")
    body = source[source.index("def forward_downstream") : source.index("def no_op_equivalence")]
    assert "logits_to_keep=1" in body
    assert "self.layers[selected_layer].register_forward_hook" in body
    assert "handle.remove()" in body
    assert "output.logits[:, 0]" in body
    assert 'tokenizer.padding_side = "left"' in body
    assert 'encoded["attention_mask"][:, -1] == 1' in body


def test_v11_codebooks_are_folded_antithetic_and_balanced() -> None:
    codebooks = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v11/configs/proxy_codebooks_v2.json").read_text(
            encoding="utf-8"
        )
    )["codebooks"]
    expected = set(range(0, 101, 10))
    assert set(codebooks["0"].values()) == expected
    assert set(codebooks["1"].values()) == expected
    folded = {
        key: [max(value, 100 - value) for value in mapping.values()]
        for key, mapping in codebooks.items()
    }
    assert sorted(folded["0"]) == sorted(folded["1"])
    assert all(left != right for left, right in zip(folded["0"], folded["1"], strict=True))
    _, proxy, _ = validate_downstream_assets(repo_root=ROOT, manifest_path=MANIFEST)
    by_identity = {}
    for row in proxy:
        key = (row["group"], row["analysis_index"])
        by_identity.setdefault(key, {"0": 0, "1": 0})[row["codebook_id"]] += 1
    # Missing source transcripts can remove at most a few realized cells; the frozen
    # assignment itself is exactly 50/50 over all 100 tasks.
    assert all(abs(counts["0"] - counts["1"]) <= 2 for counts in by_identity.values())


def test_parent_proxy_surface_locks_eligible_and_api_matched_sets() -> None:
    parent = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v11/configs/parent_proxy_surface_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert parent["counts"]["pre_missing"] == 6400
    assert parent["counts"]["local_reconstructable"] == 6387
    assert parent["counts"]["api_parent_valid"] == 6375
    assert parent["counts"]["ineligible_empty_first_assistant"] == 13
    assert parent["estimate"]["interaction_pp"] == -0.7797975559008098
    assert parent["bootstrap"] == {
        "confidence_levels": [0.9, 0.95],
        "reps": 20000,
        "seed": 20260830,
    }
    assert parent["independent_recomputation"]["passed"] is True


class _FakeTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [100 + index + len(token) for index, token in enumerate(text.split())]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        padding=False,
        return_offsets_mapping=False,
        return_tensors=None,
    ):
        del add_special_tokens, return_offsets_mapping
        if isinstance(texts, str):
            return {"input_ids": self._ids(texts)}
        encoded = [self._ids(text) for text in texts]
        width = max(map(len, encoded))
        padded = []
        masks = []
        for ids in encoded:
            missing = width - len(ids)
            if padding and self.padding_side == "left":
                padded.append([0] * missing + ids)
                masks.append([0] * missing + [1] * len(ids))
            else:
                padded.append(ids + [0] * missing)
                masks.append([1] * len(ids) + [0] * missing)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    @staticmethod
    def apply_chat_template(messages, *, tokenize=False, **kwargs):
        del tokenize, kwargs
        return messages[-1]["content"]


class _FakeModel:
    def __init__(self, layer: torch.nn.Module) -> None:
        self.layer = layer

    def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
        del attention_mask, use_cache
        assert logits_to_keep == 1
        streams = input_ids.float()[:, :, None, None].expand(-1, -1, 4, 4096).clone()
        self.layer(streams)
        final = input_ids[:, -1]
        logits = torch.zeros((input_ids.shape[0], 1, 800), dtype=torch.float32)
        logits[:, 0, 10] = final.float()
        logits[:, 0, 11] = -final.float()
        return SimpleNamespace(logits=logits)


def _fake_runtime() -> LoadedV11GLM53:
    runtime = object.__new__(LoadedV11GLM53)
    layer = torch.nn.Identity()
    runtime.layers = [layer]
    runtime.model = _FakeModel(layer)
    runtime.processor = _FakeProcessor()
    runtime.embedding_device = torch.device("cpu")
    runtime.config = {"rendering": {"reasoning_effort": "high", "clear_thinking": True}}
    return runtime


def test_mixed_length_batch_forces_left_padding_and_matches_single_rows() -> None:
    runtime = _fake_runtime()
    messages = [
        [{"role": "user", "content": "short prompt"}],
        [{"role": "user", "content": "this is the much longer prompt"}],
    ]
    singles = [
        runtime.forward_downstream(
            row, selected_layer=0, continuation=False, allowed_token_ids=[10, 11]
        )
        for row in messages
    ]
    batch = runtime.forward_downstream_batch(
        messages, selected_layer=0, continuation=False, allowed_token_ids=[10, 11]
    )
    assert runtime.processor.tokenizer.padding_side == "right"
    for expected, observed in zip(singles, batch, strict=True):
        np.testing.assert_array_equal(observed.allowed_logits, expected.allowed_logits)
        np.testing.assert_array_equal(observed.prompt_final, expected.prompt_final)
        assert observed.prompt_sha256 == expected.prompt_sha256
        assert observed.prompt_tokens == expected.prompt_tokens


def test_batch_calibration_includes_longest_row_for_every_candidate() -> None:
    runtime = _fake_runtime()
    rows = [
        {
            "sample_id": f"row-{index}",
            "messages": [{"role": "user", "content": " ".join(["token"] * length)}],
        }
        for index, length in enumerate((2, 3, 5, 9, 12, 20))
    ]
    result = calibrate_downstream_batch(
        runtime,
        rows,
        selected_layer=0,
        continuation=False,
        allowed_token_ids=[10, 11],
        candidate_batch_sizes=[1, 2, 4],
        logits_tolerance=0.0,
        activation_tolerance=0.0,
        selected_span=False,
    )
    longest = max(result["representative_token_lengths"])
    assert result["selected_batch_size"] == 4
    assert all(longest in record["token_lengths"] for record in result["candidate_results"])


def test_final_source_opening_marker_has_one_strict_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.glm53_user_eval.v11 import probes

    source_root = tmp_path / "source"
    feature_root = tmp_path / "features"
    source_root.mkdir()
    feature_root.mkdir()
    permutation_path = source_root / "permutation_analysis.json"
    permutation_path.write_text(
        json.dumps({"complete": True, "reps": 1000}) + "\n", encoding="utf-8"
    )
    readout_hash = "a" * 64
    permutation_hash = hashlib.sha256(permutation_path.read_bytes()).hexdigest()
    marker = source_root / "FINAL_SOURCE_HOLDOUT_OPENED.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": "glm53_v11_final_source_holdout_open_v1",
                "opened_once": True,
                "status": "opening",
                "readout_lock_sha256": readout_hash,
                "permutation_sha256": permutation_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fit = SimpleNamespace(report={"readout_lock_sha256": readout_hash})
    monkeypatch.setattr(probes, "load_development_fit", lambda _root: fit)
    monkeypatch.setattr(
        probes,
        "load_partition",
        lambda _root, _name: (np.zeros((1, 1, 1)), [{"sample_id": "x"}]),
    )
    monkeypatch.setattr(probes, "evaluate_source_final", lambda *args: {"passed": True})
    monkeypatch.setattr(probes, "leave_one_generator_score_gaps", lambda *args: {})
    args = Namespace(source_root=source_root, feature_root=feature_root)
    supervisor.command_evaluate_source_final(args)
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "complete"
    with pytest.raises(ValueError, match="may not be rerun"):
        supervisor.command_evaluate_source_final(args)


def test_scientific_machine_pass_never_authorizes_final_claim(tmp_path: Path) -> None:
    decision = supervisor._write_downstream_decision(
        output=tmp_path / "decision.json",
        schema="test",
        passed=True,
        pass_state="scientific_pass",
        fail_state="scientific_fail",
        checks={"science": True},
        inputs={"input": "a" * 64},
        recruitment_authorized=True,
    )
    assert decision["scientific_gate_passed"] is True
    assert decision["claim_ready"] is False
    assert decision["authorization"]["final_claim"] is False
    assert decision["manual_audit_status"] == (
        "scientific_decision_complete_manual_audit_pending"
    )
