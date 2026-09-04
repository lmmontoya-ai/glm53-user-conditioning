from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from src.glm53_user_eval.v9.datasets import EvalRow, load_eval_rows
from src.glm53_user_eval.v9.masking import (
    build_token_masks,
    resolve_cue_span,
    validate_mask_contract,
)
from src.glm53_user_eval.v9.probes import (
    _adamw_linear,
    _labels_from_rows,
    pair_preserving_labels,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "artifacts/datasets/contrastive_prompts_v2"


def test_dataset_preserves_paper_contract_and_cues() -> None:
    rows = load_eval_rows(DATASET)
    assert len(rows) == 448
    assert sum(row.label is None for row in rows) == 32
    assert sum(bool(row.cue_spans) for row in rows) == 416
    assert all(row.variant_family for row in rows)
    assert all(row.cue_span_text is not None and row.cue_spans_json is not None for row in rows)


def test_dataset_messages_preserve_system_and_user() -> None:
    rows = load_eval_rows(DATASET)
    with_system = next(row for row in rows if row.system_prompt)
    without_system = next(row for row in rows if not row.system_prompt)
    assert [message["role"] for message in with_system.messages] == ["system", "user"]
    assert [message["role"] for message in without_system.messages] == ["user"]


def test_exact_cue_is_unique() -> None:
    rendered = "prefix Request path: evaluation harness scoring model responses. suffix"
    cue = "Request path: evaluation harness scoring model responses."
    assert resolve_cue_span(rendered, cue) == (7, 64, "masked")


def test_ambiguous_cue_fails_closed() -> None:
    assert resolve_cue_span("cue x cue", "cue")[2] == "ambiguous"


def test_whitespace_normalized_resolution() -> None:
    start, end, status = resolve_cue_span("a  b\n c", "a b c")
    assert (start, end, status) == (0, 7, "masked")


def test_mask_excludes_only_overlap_padding_and_zero_offsets() -> None:
    rendered = "abc cue xyz"
    offsets = [(0, 3), (3, 4), (4, 8), (8, 9), (9, 12), (0, 0)]
    result = build_token_masks(
        rendered=rendered,
        offsets=offsets,
        attention_mask=[1, 1, 1, 1, 1, 0],
        cue_spans=("cue",),
    )
    assert result.status == "masked"
    assert result.cue.tolist() == [False, False, True, False, False, False]
    assert result.retained.tolist() == [True, True, False, True, True, False]


def test_neutral_decoy_is_not_available_but_retained() -> None:
    result = build_token_masks(
        rendered="neutral",
        offsets=[(0, 7)],
        attention_mask=[1],
        cue_spans=(),
    )
    assert result.status == "not_available"
    assert result.retained.tolist() == [True]


def test_corrupt_cue_is_not_found() -> None:
    result = build_token_masks(
        rendered="neutral",
        offsets=[(0, 7)],
        attention_mask=[1],
        cue_spans=("missing",),
    )
    assert result.status == "not_found"


def test_empty_after_mask_fails() -> None:
    result = build_token_masks(
        rendered="cue",
        offsets=[(0, 3)],
        attention_mask=[1],
        cue_spans=("cue",),
    )
    assert result.status == "empty_after_mask"


def test_mask_contract_exact_counts() -> None:
    report = validate_mask_contract(["masked"] * 416 + ["not_available"] * 32)
    assert report["passed"] is True
    assert validate_mask_contract(["masked"] * 415 + ["not_available"] * 33)["passed"] is False


def _paired_rows() -> list[dict[str, object]]:
    return [
        {"sample_id": "a0", "pair_id": "a", "label": 0},
        {"sample_id": "a1", "pair_id": "a", "label": 1},
        {"sample_id": "b0", "pair_id": "b", "label": 0},
        {"sample_id": "b1", "pair_id": "b", "label": 1},
        {"sample_id": "neutral", "pair_id": "n", "label": None},
    ]


def test_pair_preserving_null_keeps_one_class_per_pair() -> None:
    rows = _paired_rows()
    labels = pair_preserving_labels(rows, seed=1)
    assert sorted(labels[:2].tolist()) == [0, 1]
    assert sorted(labels[2:4].tolist()) == [0, 1]
    assert labels[4] == -1


def test_pair_preserving_null_is_deterministic() -> None:
    rows = _paired_rows()
    assert np.array_equal(
        pair_preserving_labels(rows, seed=17), pair_preserving_labels(rows, seed=17)
    )


def test_probe_labels_encode_neutral_decoys_as_minus_one() -> None:
    labels = _labels_from_rows(_paired_rows())
    assert labels.tolist() == [0, 1, 0, 1, -1]
    assert labels.dtype == np.int64


def test_pair_preserving_null_rejects_incomplete_pair() -> None:
    rows = [{"sample_id": "a0", "pair_id": "a", "label": 0}]
    try:
        pair_preserving_labels(rows, seed=1)
    except ValueError as error:
        assert "one eval and one deploy" in str(error)
    else:
        raise AssertionError("incomplete pair was accepted")


def test_adamw_linear_is_deterministic_and_learns() -> None:
    rng = np.random.default_rng(2)
    train_x = rng.normal(size=(64, 8)).astype(np.float32)
    train_y = (train_x[:, 0] > 0).astype(np.float32)
    val_x = rng.normal(size=(32, 8)).astype(np.float32)
    val_y = (val_x[:, 0] > 0).astype(np.float32)
    first = _adamw_linear(
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=50,
        lr=0.01,
        weight_decay=0.0001,
        batch_size=16,
        patience=10,
        seed=3,
    )
    second = _adamw_linear(
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=50,
        lr=0.01,
        weight_decay=0.0001,
        batch_size=16,
        patience=10,
        seed=3,
    )
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert first[0][0] > 0


def test_prereg_keeps_readout_and_steering_separate() -> None:
    import yaml

    config = yaml.safe_load(
        (ROOT / "pipelines/glm53_user_eval/v9/configs/prereg_v9_paper_faithful.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["representations"]["primary"] == "masked_prompt_mean"
    assert config["analysis"]["steering_gate_is_separate"] is True
    assert "steering" in config["execution"]["forbidden_in_first_paid_run"]
    assert config["probe"]["sequence_linear"]["optimizer"] == "adamw"


def test_neutral_json_cue_contract() -> None:
    row = EvalRow(
        sample_id="n",
        pair_id="n",
        family_id="n",
        split="steering_anchor",
        label=None,
        context_label="neutral_decoy",
        variant_family="neutral",
        cue_span_text="",
        cue_spans_json=json.dumps({}),
        system_prompt="",
        user_prompt="neutral",
    )
    assert row.cue_spans == ()
