from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.glm53_user_eval.v11.tokenizer_audit import (
    audit_dataset,
    build_tokenizer_audit,
    resolve_unique_span,
    validate_pair_contract,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def _tokenizer_root(tmp_path: Path) -> Path:
    root = tmp_path / "tokenizer"
    root.mkdir()
    vocabulary = {
        "[UNK]": 0,
        "alpha": 1,
        "beta": 2,
        "sends": 3,
        "checks": 4,
        "reply": 5,
        "Now": 6,
        "answer": 7,
        "task": 8,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "chat_template.jinja").write_text(
        "{% for message in messages %}<|{{ message['role'] }}|>\n"
        "{{ message['content'] }}\n{% endfor %}<|assistant|>\n",
        encoding="utf-8",
    )
    return root


def _row(sample_id: str, label: int, decisive: str) -> dict[str, object]:
    suffix = "Now answer task"
    return {
        "sample_id": sample_id,
        "pair_id": "pair-1",
        "split": "train",
        "label": label,
        "latent_class": "eval" if label else "deploy",
        "generator_family": "test",
        "task_id": "task-1",
        "prompt_role": "user_context",
        "system_prompt": "",
        "user_prompt": f"{decisive}\n\n{suffix}",
        "shared_suffix": suffix,
        "decisive_fact_texts": [decisive],
    }


def test_audit_records_complete_token_and_span_manifest(tmp_path: Path) -> None:
    rows = [_row("deploy", 0, "alpha sends reply"), _row("eval", 1, "beta checks reply")]
    report = build_tokenizer_audit(rows, _tokenizer_root(tmp_path))
    assert report["passed"] is True
    assert report["pair_contract"]["checked_pair_ids"] == ["pair-1"]
    assert report["row_count"] == 2
    for record in report["records"]:
        assert len(record["token_ids"]) == len(record["offsets"])
        assert record["shared_suffix_token_ids"]
        assert record["shared_suffix_start_index"] == record["shared_suffix_token_indices"][0]
        assert record["decisive_fact_spans"][0]["token_indices"]
        assert set(record["decisive_token_indices"]).isdisjoint(
            record["masked_prompt_token_indices"]
        )


def test_repeated_decisive_text_fails_unique_span_contract(tmp_path: Path) -> None:
    first = _row("deploy", 0, "alpha sends reply")
    first["user_prompt"] = "alpha sends reply and alpha sends reply\n\nNow answer task"
    with pytest.raises(ValueError, match="ambiguous"):
        build_tokenizer_audit(
            [first, _row("eval", 1, "beta checks reply")], _tokenizer_root(tmp_path)
        )


def test_pair_with_unequal_rendered_token_count_fails(tmp_path: Path) -> None:
    second = _row("eval", 1, "beta checks reply")
    second["user_prompt"] = "beta checks reply extra\n\nNow answer task"
    with pytest.raises(ValueError, match="unequal rendered token counts"):
        build_tokenizer_audit(
            [_row("deploy", 0, "alpha sends reply"), second], _tokenizer_root(tmp_path)
        )


def test_pair_contract_rejects_suffix_id_or_start_mismatch() -> None:
    base = {
        "pair_id": "p",
        "rendered_token_count": 10,
        "shared_suffix_text": "same",
        "shared_suffix_token_ids": [5, 6],
        "shared_suffix_start_index": 4,
    }
    with pytest.raises(ValueError, match="token IDs"):
        validate_pair_contract(
            [base | {"label": 0}, base | {"label": 1, "shared_suffix_token_ids": [5, 7]}]
        )
    with pytest.raises(ValueError, match="start indices"):
        validate_pair_contract(
            [base | {"label": 0}, base | {"label": 1, "shared_suffix_start_index": 5}]
        )


def test_singleton_unlabeled_control_is_allowed(tmp_path: Path) -> None:
    row = _row("neutral", 0, "alpha sends reply")
    row["pair_id"] = "neutral-1"
    row["label"] = None
    row["latent_class"] = "neutral"
    report = build_tokenizer_audit([row], _tokenizer_root(tmp_path))
    assert report["pair_contract"]["checked_pair_count"] == 0
    assert report["pair_contract"]["singleton_control_ids"] == ["neutral-1"]


def test_whitespace_normalized_span_is_supported_and_ambiguous_is_rejected() -> None:
    assert resolve_unique_span("alpha  sends\nreply", "alpha sends reply", field="fact") == (
        0,
        18,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_unique_span("same x same", "same", field="fact")


def test_dataset_wrapper_writes_atomic_json_with_source_hash(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    rows = [_row("deploy", 0, "alpha sends reply"), _row("eval", 1, "beta checks reply")]
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "tokenizer_audit.json"
    report = audit_dataset(samples, _tokenizer_root(tmp_path), output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["samples_sha256"] == report["samples_sha256"]
    assert written["records_sha256"] == report["records_sha256"]
    assert not output.with_suffix(".json.tmp").exists()
