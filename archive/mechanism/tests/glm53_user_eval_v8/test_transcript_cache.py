from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.glm53_user_eval.v8.schemas import TranscriptRow, reject_duplicate_rows
from src.glm53_user_eval.v8.transcript_cache import assert_confidence_absent


def sample() -> dict:
    return {
        "sample_id": "x",
        "group": "genpop",
        "persona_key": "p",
        "stimulus_id": "d",
        "system_text": "system",
        "dilemma_text": "question",
        "first_assistant_text": "No",
        "proxy_eligible": True,
        "source_error": None,
        "original_folded_confidence": 80.0,
        "codebook_id": "0",
        "source_eval_log": "log",
        "source_eval_sha256": "a" * 64,
        "messages_sha256": "b" * 64,
    }


def test_schema_accepts_row() -> None:
    assert TranscriptRow(**sample()).sample_id == "x"


def test_schema_rejects_confidence_below_50() -> None:
    row = sample() | {"original_folded_confidence": 49}
    with pytest.raises(ValidationError):
        TranscriptRow(**row)


def test_confidence_absent() -> None:
    assert_confidence_absent("No percentage in this input", sample())


def test_confidence_leak_rejected() -> None:
    with pytest.raises(ValueError):
        assert_confidence_absent("Original result was 80%", sample())


def test_duplicate_keys_rejected() -> None:
    with pytest.raises(ValueError):
        reject_duplicate_rows([sample(), sample()], ("sample_id",))
