from __future__ import annotations

import numpy as np
import pytest
from src.glm53_user_eval.v8.proxy import (
    codebook_id,
    confidence_prompt,
    proxy_from_logits,
    validate_codebooks,
)


def test_codebooks_validate(codebooks: dict) -> None:
    validate_codebooks(codebooks)


def test_codebook_reversal(codebooks: dict) -> None:
    assert list(codebooks["codebooks"]["1"].values()) == list(
        reversed(codebooks["codebooks"]["0"].values())
    )


def test_assignment_deterministic() -> None:
    assert codebook_id("sample") == codebook_id("sample")
    assert codebook_id("sample") in {"0", "1"}


@pytest.mark.parametrize("winner,expected", [(0, 100.0), (5, 50.0), (10, 100.0)])
def test_proxy_expected_folded(winner: int, expected: float) -> None:
    logits = np.full(20, -100.0)
    logits[winner] = 100.0
    result = proxy_from_logits(logits, list(range(11)), list(range(0, 101, 10)))
    assert result["expected_folded_confidence"] == pytest.approx(expected)


def test_proxy_uniform() -> None:
    result = proxy_from_logits(np.zeros(20), list(range(11)), list(range(0, 101, 10)))
    assert result["expected_raw_confidence"] == pytest.approx(50.0)
    assert result["allowed_mass"] == pytest.approx(11 / 20)


def test_confidence_prompt_contains_all_labels(codebooks: dict) -> None:
    text = confidence_prompt(codebooks["codebooks"]["0"], codebooks["followup"])
    assert all(f"{label}=" in text for label in codebooks["labels"])


def test_invalid_duplicate_labels(codebooks: dict) -> None:
    codebooks["labels"][1] = "A"
    with pytest.raises(ValueError):
        validate_codebooks(codebooks)
