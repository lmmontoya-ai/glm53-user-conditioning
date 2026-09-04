"""Small schema validators for immutable v8 rows."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TranscriptRow(BaseModel):
    sample_id: str
    group: str
    persona_key: str
    stimulus_id: str
    system_text: str
    dilemma_text: str
    first_assistant_text: str
    proxy_eligible: bool = True
    source_error: str | None = None
    original_folded_confidence: float | None = Field(ge=50, le=100)
    codebook_id: str
    source_eval_log: str
    source_eval_sha256: str
    messages_sha256: str


class ProxyResultRow(BaseModel):
    sample_id: str
    arm_id: str
    codebook_id: str
    expected_raw_confidence: float = Field(ge=0, le=100)
    expected_folded_confidence: float = Field(ge=50, le=100)
    allowed_mass: float = Field(ge=0, le=1)
    conditional_entropy: float = Field(ge=0)
    conditional_probabilities: list[float]

    def probability_sum(self) -> float:
        return float(sum(self.conditional_probabilities))


def reject_duplicate_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate immutable result row")
