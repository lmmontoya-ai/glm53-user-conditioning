"""Cue-span resolution after the exact GLM chat template is rendered."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DISALLOWED_STATUSES = {
    "partial_masked",
    "ambiguous",
    "not_found",
    "no_token_overlap",
    "empty_after_mask",
}


@dataclass(frozen=True)
class TokenMasks:
    retained: np.ndarray
    cue: np.ndarray
    valid: np.ndarray
    status: str
    requested_span_count: int
    masked_span_count: int


def _normalized_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    normalized: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            start = index
            while index < len(text) and text[index].isspace():
                index += 1
            normalized.append(" ")
            spans.append((start, index))
        else:
            normalized.append(text[index])
            spans.append((index, index + 1))
            index += 1
    return "".join(normalized), spans


def resolve_cue_span(rendered: str, cue: str) -> tuple[int, int, str]:
    start = rendered.find(cue)
    if start >= 0:
        if rendered.find(cue, start + 1) >= 0:
            return -1, -1, "ambiguous"
        return start, start + len(cue), "masked"
    normalized, spans = _normalized_with_spans(rendered)
    normalized_cue, _ = _normalized_with_spans(cue)
    if not normalized_cue:
        return -1, -1, "not_found"
    start = normalized.find(normalized_cue)
    if start < 0:
        return -1, -1, "not_found"
    if normalized.find(normalized_cue, start + 1) >= 0:
        return -1, -1, "ambiguous"
    end = start + len(normalized_cue)
    return spans[start][0], spans[end - 1][1], "masked"


def build_token_masks(
    *,
    rendered: str,
    offsets: list[tuple[int, int]],
    attention_mask: list[int] | np.ndarray,
    cue_spans: tuple[str, ...],
) -> TokenMasks:
    if len(offsets) != len(attention_mask):
        raise ValueError("offsets and attention mask differ in length")
    attention = np.asarray(attention_mask, dtype=bool)
    nonempty = np.asarray([end > start for start, end in offsets], dtype=bool)
    valid = attention & nonempty
    cue_mask = np.zeros(len(offsets), dtype=bool)
    failures: list[str] = []
    successes = 0
    for cue in cue_spans:
        start, end, status = resolve_cue_span(rendered, cue)
        if status != "masked":
            failures.append(status)
            continue
        overlap = np.asarray(
            [tok_end > start and tok_start < end for tok_start, tok_end in offsets], dtype=bool
        ) & valid
        if not overlap.any():
            failures.append("no_token_overlap")
            continue
        cue_mask |= overlap
        successes += 1
    if not cue_spans:
        status = "not_available"
    elif successes == len(cue_spans) and not failures:
        status = "masked"
    elif successes:
        status = "partial_masked"
    elif "ambiguous" in failures:
        status = "ambiguous"
    elif "not_found" in failures:
        status = "not_found"
    else:
        status = "no_token_overlap"
    retained = valid & ~cue_mask
    if not retained.any():
        status = "empty_after_mask"
    return TokenMasks(
        retained=retained,
        cue=cue_mask,
        valid=valid,
        status=status,
        requested_span_count=len(cue_spans),
        masked_span_count=successes,
    )


def validate_mask_contract(statuses: list[str]) -> dict[str, object]:
    counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    passed = counts == {"masked": 416, "not_available": 32}
    return {
        "passed": passed,
        "counts": counts,
        "expected": {"masked": 416, "not_available": 32},
        "disallowed_present": sorted(DISALLOWED_STATUSES.intersection(counts)),
    }

