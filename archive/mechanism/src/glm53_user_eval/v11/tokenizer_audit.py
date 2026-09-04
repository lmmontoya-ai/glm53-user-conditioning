"""Weight-free tokenizer audit for ``contrastive_prompts_v3``.

The audit renders the same messages that the model will receive, resolves all
scientifically meaningful spans in that rendered string, and retains the exact
token IDs and offsets needed by the later activation extractor.  It deliberately
uses the checked-out tokenizer files and never resolves or downloads a model.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def render_messages(template_path: Path, messages: Sequence[Mapping[str, str]]) -> str:
    """Render messages under the pinned GLM template without loading weights."""

    from jinja2 import Environment

    normalized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    environment = Environment(autoescape=False, extensions=["jinja2.ext.loopcontrols"])
    template = environment.from_string(template_path.read_text(encoding="utf-8"))
    rendered = template.render(
        messages=normalized,
        tools=None,
        add_generation_prompt=True,
        reasoning_effort="high",
        clear_thinking=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template produced an empty rendered prompt")
    return rendered


def messages_from_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Accept either explicit messages or the V11 system/user row fields."""

    explicit = row.get("messages")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("row messages must be a non-empty list")
        messages: list[dict[str, str]] = []
        for message in explicit:
            if not isinstance(message, Mapping):
                raise TypeError("every message must be an object")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError("message role and content must be strings")
            messages.append({"role": role, "content": content})
        return messages

    system_prompt = row.get("system_prompt", "")
    user_prompt = row.get("user_prompt")
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise TypeError("system_prompt and user_prompt must be strings")
    if not user_prompt:
        raise ValueError("user_prompt must not be empty")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _normalized_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    normalized: list[str] = []
    source_spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            start = index
            while index < len(text) and text[index].isspace():
                index += 1
            normalized.append(" ")
            source_spans.append((start, index))
        else:
            normalized.append(text[index])
            source_spans.append((index, index + 1))
            index += 1
    return "".join(normalized), source_spans


def resolve_unique_span(rendered: str, text: str, *, field: str) -> tuple[int, int]:
    """Resolve exactly one rendered character span, with V9 whitespace fallback."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field} must be a non-empty string")
    first = rendered.find(text)
    if first >= 0:
        if rendered.find(text, first + 1) >= 0:
            raise ValueError(f"{field} is ambiguous in the rendered prompt")
        return first, first + len(text)

    normalized, source_spans = _normalized_with_spans(rendered)
    normalized_text, _ = _normalized_with_spans(text)
    first = normalized.find(normalized_text)
    if first < 0:
        raise ValueError(f"{field} was not found in the rendered prompt")
    if normalized.find(normalized_text, first + 1) >= 0:
        raise ValueError(f"{field} is ambiguous after whitespace normalization")
    end = first + len(normalized_text)
    return source_spans[first][0], source_spans[end - 1][1]


def token_indices_for_span(
    offsets: Sequence[tuple[int, int]],
    char_span: tuple[int, int],
    *,
    field: str,
) -> list[int]:
    """Return nonempty tokenizer offsets that overlap a character span."""

    start, end = char_span
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]
    if not indices:
        raise ValueError(f"{field} has no overlapping tokenizer token")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{field} does not map to a contiguous token span")
    return indices


def _decisive_texts(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("decisive_fact_texts", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("decisive_fact_texts must be a list of strings")
    if len(set(raw)) != len(raw):
        raise ValueError("decisive_fact_texts contains duplicate text")
    return list(raw)


def _tokenize_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    template_path: Path,
    source_row_index: int,
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    pair_id = row.get("pair_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("every row needs a non-empty sample_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError(f"row {sample_id} needs a non-empty pair_id")

    rendered = render_messages(template_path, messages_from_row(row))
    encoded = tokenizer.encode(rendered, add_special_tokens=False)
    token_ids = [int(token_id) for token_id in encoded.ids]
    offsets = [(int(start), int(end)) for start, end in encoded.offsets]
    if len(token_ids) != len(offsets) or not token_ids:
        raise ValueError(f"row {sample_id} has invalid tokenizer output")
    valid_indices = [index for index, (start, end) in enumerate(offsets) if end > start]
    if not valid_indices:
        raise ValueError(f"row {sample_id} has no content-token offsets")

    suffix = row.get("shared_suffix")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError(f"row {sample_id} needs a non-empty shared_suffix")
    suffix_char_span = resolve_unique_span(rendered, suffix, field=f"{sample_id}.shared_suffix")
    suffix_indices = token_indices_for_span(
        offsets, suffix_char_span, field=f"{sample_id}.shared_suffix"
    )

    decisive_records: list[dict[str, Any]] = []
    decisive_indices: set[int] = set()
    occupied_char_spans: list[tuple[int, int]] = []
    for fact_index, fact_text in enumerate(_decisive_texts(row)):
        field = f"{sample_id}.decisive_fact_texts[{fact_index}]"
        char_span = resolve_unique_span(rendered, fact_text, field=field)
        if any(char_span[1] > other[0] and char_span[0] < other[1] for other in occupied_char_spans):
            raise ValueError(f"{field} overlaps another decisive fact span")
        if char_span[1] > suffix_char_span[0] and char_span[0] < suffix_char_span[1]:
            raise ValueError(f"{field} overlaps the shared suffix")
        occupied_char_spans.append(char_span)
        indices = token_indices_for_span(offsets, char_span, field=field)
        decisive_indices.update(indices)
        decisive_records.append(
            {
                "fact_index": fact_index,
                "text": fact_text,
                "text_sha256": _sha256_text(fact_text),
                "char_start": char_span[0],
                "char_end": char_span[1],
                "token_indices": indices,
                "token_ids": [token_ids[index] for index in indices],
            }
        )

    masked_prompt_indices = [index for index in valid_indices if index not in decisive_indices]
    if not masked_prompt_indices:
        raise ValueError(f"row {sample_id} is empty after decisive-fact masking")
    return {
        "source_row_index": source_row_index,
        "sample_id": sample_id,
        "pair_id": pair_id,
        "split": row.get("split"),
        "label": row.get("label"),
        "latent_class": row.get("latent_class"),
        "generator_family": row.get("generator_family"),
        "task_id": row.get("task_id"),
        "prompt_role": row.get("prompt_role"),
        "rendered_sha256": _sha256_text(rendered),
        "rendered_token_count": len(token_ids),
        "token_ids": token_ids,
        "offsets": [[start, end] for start, end in offsets],
        "valid_token_indices": valid_indices,
        "prompt_final_index": valid_indices[-1],
        "decisive_fact_spans": decisive_records,
        "decisive_token_indices": sorted(decisive_indices),
        "masked_prompt_token_indices": masked_prompt_indices,
        "shared_suffix_text": suffix,
        "shared_suffix_sha256": _sha256_text(suffix),
        "shared_suffix_char_span": [suffix_char_span[0], suffix_char_span[1]],
        "shared_suffix_token_indices": suffix_indices,
        "shared_suffix_start_index": suffix_indices[0],
        "shared_suffix_token_ids": [token_ids[index] for index in suffix_indices],
    }


def validate_pair_contract(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail unless every labeled pair has token-identical aligned suffixes."""

    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[str(record["pair_id"])].append(record)

    checked_pair_ids: list[str] = []
    singleton_control_ids: list[str] = []
    for pair_id, members in sorted(by_pair.items()):
        labels = [member.get("label") for member in members]
        has_label = any(label is not None for label in labels)
        if has_label:
            if len(members) != 2 or sorted(labels) != [0, 1]:
                raise ValueError(f"labeled pair {pair_id} must contain one label 0 and one label 1")
        elif len(members) == 1:
            singleton_control_ids.append(pair_id)
            continue

        reference = members[0]
        for member in members[1:]:
            if member["rendered_token_count"] != reference["rendered_token_count"]:
                raise ValueError(f"pair {pair_id} has unequal rendered token counts")
            if member["shared_suffix_text"] != reference["shared_suffix_text"]:
                raise ValueError(f"pair {pair_id} has unequal shared-suffix text")
            if member["shared_suffix_token_ids"] != reference["shared_suffix_token_ids"]:
                raise ValueError(f"pair {pair_id} has unequal shared-suffix token IDs")
            if member["shared_suffix_start_index"] != reference["shared_suffix_start_index"]:
                raise ValueError(f"pair {pair_id} has unequal shared-suffix start indices")
        checked_pair_ids.append(pair_id)

    return {
        "passed": True,
        "checked_pair_count": len(checked_pair_ids),
        "checked_pair_ids": checked_pair_ids,
        "singleton_control_count": len(singleton_control_ids),
        "singleton_control_ids": singleton_control_ids,
    }


def build_tokenizer_audit(
    rows: Iterable[Mapping[str, Any]], tokenizer_root: Path
) -> dict[str, Any]:
    """Build a complete token manifest using only local tokenizer artifacts."""

    from tokenizers import Tokenizer

    tokenizer_path = tokenizer_root / "tokenizer.json"
    template_path = tokenizer_root / "chat_template.jinja"
    if not tokenizer_path.is_file() or not template_path.is_file():
        raise FileNotFoundError("tokenizer_root must contain tokenizer.json and chat_template.jinja")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    row_list = list(rows)
    if not row_list:
        raise ValueError("tokenizer audit needs at least one row")
    sample_ids = [row.get("sample_id") for row in row_list]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("tokenizer audit sample_ids must be unique")
    records = [
        _tokenize_row(
            row,
            tokenizer=tokenizer,
            template_path=template_path,
            source_row_index=index,
        )
        for index, row in enumerate(row_list)
    ]
    pair_contract = validate_pair_contract(records)
    return {
        "schema_version": "glm53_v11_tokenizer_audit_v1",
        "dataset_id": "contrastive_prompts_v3",
        "passed": True,
        "row_count": len(records),
        "tokenizer_json_sha256": _sha256_bytes(tokenizer_path.read_bytes()),
        "chat_template_sha256": _sha256_bytes(template_path.read_bytes()),
        "pair_contract": pair_contract,
        "records_sha256": _sha256_text(_canonical_json(records)),
        "records": records,
    }


def write_tokenizer_audit(
    rows: Iterable[Mapping[str, Any]], tokenizer_root: Path, output_path: Path
) -> dict[str, Any]:
    report = build_tokenizer_audit(rows, tokenizer_root)
    _atomic_json(output_path, report)
    return report


def audit_dataset(samples_path: Path, tokenizer_root: Path, output_path: Path) -> dict[str, Any]:
    """Load V11 JSONL rows, audit them, and atomically save the token manifest."""

    rows: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"samples line {line_number} is not an object")
            rows.append(row)
    report = build_tokenizer_audit(rows, tokenizer_root)
    report["samples_path"] = samples_path.as_posix()
    report["samples_sha256"] = _sha256_bytes(samples_path.read_bytes())
    _atomic_json(output_path, report)
    return report
