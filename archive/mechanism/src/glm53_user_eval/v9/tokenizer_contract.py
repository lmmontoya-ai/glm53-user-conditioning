"""Local, weight-free verification of the pinned GLM tokenizer and chat template."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .datasets import EvalRow
from .masking import build_token_masks, validate_mask_contract


def render_messages(template_path: Path, messages: list[dict[str, str]]) -> str:
    from jinja2 import Environment

    environment = Environment(autoescape=False, extensions=["jinja2.ext.loopcontrols"])
    template = environment.from_string(template_path.read_text(encoding="utf-8"))
    return template.render(
        messages=messages,
        tools=None,
        add_generation_prompt=True,
        reasoning_effort="high",
        clear_thinking=True,
    )


def validate_rows(rows: list[EvalRow], tokenizer_root: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_root / "tokenizer.json"))
    template_path = tokenizer_root / "chat_template.jinja"
    statuses: list[str] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        rendered = render_messages(template_path, row.messages)
        encoded = tokenizer.encode(rendered, add_special_tokens=False)
        masks = build_token_masks(
            rendered=rendered,
            offsets=[(int(start), int(end)) for start, end in encoded.offsets],
            attention_mask=[1] * len(encoded.ids),
            cue_spans=row.cue_spans,
        )
        statuses.append(masks.status)
        records.append(
            {
                "sample_id": row.sample_id,
                "status": masks.status,
                "prompt_tokens": len(encoded.ids),
                "valid_tokens": int(masks.valid.sum()),
                "masked_tokens": int(masks.cue.sum()),
                "retained_tokens": int(masks.retained.sum()),
                "requested_spans": masks.requested_span_count,
                "masked_spans": masks.masked_span_count,
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "input_ids_sha256": hashlib.sha256(
                    b"".join(int(token).to_bytes(4, "little") for token in encoded.ids)
                ).hexdigest(),
            }
        )
    contract = validate_mask_contract(statuses)
    return {
        "schema_version": "glm53_v9_local_tokenizer_contract_v1",
        "passed": contract["passed"],
        "mask_contract": contract,
        "tokenizer_json_sha256": hashlib.sha256(
            (tokenizer_root / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "records": records,
    }
