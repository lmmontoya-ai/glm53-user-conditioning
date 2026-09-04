"""One-token confidence proxy and codebook checks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_contract_tokenizer(model_id: str, revision: str) -> Any:
    """Load the exact tokenizer/template without requiring multimodal classes locally."""
    import json

    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    tokenizer_path = hf_hub_download(model_id, "tokenizer.json", revision=revision)
    config_path = hf_hub_download(model_id, "tokenizer_config.json", revision=revision)
    template_path = hf_hub_download(model_id, "chat_template.jinja", revision=revision)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    template = Path(template_path).read_text(encoding="utf-8")
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        chat_template=template,
        eos_token=config["eos_token"],
        pad_token=config["pad_token"],
        clean_up_tokenization_spaces=config.get("clean_up_tokenization_spaces", False),
        model_max_length=config.get("model_max_length", int(1e30)),
    )


def codebook_id(sample_id: str, version: str = "v1") -> str:
    digest = hashlib.sha256(f"glm53-v8-codebook-{version}|{sample_id}".encode()).digest()
    return str(digest[0] & 1)


def validate_codebooks(payload: dict[str, Any]) -> None:
    labels = payload["labels"]
    if len(labels) != 11 or len(set(labels)) != 11:
        raise ValueError("proxy requires eleven distinct labels")
    zero = payload["codebooks"]["0"]
    one = payload["codebooks"]["1"]
    if list(zero) != labels or list(one) != labels:
        raise ValueError("codebook keys must match label order")
    values = [zero[label] for label in labels]
    if values != list(range(0, 101, 10)):
        raise ValueError("codebook 0 is not the frozen 0..100 mapping")
    if [one[label] for label in labels] != list(reversed(values)):
        raise ValueError("codebook 1 is not an exact reversal")


def confidence_prompt(codebook: dict[str, int], followup: str) -> str:
    mapping = ", ".join(f"{label}={value}%" for label, value in codebook.items())
    return f"{followup}\nCodebook: {mapping}"


def proxy_messages(
    *,
    system_text: str,
    dilemma_text: str,
    first_assistant_text: str,
    codebook: dict[str, int],
    followup: str,
    assistant_prefix: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": dilemma_text},
        {"role": "assistant", "content": first_assistant_text},
        {"role": "user", "content": confidence_prompt(codebook, followup)},
        {"role": "assistant", "content": assistant_prefix},
    ]


def proxy_from_logits(
    logits: Sequence[float], label_ids: Sequence[int], codebook_values: Sequence[float]
) -> dict[str, Any]:
    full = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(label_ids, dtype=np.int64)
    values = np.asarray(codebook_values, dtype=np.float64)
    if full.ndim != 1 or len(ids) != 11 or len(values) != 11:
        raise ValueError("expected one vocabulary vector and eleven labels")
    if len(set(ids.tolist())) != 11 or np.any(ids < 0) or np.any(ids >= len(full)):
        raise ValueError("invalid label IDs")
    allowed = full[ids]
    shifted = allowed - np.max(allowed)
    conditional = np.exp(shifted) / np.exp(shifted).sum()
    full_max = float(np.max(full))
    log_full = full_max + math.log(float(np.exp(full - full_max).sum()))
    allowed_max = float(np.max(allowed))
    log_allowed = allowed_max + math.log(float(np.exp(allowed - allowed_max).sum()))
    folded_values = np.maximum(values, 100.0 - values)
    entropy = -float(np.sum(np.where(conditional > 0, conditional * np.log(conditional), 0.0)))
    winner = int(np.argmax(allowed))
    return {
        "conditional_probabilities": conditional.tolist(),
        "expected_raw_confidence": float(conditional @ values),
        "expected_folded_confidence": float(conditional @ folded_values),
        "allowed_mass": float(math.exp(log_allowed - log_full)),
        "conditional_entropy": entropy,
        "argmax_label_position": winner,
        "argmax_raw_confidence": float(values[winner]),
        "argmax_folded_confidence": float(folded_values[winner]),
        "full_vocab_argmax_allowed": int(np.argmax(full)) in set(ids.tolist()),
    }


def validate_label_tokens(
    processor: Any, messages: list[dict[str, str]], labels: list[str]
) -> dict[str, Any]:
    # A multimodal AutoProcessor may return a BatchFeature when asked to
    # tokenize the chat directly. Its length counts fields, not token IDs.
    # Render first, then tokenize with the exact underlying text tokenizer.
    base_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
        reasoning_effort="high",
        clear_thinking=True,
    )
    base = processor.tokenizer(base_text, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    for label in labels:
        changed = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + label}]
        changed_text = processor.apply_chat_template(
            changed,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            reasoning_effort="high",
            clear_thinking=True,
        )
        rendered = processor.tokenizer(changed_text, add_special_tokens=False)[
            "input_ids"
        ]
        if len(rendered) != len(base) + 1 or rendered[:-1] != base:
            raise ValueError(f"label {label!r} is not one continuation token")
        ids.append(int(rendered[-1]))
    if len(set(ids)) != len(labels):
        raise ValueError("answer labels do not have distinct token IDs")
    return {"label_ids": dict(zip(labels, ids, strict=True)), "base_token_count": len(base)}
