"""Static GLM-5.3 runtime-contract validation."""

from __future__ import annotations

from typing import Any


def text_config(model_config: dict[str, Any]) -> dict[str, Any]:
    value = model_config.get("text_config")
    if not isinstance(value, dict):
        raise ValueError("GLM-5.3 config does not contain a text_config mapping")
    return value


def validate_glm53_config(model_config: dict[str, Any]) -> dict[str, int]:
    config = text_config(model_config)
    expected = {"num_hidden_layers": 45, "hidden_size": 4096, "hc_mult": 4}
    for key, value in expected.items():
        if int(config.get(key, -1)) != value:
            raise ValueError(f"unexpected GLM-5.3 {key}: {config.get(key)!r}")
    return {
        "text_layers": int(config["num_hidden_layers"]),
        "hidden_size": int(config["hidden_size"]),
        "hc_mult": int(config["hc_mult"]),
    }


def prompt_final_indices(attention_mask: Any) -> list[int]:
    rows = attention_mask.tolist()
    indices: list[int] = []
    for row in rows:
        active = [index for index, value in enumerate(row) if int(value) != 0]
        if not active:
            raise ValueError("attention mask contains an empty prompt")
        indices.append(active[-1])
    return indices
