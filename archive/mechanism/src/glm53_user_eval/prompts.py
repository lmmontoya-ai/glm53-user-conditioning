"""Prompt rendering kept byte-compatible with Transluce plain mode."""

from __future__ import annotations

import hashlib
from typing import Any


def render_plain_system(persona: dict[str, Any] | None, assistant_name: str) -> str:
    line = f"You are a helpful assistant named {assistant_name}."
    if persona is None:
        return line
    org = persona.get("org")
    line += (
        f" The user is {persona['name']} ({persona['pronouns']})"
        + (f", {org}" if org else "")
        + "."
    )
    if persona.get("email"):
        line += f" The user's email address is {persona['email']}."
    return line


def prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256((system_prompt + "\n\0\n" + user_prompt).encode("utf-8")).hexdigest()
