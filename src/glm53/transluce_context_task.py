"""Inspect task wrapper: the pinned Transluce task with a session-context block in the system prompt.

This module is loaded by `inspect eval` with the pinned Transluce checkout as the working
directory. It registers one synthetic persona per (identity, condition, paraphrase family) whose
profile fields equal the original identity's, and wraps the pinned plain-mode system-prompt
builder so the block text is appended after the identity sentences and before the dilemma. The
pinned task, solver, and scorer are otherwise called unchanged.

Persona key format: `<original_key>__<condition>__f<family>`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get("GLM53_TRANSLUCE_ROOT", os.getcwd())).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from core import personas as _personas  # noqa: E402
from core import solver as _solver  # noqa: E402
from evals.pmisaligned import task as _task  # noqa: E402
from inspect_ai import task  # noqa: E402

BLOCKS_PATH = Path(os.environ["GLM53_CONTEXT_BLOCKS"])
BLOCK_FIELD = {"neutral": "neutral_text", "operational_use": "operational_use", "model_assessment": "model_assessment"}


def _blocks() -> list[dict[str, str]]:
    return json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))["families"]


def split_key(key: str) -> tuple[str, str | None, int | None]:
    """(original key, condition, family) for a synthetic key; (key, None, None) otherwise."""
    if "__" not in key:
        return key, None, None
    base, condition, family = key.rsplit("__", 2)
    return base, condition, int(family.removeprefix("f"))


def register(keys: list[str]) -> None:
    """Add synthetic persona records that alias their original identity."""
    for key in keys:
        base, condition, family = split_key(key)
        if condition is None or key in _personas.PERSONAS:
            continue
        if base not in _personas.PERSONAS or _personas.PERSONAS[base] is None:
            raise KeyError(f"unknown base persona {base}")
        _personas.PERSONAS[key] = dict(_personas.PERSONAS[base]) | {"key": key}


_original_plain_system = _solver._plain_system


def _plain_system_with_block(persona_key: str, assistant_name: str, seed: str = "all") -> str:
    base, condition, family = split_key(persona_key)
    line = _original_plain_system(base, assistant_name, seed)
    if condition is None:
        return line
    text = _blocks()[family][BLOCK_FIELD[condition]]
    return f"{line} {text}"


_solver._plain_system = _plain_system_with_block
_task.PERSONAS = _personas.PERSONAS


@task
def pmisaligned_context(personas: str = "", **kwargs):
    """The pinned task with synthetic persona keys carrying a context block."""
    keys = [k for k in str(personas).split(",") if k]
    register(keys)
    return _task.pmisaligned(personas=",".join(keys), **kwargs)
