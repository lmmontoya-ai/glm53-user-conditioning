from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from pipelines.glm53_user_eval.v11 import run as supervisor

LEXICAL_CHECKS = {
    "structural": True,
    "tokenizer": True,
    "metadata_ordinary_le_060": True,
    "metadata_final_le_060": True,
    "keyword_final_le_060": True,
    "word_final_le_065": True,
    "char_final_le_065": True,
    "deleted_word_final_le_060": True,
    "deleted_char_final_le_060": True,
}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _args(audit_root: Path) -> argparse.Namespace:
    return argparse.Namespace(audit_root=audit_root)


@pytest.mark.parametrize(
    ("failed_artifact", "failed_check"),
    [
        ("lexical_decision.json", "lexical_decision"),
        ("semantic_validation.json", "semantic"),
        ("manual_audit.json", "manual"),
        ("verification.json", "independent_verification"),
    ],
)
def test_text_gate_checks_record_completed_failures(
    tmp_path: Path,
    failed_artifact: str,
    failed_check: str,
) -> None:
    artifacts = {
        "lexical_decision.json": {"passed": True, "checks": LEXICAL_CHECKS},
        "semantic_validation.json": {"passed": True},
        "manual_audit.json": {"passed": True},
        "verification.json": {"passed": True},
    }
    artifacts[failed_artifact]["passed"] = False
    for name, value in artifacts.items():
        _write_json(tmp_path / name, value)

    checks = supervisor._text_gate_checks(_args(tmp_path))

    assert checks[failed_check] is False
    assert sum(value is False for value in checks.values()) == 1


def test_decide_text_writes_fail_closed_terminal_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = {
        "lexical_decision.json": {"passed": True, "checks": LEXICAL_CHECKS},
        "semantic_validation.json": {"passed": False},
        "manual_audit.json": {"passed": True},
        "verification.json": {"passed": False},
    }
    for name, value in artifacts.items():
        _write_json(tmp_path / name, value)
    bound_inputs = {
        "semantic": tmp_path / "semantic_validation.json",
        "manual": tmp_path / "manual_audit.json",
        "verification": tmp_path / "verification.json",
    }
    monkeypatch.setattr(supervisor, "_require_frozen_paid_paths", lambda _args: None)
    monkeypatch.setattr(supervisor, "_text_decision_input_paths", lambda _args: bound_inputs)

    supervisor.command_decide_text(_args(tmp_path))

    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["passed"] is False
    assert decision["decision"] == "source_text_instrument_invalid_stop_before_glm"
    assert decision["failed_checks"] == ["independent_verification", "semantic"]
    assert all(value is False for value in decision["authorization"].values())
    assert set(decision["inputs"]) == set(bound_inputs)


def test_decide_text_rejects_incomplete_gate_artifact(tmp_path: Path) -> None:
    artifacts = {
        "lexical_decision.json": {"passed": True, "checks": LEXICAL_CHECKS},
        "semantic_validation.json": {},
        "manual_audit.json": {"passed": True},
        "verification.json": {"passed": True},
    }
    for name, value in artifacts.items():
        _write_json(tmp_path / name, value)

    with pytest.raises(TypeError, match="lacks a Boolean passed field"):
        supervisor._text_gate_checks(_args(tmp_path))
