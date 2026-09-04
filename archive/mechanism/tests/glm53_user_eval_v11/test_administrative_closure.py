from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from src.glm53_user_eval.v11.administrative_closure import (
    NO_AUTHORIZATION,
    TERMINAL_DECISION,
    AdministrativeClosureError,
    close_v11,
    sha256_file,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    prereg = repo / "prereg.yaml"
    prereg.parent.mkdir(parents=True)
    prereg.write_text("project_id: glm53_user_eval_source_instrument_v11\n", encoding="utf-8")
    analysis = repo / "analysis.json"
    semantic = repo / "semantic.json"
    stop = repo / "stop.json"
    summary = repo / "summary.json"
    _write_json(
        analysis,
        {
            "state": "semantic_validation_failed_manual_review_pending",
            "passed": False,
            "authorization": NO_AUTHORIZATION,
        },
    )
    _write_json(
        semantic,
        {
            "passed": False,
            "controls": {
                "acceptable": 77,
                "count": 96,
                "acceptance_rate": 77 / 96,
                "threshold": 0.90,
            },
        },
    )
    _write_json(stop, {"passed": False})
    _write_json(summary, {"passed": False})
    diagnostic = repo / "diagnostic.json"
    _write_json(
        diagnostic,
        {
            "schema_version": "glm53_v11_ai_diagnostic_review_summary_v1",
            "reviewer_type": "ai",
            "scientific_role": "nonhuman_non_gating_diagnostic",
            "human_review_requirement_satisfied": False,
            "official_human_merge_eligible": False,
            "authorization": NO_AUTHORIZATION,
            "primary": {
                "row_count": 128,
                "label_counts": {
                    "eval": 38,
                    "deploy": 46,
                    "ambiguous": 18,
                    "no_status": 26,
                },
            },
            "supplemental": {
                "row_count": 6,
                "label_counts": {"eval": 3, "no_status": 3},
            },
        },
    )
    locks = {
        name: {
            "path": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in {
            "preregistration": prereg,
            "combined_analysis": analysis,
            "semantic_validation": semantic,
            "semantic_stop_evidence": stop,
            "semantic_stop_summary": summary,
        }.items()
    }
    amendment = repo / "amendment.yaml"
    amendment.write_text(
        yaml.safe_dump(
            {
                "schema_version": (
                    "glm53_user_eval_v11_administrative_closure_amendment_v1"
                ),
                "project_id": "glm53_user_eval_source_instrument_v11",
                "terminal_decision": TERMINAL_DECISION,
                "human_review_requirement": "unmet",
                "paid_compute_authorized": False,
                "locks": locks,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "amendment": amendment,
        "diagnostic": diagnostic,
        "decision": repo / "decision.json",
        "evidence": repo / "evidence.json",
    }


def test_closure_writes_terminal_failed_decision(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decision, evidence = close_v11(
        repo_root=paths["repo"],
        amendment_path=paths["amendment"],
        diagnostic_path=paths["diagnostic"],
        decision_path=paths["decision"],
        evidence_path=paths["evidence"],
    )
    assert decision["passed"] is False
    assert decision["decision"] == TERMINAL_DECISION
    assert decision["human_review_requirement"] == "unmet"
    assert decision["authorization"] == NO_AUTHORIZATION
    assert evidence["terminal"] is True
    assert evidence["decision_sha256"] == sha256_file(paths["decision"])


def test_closure_rejects_ai_review_misrepresented_as_human(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    diagnostic = json.loads(paths["diagnostic"].read_text(encoding="utf-8"))
    diagnostic["human_review_requirement_satisfied"] = True
    _write_json(paths["diagnostic"], diagnostic)
    with pytest.raises(AdministrativeClosureError, match="diagnostic role"):
        close_v11(
            repo_root=paths["repo"],
            amendment_path=paths["amendment"],
            diagnostic_path=paths["diagnostic"],
            decision_path=paths["decision"],
            evidence_path=paths["evidence"],
        )


def test_closure_rejects_changed_locked_input(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    analysis = paths["repo"] / "analysis.json"
    analysis.write_text("{}\n", encoding="utf-8")
    with pytest.raises(AdministrativeClosureError, match="hash differs"):
        close_v11(
            repo_root=paths["repo"],
            amendment_path=paths["amendment"],
            diagnostic_path=paths["diagnostic"],
            decision_path=paths["decision"],
            evidence_path=paths["evidence"],
        )
