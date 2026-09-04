from __future__ import annotations

import ast
from pathlib import Path

from src.glm53_user_eval.v8.artifacts import atomic_json, sha256_file
from src.glm53_user_eval.v8.independent import _verify_decision_lineage

ROOT = Path(__file__).resolve().parents[2]


def test_independent_verifier_does_not_import_primary_analysis_modules() -> None:
    path = ROOT / "src/glm53_user_eval/v8/independent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden = {
        "src.glm53_user_eval.v8.science",
        "src.glm53_user_eval.v8.decisions",
        "src.glm53_user_eval.v8.statistics",
        "src.glm53_user_eval.v8.probes",
    }
    assert imports.isdisjoint(forbidden)


def test_independent_verifier_checks_decision_input_hashes(tmp_path) -> None:
    source = tmp_path / "source.json"
    atomic_json(source, {"value": 1})
    atomic_json(
        tmp_path / "decisions/m2_decision.json",
        {
            "gate": "M2",
            "passed": True,
            "checks": {"runtime": True},
            "inputs": {
                "source": {
                    "path": source.as_posix(),
                    "sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                }
            },
        },
    )
    checks = _verify_decision_lineage(tmp_path)
    assert checks["m2_classification"] is True
    assert checks["m2_inputs_present"] is True
    assert checks["m2_input_source"] is True


def test_independent_verifier_rejects_tampered_decision_input(tmp_path) -> None:
    source = tmp_path / "source.json"
    atomic_json(source, {"value": 1})
    original_hash = sha256_file(source)
    original_size = source.stat().st_size
    atomic_json(
        tmp_path / "decisions/m2_decision.json",
        {
            "gate": "M2",
            "passed": True,
            "checks": {"runtime": True},
            "inputs": {
                "source": {
                    "path": source.as_posix(),
                    "sha256": original_hash,
                    "size_bytes": original_size,
                }
            },
        },
    )
    atomic_json(source, {"value": 2})
    checks = _verify_decision_lineage(tmp_path)
    assert checks["m2_input_source"] is False
