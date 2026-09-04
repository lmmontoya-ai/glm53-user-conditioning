"""Build the compact final v7 evidence manifest and plain-language report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def newest_success(path: Path) -> Path:
    logs = sorted(path.glob("*.eval"), key=lambda value: value.stat().st_mtime)
    for log in reversed(logs):
        if str(read_eval_log(log, header_only=True).status) == "success":
            return log
    raise ValueError(f"no successful log in {path}")


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--technical-audit", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    args = parser.parse_args()
    analysis_path = args.report_root / "analysis.json"
    verification_path = args.report_root / "verification.json"
    decision_path = args.report_root / "decision.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not verification.get("passed"):
        raise ValueError("cannot build final evidence before verification passes")
    manifest = json.loads((args.run_root / "schedule_manifest.json").read_text(encoding="utf-8"))
    logs = [
        newest_success(args.run_root / "eval_logs" / str(spec["shard_id"]))
        for spec in manifest["shards"]
    ]
    if len(logs) != 100:
        raise ValueError("final evidence requires 100 successful logs")
    point = float(analysis["primary"]["interaction_pp"])
    low, high = [float(value) for value in analysis["primary"]["ci95_pp"]]
    report = f"""# GLM-5.3 Transluce interaction confirmation v7

## Result

The fresh full-population run produced an AI-specific fame interaction of **{point:.4f} percentage points** (95% crossed-bootstrap interval **[{low:.4f}, {high:.4f}]**). The machine decision is **`{decision["decision"]}`**. The white-box resource gate is **{str(bool(decision["whitebox_green_light"])).lower()}**.

The interaction is `(Famous-AI - Unknown-AI) - (Famous-non-AI - GenPop)`, using person-weighted, dilemma-centered folded confidence. The v6 result generated this hypothesis; only fresh v7 calls determine confirmation.

## Execution

- Model: `z-ai/glm-5.3-flash`
- Provider: Novita, FP8, fallbacks disabled
- Reasoning: high
- Population: five complete Transluce groups across all 100 dilemmas
- Scientific rows: {analysis["row_count"]:,}
- Valid scores: {analysis["valid_score_count"]:,}
- Primary bootstrap: 20,000 crossed person/dilemma draws
- Independent verification: passed

## Components

- Famous-AI minus Unknown-AI: {analysis["components"]["famous_ai_minus_unknown_ai"]["point_pp"]:.4f} pp
- Famous-non-AI minus GenPop: {analysis["components"]["famous_nonai_minus_genpop"]["point_pp"]:.4f} pp
- Unknown-AI minus GenPop: {analysis["components"]["unknown_ai_minus_genpop"]["point_pp"]:.4f} pp
- Matched public minus constructed address: {analysis["matched_public_minus_constructed_address"]["point_pp"]:.4f} pp

## Limitation

This population contrast is not a randomized 2×2 fame-by-AI experiment. It tests whether the four released identity populations differ in the preregistered pattern; it does not by itself establish evaluation awareness or a causal internal mechanism.
"""
    report_path = args.report_root / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    files = [
        args.prereg,
        args.run_root / "schedule_manifest.json",
        args.report_root / "raw_scores.jsonl",
        args.report_root / "person_deltas.json",
        analysis_path,
        verification_path,
        args.technical_audit,
        args.manual_audit,
        decision_path,
        report_path,
    ]
    evidence = {
        "schema_version": "glm53_transluce_interaction_v7_final_evidence_v1",
        "project_id": "glm53_user_eval_interaction_confirmatory_v7",
        "decision": decision["decision"],
        "whitebox_green_light": decision["whitebox_green_light"],
        "interaction_pp": point,
        "interaction_ci95_pp": [low, high],
        "compact_files": [record(path) for path in files],
        "successful_eval_logs": [record(path) for path in logs],
        "raw_logs_committed": False,
        "raw_logs_retained_locally_and_hashed": True,
        "self_hash_note": "final_evidence.json cannot contain its own SHA-256; hash it after creation",
    }
    evidence_path = args.report_root / "final_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "interaction_pp": point,
                "final_evidence_sha256": sha256(evidence_path),
                "final_report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
