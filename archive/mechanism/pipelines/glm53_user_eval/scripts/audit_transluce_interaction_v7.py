"""Score-blind technical audit for v7, including completeness and route integrity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from inspect_ai.log import read_eval_log_sample_summaries
from pipelines.glm53_user_eval.scripts.audit_transluce_exact_v6 import (
    atomic_json,
    audit_log,
    newest_success,
)

GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_ai_real", "famous_nonai")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prereg", type=Path)
    parser.add_argument("--expected-completed-shards", type=int)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    prereg_path = args.prereg or Path(
        "pipelines/glm53_user_eval/configs/prereg_v7_interaction_confirmatory.yaml"
    )
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    thresholds = prereg["technical_validity"]
    manifest = json.loads((args.run_root / "schedule_manifest.json").read_text(encoding="utf-8"))
    expected = {row["shard_id"]: row for row in manifest["shards"]}
    reports: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicates: list[tuple[str, str, str]] = []
    total_rows = 0
    total_valid = 0
    group_total = {group: 0 for group in GROUPS}
    group_valid = {group: 0 for group in GROUPS}
    for shard_id, spec in expected.items():
        log_path = newest_success(args.run_root / "eval_logs" / shard_id)
        if log_path is None:
            continue
        report = audit_log(log_path, int(spec["expected_rows"]))
        report["shard_id"] = shard_id
        reports.append(report)
        group = str(spec["group"])
        for sample in read_eval_log_sample_summaries(log_path):
            metadata = sample.metadata or {}
            persona = str(metadata.get("persona") or "")
            stimulus = str(metadata.get("scenario_id") or "")
            key = (group, persona, stimulus)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            total_rows += 1
            scores = list((sample.scores or {}).values())
            valid = False
            if sample.error is None and scores:
                try:
                    value = float(scores[0].value)
                    valid = math.isfinite(value) and 0.0 <= value <= 1.0
                except (TypeError, ValueError):
                    pass
            total_valid += int(valid)
            if persona != "anon":
                group_total[group] += 1
                group_valid[group] += int(valid)
    all_expected = len(reports) == int(thresholds["required_shards"])
    expected_count_ok = total_rows == int(thresholds["required_scientific_rows"])
    overall_valid_rate = total_valid / total_rows if total_rows else 0.0
    group_rates = {
        group: group_valid[group] / group_total[group] if group_total[group] else 0.0
        for group in GROUPS
    }
    missingness_spread = max(1.0 - value for value in group_rates.values()) - min(
        1.0 - value for value in group_rates.values()
    )
    requested_count_ok = (
        args.expected_completed_shards is None or len(reports) == args.expected_completed_shards
    )
    contract_ok = bool(reports) and all(report["passed"] for report in reports)
    technical_checks = {
        "requested_completed_shards": requested_count_ok,
        "all_audited_shards_pass_contract": contract_ok,
        "duplicate_scientific_keys_allowed": len(duplicates)
        <= int(thresholds["duplicate_scientific_keys_allowed"]),
        "overall_valid_rate": overall_valid_rate >= float(thresholds["total_valid_score_rate_min"]),
        "per_group_valid_rates": all(
            value >= float(thresholds["per_group_nonanonymous_valid_rate_min"])
            for value in group_rates.values()
        ),
        "missingness_spread": missingness_spread
        <= float(thresholds["condition_missingness_spread_max"]),
    }
    if args.require_all:
        technical_checks |= {
            "all_expected_present": all_expected,
            "exact_scientific_row_count": expected_count_ok,
        }
    passed = all(technical_checks.values())
    payload = {
        "schema_version": "glm53_transluce_interaction_v7_technical_audit_v1",
        "audited_shards": len(reports),
        "audited_rows": total_rows,
        "valid_score_count": total_valid,
        "overall_valid_score_rate": overall_valid_rate,
        "group_nonanonymous_valid_rates": group_rates,
        "condition_missingness_spread": missingness_spread,
        "duplicate_scientific_key_count": len(duplicates),
        "api_cost_usd": sum(float(report["api_cost_usd"]) for report in reports),
        "retry_error_event_count": sum(
            int(report["retry_error_event_count"]) for report in reports
        ),
        "expected_shards": int(thresholds["required_shards"]),
        "required_all": args.require_all,
        "all_expected_present": all_expected,
        "all_expected_rows_present": expected_count_ok,
        "checks": technical_checks,
        "passed": passed,
        "reports": reports,
    }
    atomic_json(args.output, payload)
    marker = args.run_root / "FULL_RUN_AUDIT_PASS.json"
    if args.require_all and passed:
        atomic_json(
            marker,
            {
                "schema_version": "glm53_v7_full_run_audit_pass_v1",
                "technical_audit": str(args.output.resolve()),
                "passed": True,
            },
        )
    elif marker.exists():
        marker.unlink()
    print(
        json.dumps(
            {
                "audited_shards": len(reports),
                "audited_rows": total_rows,
                "valid_score_rate": overall_valid_rate,
                "api_cost_usd": payload["api_cost_usd"],
                "passed": passed,
                "scores_aggregated": False,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
