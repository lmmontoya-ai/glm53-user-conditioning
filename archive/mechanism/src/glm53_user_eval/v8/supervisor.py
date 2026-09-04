"""Fail-closed gate and paid-run supervision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GATES = [f"M{index}" for index in range(9)]


def require_previous(decision_root: Path, gate: str) -> None:
    index = GATES.index(gate)
    if index == 0:
        return
    prior = decision_root / f"{GATES[index - 1].lower()}_decision.json"
    if not prior.is_file():
        raise ValueError(f"missing preceding decision: {prior}")
    payload = json.loads(prior.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise ValueError(f"preceding gate {GATES[index - 1]} did not pass")


def projected_cost(
    hourly_rate: float, gpu_count: int, hours: float, headroom: float = 1.2
) -> float:
    values = (hourly_rate, hours, headroom)
    if any(value < 0 for value in values) or gpu_count <= 0:
        raise ValueError("invalid cost inputs")
    return float(hourly_rate * gpu_count * hours * headroom)


def validate_paid_launch(
    *,
    v7_green_light: bool,
    prereg_tag_commit: str | None,
    tree_clean: bool,
    volume_id: str,
    expected_volume_id: str,
    volume_size_gb: int,
    live_hourly_rate: float,
    gpu_count: int,
    projected_hours: float,
    hard_cap: float,
    balance: float,
    reserve: float,
    terminate_after_hours: float,
) -> dict[str, Any]:
    checks = {
        "v7_green_light": v7_green_light,
        "prereg_tag_exists": bool(prereg_tag_commit),
        "tree_clean": tree_clean,
        "volume_identity": volume_id == expected_volume_id,
        "volume_size": volume_size_gb >= 500,
        "termination_deadline": terminate_after_hours > 0,
    }
    estimate = projected_cost(live_hourly_rate, gpu_count, projected_hours)
    checks["under_hard_cap"] = estimate <= hard_cap
    checks["reserve_preserved"] = balance - estimate >= reserve
    if not all(checks.values()):
        raise ValueError(f"paid launch blocked: {checks}")
    return {"checks": checks, "projected_cost_usd": estimate}


def decision_payload(
    gate: str, checks: dict[str, bool], inputs: dict[str, Any], estimates: dict[str, Any]
) -> dict[str, Any]:
    passed = bool(checks) and all(checks.values())
    return {
        "schema_version": "glm53_v8_decision_v1",
        "gate": gate,
        "passed": passed,
        "decision": "proceed" if passed else "stop",
        "checks": checks,
        "inputs": inputs,
        "estimates": estimates,
    }
