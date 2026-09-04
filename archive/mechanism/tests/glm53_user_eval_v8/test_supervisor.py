from __future__ import annotations

import json

import pytest
from src.glm53_user_eval.v8.supervisor import (
    decision_payload,
    projected_cost,
    require_previous,
    validate_paid_launch,
)


def test_projected_cost() -> None:
    assert projected_cost(2.0, 4, 1.0) == pytest.approx(9.6)


def test_invalid_cost() -> None:
    with pytest.raises(ValueError):
        projected_cost(-1, 4, 1)


def test_paid_launch_passes() -> None:
    result = validate_paid_launch(
        v7_green_light=True,
        prereg_tag_commit="a",
        tree_clean=True,
        volume_id="v",
        expected_volume_id="v",
        volume_size_gb=500,
        live_hourly_rate=2,
        gpu_count=4,
        projected_hours=1,
        hard_cap=20,
        balance=30,
        reserve=10,
        terminate_after_hours=9,
    )
    assert all(result["checks"].values())


def test_paid_launch_blocks_missing_tag() -> None:
    with pytest.raises(ValueError):
        validate_paid_launch(
            v7_green_light=True,
            prereg_tag_commit=None,
            tree_clean=True,
            volume_id="v",
            expected_volume_id="v",
            volume_size_gb=500,
            live_hourly_rate=2,
            gpu_count=4,
            projected_hours=1,
            hard_cap=20,
            balance=30,
            reserve=10,
            terminate_after_hours=9,
        )


def test_previous_gate_required(tmp_path) -> None:
    with pytest.raises(ValueError):
        require_previous(tmp_path, "M1")


def test_previous_gate_passes(tmp_path) -> None:
    (tmp_path / "m0_decision.json").write_text(json.dumps({"passed": True}))
    require_previous(tmp_path, "M1")


def test_decision_fails_closed() -> None:
    assert decision_payload("M0", {"a": True, "b": False}, {}, {})["passed"] is False
