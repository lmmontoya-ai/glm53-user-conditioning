"""Immutable inputs and fail-closed validation for V20."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from src.glm53_user_eval.v17.contract import (
    atomic_json,
    atomic_jsonl,
    canonical_sha256,
    read_json,
    read_jsonl,
    read_yaml,
    sha256_file,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_signs(derivation_index: int) -> list[int]:
    ranked = sorted(
        range(16),
        key=lambda pair_index: hashlib.sha256(
            f"glm53-v20-null|{derivation_index}|{pair_index + 1}".encode()
        ).hexdigest(),
    )
    positive = set(ranked[:8])
    return [1 if pair_index in positive else -1 for pair_index in range(16)]


def validate_v20_prereg(root: Path, path: Path, *, verify_git: bool = True) -> dict[str, Any]:
    prereg = read_yaml(path)
    _require(
        prereg.get("schema_version") == "glm53_user_eval_v20_prereg_v1",
        "wrong V20 schema",
    )
    _require(
        prereg.get("project_id") == "glm53_user_eval_direct_hua_v20",
        "wrong V20 project",
    )
    for name, record in prereg["immutable_inputs"].items():
        source = root / record["path"]
        _require(source.is_file(), f"missing V20 input: {name}")
        _require(
            sha256_file(source) == record["sha256"],
            f"V20 input hash mismatch: {name}",
        )

    parent = prereg["parent"]
    if verify_git:
        tagged = subprocess.check_output(
            ["git", "rev-list", "-n", "1", parent["final_tag"]],
            cwd=root,
            text=True,
        ).strip()
        _require(tagged == parent["final_commit"], "V19 final tag moved")
    parent_decision = read_json(
        root / prereg["immutable_inputs"]["v19_positive_control_decision"]["path"]
    )
    _require(
        parent_decision["decision"] == "stop_hua_direction_did_not_transfer",
        "wrong V19 parent decision",
    )
    _require(parent_decision["passed"] is False, "V19 unexpectedly passed")
    _require(
        parent_decision["authorization"]
        == {"causal_user_test": False, "local_proxy_parity": False},
        "V19 authorization record changed",
    )

    directions_path = root / prereg["immutable_inputs"]["v19_directions"]["path"]
    with np.load(directions_path) as arrays:
        _require(
            set(arrays.files)
            == {
                "hua_pair_differences",
                "formality_pair_differences",
                "all16",
                "status",
                "formality",
            },
            "V19 direction keys changed",
        )
        _require(
            arrays["hua_pair_differences"].shape == (16, 45, 4096),
            "V19 Hua pair differences have the wrong shape",
        )
        for name in ("all16", "status", "formality"):
            _require(arrays[name].shape == (45, 4096), f"V19 {name} direction has the wrong shape")
            _require(np.isfinite(arrays[name]).all(), f"V19 {name} direction is not finite")
        _require(
            np.array_equal(
                arrays["all16"], arrays["hua_pair_differences"].mean(axis=0).astype(np.float32)
            ),
            "V19 all-16 direction is not the exact paired mean",
        )

    parent_design = read_json(root / prereg["immutable_inputs"]["parent_design"]["path"])
    _require(len(parent_design["tasks"]) == 22, "V20 requires the frozen 22 tasks")
    _require(
        all(len(values) == 16 for values in parent_design["identities"].values()),
        "V20 requires 16 identities per group",
    )
    _require(
        parent_design["reconstructable_base_rows"] == 1404,
        "V20 parent row count changed",
    )

    design = read_json(root / prereg["immutable_inputs"]["design"]["path"])
    _require(
        design["schema_version"] == "glm53_v20_direct_hua_design_v1",
        "wrong V20 design schema",
    )
    _require(
        design["full_surface"]["conditions"]
        == [
            "baseline",
            "all16_deploy",
            "all16_eval",
            "status_deploy",
            "actor_deploy",
            "formality",
        ],
        "V20 full condition order changed",
    )
    _require(design["full_surface"]["reconstructable_rows"] == 1404, "wrong V20 full rows")
    _require(design["signflip_control_surface"]["rows_per_control"] == 80, "wrong V20 null rows")
    _require(design["signflip_control_surface"]["controls"] == 20, "wrong V20 null count")
    _require(design["planned_scientific_prompt_evaluations"] == 10024, "wrong V20 prompt count")

    nulls = read_json(root / prereg["immutable_inputs"]["null_signs"]["path"])
    controls = nulls["controls"]
    _require(len(controls) == 20, "V20 needs 20 sign-flip controls")
    observed: set[tuple[int, ...]] = set()
    for control_index, control in enumerate(controls):
        _require(control["control_id"] == f"signflip_{control_index:02d}", "V20 null ID changed")
        signs = [int(value) for value in control["signs"]]
        _require(
            signs == expected_signs(int(control["derivation_index"])), "V20 null derivation changed"
        )
        _require(signs.count(1) == signs.count(-1) == 8, "V20 null is not balanced")
        key = tuple(signs)
        _require(
            key not in observed and tuple(-value for value in signs) not in observed,
            "duplicate V20 null",
        )
        observed.add(key)

    runtime = read_yaml(root / prereg["immutable_inputs"]["runtime"]["path"])
    _require(runtime["runpod"]["gpu_id"] == "NVIDIA B300 SXM6 AC", "V20 hardware changed")
    _require(runtime["runpod"]["gpu_count"] == 2, "V20 needs two B300s")
    _require(runtime["runpod"]["allow_gpu_type_fallback"] is False, "V20 GPU fallback is forbidden")
    _require(
        runtime["forward"]["candidate_batch_sizes"] == [1, 2, 4], "V20 batch candidates changed"
    )
    _require(
        runtime["throughput_gate"]["planned_scientific_prompt_evaluations"] == 10024,
        "V20 runtime prompt count changed",
    )
    _require(prereg["intervention"]["layers"] == [10, 14, 18, 22, 26, 30], "V20 layers changed")
    _require(prereg["intervention"]["deployment_alpha"] == 0.6, "V20 deployment alpha changed")
    _require(prereg["intervention"]["evaluation_alpha"] == -0.6, "V20 evaluation alpha changed")
    _require(prereg["statistics"]["bootstrap_reps"] == 20000, "V20 bootstrap count changed")
    _require(
        prereg["scope"]["explicit_status_positive_control_gate"] is False,
        "V20 reinstated the V19 gate",
    )
    return {
        "schema_version": "glm53_v20_prereg_validation_v1",
        "passed": True,
        "prereg_sha256": sha256_file(path),
        "immutable_input_count": len(prereg["immutable_inputs"]),
        "planned_scientific_prompt_evaluations": design["planned_scientific_prompt_evaluations"],
        "signflip_controls": len(controls),
    }


__all__ = [
    "atomic_json",
    "atomic_jsonl",
    "canonical_sha256",
    "expected_signs",
    "read_json",
    "read_jsonl",
    "read_yaml",
    "sha256_file",
    "validate_v20_prereg",
]
