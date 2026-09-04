from __future__ import annotations

import json

import numpy as np
from src.glm53_user_eval.v8.probes import grouped_permutation

from .conftest import ROOT


def test_split_hashes_exist() -> None:
    payload = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v8/configs/direction_splits_v1.json").read_text()
    )
    assert len(payload["samples_sha256"]) == 64
    assert len(payload["splits_sha256"]) == 64


def test_split_counts_total() -> None:
    payload = json.loads(
        (ROOT / "pipelines/glm53_user_eval/v8/configs/direction_splits_v1.json").read_text()
    )
    assert sum(payload["split_counts"].values()) == 448


def test_grouped_permutation_constant_within_group() -> None:
    labels = np.array([0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b"])
    result = grouped_permutation(labels, groups, 3)
    assert result[0] == result[1] and result[2] == result[3]


def test_grouped_permutation_reproducible() -> None:
    labels = np.array([0, 0, 1, 1])
    groups = np.array([0, 0, 1, 1])
    assert np.array_equal(
        grouped_permutation(labels, groups, 8), grouped_permutation(labels, groups, 8)
    )


def test_target_pairs_are_sixteen(schedule: dict) -> None:
    assert len(schedule["pairs"]) == 16


def test_target_sets_are_eight_each(schedule: dict) -> None:
    assert sum(row["set"] == "enriched" for row in schedule["pairs"]) == 8
    assert sum(row["set"] == "prospective" for row in schedule["pairs"]) == 8
