import json
from pathlib import Path

import pytest

from src.glm53_user_eval.personas import (
    load_glm52_deltas,
    load_roster,
    reproduce_delta_cache,
    select_personas,
    validate_all_twin_indices,
    validate_twin_index,
)


SOURCE = Path(__file__).resolve().parents[3] / "reference/transluce-user-awareness"


@pytest.fixture(scope="module")
def roster():
    return load_roster(SOURCE / "core/personas2.json")


@pytest.fixture(scope="module")
def deltas():
    return load_glm52_deltas(SOURCE / "cache/aggregates/s2glm52_deltas_conf.json")


def test_primary_groups_have_70_rows(roster) -> None:
    assert all(
        len(roster[group]) == 70 for group in ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
    )


def test_all_twin_indices_validate(roster) -> None:
    validate_all_twin_indices(roster)


def test_mismatched_same_name_fails(roster) -> None:
    changed = json.loads(json.dumps(roster))
    changed["genpop"][0]["name"] = "Different Person"
    with pytest.raises(ValueError, match="names differ"):
        validate_twin_index(changed, 0)


def test_delta_cache_person_means_reproduce(deltas) -> None:
    report = reproduce_delta_cache(deltas)
    assert report["entry_count"] == 339
    assert report["max_person_mean_error"] < 1e-9


def test_selection_is_deterministic(roster, deltas) -> None:
    first = select_personas(roster, deltas, seed=20260828)
    second = select_personas(roster, deltas, seed=20260828)
    assert first == second


def test_enriched_selection_uses_glm52_rank(roster, deltas) -> None:
    pairs, _ = select_personas(roster, deltas, seed=20260828)
    enriched = [pair for pair in pairs if pair.selection_role == "enriched_target"]
    assert len(enriched) == 8
    assert enriched[0].famous_key == "fai2_neel_nanda"
    assert [pair.prior_glm52_name_effect_pp for pair in enriched] == sorted(
        pair.prior_glm52_name_effect_pp for pair in enriched
    )


def test_primary_four_are_fixed_by_prior_rank(roster, deltas) -> None:
    pairs, _ = select_personas(roster, deltas, seed=20260828)
    primary = [pair for pair in pairs if pair.primary_intervention]
    assert [pair.famous_key for pair in primary] == [
        "fai2_neel_nanda",
        "fai2_dylan_hadfieldmenell",
        "fai2_geoffrey_irving",
        "fai2_owain_evans",
    ]


def test_control_counts_are_fixed(roster, deltas) -> None:
    _, controls = select_personas(roster, deltas, seed=20260828)
    assert sum(control.group == "famous_nonai" for control in controls) == 8
    assert sum(control.group == "genpop" for control in controls) == 8
