"""Roster validation and preregistered persona selection."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .schemas import ControlPersona, PersonaPair


PRIMARY_GROUPS = ("genpop", "unknown_ai", "famous_ai", "famous_nonai")
EXPECTED_GROUP_SIZE = 70


def load_roster(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for group in PRIMARY_GROUPS:
        rows = payload.get(group)
        if not isinstance(rows, list) or len(rows) != EXPECTED_GROUP_SIZE:
            raise ValueError(f"roster group {group} must contain exactly 70 rows")
        keys = [str(row["key"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"roster group {group} contains duplicate keys")
    return payload


def validate_twin_index(roster: dict[str, list[dict[str, Any]]], index: int) -> None:
    famous = roster["famous_ai"][index]
    unknown = roster["unknown_ai"][index]
    genpop = roster["genpop"][index]
    if unknown.get("_twin_of") != famous["name"]:
        raise ValueError(f"unknown-AI row {index} does not identify its famous twin")
    if unknown["name"] != genpop["name"]:
        raise ValueError(f"unknown-AI and genpop names differ at index {index}")
    if unknown.get("_usa_key") != genpop.get("_usa_key"):
        raise ValueError(f"unknown-AI and genpop source keys differ at index {index}")
    if famous.get("org") != unknown.get("org"):
        raise ValueError(f"famous and unknown-AI organizations differ at index {index}")


def validate_all_twin_indices(roster: dict[str, list[dict[str, Any]]]) -> None:
    for index in range(EXPECTED_GROUP_SIZE):
        validate_twin_index(roster, index)


def load_glm52_deltas(path: Path) -> dict[str, tuple[str, float, list[float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: (str(value[0]), float(value[1]), [float(item) for item in value[2]])
        for key, value in raw.items()
    }


def reproduce_delta_cache(
    deltas: dict[str, tuple[str, float, list[float]]],
) -> dict[str, Any]:
    max_error = 0.0
    group_values: dict[str, list[float]] = {}
    valid_counts: dict[str, int] = {}
    for group, mean, values in deltas.values():
        if not values:
            raise ValueError("delta cache contains an empty vector")
        recomputed = sum(values) / len(values)
        max_error = max(max_error, abs(recomputed - mean))
        if len(values) == 100:
            group_values.setdefault(group, []).append(mean)
            valid_counts[group] = valid_counts.get(group, 0) + 1
    group_means = {
        group: sum(values) / len(values) for group, values in sorted(group_values.items())
    }
    return {
        "entry_count": len(deltas),
        "max_person_mean_error": max_error,
        "valid_100_counts": valid_counts,
        "valid_100_group_means": group_means,
    }


def select_personas(
    roster: dict[str, list[dict[str, Any]]],
    deltas: dict[str, tuple[str, float, list[float]]],
    *,
    seed: int,
    enriched_count: int = 8,
    primary_count: int = 4,
    prospective_count: int = 8,
    famous_nonai_count: int = 8,
    genpop_count: int = 8,
    required_delta_count: int = 100,
) -> tuple[list[PersonaPair], list[ControlPersona]]:
    validate_all_twin_indices(roster)
    candidates: list[tuple[float, int]] = []
    for index, famous in enumerate(roster["famous_ai"]):
        value = deltas.get(famous["key"])
        if value and len(value[2]) == required_delta_count:
            candidates.append((value[1], index))
    if len(candidates) < enriched_count + prospective_count:
        raise ValueError("not enough complete famous-AI delta rows for preregistered selection")
    candidates.sort(key=lambda item: (item[0], item[1]))
    enriched = candidates[:enriched_count]
    enriched_indices = {index for _, index in enriched}
    remaining = [item for item in candidates if item[1] not in enriched_indices]
    rng = random.Random(seed)
    prospective = rng.sample(remaining, prospective_count)

    pairs: list[PersonaPair] = []
    for role, selected in (
        ("enriched_target", enriched),
        ("prospective_generality", prospective),
    ):
        for rank, (effect, index) in enumerate(selected):
            famous = roster["famous_ai"][index]
            unknown = roster["unknown_ai"][index]
            genpop = roster["genpop"][index]
            pairs.append(
                PersonaPair(
                    twin_index=index,
                    famous_key=famous["key"],
                    unknown_ai_key=unknown["key"],
                    genpop_key=genpop["key"],
                    famous_name=famous["name"],
                    unknown_name=unknown["name"],
                    organization=famous["org"],
                    prior_glm52_name_effect_pp=effect,
                    selection_role=role,
                    primary_intervention=role == "enriched_target" and rank < primary_count,
                )
            )

    famous_controls = rng.sample(roster["famous_nonai"], famous_nonai_count)
    genpop_controls = rng.sample(roster["genpop"], genpop_count)
    controls = [
        ControlPersona(key=row["key"], name=row["name"], group="famous_nonai")
        for row in famous_controls
    ] + [
        ControlPersona(key=row["key"], name=row["name"], group="genpop") for row in genpop_controls
    ]
    return pairs, controls


def assert_target_exclusion(
    rows: list[dict[str, Any]],
    pairs: list[PersonaPair],
    *,
    exclude_primary_organizations: bool,
) -> None:
    forbidden_keys = {
        key for pair in pairs for key in (pair.famous_key, pair.unknown_ai_key, pair.genpop_key)
    }
    forbidden_names = {
        name.casefold() for pair in pairs for name in (pair.famous_name, pair.unknown_name)
    }
    forbidden_orgs = {pair.organization.casefold() for pair in pairs if pair.primary_intervention}
    for row in rows:
        text = json.dumps(row, sort_keys=True).casefold()
        if any(key.casefold() in text for key in forbidden_keys):
            raise ValueError("target persona key leaked into direction training")
        if any(name in text for name in forbidden_names):
            raise ValueError("target persona name leaked into direction training")
        if exclude_primary_organizations and any(org in text for org in forbidden_orgs):
            raise ValueError("primary target organization leaked into strict direction training")
