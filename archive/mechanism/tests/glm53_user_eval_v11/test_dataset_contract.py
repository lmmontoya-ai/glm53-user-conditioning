from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter

import pytest
from src.glm53_user_eval.v11.builder import (
    _binary_rows,
    _factorial_calibration_rows,
    _neutral_rows,
    build_dataset,
)
from src.glm53_user_eval.v11.dataset import NUISANCE_KEYS, validate_rows
from src.glm53_user_eval.v11.spec import (
    ASSESSMENT_MODES,
    FINAL_BINARY_SOURCE,
    FINAL_NEUTRAL_SOURCE,
    GENERATOR_FAMILIES,
    OPERATIONAL_ACTORS,
    OPERATIONAL_OUTCOMES,
    PROMPT_ROLES,
    SPLIT_PAIR_COUNTS,
    TASKS,
)


def rows() -> list[dict]:
    result: list[dict] = []
    for split, count in SPLIT_PAIR_COUNTS.items():
        result.extend(_binary_rows(split, count))
    result.extend(_neutral_rows())
    result.extend(_factorial_calibration_rows())
    return result


def test_complete_dataset_passes_structural_contract() -> None:
    report = validate_rows(rows())
    assert report["passed"] is True
    assert report["row_count"] == 576
    assert report["split_counts"]["train"] == 256
    assert report["split_counts"]["final_counterfactual"] == 64


def test_builder_writes_576_row_manifest_with_frozen_source_hashes(tmp_path) -> None:
    manifest = build_dataset(tmp_path)
    assert manifest["row_count"] == 576
    assert manifest["factorial_calibration_row_count"] == 32
    assert manifest["split_counts"]["factorial_calibration"] == 32
    assert set(manifest["frozen_sources"]) == {FINAL_BINARY_SOURCE, FINAL_NEUTRAL_SOURCE}
    assert (tmp_path / "samples.jsonl").read_text(encoding="utf-8").count("\n") == 576


def test_all_binary_pairs_have_identical_nuisance_and_suffix() -> None:
    by_pair: dict[str, list[dict]] = {}
    for row in rows():
        if row["split"] not in SPLIT_PAIR_COUNTS:
            continue
        by_pair.setdefault(row["pair_id"], []).append(row)
    assert len(by_pair) == 240
    for pair in by_pair.values():
        assert {item["label"] for item in pair} == {0, 1}
        assert pair[0]["nuisance"] == pair[1]["nuisance"]
        assert pair[0]["shared_suffix"] == pair[1]["shared_suffix"]


def test_generators_and_tasks_are_disjoint_across_binary_splits() -> None:
    materialized = rows()
    seen_generators: set[str] = set()
    seen_tasks: set[str] = set()
    for split in SPLIT_PAIR_COUNTS:
        split_rows = [row for row in materialized if row["split"] == split]
        generators = {row["generator_family"] for row in split_rows}
        tasks = {row["task_id"] for row in split_rows}
        assert generators == set(GENERATOR_FAMILIES[split])
        assert not generators & seen_generators
        assert not tasks & seen_tasks
        seen_generators |= generators
        seen_tasks |= tasks


def test_orthogonal_nuisance_ledger_is_balanced_within_generators() -> None:
    materialized = rows()
    for split, families in GENERATOR_FAMILIES.items():
        for family in families:
            pair_rows = [
                row
                for row in materialized
                if row["split"] == split
                and row["generator_family"] == family
                and row["label"] == 1
            ]
            for key in NUISANCE_KEYS:
                assert sum(row["nuisance"][key] for row in pair_rows) == len(pair_rows) // 2


def test_factorial_calibration_covers_every_boolean_cell() -> None:
    calibration = [row for row in rows() if row["split"] == "factorial_calibration"]
    cells = {
        (
            row["factors"]["real_external_recipient"],
            row["factors"]["operational_consequence"],
            row["factors"]["model_scoring"],
            row["factors"]["comparison_among_models"],
        )
        for row in calibration
    }
    assert len(cells) == 16
    for cell in cells:
        members = [
            row
            for row in calibration
            if (
                row["factors"]["real_external_recipient"],
                row["factors"]["operational_consequence"],
                row["factors"]["model_scoring"],
                row["factors"]["comparison_among_models"],
            )
            == cell
        ]
        assert len(members) == 2
        assert {row["register"] for row in members} == {"formal", "casual"}
        assert len({row["prompt_role"] for row in members}) == 2
        assert len({row["task_id"] for row in members}) == 2


def test_factorial_calibration_orthogonalizes_role_and_task() -> None:
    calibration = [row for row in rows() if row["split"] == "factorial_calibration"]
    factor_names = (
        "real_external_recipient",
        "operational_consequence",
        "model_scoring",
        "comparison_among_models",
    )
    expected_roles = Counter({role: 4 for role in PROMPT_ROLES})
    expected_tasks = Counter({task[0]: 4 for task in TASKS["neutral"]})
    expected_pairs = Counter(
        {(role, task[0]): 1 for role in PROMPT_ROLES for task in TASKS["neutral"]}
    )
    for factor_name in factor_names:
        for factor_value in (False, True):
            subset = [
                row
                for row in calibration
                if row["factors"][factor_name] is factor_value
            ]
            assert Counter(row["prompt_role"] for row in subset) == expected_roles
            assert Counter(row["task_id"] for row in subset) == expected_tasks
            assert Counter(
                (row["prompt_role"], row["task_id"]) for row in subset
            ) == expected_pairs
    for register in ("formal", "casual"):
        subset = [row for row in calibration if row["register"] == register]
        assert Counter(
            (row["prompt_role"], row["task_id"]) for row in subset
        ) == expected_pairs


def test_factorial_calibration_semantic_labels_match_construct() -> None:
    calibration = [row for row in rows() if row["split"] == "factorial_calibration"]
    for row in calibration:
        factors = row["factors"]
        operational = (
            factors["real_external_recipient"]
            or factors["operational_consequence"]
        )
        assessment = factors["model_scoring"] or factors["comparison_among_models"]
        if operational and not assessment:
            expected = ["deploy"]
        elif assessment and not operational:
            expected = ["eval"]
        elif operational and assessment:
            expected = ["ambiguous"]
        else:
            expected = ["no_status", "ambiguous"]
        assert row["acceptable_judge_labels"] == expected


def test_each_neutral_family_has_disjoint_development_and_final_halves() -> None:
    neutral = [row for row in rows() if row["split"] == "neutral_controls"]
    for family in {row["generator_family"] for row in neutral}:
        family_rows = [row for row in neutral if row["generator_family"] == family]
        assert sum(row["control_partition"] == "development" for row in family_rows) == 8
        assert sum(row["control_partition"] == "final" for row in family_rows) == 8
        development = {
            row["scenario_text"]
            for row in family_rows
            if row["control_partition"] == "development"
        }
        final = {
            row["scenario_text"]
            for row in family_rows
            if row["control_partition"] == "final"
        }
        assert len(development) == 8
        assert len(final) == 8
        assert development.isdisjoint(final)


def test_binary_labels_follow_the_latent_relation_not_nuisance_words() -> None:
    for row in rows():
        if row["split"] not in SPLIT_PAIR_COUNTS:
            continue
        if row["label"] == 1:
            assert row["factors"]["primary_model_assessment"] is True
            assert row["factors"]["operational_consequence"] is False
            assert row["factors"]["real_external_recipient"] is False
        else:
            assert row["factors"]["primary_model_assessment"] is False
            assert row["factors"]["operational_consequence"] is True


def test_aliases_are_pair_unique_and_pair_lexical_inventory_is_exact() -> None:
    aliases_seen: set[str] = set()
    by_pair: dict[str, list[dict]] = {}
    for row in rows():
        if row["split"] in SPLIT_PAIR_COUNTS:
            by_pair.setdefault(row["pair_id"], []).append(row)
    for pair in by_pair.values():
        aliases = set(pair[0]["opaque_aliases"])
        assert len(aliases) == 4
        assert aliases.isdisjoint(aliases_seen)
        aliases_seen |= aliases
        inventories = [
            Counter(
                re.findall(
                    r"\w+|[^\w\s]",
                    f"{row['system_prompt']} {row['user_prompt']}".casefold(),
                )
            )
            for row in pair
        ]
        assert inventories[0] == inventories[1]


def test_assessment_subfactors_are_rendered_and_counterrotated() -> None:
    materialized = [
        row
        for row in rows()
        if row["split"] in SPLIT_PAIR_COUNTS and row["label"] == 1
    ]
    for row in materialized:
        contract = ASSESSMENT_MODES[row["assessment_design"]]
        assert contract["actor"] in row["scenario_text"]
        assert contract["outcome"] in row["scenario_text"]
        assert row["factors"]["model_scoring"] is contract["model_scoring"]
        assert (
            row["factors"]["comparison_among_models"]
            is contract["comparison_among_models"]
        )
    for split in SPLIT_PAIR_COUNTS:
        split_rows = [row for row in materialized if row["split"] == split]
        for field in ("prompt_role", "task_id", "assessment_design"):
            for value in {row[field] for row in split_rows}:
                subset = [row for row in split_rows if row[field] == value]
                for key in NUISANCE_KEYS:
                    assert {row["nuisance"][key] for row in subset} == {False, True}


def test_operational_and_assessment_actor_outcome_bundles_are_coherent() -> None:
    binary = [
        row
        for row in rows()
        if row["split"] in SPLIT_PAIR_COUNTS and row["label"] == 1
    ]
    assert len(binary) == 240
    for row in binary:
        mapping = row["decisive_fact_texts"][0]
        actor_indices = [
            index
            for index, (actor, _) in enumerate(OPERATIONAL_ACTORS)
            if actor in mapping
        ]
        outcome_indices = [
            index
            for index, outcome in enumerate(OPERATIONAL_OUTCOMES)
            if outcome in mapping
        ]
        assert actor_indices == outcome_indices
        assert len(actor_indices) == 1

        assessment = ASSESSMENT_MODES[row["assessment_design"]]
        assert assessment["actor"] in mapping
        assert assessment["outcome"] in mapping


def test_bundle_coherence_cannot_predict_the_binary_label() -> None:
    by_label: dict[int, list[bool]] = {0: [], 1: []}
    for row in rows():
        if row["split"] not in SPLIT_PAIR_COUNTS:
            continue
        mapping = row["decisive_fact_texts"][0]
        if row["label"] == 1:
            assessment = ASSESSMENT_MODES[row["assessment_design"]]
            coherent = assessment["actor"] in mapping and assessment["outcome"] in mapping
        else:
            actor_indices = [
                index
                for index, (actor, _) in enumerate(OPERATIONAL_ACTORS)
                if actor in mapping
            ]
            outcome_indices = [
                index
                for index, outcome in enumerate(OPERATIONAL_OUTCOMES)
                if outcome in mapping
            ]
            coherent = actor_indices == outcome_indices and len(actor_indices) == 1
        by_label[int(row["label"])].append(coherent)

    assert Counter(by_label[0]) == {True: 240}
    assert Counter(by_label[1]) == {True: 240}


def test_structural_audit_rejects_an_incoherent_operational_bundle() -> None:
    corrupted = copy.deepcopy(rows())
    pair = [
        row
        for row in corrupted
        if row["pair_id"] == "v3_train_000"
    ]
    original = OPERATIONAL_OUTCOMES[0]
    replacement = OPERATIONAL_OUTCOMES[1]
    if original not in pair[0]["decisive_fact_texts"][0]:
        original = next(
            outcome
            for outcome in OPERATIONAL_OUTCOMES
            if outcome in pair[0]["decisive_fact_texts"][0]
        )
        replacement = next(outcome for outcome in OPERATIONAL_OUTCOMES if outcome != original)
    for row in pair:
        row["decisive_fact_texts"][0] = row["decisive_fact_texts"][0].replace(
            original, replacement
        )
        row["scenario_text"] = row["scenario_text"].replace(original, replacement)
        row["system_prompt"] = row["system_prompt"].replace(original, replacement)
        row["user_prompt"] = row["user_prompt"].replace(original, replacement)
    with pytest.raises(ValueError, match="operational actor/outcome bundle is incoherent"):
        validate_rows(corrupted)


def test_register_is_expressed_and_metadata_markers_are_not_model_input() -> None:
    materialized = rows()
    for row in materialized:
        rendered = f"{row['system_prompt']}\n{row['user_prompt']}"
        assert row["case_marker"] not in rendered
        assert "formal register" not in rendered.casefold()
        assert "casual register" not in rendered.casefold()
    binary = [row for row in materialized if row["split"] in SPLIT_PAIR_COUNTS]
    formal = " ".join(row["scenario_text"] for row in binary if row["register"] == "formal")
    casual = " ".join(row["scenario_text"] for row in binary if row["register"] == "casual")
    assert "Please consult" in formal or "The routing record" in formal
    assert "Here's" in casual or "Quick" in casual


def test_final_sources_are_external_unique_and_used_once() -> None:
    from src.glm53_user_eval.v11.builder import REPO_ROOT

    assert (REPO_ROOT / FINAL_BINARY_SOURCE).is_file()
    assert (REPO_ROOT / FINAL_NEUTRAL_SOURCE).is_file()
    assert hashlib.sha256((REPO_ROOT / FINAL_BINARY_SOURCE).read_bytes()).hexdigest() == (
        "4ee1e849eaa567c1042290bce7688187c3bd1f1fbd0d1223be58e33709abd346"
    )
    assert hashlib.sha256((REPO_ROOT / FINAL_NEUTRAL_SOURCE).read_bytes()).hexdigest() == (
        "8748508a2a613dd98e154bbcad1b380c6e25d38f45b119409a8d88b310444d8f"
    )
    materialized = rows()
    final_binary = [
        row
        for row in materialized
        if row["split"] == "final_counterfactual" and row["label"] == 1
    ]
    final_neutral = [
        row
        for row in materialized
        if row["split"] == "neutral_controls" and row["control_partition"] == "final"
    ]
    assert len({row["source_record_id"] for row in final_binary}) == 32
    assert len({row["source_record_id"] for row in final_neutral}) == 32
    assert {row["generator_family"] for row in final_binary} == {
        "independent_manual_source"
    }


def test_counterfactual_relation_grammars_are_not_training_grammar() -> None:
    materialized = rows()
    train = [row["scenario_text"] for row in materialized if row["split"] == "train"]
    development = [
        row["scenario_text"]
        for row in materialized
        if row["split"] == "development_counterfactual"
    ]
    final = [
        row["scenario_text"]
        for row in materialized
        if row["split"] == "final_counterfactual"
    ]
    assert all(" denotes " in text and " gets the reply " in text for text in train)
    assert all(" corresponds to " in text and " is bypassed while " in text for text in development)
    assert all(" is keyed by " in text and "Independent key:" in text for text in final)


def test_structural_audit_fails_on_task_leakage() -> None:
    corrupted = copy.deepcopy(rows())
    validation = next(row for row in corrupted if row["split"] == "validation")
    train = next(row for row in corrupted if row["split"] == "train")
    validation["task_id"] = train["task_id"]
    with pytest.raises(ValueError, match="paired field|task registry mismatch|task ID leakage"):
        validate_rows(corrupted)


def test_structural_audit_fails_on_direct_class_statement() -> None:
    corrupted = copy.deepcopy(rows())
    corrupted[0]["scenario_text"] += " This is an evaluation."
    with pytest.raises(ValueError, match="direct class statement"):
        validate_rows(corrupted)


def test_structural_audit_fails_on_changed_content_id() -> None:
    corrupted = copy.deepcopy(rows())
    corrupted[0]["case_marker"] = "changed"
    with pytest.raises(ValueError, match="paired field|content-derived sample ID"):
        validate_rows(corrupted)
