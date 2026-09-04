"""Loading and fail-closed structural validation for contrastive_prompts_v3."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .builder import canonical_json, sha256_text
from .spec import (
    ASSESSMENT_MODES,
    GENERATOR_FAMILIES,
    OPERATIONAL_ACTORS,
    OPERATIONAL_OUTCOMES,
    PROMPT_ROLES,
    SPLIT_PAIR_COUNTS,
    TASKS,
)

BINARY_SPLITS = tuple(SPLIT_PAIR_COUNTS)
EXPECTED_SPLIT_ROWS = {
    **{split: count * 2 for split, count in SPLIT_PAIR_COUNTS.items()},
    "neutral_controls": 64,
    "factorial_calibration": 32,
}
NUISANCE_KEYS = (
    "formal_register",
    "ai_vocabulary",
    "benchmark_vocabulary",
    "production_vocabulary",
    "logging",
    "human_review",
)
BANNED_DIRECT_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bthis is an evaluation\b",
        r"\bthis is a deployment\b",
        r"\byou are being evaluated\b",
        r"\bevaluation item\b",
        r"\bdeployment item\b",
        r"\blive user\b",
        r"\bbenchmark item\b",
        r"\bevaluation harness\b",
        r"\bdeployed service\b",
    )
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row {line_number} is not an object")
            rows.append(row)
    return rows


def rendered_text(row: dict[str, Any]) -> str:
    return f"<system>\n{row['system_prompt']}\n<user>\n{row['user_prompt']}"


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("split") for row in rows)
    _require(len(rows) == 576, f"expected 576 rows, observed {len(rows)}")
    _require(dict(counts) == EXPECTED_SPLIT_ROWS, f"unexpected split counts: {dict(counts)}")
    sample_ids = [row.get("sample_id") for row in rows]
    _require(None not in sample_ids, "sample_id is required")
    _require(len(sample_ids) == len(set(sample_ids)), "sample IDs are not unique")
    return dict(counts)


def _assert_binary_pairs(rows: list[dict[str, Any]]) -> None:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] in BINARY_SPLITS:
            pairs[row["pair_id"]].append(row)
    _require(len(pairs) == sum(SPLIT_PAIR_COUNTS.values()), "unexpected binary pair count")
    all_aliases: set[str] = set()
    final_source_ids: set[str] = set()
    for pair_id, pair_rows in pairs.items():
        _require(len(pair_rows) == 2, f"{pair_id} does not have two rows")
        by_label = {row["label"]: row for row in pair_rows}
        _require(set(by_label) == {0, 1}, f"{pair_id} is not one eval/deploy pair")
        deploy, evaluate = by_label[0], by_label[1]
        for key in (
            "split",
            "generator_family",
            "task_id",
            "task_domain",
            "prompt_role",
            "register",
            "variant_order",
            "case_marker",
            "source_record_id",
            "shared_suffix",
            "shared_suffix_sha256",
            "nuisance",
            "opaque_aliases",
            "role_alias",
            "mapping_order",
            "assessment_design",
            "control_partition",
        ):
            _require(deploy[key] == evaluate[key], f"{pair_id} differs on paired field {key}")
        _require(deploy["latent_class"] == "deploy", f"{pair_id} label 0 is not deploy")
        _require(evaluate["latent_class"] == "eval", f"{pair_id} label 1 is not eval")
        _require(deploy["factors"]["operational_consequence"], f"{pair_id} deploy is not operational")
        _require(not deploy["factors"]["primary_model_assessment"], f"{pair_id} deploy is assessment")
        _require(not evaluate["factors"]["operational_consequence"], f"{pair_id} eval is operational")
        _require(evaluate["factors"]["primary_model_assessment"], f"{pair_id} eval is not assessment")
        _require(not evaluate["factors"]["real_external_recipient"], f"{pair_id} eval has external recipient")
        mode = str(evaluate["assessment_design"])
        _require(mode in ASSESSMENT_MODES, f"{pair_id} has an unknown assessment design")
        mode_contract = ASSESSMENT_MODES[mode]
        mapping = str(deploy["decisive_fact_texts"][0])
        operational_actor_indices = [
            index
            for index, (actor, _) in enumerate(OPERATIONAL_ACTORS)
            if actor in mapping
        ]
        operational_outcome_indices = [
            index
            for index, outcome in enumerate(OPERATIONAL_OUTCOMES)
            if outcome in mapping
        ]
        _require(
            len(operational_actor_indices) == 1
            and len(operational_outcome_indices) == 1
            and operational_actor_indices[0] == operational_outcome_indices[0],
            f"{pair_id} operational actor/outcome bundle is incoherent",
        )
        _require(
            str(mode_contract["actor"]) in mapping
            and str(mode_contract["outcome"]) in mapping,
            f"{pair_id} assessment actor/outcome bundle is incoherent",
        )
        _require(
            evaluate["factors"]["assessment_mode"] == mode
            and evaluate["factors"]["model_scoring"] == mode_contract["model_scoring"]
            and evaluate["factors"]["comparison_among_models"]
            == mode_contract["comparison_among_models"],
            f"{pair_id} rendered assessment design and eval factors differ",
        )
        _require(
            deploy["factors"]["assessment_mode"] == "none"
            and not deploy["factors"]["model_scoring"]
            and not deploy["factors"]["comparison_among_models"],
            f"{pair_id} deployment row activates model assessment",
        )
        aliases = [str(alias) for alias in deploy["opaque_aliases"]]
        _require(len(aliases) == 4 and len(set(aliases)) == 4, f"{pair_id} aliases are not unique")
        _require(not all_aliases.intersection(aliases), f"{pair_id} reuses aliases from another pair")
        all_aliases.update(aliases)
        _require(
            Counter(re.findall(r"\w+|[^\w\s]", rendered_text(deploy).casefold()))
            == Counter(re.findall(r"\w+|[^\w\s]", rendered_text(evaluate).casefold())),
            f"{pair_id} pair members do not share a lexical inventory",
        )
        if deploy["split"] == "final_counterfactual":
            source_id = str(deploy["source_record_id"])
            _require(source_id not in final_source_ids, f"duplicate final source record {source_id}")
            final_source_ids.add(source_id)
        for row in pair_rows:
            _require(
                row["shared_suffix_sha256"] == sha256_text(row["shared_suffix"]),
                f"{row['sample_id']} has a bad suffix hash",
            )
            _require(
                row["shared_suffix"] in row["user_prompt"],
                f"{row['sample_id']} lacks the shared suffix in the user message",
            )
            for fact in row["decisive_fact_texts"]:
                _require(
                    row["scenario_text"].count(fact) == 1,
                    f"{row['sample_id']} decisive fact is absent or ambiguous",
                )
            _require(
                str(row["case_marker"]) not in rendered_text(row),
                f"{row['sample_id']} exposes its metadata marker in model input",
            )
    _require(len(final_source_ids) == 32, "final binary source does not contain 32 unique records")


def _assert_split_isolation(rows: list[dict[str, Any]]) -> None:
    generators: dict[str, set[str]] = {}
    task_ids: dict[str, set[str]] = {}
    task_texts: dict[str, set[str]] = {}
    for split in BINARY_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        generators[split] = {row["generator_family"] for row in split_rows}
        task_ids[split] = {row["task_id"] for row in split_rows}
        task_texts[split] = {normalized_text(row["shared_suffix"]) for row in split_rows}
        _require(
            generators[split] == set(GENERATOR_FAMILIES[split]),
            f"{split} generator registry mismatch",
        )
        registered_tasks = {task_id for task_id, _, _ in TASKS[split]}
        _require(task_ids[split] == registered_tasks, f"{split} task registry mismatch")
    for index, left in enumerate(BINARY_SPLITS):
        for right in BINARY_SPLITS[index + 1 :]:
            _require(not generators[left] & generators[right], f"generator leakage: {left}/{right}")
            _require(not task_ids[left] & task_ids[right], f"task ID leakage: {left}/{right}")
            _require(not task_texts[left] & task_texts[right], f"task text leakage: {left}/{right}")


def _assert_counterbalance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for split in BINARY_SPLITS:
        families = sorted(GENERATOR_FAMILIES[split])
        reports[split] = {}
        for family in families:
            pair_rows = [
                row
                for row in rows
                if row["split"] == split
                and row["generator_family"] == family
                and row["label"] == 1
            ]
            pair_count = len(pair_rows)
            _require(pair_count % 8 == 0, f"{split}/{family} is not an eight-row OA block")
            factor_counts = {
                key: sum(bool(row["nuisance"][key]) for row in pair_rows)
                for key in NUISANCE_KEYS
            }
            _require(
                all(value == pair_count // 2 for value in factor_counts.values()),
                f"{split}/{family} nuisance marginals are not balanced",
            )
            for left_index, left in enumerate(NUISANCE_KEYS):
                for right in NUISANCE_KEYS[left_index + 1 :]:
                    cells = Counter(
                        (bool(row["nuisance"][left]), bool(row["nuisance"][right]))
                        for row in pair_rows
                    )
                    _require(
                        set(cells) == {(False, False), (False, True), (True, False), (True, True)}
                        and set(cells.values()) == {pair_count // 4},
                        f"{split}/{family} lacks pairwise balance for {left}/{right}",
                    )
            orders = Counter(row["variant_order"] for row in pair_rows)
            _require(
                orders == {"deploy_first": pair_count // 2, "eval_first": pair_count // 2},
                f"{split}/{family} variant order is not balanced",
            )
            _require(
                all(
                    row["register"]
                    == ("formal" if row["nuisance"]["formal_register"] else "casual")
                    for row in pair_rows
                ),
                f"{split}/{family} register disagrees with its nuisance ledger",
            )
            for field in ("prompt_role", "assessment_design"):
                for value in {row[field] for row in pair_rows}:
                    subset = [row for row in pair_rows if row[field] == value]
                    for key in NUISANCE_KEYS:
                        _require(
                            {bool(row["nuisance"][key]) for row in subset} == {False, True},
                            f"{split}/{family} {field}={value} fixes nuisance {key}",
                        )
            reports[split][family] = {
                "pair_count": pair_count,
                "factor_true_counts": factor_counts,
                "variant_order_counts": dict(orders),
            }
    return reports


def _assert_text_integrity(rows: list[dict[str, Any]]) -> None:
    full_prompts = [normalized_text(rendered_text(row)) for row in rows]
    _require(len(full_prompts) == len(set(full_prompts)), "normalized full prompts are not unique")
    for row in rows:
        _require(row["schema_version"] == "contrastive_prompts_v3_row_v1", "row schema mismatch")
        _require(row["dataset_id"] == "contrastive_prompts_v3", "dataset ID mismatch")
        scenario = row["scenario_text"]
        _require(
            "formal register" not in scenario.casefold()
            and "casual register" not in scenario.casefold(),
            f"register is declared rather than expressed in {row['sample_id']}",
        )
        for pattern in BANNED_DIRECT_PATTERNS:
            _require(not pattern.search(scenario), f"direct class statement in {row['sample_id']}")
        reconstructed = dict(row)
        sample_id = reconstructed.pop("sample_id")
        _require(
            sample_id == sha256_text(canonical_json(reconstructed))[:20],
            f"content-derived sample ID mismatch for {sample_id}",
        )


def _assert_calibration(rows: list[dict[str, Any]]) -> None:
    calibration = [row for row in rows if row["split"] == "factorial_calibration"]
    factor_names = (
        "real_external_recipient",
        "operational_consequence",
        "model_scoring",
        "comparison_among_models",
    )
    cells = Counter(
        (
            *(row["factors"][name] for name in factor_names),
        )
        for row in calibration
    )
    _require(len(cells) == 16 and set(cells.values()) == {2}, "factorial calibration is incomplete")
    for factor_cell in cells:
        cell_rows = [
            row
            for row in calibration
            if (
                row["factors"]["real_external_recipient"],
                row["factors"]["operational_consequence"],
                row["factors"]["model_scoring"],
                row["factors"]["comparison_among_models"],
            )
            == factor_cell
        ]
        _require(
            {row["register"] for row in cell_rows} == {"formal", "casual"},
            f"factorial cell {factor_cell} is not counterrotated over register",
        )
        _require(
            len({row["prompt_role"] for row in cell_rows}) == 2
            and len({row["task_id"] for row in cell_rows}) == 2,
            f"factorial cell {factor_cell} is not counterrotated over role/task",
        )
        recipient, consequence, scoring, comparison = factor_cell
        if not scoring and not comparison and (recipient or consequence):
            expected_labels = ["deploy"]
        elif not recipient and not consequence and (scoring or comparison):
            expected_labels = ["eval"]
        elif not any(factor_cell):
            expected_labels = ["no_status", "ambiguous"]
        else:
            expected_labels = ["ambiguous"]
        _require(
            all(row["acceptable_judge_labels"] == expected_labels for row in cell_rows),
            f"factorial cell {factor_cell} has an invalid semantic label contract",
        )
    expected_roles = Counter({role: 4 for role in PROMPT_ROLES})
    expected_tasks = Counter({task[0]: 4 for task in TASKS["neutral"]})
    expected_role_task = Counter(
        {(role, task[0]): 1 for role in PROMPT_ROLES for task in TASKS["neutral"]}
    )
    for factor_name in factor_names:
        for factor_value in (False, True):
            subset = [
                row
                for row in calibration
                if row["factors"][factor_name] is factor_value
            ]
            _require(
                Counter(row["prompt_role"] for row in subset) == expected_roles,
                f"factorial {factor_name}={factor_value} is confounded with prompt role",
            )
            _require(
                Counter(row["task_id"] for row in subset) == expected_tasks,
                f"factorial {factor_name}={factor_value} is confounded with task",
            )
            _require(
                Counter((row["prompt_role"], row["task_id"]) for row in subset)
                == expected_role_task,
                f"factorial {factor_name}={factor_value} is confounded with role/task pairs",
            )
    for register in ("formal", "casual"):
        subset = [row for row in calibration if row["register"] == register]
        _require(
            Counter((row["prompt_role"], row["task_id"]) for row in subset)
            == expected_role_task,
            f"factorial register {register} is confounded with role/task pairs",
        )
    neutral = [row for row in rows if row["split"] == "neutral_controls"]
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in neutral:
        by_family[row["generator_family"]][row["control_partition"]] += 1
    _require(
        all(counts == {"development": 8, "final": 8} for counts in by_family.values()),
        "neutral-control development/final partitions are not balanced",
    )
    for family in by_family:
        development = {
            normalized_text(row["scenario_text"])
            for row in neutral
            if row["generator_family"] == family and row["control_partition"] == "development"
        }
        final = {
            normalized_text(row["scenario_text"])
            for row in neutral
            if row["generator_family"] == family and row["control_partition"] == "final"
        }
        _require(len(development) == 8, f"{family} development controls repeat")
        _require(len(final) == 8, f"{family} final controls repeat")
        _require(not development & final, f"{family} final controls reuse development text")


def validate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    split_counts = _assert_counts(materialized)
    _assert_binary_pairs(materialized)
    _assert_split_isolation(materialized)
    counterbalance = _assert_counterbalance(materialized)
    _assert_text_integrity(materialized)
    _assert_calibration(materialized)
    return {
        "schema_version": "contrastive_prompts_v3_structural_audit_v1",
        "passed": True,
        "row_count": len(materialized),
        "split_counts": split_counts,
        "counterbalance": counterbalance,
    }
