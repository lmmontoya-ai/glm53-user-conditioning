from pathlib import Path

from src.glm53_user_eval.analysis import (
    analyze_g1_behavior,
    analyze_g3_local_behavior,
    manual_review_complete,
)


def _synthetic_rows():
    schedule = []
    results = []
    selection = {"pairs": []}
    for pair_index in range(16):
        role = "enriched_target" if pair_index < 8 else "prospective_generality"
        selection["pairs"].append(
            {
                "twin_index": pair_index,
                "selection_role": role,
            }
        )
        for scenario in ("s0", "s1"):
            for condition, confidence in (
                ("famous_coherent", 70.0),
                ("unknown_same_org", 75.0),
                ("unknown_general", 76.0),
            ):
                sample_id = f"p{pair_index}-{scenario}-{condition}"
                schedule.append(
                    {
                        "sample_id": sample_id,
                        "pair_index": pair_index,
                        "condition": condition,
                        "scenario_id": scenario,
                    }
                )
                results.append(
                    {
                        "sample_id": sample_id,
                        "condition": condition,
                        "scenario_id": scenario,
                        "confidence_p": confidence,
                        "parse_valid": True,
                        "realized_reasoning_tokens": 10,
                        "provider_metadata": {
                            "provider": "zai_first_party",
                            "requested_model": "glm-5.3-flash",
                            "main_response_model": "glm-5.3-flash",
                            "followup_response_model": "glm-5.3-flash",
                        },
                    }
                )
    for identity, condition, confidence in (
        ("f0", "famous_nonai_control", 75.5),
        ("g0", "genpop_control", 75.0),
    ):
        for scenario in ("s0", "s1"):
            sample_id = f"{identity}-{scenario}"
            schedule.append(
                {
                    "sample_id": sample_id,
                    "pair_index": None,
                    "condition": condition,
                    "scenario_id": scenario,
                }
            )
            results.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "scenario_id": scenario,
                    "confidence_p": confidence,
                    "parse_valid": True,
                    "realized_reasoning_tokens": 10,
                    "provider_metadata": {
                        "provider": "zai_first_party",
                        "requested_model": "glm-5.3-flash",
                        "main_response_model": "glm-5.3-flash",
                        "followup_response_model": "glm-5.3-flash",
                    },
                }
            )
    return results, schedule, selection


def test_g1_analysis_recovers_matched_effect() -> None:
    results, schedule, selection = _synthetic_rows()
    estimates, checks = analyze_g1_behavior(
        results,
        schedule,
        selection,
        bootstrap_reps=100,
        seed=1,
        reading_log=None,
    )
    assert estimates["enriched_name_effect_pp"] == -5.0
    assert estimates["enriched_negative_pair_count"] == 8
    assert checks["enriched_effect_negative_90ci"]
    assert checks["fame_control_smaller"]
    assert not checks["manual_review_complete"]


def test_manual_review_requires_40_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "reading_log.csv"
    path.write_text(
        "sample_id,reviewed\n" + "".join(f"s{i},true\n" for i in range(40)),
        encoding="utf-8",
    )
    assert manual_review_complete(path)


def test_g3_local_analysis_recovers_frozen_effect(tmp_path: Path) -> None:
    schedule = []
    results = []
    selection = {"pairs": []}
    for pair_index in range(4):
        selection["pairs"].append(
            {
                "twin_index": pair_index,
                "primary_intervention": True,
            }
        )
    for task_index in range(50):
        for pair_index in range(4):
            for condition, confidence in (
                ("famous_coherent", 70.0),
                ("unknown_same_org", 75.0),
                ("unknown_general", 76.0),
            ):
                sample_id = f"t{task_index}-p{pair_index}-{condition}"
                schedule.append(
                    {
                        "sample_id": sample_id,
                        "scenario_id": f"t{task_index}",
                        "pair_index": pair_index,
                        "condition": condition,
                        "analysis_block": task_index % 5,
                    }
                )
                results.append(
                    {
                        "sample_id": sample_id,
                        "scenario_id": f"t{task_index}",
                        "condition": condition,
                        "confidence_p": confidence,
                        "parse_valid": True,
                        "provider_metadata": {
                            "provider": "local_official_fp8",
                            "model_revision": "b" * 40,
                            "runtime_hash": "a" * 64,
                        },
                    }
                )
    reading_log = tmp_path / "reading_log.csv"
    reading_log.write_text(
        "sample_id,reviewed\n" + "".join(f"s{i},true\n" for i in range(20)),
        encoding="utf-8",
    )
    estimates, checks = analyze_g3_local_behavior(
        results,
        schedule,
        selection,
        bootstrap_reps=100,
        seed=1,
        reading_log=reading_log,
    )
    assert estimates["primary_name_effect_pp"] == -5.0
    assert estimates["primary_affiliation_effect_pp"] == -1.0
    assert estimates["primary_negative_pair_count"] == 4
    assert all(checks.values())
