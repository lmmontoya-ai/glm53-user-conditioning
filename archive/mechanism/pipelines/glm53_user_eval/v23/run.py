from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v23.analysis import (
    annotation_analysis,
    build_human_packet,
    category_analysis,
    category_effects,
    deterministic_analysis,
)
from src.glm53_user_eval.v23.artifacts import atomic_json, atomic_text, sha256_file
from src.glm53_user_eval.v23.contract import (
    require_preregistered_tag,
    validate_prereg,
    write_validation,
)
from src.glm53_user_eval.v23.judges import (
    build_category_packet,
    judge_preflight,
    run_judges,
)
from src.glm53_user_eval.v23.transcripts import (
    build_annotation_packet,
    build_development_packet,
    extract_run,
)
from src.glm53_user_eval.v23.verification import verify

PIPE = ROOT / "pipelines/glm53_user_eval/v23"
CONFIG = PIPE / "configs"
STATUS = PIPE / "status"
ART = ROOT / "artifacts/glm53_user_eval/v23"
DERIVED = ART / "derived"
REPORTS = ART / "reports"
JUDGMENTS = ART / "judgments"
PREREG = CONFIG / "prereg_v23_behavioral_decomposition.yaml"
ROSTER = ROOT / "artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json"
V6_RAW = ROOT / "artifacts/glm53_user_eval/reports/transluce_exact_v6/raw_scores.jsonl"
V7_RAW = ROOT / "artifacts/glm53_user_eval/reports/transluce_interaction_v7/raw_scores.jsonl"
V6_RUN = ROOT / "artifacts/glm53_user_eval/behavior_api/transluce_exact_v6/eval_logs"
V7_RUN = ROOT / "artifacts/glm53_user_eval/runs/transluce_interaction_v7/eval_logs"


def _locked() -> None:
    validate_prereg(ROOT, PREREG)
    require_preregistered_tag(ROOT, PREREG)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def build_evidence() -> dict[str, object]:
    required = {
        "deterministic": REPORTS / "deterministic_analysis.json",
        "annotations": REPORTS / "annotation_analysis.json",
        "categories": REPORTS / "category_analysis.json",
        "category_effects": REPORTS / "category_effects.json",
        "verification": REPORTS / "independent_verification.json",
        "human_packet": REPORTS / "human_audit_packet.manifest.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    deterministic = json.loads(required["deterministic"].read_text(encoding="utf-8"))
    annotations = json.loads(required["annotations"].read_text(encoding="utf-8"))
    categories = json.loads(required["categories"].read_text(encoding="utf-8"))
    category_effects = json.loads(required["category_effects"].read_text(encoding="utf-8"))
    verification = json.loads(required["verification"].read_text(encoding="utf-8"))
    choice = deterministic["first_turn_choice_interaction_pp"]
    adjusted = deterministic["choice_standardized_folded_confidence"]
    eval_metric = annotations["dimensions"]["explicit_evaluation_or_scrutiny_inference"]
    composite = annotations["caution_deference_composite"]
    deference = annotations["dimensions"]["deference_to_user_expertise"]
    if deterministic["choice_equivalent_to_zero"] and adjusted["ci95_pp"][1] < 0:
        behavioral_state = "confidence_shift_persists_and_choice_effect_is_practically_equivalent_to_zero"
    elif adjusted["ci95_pp"][1] < 0 and choice["ci95"][0] <= 0 <= choice["ci95"][1]:
        behavioral_state = "confidence_shift_persists_choice_effect_not_detected_but_not_excluded"
    elif choice["ci95"][1] < 0 or choice["ci95"][0] > 0:
        behavioral_state = "choice_and_confidence_may_both_shift"
    else:
        behavioral_state = "choice_decomposition_unresolved"
    content_state = "human_audit_pending"
    if not eval_metric["combined_interpretable"] and not composite["combined_interpretable"]:
        content_state = "judge_agreement_inadequate_and_human_audit_pending"
    elif (
        eval_metric["combined_interpretable"]
        and deference["combined_interpretable"]
        and eval_metric["combined"]["ci95"][0] <= 0 <= eval_metric["combined"]["ci95"][1]
        and deference["combined"]["ci95"][0] <= 0 <= deference["combined"]["ci95"][1]
    ):
        content_state = "no_detected_scrutiny_or_deference_interaction_human_audit_pending"
    decision = {
        "schema_version": "glm53_v23_final_decision_v1",
        "decision": "offline_behavioral_decomposition_complete_human_audit_pending",
        "behavioral_state": behavioral_state,
        "content_state": content_state,
        "new_glm_calls": 0,
        "runpod_spend_usd": 0,
        "human_review_complete": False,
        "unqualified_positive_transcript_claim_authorized": False,
        "independent_verification_passed": verification["passed"],
        "artifact_hashes": {name: sha256_file(path) for name, path in required.items()},
    }
    atomic_json(REPORTS / "decision.json", decision)
    atomic_json(STATUS / "FINAL_DECISION.json", decision)
    length = deterministic["reasoning_and_response_length_interactions"]
    luna_categories = category_effects["judge_specific_results"]["luna_max"]
    report = f"""# GLM-5.3 V23 — Decision versus self-calibration

V23 completed the offline decomposition without new GLM calls or RunPod spend.

## Deterministic results

- First-turn choice interaction: **{choice["interaction"]:.3f} percentage points**, 95% interval **[{choice["ci95"][0]:.3f}, {choice["ci95"][1]:.3f}]**.
- Original folded-confidence interaction: **{deterministic["folded_confidence_interaction_pp"]["interaction"]:.3f} pp**, 95% interval **[{deterministic["folded_confidence_interaction_pp"]["ci95"][0]:.3f}, {deterministic["folded_confidence_interaction_pp"]["ci95"][1]:.3f}]**.
- Choice-standardized folded-confidence interaction: **{adjusted["interaction_pp"]:.3f} pp**, 95% interval **[{adjusted["ci95_pp"][0]:.3f}, {adjusted["ci95_pp"][1]:.3f}]**.

Machine interpretation: `{behavioral_state}`.

The first-turn choice estimate is close to zero, but its interval is too wide to establish practical equivalence under the frozen ±1 percentage-point margin. The confidence interaction remains negative after standardizing for first-turn choice, so the data support an identity-conditioned self-calibration effect more directly than a decision effect; they do not rule out a smaller decision effect.

## Blinded AI annotations

- Explicit evaluation/scrutiny interaction: **{eval_metric["combined"]["interaction"]:.3f}**, 95% interval **[{eval_metric["combined"]["ci95"][0]:.3f}, {eval_metric["combined"]["ci95"][1]:.3f}]**; inter-judge kappa **{annotations["agreement"]["explicit_evaluation_or_scrutiny_inference"]["kappa"]:.3f}**.
- Deference-to-expertise interaction: **{deference["combined"]["interaction"]:.3f}**, 95% interval **[{deference["combined"]["ci95"][0]:.3f}, {deference["combined"]["ci95"][1]:.3f}]**; inter-judge kappa **{annotations["agreement"]["deference_to_user_expertise"]["kappa"]:.3f}**.
- The caution/deference composite is exploratory because V6 development showed unreliable caution thresholds. Its interaction is **{composite["interaction"]:.3f}**, and combined interpretation is allowed only if both dimensions pass the frozen agreement rule: **{composite["combined_interpretable"]}**.

Neither primary transcript-content interaction was resolved after Holm correction. Deference was rare: its raw weighted kappa was **{annotations["agreement"]["deference_to_user_expertise"]["kappa"]:.3f}**, and it met the frozen agreement rule only through **{annotations["agreement"]["deference_to_user_expertise"]["exact_agreement"]:.1%}** exact agreement. Two exploratory deterministic measures did differ: the confidence-turn visible response was **{length["confidence_turn_visible_tokens"]["interaction"]:.1f} tokens** longer in the four-group interaction (95% interval **[{length["confidence_turn_visible_tokens"]["ci95"][0]:.1f}, {length["confidence_turn_visible_tokens"]["ci95"][1]:.1f}]**), and had **{length["confidence_visible_sentence_count"]["interaction"]:.2f}** more sentences (95% interval **[{length["confidence_visible_sentence_count"]["ci95"][0]:.2f}, {length["confidence_visible_sentence_count"]["ci95"][1]:.2f}]**).

These are AI judgments from independent Luna-max and Terra-high runs with fast mode disabled. A 160-row audit packet has been prepared, but Luis has not yet reviewed it. Accordingly, V23 does not authorize an unqualified positive transcript-content claim.

## Famous-non-AI control audit

The two judges agreed on **{categories["agreement_rate"]:.1%}** of 70 identity categories. The roster is overwhelmingly athletes (**{categories["consensus_counts"].get("athlete", 0)}**) and entertainers (**{categories["consensus_counts"].get("entertainer", 0)}**), with one unresolved category disagreement. The folded-confidence difference from the dilemma-specific general-population center was positive for both large categories: **{luna_categories["athlete"]["mean_delta_from_genpop_pp"]:.3f} pp** for athletes and **{luna_categories["entertainer"]["mean_delta_from_genpop_pp"]:.3f} pp** for entertainers. This exploratory audit suggests that the Famous-non-AI component was not carried by only one of those two large categories, but the roster cannot answer whether technical fame is the relevant comparison.

## Scope

V7 is a locked held-out analysis for this decomposition, not a pristine untouched dataset, because small subsets were manually audited during earlier work. Famous-non-AI occupational analyses are exploratory. V23 makes no activation-mechanism or causal-mediation claim.
"""
    atomic_text(STATUS / "FINAL_REPORT.md", report)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-prereg")
    sub.add_parser("build-v6-transcripts")
    sub.add_parser("build-v6-development-packet")
    dev_judge = sub.add_parser("judge-v6-development")
    dev_judge.add_argument("--max-new-per-judge", type=int)
    sub.add_parser("judge-preflight")
    sub.add_parser("build-v7-transcripts")
    sub.add_parser("build-v7-annotation-packet")
    judge = sub.add_parser("judge-v7-content")
    judge.add_argument("--max-new-per-judge", type=int)
    sub.add_parser("analyze-deterministic")
    sub.add_parser("analyze-annotations")
    category = sub.add_parser("judge-fn-categories")
    category.add_argument("--max-new-per-judge", type=int)
    sub.add_parser("analyze-categories")
    sub.add_parser("analyze-category-effects")
    sub.add_parser("build-human-packet")
    sub.add_parser("verify-independent")
    sub.add_parser("build-evidence")
    args = parser.parse_args()

    if args.command == "validate-prereg":
        result = write_validation(ROOT, PREREG, STATUS / "prereg_validation.json")
    elif args.command == "build-v6-transcripts":
        result = extract_run(
            run_root=V6_RUN,
            raw_scores_path=V6_RAW,
            roster_path=ROSTER,
            output_path=DERIVED / "v6_transcripts.jsonl",
            development_rows_per_group=22,
            development_salt="glm53-v23-v6-development-v1",
        )
    elif args.command == "build-v6-development-packet":
        result = build_development_packet(
            transcript_path=DERIVED / "v6_transcripts.jsonl",
            output_path=DERIVED / "v6_development_packet.jsonl",
            rows_per_group=20,
            salt="glm53-v23-v6-development-v1",
        )
    elif args.command == "judge-v6-development":
        result = {
            "records": len(
                run_judges(
                    packet_path=DERIVED / "v6_development_packet.jsonl",
                    output_root=JUDGMENTS / "v6_development_v2",
                    schema_path=CONFIG / "content_judgment.schema.json",
                    task="content",
                    max_new_per_judge=args.max_new_per_judge,
                )
            )
        }
    elif args.command == "judge-preflight":
        result = judge_preflight(STATUS / "judge_preflight.json")
    else:
        _locked()
        if args.command == "build-v7-transcripts":
            result = extract_run(
                run_root=V7_RUN,
                raw_scores_path=V7_RAW,
                roster_path=ROSTER,
                output_path=DERIVED / "v7_transcripts.jsonl",
            )
        elif args.command == "build-v7-annotation-packet":
            result = build_annotation_packet(
                transcript_path=DERIVED / "v7_transcripts.jsonl",
                output_path=DERIVED / "v7_annotation_packet.jsonl",
                matched_cells=500,
                salt="glm53-v23-annotation-cells-v1",
            )
        elif args.command == "judge-v7-content":
            result = {
                "records": len(
                    run_judges(
                        packet_path=DERIVED / "v7_annotation_packet.jsonl",
                        output_root=JUDGMENTS / "v7_content",
                        schema_path=CONFIG / "content_judgment.schema.json",
                        task="content",
                        max_new_per_judge=args.max_new_per_judge,
                    )
                )
            }
        elif args.command == "analyze-deterministic":
            result = deterministic_analysis(
                DERIVED / "v7_transcripts.jsonl",
                REPORTS / "deterministic_analysis.json",
                reps=20000,
                seed=20260923,
            )
        elif args.command == "analyze-annotations":
            result = annotation_analysis(
                transcript_path=DERIVED / "v7_transcripts.jsonl",
                judgment_root=JUDGMENTS / "v7_content",
                deterministic_path=REPORTS / "deterministic_analysis.json",
                output_path=REPORTS / "annotation_analysis.json",
                reps=20000,
                seed=20260923 + 1000,
            )
        elif args.command == "judge-fn-categories":
            packet = DERIVED / "famous_nonai_category_packet.jsonl"
            build_category_packet(ROSTER, packet)
            result = {
                "records": len(
                    run_judges(
                        packet_path=packet,
                        output_root=JUDGMENTS / "categories",
                        schema_path=CONFIG / "category_judgment.schema.json",
                        task="category",
                        max_new_per_judge=args.max_new_per_judge,
                    )
                )
            }
        elif args.command == "analyze-categories":
            result = category_analysis(
                judgment_root=JUDGMENTS / "categories",
                transcript_path=DERIVED / "v7_transcripts.jsonl",
                output_path=REPORTS / "category_analysis.json",
            )
        elif args.command == "analyze-category-effects":
            result = category_effects(
                judgment_root=JUDGMENTS / "categories",
                transcript_path=DERIVED / "v7_transcripts.jsonl",
                output_path=REPORTS / "category_effects.json",
            )
        elif args.command == "build-human-packet":
            result = build_human_packet(
                transcript_path=DERIVED / "v7_transcripts.jsonl",
                judgment_root=JUDGMENTS / "v7_content",
                output_path=REPORTS / "human_audit_packet.jsonl",
                seed_salt="glm53-v23-human-audit-v1",
            )
        elif args.command == "verify-independent":
            result = verify(
                transcript_path=DERIVED / "v7_transcripts.jsonl",
                deterministic_path=REPORTS / "deterministic_analysis.json",
                annotation_path=REPORTS / "annotation_analysis.json",
                output_path=REPORTS / "independent_verification.json",
            )
        elif args.command == "build-evidence":
            result = build_evidence()
        else:
            raise AssertionError(args.command)
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
