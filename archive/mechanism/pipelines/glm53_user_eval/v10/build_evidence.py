"""Build compact terminal evidence for the completed v10 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    fixed = analysis["fixed_split_trainer_comparison"]

    display = [
        ("V9 AdamW", "v9_adamw"),
        ("Historical full-batch", "historical_full_batch"),
        ("Paired mean", "paired_mean"),
        ("Ridge logistic", "ridge_logistic"),
        ("Shrinkage LDA", "shrinkage_lda"),
        ("PCA + ridge", "pca_ridge"),
    ]
    table = []
    for label, key in display:
        row = fixed[key]
        table.append(
            f"| {label} | {row['selected_layer']} | {row['test']['auroc']:.3f} | "
            f"{row['holdout_realism']['auroc']:.3f} | {row['holdout_hardneg']['auroc']:.3f} | "
            f"{row['holdout_hardneg']['fpr_at_80_tpr']:.3f} | "
            f"{row['holdout_hardneg']['neutral_false_positive_rate']:.3f} |"
        )
    lofo = analysis["leave_family_out"]
    blocked = analysis["family_and_task_blocked"]
    stability = analysis["direction_stability"]
    hard = analysis["hard_negative_decomposition"]["v9_adamw"]
    report = f"""# GLM-5.3 v10 offline construct audit

## Decision

`{decision["decision"]}`

The v9 code successfully learned label-associated information, but the preserved
source bank is not a valid instrument for a robust latent evaluation/deployment
state. The failure is not explained by the v9 AdamW trainer or by simple saved
length, role, and mask metadata. No new model forward or paid compute was used.

## Fixed-split trainer comparison

| Method | Layer | Core test AUROC | Realism AUROC | Hard-negative AUROC | Hard FPR@80% TPR | Neutral FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

The preregistered hard-negative gate required FPR@80% TPR at or below 0.30.
Every method failed. The historical trainer reached {fixed["historical_full_batch"]["holdout_hardneg"]["fpr_at_80_tpr"]:.3f},
worse than v9 AdamW at {fixed["v9_adamw"]["holdout_hardneg"]["fpr_at_80_tpr"]:.3f}.

## Distribution and leakage diagnostics

- Leave-one-family-out paired-mean transfer: median AUROC {lofo["median_auroc"]:.3f}, minimum {lofo["minimum_auroc"]:.3f} across {lofo["fold_count"]} ordered train/validation/test configurations.
- Family-and-task-blocked transfer: median AUROC {blocked["median_auroc"]:.3f}, minimum {blocked["minimum_auroc"]:.3f}; {100 * blocked["fraction_perfect"]:.1f}% of the 56 tiny held-out folds were perfect.
- Structural metadata baseline: test AUROC {analysis["metadata_baseline"]["test"]["auroc"]:.3f}. Saved prompt length, retained and masked token counts, cue count, prompt role, cue location, and within-pair length gap do not explain the activation result.
- V9 hard-negative decomposition: benchmark-language deployment AUROC {hard["benchmark_lexicon_deploy"]["auroc"]:.3f}; production-language evaluation AUROC {hard["production_lexicon_eval"]["auroc"]:.3f}. The metadata-header role was weakest at AUROC {hard["by_prompt_role_binary"]["metadata_header"]["auroc"]:.3f}.
- Neutral controls: formal/casual FPR {hard["formal_casual_neutral"]["false_positive_rate"]:.3f}; AI/human FPR {hard["ai_human_neutral"]["false_positive_rate"]:.3f}. Generic AI wording was not the main neutral failure in this run.
- Pair-bootstrap stability: median direction cosine to the full paired-mean model {stability["cosine_to_full"]["median"]:.3f}, but minimum {stability["cosine_to_full"]["minimum"]:.3f}. Test AUROC ranged down to {stability["test_auroc"]["minimum"]:.3f}. The direction has an unstable tail even though its median stability clears 0.80.

## What this establishes

The ordinary core task remains easy after holding out framing families and task
identities. That makes classic memorization of training rows an implausible main
explanation. Yet changing the optimizer or using far stronger regularization
does not produce counterfactual robustness. The best-supported diagnosis is
construct overfit: the bank defines a broadly propagated explicit-framing axis
that transfers within its construction grammar but does not reliably track the
semantically decisive eval/deploy relation.

This diagnosis is still an inference. A direct cue-deletion, neutral-replacement,
and cue-swap experiment would measure propagated cue dependence, but it requires
new GLM forwards and was intentionally not run here.

## Checks that could not be completed offline

- The historical-paper activation matrix is not present in the local frozen cache, so the historical-feature cross-implementation cell remains unavailable.
- V9 token bags omit token IDs and retained-position indices, so exact identical-task-suffix pooling cannot be reconstructed from the archive.
- The planned 1,000-permutation extension was not run. The existing 100-draw null already establishes label association; more p-value resolution cannot repair construct validity.

## Decision and next step

The independent verifier reproduced the v9 test and hard-negative metrics plus
all 56 family-only and 56 family-and-task-blocked summaries. User recruitment,
early-CoT transfer, and steering remain locked.

The next scientific step is to design `contrastive_prompts_v3`: latent factorial
scenarios, counterbalanced evaluation and production vocabulary, matched length
and cue position, disjoint generators and downstream tasks, and a fresh untouched
counterfactual holdout. No further RunPod spend is justified on v2.
"""
    atomic_text(args.report, report)
    evidence_inputs = {
        "preregistration": args.prereg,
        "analysis": args.analysis,
        "verification": args.verification,
        "decision": args.decision,
        "report": args.report,
    }
    evidence = {
        "schema_version": "glm53_v10_final_evidence_v1",
        "project_id": analysis["project_id"],
        "decision": decision["decision"],
        "verification_passed": verification["passed"],
        "paid_compute_used": False,
        "files": {
            name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in evidence_inputs.items()
        },
    }
    atomic_json(args.evidence, evidence)
    print(json.dumps({"report": str(args.report), "evidence": str(args.evidence)}, indent=2))


if __name__ == "__main__":
    main()
