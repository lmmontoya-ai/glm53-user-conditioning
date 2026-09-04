# GLM-5.3 v10 offline construct audit

## Decision

`construct_overfit_source_instrument_invalid`

The v9 code successfully learned label-associated information, but the preserved
source bank is not a valid instrument for a robust latent evaluation/deployment
state. The failure is not explained by the v9 AdamW trainer or by simple saved
length, role, and mask metadata. No new model forward or paid compute was used.

## Fixed-split trainer comparison

| Method | Layer | Core test AUROC | Realism AUROC | Hard-negative AUROC | Hard FPR@80% TPR | Neutral FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V9 AdamW | 44 | 1.000 | 1.000 | 0.719 | 0.500 | 0.500 |
| Historical full-batch | 42 | 1.000 | 0.957 | 0.688 | 0.625 | 0.719 |
| Paired mean | 26 | 0.875 | 0.934 | 0.695 | 0.562 | 0.281 |
| Ridge logistic | 26 | 0.992 | 0.957 | 0.645 | 0.750 | 0.062 |
| Shrinkage LDA | 27 | 0.969 | 0.977 | 0.684 | 0.750 | 0.031 |
| PCA + ridge | 27 | 0.996 | 0.977 | 0.699 | 0.562 | 0.281 |

The preregistered hard-negative gate required FPR@80% TPR at or below 0.30.
Every method failed. The historical trainer reached 0.625,
worse than v9 AdamW at 0.500.

## Distribution and leakage diagnostics

- Leave-one-family-out paired-mean transfer: median AUROC 0.990, minimum 0.801 across 56 ordered train/validation/test configurations.
- Family-and-task-blocked transfer: median AUROC 1.000, minimum 0.750; 92.9% of the 56 tiny held-out folds were perfect.
- Structural metadata baseline: test AUROC 0.500. Saved prompt length, retained and masked token counts, cue count, prompt role, cue location, and within-pair length gap do not explain the activation result.
- V9 hard-negative decomposition: benchmark-language deployment AUROC 1.000; production-language evaluation AUROC 0.875. The metadata-header role was weakest at AUROC 0.562.
- Neutral controls: formal/casual FPR 0.812; AI/human FPR 0.188. Generic AI wording was not the main neutral failure in this run.
- Pair-bootstrap stability: median direction cosine to the full paired-mean model 0.819, but minimum 0.032. Test AUROC ranged down to 0.637. The direction has an unstable tail even though its median stability clears 0.80.

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
