# GLM-5.3 v9 terminal state

Date: 2026-08-31

Project: `glm53_user_eval_paper_faithful_probe_v9`

## Outcome

V9 successfully corrected the v8 representation mismatch and ran the
paper-faithful cue-masked prompt probe on the exact official FP8
GLM-5.3-Flash checkpoint. The readout gate failed its adversarial hard-negative
control, so the branch stops before user-context recruitment or steering.

This is not evidence that GLM-5.3 lacks an eval/deploy representation. It means
the tested cue-masked linear instrument did not meet the preregistered transfer
standard required to apply it to the user-awareness question.

## Completed work

- Verified the exact tokenizer and chat template on all 448 governed rows.
- Obtained exactly 416 `masked` rows and 32 neutral `not_available` rows, with
  no ambiguous, partial, missing, zero-overlap, or empty masks.
- Loaded official FP8 `zai-org/GLM-5.3-Flash` revision
  `04c4e9e95c5da8862dced7e5056455116f83a7e0`.
- Rehashed all 62 weight shards totaling 328,337,455,672 bytes.
- Verified 45 decoder layers, including 34 linear-attention and 11
  sparse-attention layers.
- Extracted all 448 rows at all 45 layers:
  `masked_prompt_mean`, `prompt_final`, `last_unmasked_prompt_token`,
  `cue_token_mean`, and cue-masked token bags.
- Matched the preserved v8 prompt-final features exactly: all 448 prompt hashes
  matched and maximum activation error across all layers was 0.0.
- Fit the declared AdamW sequence-linear probe and top-four-layer token-MIL.
- Ran 100 pair-preserving full-pipeline permutations.
- Independently recomputed all primary metrics and the machine decision with
  exact agreement.

## Primary result

The cue-masked prompt-mean sequence probe selected layer 44 using validation
data only.

| Surface | AUROC | FPR at 80% TPR |
| --- | ---: | ---: |
| Core test | 1.000 | 0.000 |
| Realism holdout | 1.000 | 0.000 |
| Hard-negative holdout | 0.719 | 0.500 |
| MATS auxiliary holdout | 0.926 | 0.125 |

The full-pipeline paired permutation test had zero of 100 null runs equal to or
above the observed test AUROC, giving the add-one empirical value
`p = 0.00990099`. That control passed.

The hard-negative requirement was `FPR@80% TPR <= 0.30`. The observed value was
`0.50`, so the readout gate failed. Neutral-decoy FPR was also `0.50`.

## Secondary results

- `cue_token_mean` was perfect across the binary surfaces. This is the expected
  direct-cue positive control.
- `prompt_final` and `last_unmasked_prompt_token` selected layer 25 and reached
  about 0.95 core-test AUROC, but hard-negative FPR remained about 0.38.
- Token-MIL did not rescue the control. Its four candidate layers had
  hard-negative FPR values from 0.44 to 0.50, all above the 0.30 gate.

Masking therefore changed the instrument and removed the v8 methodological
regression, but it did not produce a readout robust enough for the planned
cross-domain recruitment claim.

## Machine decision

```text
decision: stop_readout_branch
readout_passed: false
steering_status: not_run_separate_gate
user_recruitment: not_run
early_CoT_transfer: not_run
steering: not_run
```

The readout and steering gates remained separate as intended. Steering did not
cause this stop; the readout failed before steering was eligible.

## Technical amendments

All amendments were made before the affected scientific output existed.

1. V1.1 changed Windows working-tree hashes to canonical Git-blob hashes after
   Linux validation stopped before model loading.
2. V1.2 corrected the expected forget-gate inverse-scale count from 90 to 68.
   The locked architecture has 34 linear-attention layers with two projections
   each and 11 sparse-attention layers without those projections. No model
   forward had run.
3. V1.3 encoded governed neutral-decoy labels as `-1` in the sequence fitter,
   consistent with the other analysis paths and the declared separate-decoy
   analysis. The first fit had stopped before producing any metric or selected
   layer.

## Infrastructure and cost

- Pod: `v6w0o4uykl6csw`
- Hardware: 2 x NVIDIA B300 SXM6 AC
- Rate: USD 15.78/hour
- Billed uptime captured before deletion: 3,035 seconds
- Balance before launch: USD 58.6749472482
- Balance after deletion: USD 44.9301634705
- Observed balance change, including contemporaneous storage: USD 13.7447837777
- Pod deletion verified: `runpodctl pod list --all` returned `[]`
- Remaining non-Pod spend: USD 0.049/hour for retained storage

The 7.7 GB scientific bundle was compressed to 6,218,291,465 bytes, copied off
the Pod, and verified before deletion. Its SHA-256 is
`705abef6295b79dbb429bda6abf9db7ec55758d30a35af5f23528c9efe336cde`.

## Scientific interpretation

V8 did test the wrong probe surface. V9 fixed that. The correction did not,
however, yield a claim-grade transferable readout under the existing hard
negative set. The ordinary held-out and realism results show that eval/deploy
information remains very easy to decode after removing the governed cue spans.
The adversarial inversion shows that the learned boundary still relies on
correlates that do not survive the strongest counterexamples.

The next scientifically defensible step, if pursued, is a new preregistered
representation study with a redesigned training bank that includes
counterbalanced hard negatives during training while reserving a new untouched
adversarial set. Reusing the present hard negatives for training would change
the estimand and cannot be called a continuation of this confirmatory gate.
