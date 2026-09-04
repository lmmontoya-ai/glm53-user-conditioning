# GLM-5.3 v8 terminal state, 2026-08-31

## Outcome

The exact official FP8 GLM-5.3-Flash checkpoint ran successfully on two B300
GPUs, passed M2, and stopped at M3 under the preregistered fail-closed rule.
M4 through M8 did not run.

This is a scientific gate failure, not another storage or scheduler failure.
The model loaded, prompt forwards completed, all-layer features were extracted,
and the independent eval/deploy readout was fitted. The readout was highly
predictive on ordinary held-out data, but it did not pass all controls required
to freeze it as the mechanism candidate.

## Exact subject and runtime

- Model: `zai-org/GLM-5.3-Flash`
- Revision: `04c4e9e95c5da8862dced7e5056455116f83a7e0`
- Weights: official FP8, 62 shards, 328,337,455,672 bytes
- Transformers commit: `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`
- Runtime: PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1
- Hardware: 2 x NVIDIA B300 SXM6 AC
- Preregistration tag: `glm53-user-eval-v8-preregistered-v1.19`
- Preregistration commit: `71e691ca`

The CUDA 13 runtime amendment was made before the successful model forward and
did not change the checkpoint, data, prompts, splits, estimands, thresholds, or
gate logic.

## M2 result

M2 passed every required check:

- exact snapshot, byte count, and shard count;
- 45 decoder layers and expected mHC tensor shape;
- alpha-zero logits and prompt features;
- hook cleanup;
- batch and budget checks.

Batch size 1 was selected at 2.729 prompts/second. Larger padded batches were
faster in some cases but exceeded the strict preregistered batch-versus-single
numerical tolerance, so they were rejected. The projected remaining ladder cost
was $40.95 at the measured rate.

## M3 result

The preregistered selected representation was layer 42, prompt-final view, with
the paired-mean direction. The main readout numbers were strong:

- core held-out AUROC: 0.9961;
- realism-holdout AUROC: 1.0000;
- observed AUROC exceeded the 99th percentile of the shuffled controls.

Three required controls failed:

1. Hard-negative false-positive rate at 80% TPR was 0.3333, above the 0.30 cap.
2. Shuffled-label median AUROC was 0.34375, outside the required [0.45, 0.55]
   calibration band.
3. The Hua steering-anchor effects had the wrong sign for both candidates:
   -0.109375 for paired mean and -0.078125 for raw logistic.

The machine decision was therefore `stop`, with `passed: false`. The high
held-out AUROC is not enough to override these controls. In particular, the
wrong-sign Hua anchor means the fitted vector was not independently validated
as an intervention direction.

## Scientific interpretation

The experiment established that eval/deploy context is highly linearly readable
in this exact local checkpoint under the tested dataset. It did not establish a
valid causal eval/deploy direction under the preregistered control suite.

No user-population recruitment test, local confidence-proxy parity test, or
causal user intervention ran. Consequently, this run provides no evidence for
or against the hypothesis that the v7 AI-specific fame interaction recruits the
eval/deploy representation. It stops one rung earlier: the proposed independent
mechanism instrument did not meet its own validity gate.

Post-hoc layer substitution or sign flipping is not permitted. Any revised
direction construction, shuffle procedure, hard-negative threshold, or anchor
surface would require a new preregistration and should be treated as a new
study.

## Preserved evidence

Compact evidence is stored under
`pipelines/glm53_user_eval/v8/status/evidence/recovery_20260831/v119_terminal/`.
It includes the M2 and M3 decisions, runtime report, direction report, feature
row metadata, supervisor summary, and log.

The complete 169 MB M3 feature/direction bundle is preserved outside Git at:

`D:\research\mats\neel_2026\glm53_v8_terminal_v119\m3`

Load-bearing hashes:

- `eval_features.npz`: `a357f42f2fd9d9f4ef09e688bb16f4e82f8eb1cc6d40a07bd9ca137075442025`
- `eval_directions.npz`: `6bb47c91c6f21d5c9c6a3c8ee8dabe509b6a7bda53ac167e94264300d04e56e5`
- `eval_features.jsonl`: `36819fdf06e1b76c6e3a4cc82e767b7bbe30677a0aed0fc8f3ccb7952b397548`
- `eval_direction_report.json`: `fecce64e929ebe3e65c3dae84c8c5bf52f7b02bd9cf5640ea517a4d541724728`
- `m3_decision.json`: `3a9fda6876b626c7e7baad30811022055d6b6c200749d6792251aff1e2272d28`
- `supervisor_v119.log`: `92a7cca3a59d715a94a21e311d1abd566ff0972aec6de54943e0ed7dfcb4e12a`

## Infrastructure closure

- Pod `6f8skqxvd5n9xd` was deleted after evidence transfer.
- `runpodctl pod list` returned an empty list.
- Account balance immediately after deletion was $59.41, subject to final
  metering settlement.
- The network volume remains preserved and continues to incur storage charges.

