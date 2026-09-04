# V21 final report

V21 completed the user-facing Hua intervention ladder that V20 had stopped
before running. It is explicitly exploratory because V20 failed one component
of its local-parity gate: Famous-non-AI minus GenPop was negative rather than
nonnegative. V20 remains failed and unchanged.

## What ran

The run used the exact official FP8 `zai-org/GLM-5.3-Flash` checkpoint at
revision `04c4e9e95c5da8862dced7e5056455116f83a7e0` on two NVIDIA B300 SXM6 AC
GPUs. All 62 weight shards, totaling 328,337,455,672 bytes, matched the frozen
manifest. The live process reproduced four saved V20 baseline rows with zero
logit error, and an alpha-zero intervention reproduced the full logits exactly.

V21 reused the 1,404 immutable V20 baseline rows and scored 8,620 new rows:

- 1,404 rows under the complete Hua deployment direction;
- 1,404 under the reversed evaluation direction;
- 1,404 under a status-only deployment direction;
- 1,404 under an actor-only deployment direction;
- 1,404 under a formality-control direction;
- 1,600 rows under 20 fixed sign-flip null directions.

The measured intervention rate was 2.253 prompts per second. Model-runtime work
took 54.99 minutes, and the whole Pod existed for about 64 minutes.

## Primary result

The reused local baseline interaction was:

```text
(Famous-AI - Unknown-AI) - (Famous-non-AI - GenPop) = -0.402 pp
```

Under the complete Hua deployment direction, the interaction became
`-0.127 pp`, a point change of `+0.274 pp`. This corresponds to 68.3% attenuation
at the point estimate, but the paired 95% interval was
`[-1.099, +1.650] pp`. The independent bootstrap returned
`[-1.056, +1.674] pp`. Both intervals include zero by a wide margin.

The machine decision was:

```text
no_resolved_deployment_attenuation
```

## Why the point estimate is not persuasive

The apparent attenuation did not come from the target Famous-AI versus
Unknown-AI comparison. That component changed by only `+0.031 pp`. The
Famous-non-AI versus GenPop component changed by `-0.243 pp`, accounting for
most of the `+0.274 pp` interaction movement. The intervention therefore moved
the control side of the contrast, not primarily the AI-user side.

The reversed evaluation intervention also moved the interaction toward and
past zero. Its change was `+1.138 pp`; the prediction was a negative change.
This failure of sign reversal is inconsistent with a clean deployment-versus-
evaluation mechanism.

The diagnostic directions were not selective:

| Arm | Interaction (pp) | Change from baseline (pp) | 95% interval for change (pp) |
| --- | ---: | ---: | ---: |
| Complete Hua deployment | -0.127 | +0.274 | [-1.099, +1.650] |
| Complete Hua evaluation | +0.737 | +1.138 | [-0.681, +2.866] |
| Status-only deployment | +0.015 | +0.417 | [-1.102, +1.934] |
| Actor-only deployment | +0.183 | +0.585 | [-1.606, +2.699] |
| Formality control | +0.357 | +0.759 | [-1.919, +3.523] |

Actor-only and formality interventions produced larger point changes than the
complete Hua deployment direction. Those diagnostic arms also damaged the
confidence-output surface, so their magnitudes are not reliable causal effects.
The complete Hua deployment and evaluation arms themselves retained acceptable
output integrity.

On the frozen 80-row null-control surface, the complete Hua deployment
direction changed the interaction by `-2.458 pp`, opposite the prediction. It
ranked 21st out of the candidate plus 20 sign-flip null directions. The null
changes ranged from `-1.582` to `+2.144 pp`, and the add-one empirical p-value
was `1.0`.

Both confidence-codebook orientations did give positive full-sample point
changes (`+0.284` and `+0.216 pp`). This rules out a simple letter-mapping
artifact but does not overcome the unresolved interval, wrong reverse effect,
control-group source of the movement, or null-direction failure.

## Interpretation

V21 does not provide evidence that the fixed Hua deployment direction
selectively attenuates GLM-5.3's AI-specific user-conditioning effect. The
small full-sample point movement is compatible with noise or nonspecific
activation perturbation and is not localized to the target F-U component.

This does not show that GLM-5.3 lacks evaluation awareness. It shows that this
published, externally fixed intervention did not give a resolved or selective
causal effect on this local confidence proxy. The conclusion is additionally
limited because V21 was run after V20 failed its strict local-parity component
pattern.

The strongest completed scientific result remains the fresh V7 behavioral
interaction on the provider-served model. V21 should be reported as an honest
negative mechanistic follow-up, not as evidence for shared causal mediation.

## Verification, cost, and cleanup

The primary output contained 8,424 paired baseline/full-arm rows and 1,600 null
rows, with unique scientific keys. The independent implementation reproduced
every point estimate, null comparison, decision classification, and bootstrap
interval within the frozen tolerances. A second local rerun of that verifier
also passed every check.

All 575 files named in the Pod evidence manifest were copied locally and
matched their SHA-256 hashes, with zero missing files and zero mismatches. The
complete recovery copy contains 579 S3 objects and about 80 MB, including
heartbeats and terminal records outside the evidence manifest.

The balance change during V21 was approximately `$16.84`, leaving about
`$38.38`. The Pod self-deleted after the terminal evidence upload. The temporary
20 GB transport volume was deleted after local verification. Final checks found
zero Pods, zero Serverless endpoints, and zero network volumes.

The temporary S3 credentials must now be rotated because they were exposed
earlier in the project history.

## Recommended next action

Do not spend more compute trying to rescue this intervention before the
application. Write up V7 as the confirmed behavioral result, then use V19 and
V21 as bounded negative mechanistic tests: the Hua direction was not a robust
explicit-status intervention in V19 and did not selectively control the natural
user-conditioning phenotype in V21.
