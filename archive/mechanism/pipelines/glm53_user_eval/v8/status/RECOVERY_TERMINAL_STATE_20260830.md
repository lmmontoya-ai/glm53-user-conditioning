# GLM-5.3 v8 recovery terminal state

Date: 2026-08-30

## Decision

The exact-FP8 infrastructure branch is stopped at M2. The recovery attempts did
not complete the runtime gate, and the final v1.10 placement attempt failed
before producing logits. Per the recovery preregistration, do not launch another
exact-FP8 Pod or Serverless endpoint under v8.

This is an infrastructure/runtime failure, not a negative scientific result.
The confirmed v7 behavioral interaction remains unchanged.

## Completed work

- M0 passed: the v7 parent evidence, model revision, selections, datasets,
  codebooks, task splits, analysis rules, and gates are frozen and hash-locked.
- M1 passed: the transcript cache and one-token proxy contract were built and
  validated. No original confidence response enters the local model input.
- The official `zai-org/GLM-5.3-Flash` FP8 revision
  `04c4e9e95c5da8862dced7e5056455116f83a7e0` was staged to Pod-local NVMe.
- All 62 shards, totaling 328,337,455,672 bytes, matched the frozen hashes.
  Download took 290.90 seconds and full verification took 338.73 seconds.
- The pinned fine-grained FP8 kernel revision
  `29083040812e244b390757d6198e2889fe551d13` was cached and passed an offline
  import/load smoke.
- The exact model loaded far enough under multiple placements to isolate the
  memory-topology failures described below.

## What did not complete

- M2 did not pass and no `m2_decision.json` exists.
- No successful v8 logits, prompt activations, proxy scores, or intervention
  outputs were produced.
- No v8 scientific row was produced.
- M3 through M8 were not run.

## Failure sequence

1. v1.7, default balanced placement: model loading completed, but the first M2
   calibration forward requested another 11.00 GiB on GPU 0 with 9.86 GiB free.
2. v1.8, `balanced_low_0` with uniform 115 GiB caps: aggregate capped memory was
   too low for the expanded MoE representation, so Accelerate requested disk
   offload during loading. Disk offload was outside the runtime contract.
3. v1.9, `balanced_low_0` with auto-detected capacity: GPUs 1 and 2 filled to
   roughly 138 and 140 GiB. Final conversion requested 4.50 GiB on GPU 1 with
   1.71 GiB free and failed.
4. v1.10, measured 130/134/134 GiB caps: the aggregate cap again caused
   Accelerate to request disk offload before model finalization.

These failures show that three H200s have enough nominal aggregate capacity for
the checkpoint bytes, but the pinned Transformers loader cannot simultaneously
place the expanded MoE weights and preserve the working memory required by the
M2 forward without offload under the tested maps.

## Infrastructure state after stop

- Pod `tnlgt7adfbgtp6` was deleted successfully.
- `runpodctl pod list` returned an empty list.
- `runpodctl serverless list` returned an empty list.
- RunPod balance immediately before deletion was USD 85.6140180584.
- Account spend after deletion was USD 0.049/hour, attributable to retained
  storage rather than compute.
- Network volume `a9diryunoj` remains intact.
- The local temporary S3 credential file was deleted. The Pod copy was destroyed
  with the Pod. The S3 key must still be rotated because it appeared in earlier
  tool output.

## Evidence

Compact evidence is committed under
`status/evidence/recovery_20260830/`. The complete downloaded Pod bundle is kept
outside Git at:

`D:\research\mats\neel_2026\glm53-v8-v1.10-final-pod-evidence.tar.gz`

Bundle SHA-256:

`05ad4837cbb3403cad6b9a533d31f6f20974a543fd9089f9fc88c3c3026cfe1f`

Incremental copies were also written to the retained volume's S3 endpoint under
the versioned `glm53-v8-results/v1.7` through `v1.10` prefixes.

## Frozen recovery tags

- `glm53-user-eval-v8-preregistered-v1.8` at `2584f042`
- `glm53-user-eval-v8-preregistered-v1.9` at `d3d09934`
- `glm53-user-eval-v8-preregistered-v1.10` at `808fc107`

## Permitted next work

Choose a new preregistration and new project identity for one of:

1. a quantized, instrumented GLM-5.3 checkpoint with claims explicitly scoped
   to that checkpoint; or
2. an API-only causal cue-decomposition study using the confirmed v7 behavior.

Do not describe v8 as a mechanistic null. It never reached the first valid
white-box forward under the final pipeline.
