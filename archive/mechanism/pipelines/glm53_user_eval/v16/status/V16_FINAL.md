# GLM-5.3 V16 final state

V16 completed the exact-FP8 source extraction and stopped at the preregistered source-instrument gate. Local behavioral parity and prompt recruitment were not run.

## What ran

- Loaded the official `zai-org/GLM-5.3-Flash` FP8 checkpoint at revision `04c4e9e95c5da8862dced7e5056455116f83a7e0` on two NVIDIA B300 GPUs.
- Downloaded and fully hash-verified the 328 GB checkpoint in 100.94 seconds.
- Passed the runtime checks: 62 shards, 45 text layers, four 4096-dimensional mHC streams, 68 finite FP8 scale tensors, exact no-op logits and features, and hook cleanup.
- Extracted four all-layer activation views for all 576 V15 source prompts.
- Fit the preregistered ridge-logistic and paired-mean readouts using only development data.
- Selected layer 26 and ridge `C = 0.1` without exposing the final binary, factorial, or fresh-control rows.
- Completed the source analysis and machine stop on the Pod.

## Scientific result

The readout generalized well on the binary tests:

- ordinary-test AUROC: `0.898`;
- final-counterfactual AUROC: `0.932`;
- final-counterfactual false-positive rate at 80% recall: `0.094`.

It nevertheless failed the untouched control bank:

- clean eval-versus-deploy control AUROC: `0.719`, below the required `0.80`;
- evaluation true-positive rate: `0.25`, below the required `0.75`;
- no-status abstention rate: `0.00`, below the required `0.50`;
- mixed-purpose scores were not ordered between clean deployment and clean evaluation as required;
- the operational-use coefficient had the wrong sign;
- formality and AI-vocabulary effects were too large relative to the decisive factors.

The whole factorial calibration also failed the nuisance-subordination rule because the formality coefficient was larger than at least one decisive-factor coefficient. The paired-mean readout failed additional checks.

The machine decision was therefore:

```text
stop_before_local_parity
```

This means the new scenario bank contains a strong, stable binary signal, but the learned direction is still not selective enough to serve as a validated evaluation-versus-deployment instrument. It must not be transferred to the user-identity experiment.

## Downstream work not run

- local one-token confidence parity: not run;
- Famous-AI prompt recruitment: not run;
- early chain-of-thought transfer: out of scope and not run;
- steering: out of scope and not run.

The V7 behavioral result remains intact. V16 only says that this particular source readout is not valid enough to test whether that behavioral effect shares an eval/deploy representation.

## Infrastructure and recovery

Run ID: `glm53-v16-source-20260902T015329Z-8c254763`

The scientific process ran for 2,568 seconds and the Pod self-deleted. RunPod's billing history records $3.0556 for this final Pod. No Pods or Serverless endpoints remained afterward. The exact row-level activation archive was recovered locally: 576 activation files plus 576 manifests, 851,123,314 bytes total, with zero missing or mismatched hashes.

The S3 uploader rejected a large consolidated NumPy archive with HTTP 413. That happened after all row-level activations and manifests were uploaded, but before the compact source reports were copied. The signed Pod evidence preserved the report hashes and terminal decision. Final source metrics were deterministically recomputed from the exact recovered row files under the frozen V16 implementation. The missing numeric permutation report cannot rescue the source gate because multiple independently recovered non-permutation checks fail.

After the local recovery and hash audit passed, the temporary 20 GB S3 transport volume `yqs9l7a86w` was deleted. RunPod then reported zero Pods, zero Serverless endpoints, zero network volumes, no active hourly spend, and a remaining balance of $26.675.

Compact evidence is under `artifacts/glm53_user_eval/v16/reports/source_readout/` and `artifacts/glm53_user_eval/v16/infrastructure/final_run/`. The 851 MB raw activation archive remains local and is bound by `recovery_integrity.json` rather than committed to Git.
