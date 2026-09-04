# V19 final report

V19 tested whether the published Hua deployment/evaluation activation
intervention transfers to the exact official FP8 GLM-5.3-Flash checkpoint. The
test stopped at its independent positive control. It did not score the local
user-behavior replication or the causal user experiment.

## What ran

The final run used two NVIDIA B300 SXM6 AC GPUs and the official 62-shard,
328,337,455,672-byte checkpoint at revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. Every shard matched the frozen
manifest. Alpha-zero output was exact, and the measured rate was 1.0955 prompts
per second.

The run then extracted Hua's 16 paired deployment/evaluation directions across
all 45 GLM text layers. It tested two fixed six-layer schedules on 32 independent
scenarios under two reversed label mappings, with neutral-output damage checks.

## Result

Baseline direct classification was perfect, so the control task itself was
well-defined for the model.

The literal Hua schedule moved the average in both predicted directions. The
deployment-directed change was -0.728 log-odds, but its 95% interval was
[-1.958, 0.493], and only 50% of scenarios moved in the predicted direction.
The reverse evaluation-directed change was +2.402 with a 95% interval of
[1.349, 3.473], but only 65.6% of scenarios moved correctly. The frozen gate
required at least 75% consistency in each direction and both intervals on the
predicted side of zero.

The depth-mapped schedule failed more clearly. Its average effects were small,
the two label mappings disagreed, and median neutral-output divergence was
1.200, compared with the frozen maximum of 0.10.

The machine decision was:

```text
stop_hua_direction_did_not_transfer
```

## Meaning

The literal schedule shows some average steering sensitivity, especially in
the evaluation direction, but the effect is too uneven to treat this direction
as a validated GLM intervention. Testing it on the Famous-AI user effect would
have made any positive or negative result hard to interpret. The pipeline
therefore stopped before local behavioral parity and before all user-facing
intervention rows.

This result does not weaken the confirmed V7 behavioral interaction. It says
only that Hua's published activation-addition recipe did not transfer robustly
enough to this checkpoint under the preregistered GLM adaptation.

## Evidence

The final scientific run is
`glm53-v19-hua-20260903T054118Z-941c61df`. Its independent verifier passed, and
all 16 files in the evidence manifest match their recorded SHA-256 hashes. The
committed evidence includes the raw 320 positive-control rows, all damage rows,
the extracted directions, analysis, machine decision, independent verification,
runtime calibration, exact-checkpoint verification, launch record, and watchdog
deletion record.

The Pod was deleted after backup. No RunPod Pod or Serverless endpoint remained
active. The temporary 20 GB S3 evidence volume `950vl0d4x4` was deleted after
the committed evidence hashes were verified, leaving zero network volumes. The
account balance observed after the terminal decision was $68.68. The temporary
S3 credentials must now be rotated.

## Next action

Do not spend more compute trying to make this intervention pass. Use the V7
behavioral confirmation as the main result and report V19 as a preregistered
negative transfer result. It is evidence that the confirmed user-conditioning
effect cannot currently be connected to Hua's intervention on the exact local
GLM checkpoint.
