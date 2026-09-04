# v8 infrastructure amendment v1.7

Frozen on 2026-08-30 before M2, any v8 model forward, and any v8 scientific
row.

Preregistration SHA-256:
`29a7fb46cd6d1665cb7e7f602c7c24cbcfb928db48a37e1f83a5c6e6691264f7`.

The fourth Serverless lifecycle is intentionally different from attempts 1--3.
It does not attach network volume `a9diryunoj`, so RunPod may place the worker
outside `US-KS-2`. RunPod's managed Hugging Face cache still supplies the exact
FP8 snapshot. Compact inputs and completed gate artifacts move through the
volume's S3-compatible endpoint.

The endpoint contract is:

- exactly three NVIDIA H200 GPUs;
- one endpoint attempt;
- one worker minimum and maximum;
- one 90-second non-scientific rate probe;
- no model load or forward during that probe;
- no scientific job unless the observed GPU rate is at most USD 14.50/hour;
- the scientific job must run on the same worker ID returned by the probe;
- no scale-to-zero step between probe and science;
- one scientific job at most;
- 90 minutes maximum to obtain a ready worker;
- USD 90 compute cap, USD 15 reserve, and USD 2 storage allowance.

The amendment changes no model, data, identity, task, codebook, estimand,
threshold, bootstrap, intervention, or claim rule. M0 and M1 remain passed. M2
remains unopened.

## Prepared state

- Preregistration commit: `bfb5fc56b4e8921a9eb42dbd7ca788558a416402`.
- Preregistration tag: `glm53-user-eval-v8-preregistered-v1.7`.
- Non-billable Serverless template: `8clavla4f2`.
- Template image: official pinned RunPod PyTorch image.
- Template startup checks out only the v1.7 tag and verifies the exact commit.
- No endpoint or Pod was created while preparing v1.7.

RunPod S3 credentials are the remaining launch prerequisite. The CLI and
official API do not create these keys. The account owner must create a
dedicated key under RunPod Settings > S3 API Keys and expose it only through
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Do not commit either value.
