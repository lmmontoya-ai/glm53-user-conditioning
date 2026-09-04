# V20 runtime amendment after a pre-forward failure

The first paid V20 attempt (`glm53-v20-hua-20260903T131208Z-602b5cb1`, Pod
`93sgji2j6sn2uk`) staged the exact official checkpoint in 146.91 seconds and
verified all 62 weight shards and static model files. It then exited before
model construction and before any model forward with:

```text
KeyError: 'runtime_checks'
```

The inherited V11 loader expects five compatibility values that the compact
V20 runtime document did not expose: expected CUDA-device count, expected GPU
name, linear-attention-layer count, sparse-attention-layer count, and the
already-present forget-gate scale count. The missing values are fixed facts of
the preregistered hardware and exact checkpoint:

```text
CUDA devices: 2
GPU: NVIDIA B300 SXM6 AC
linear-attention layers: 34
sparse-attention layers: 11
forget-gate scale-inverse tensors: 68
```

This infrastructure-only amendment constructs a loader-facing copy of the
immutable V20 runtime configuration and supplies those values. It does not
change the model, checkpoint, prompts, identities, dilemmas, directions,
layers, coefficients, codebooks, estimands, thresholds, or stopping rules.
The failed attempt produced zero local-parity rows and zero intervention rows,
so no target outcome was available when this amendment was made.

The first Pod self-deleted under the pre-existing fail-safe. For a later
unexpected pre-forward software error, the operator should preserve the live
Pod when safe long enough to apply a small audited patch, rather than repeat
checkpoint staging. Scientific gate failures still require prompt termination.
