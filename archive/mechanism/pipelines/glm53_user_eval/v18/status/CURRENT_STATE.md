# V18 current state

V18 is terminal. It used exactly two NVIDIA B300 SXM6 AC GPUs.

What passed:

- exact 328 GB official FP8 checkpoint staging and complete rehash;
- all 62 weight shards;
- exact CUDA, Torch, and transported Transformers source;
- two-B300 hardware identity;
- 45-layer model load;
- all 68 required FP8 scale tensors present and finite;
- alpha-zero intervention equivalence.

What stopped the run:

- batch size four differed from single-row scoring by 1.25 logits, versus the
  frozen 0.002 maximum;
- measured throughput projected roughly 14.5 hours for the full ladder, well
  beyond the 110-minute bound.

What was not run:

- Hua direction extraction;
- the independent eval/deploy positive control;
- local behavioral parity;
- causal Famous-AI user interventions.

Cleanup is complete: zero Pods, zero Serverless endpoints, zero persistent
volumes, and zero current hourly spend. The recovered immutable runtime files
are under `artifacts/glm53_user_eval/v18/run/`.

