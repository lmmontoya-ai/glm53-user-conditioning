# V15 final state

V15 completed successfully and is the terminal semantic-validation result for
the repaired latent evaluation/deployment source bank.

The exact residual V14 failure was repaired prospectively: all 64 old neutral
controls were retired and replaced by 64 fresh controls balanced across
`no_status`, `deploy`, `eval`, and `ambiguous`. The other 512 rows were unchanged.
Their prior judgments were reused only after exact sample, scenario, prompt,
schema, model, effort, request-hash, and non-fast checks. Both judges then made
64 fresh judgments, for 128 new scientific calls.

Runtime contract:

- Luna: `gpt-5.6-luna`, maximum reasoning;
- Terra: `gpt-5.6-terra`, high reasoning;
- ChatGPT subscription authentication;
- standard inference tier, with `fast_mode` explicitly disabled;
- at most 24 concurrent isolated sessions;
- atomic per-row checkpoints and hash-based resume.

Results:

| Metric | Luna max | Terra high |
|---|---:|---:|
| Five-factor accuracy | 99.49% | 99.41% |
| Clean binary label accuracy | 97.29% | 98.12% |
| Fresh final-holdout accuracy | 92.19% | 100.00% |
| Mixed-purpose acceptance | 100.00% | 100.00% |
| No-status acceptance | 100.00% | 100.00% |
| Fresh control accuracy | 100.00% | 100.00% |
| Evidence-span validity | 99.25% | 99.90% |

Every frozen threshold passed independently for each judge. The independent
verifier reproduced the result exactly.

Machine decision:

```text
fresh_control_bank_validated_by_both_codex_judges
```

This validates the text bank as the source instrument for the next stage. It
does not itself establish an internal GLM-5.3 representation or explain the V7
behavior. The only newly authorized step is bounded exact-FP8 source-feature
extraction. Local parity, user recruitment, CoT transfer, and steering remain
locked until their own gates pass.
