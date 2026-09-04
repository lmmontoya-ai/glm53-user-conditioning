# V14 final state

V14 is complete and remains a failed historical semantic gate. It is not being
overridden.

Both local judges completed all 576 scenarios through ChatGPT subscription
authentication on the standard Codex tier. `fast_mode` was explicitly disabled.
Luna used maximum reasoning and Terra used high reasoning. The 1,152 completed
rows passed the technical audit and the independent recomputation agreed.

The systematic binary-pair repair worked:

| Metric | Luna max | Terra high |
|---|---:|---:|
| Five-factor accuracy | 99.49% | 99.41% |
| Clean binary label accuracy | 97.29% | 98.12% |
| Fresh final-holdout accuracy | 92.19% | 100.00% |
| Evidence-span validity | 99.16% | 99.89% |

Both judges nevertheless failed the frozen 90% neutral-control threshold:

| Neutral-control accuracy | Result |
|---|---:|
| Luna max | 82.81% |
| Terra high | 79.69% |

The failures were concentrated in the unchanged `logging_review_mixed` controls.
Those rows state that a person, user, or recipient uses the answer, but do not say
that this recipient is outside the operating organization. V14's corrected fact
definition explicitly forbids inferring external-recipient status from internal
or unspecified users. Both judges therefore returned `no_status`, while the old
acceptable-label contract allowed only `deploy` or `ambiguous`.

Machine decision:

```text
final_semantic_stop_both_judges_failed
```

No GLM forward, RunPod compute, local parity, recruitment, CoT, or steering work
was authorized by V14.

A later dataset version may proceed only as a separately preregistered experiment
under explicit user authorization. It must preserve V14 unchanged, replace the
entire neutral-control surface rather than deleting only failed rows, and use a
fresh control set whose status is explicit under the five-factor contract.
