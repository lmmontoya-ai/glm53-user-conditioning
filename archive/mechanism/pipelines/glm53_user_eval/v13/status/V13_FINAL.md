# V13 final state

V13 tested the unchanged 576-row V12 scenario bank with two independent local
Codex judges authenticated through the ChatGPT subscription:

- `gpt-5.6-luna` at `max` reasoning;
- `gpt-5.6-terra` at `high` reasoning.

Neither judge used the optional fast or priority tier. Each scenario ran in a
fresh ephemeral read-only session with user configuration, tools, plugins,
shell access, web access, and API-key environment variables disabled. The run
completed 1,152 valid judgments. The primary and independently implemented
analyses agreed.

## Result

Both judges failed the frozen V12 semantic-validity gates.

| Metric | Luna max | Terra high | Required |
|---|---:|---:|---:|
| Individual-factor accuracy | 87.79% | 88.87% | 95% |
| Clean binary-label accuracy | 74.79% | 78.54% | 95% |
| Final counterfactual accuracy | 75.00% | 78.13% | 90% |

The unchanged bank therefore remains invalid as a source instrument. No GLM
forward pass, RunPod compute, recruitment test, or steering test is authorized.

## Diagnosis

The failure audit flagged 291 rows spanning 215 of 336 pair IDs. Errors were
concentrated in two factors:

- `real_external_recipient`: 115 rows were wrong for both judges;
- `operational_consequence`: 51 rows were wrong for both judges.

The common text pattern uses opaque aliases for an operational actor, an
assessment actor, an operational outcome, and an assessment outcome. The
judges often treated internal operational staff as external recipients and
treated model-ranking or model-scoring outcomes as ordinary operational
consequences. The V12 instruction did not define those boundaries precisely
enough, while the alias-heavy scenarios made the intended relations needlessly
difficult to recover.

The correct next action is the frozen V13 repair branch: create a new V14
dataset version, rewrite complete matched pairs under a symmetric rule, clarify
the factor boundaries, create a fresh untouched final holdout, and rerun both
judges. Existing successful rows cannot be reused because the text and prompt
contract will change.

