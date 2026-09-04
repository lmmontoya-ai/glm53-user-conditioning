# GLM-5.3-Flash exact Transluce population-confidence replication

## Result

The full replication is complete, but the preregistered decision is
**ambiguous complete replication**: it is neither replication-positive nor a clean negative.

For the primary famous-AI group, GLM-5.3-Flash produced a median folded-confidence shift of
**−0.270 percentage points** relative to the per-dilemma general-population baseline. The
direction matches the released GLM-5.2 result, but the original tie-aware Mann–Whitney test
did not survive the fixed 28-test correction (unadjusted `p=0.0441`, adjusted `p=1.0`). The
secondary mean was **−0.365 points**, with a two-way cluster-bootstrap 95% interval of
**[−0.853, +0.121]**.

The same-index famous-person versus unknown affiliation-twin contrast was **−0.370 points**,
95% CI **[−0.801, +0.062]**. This is directionally consistent with name recognition reducing
confidence, but it is not statistically resolved.

The released GLM-5.2 effects were larger: famous-AI median **−1.473 points**, mean **−1.387
points**, and paired famous–unknown **−0.719 points**. The GLM-5.3 primary mean interval
excludes the released GLM-5.2 mean, indicating attenuation. However, the paired interval still
contains the released paired estimate, so the preregistered clean-negative gate also fails.

## What was tested

The hypothesis was that recognized AI-world users make the model less confident in its own
predicted behavior, potentially because such users make the interaction feel more evaluative.
The released Transluce behavioral self-prediction task presents agentic dilemmas, asks for a
yes/no prediction of what the model would do, then asks how confident it is that it would
actually behave that way.

This run directly executed the pinned Transluce task and scorer on:

- all 100 released DailyDilemmas-derived items;
- 70 general-population identities;
- 70 unknown names assigned AI affiliations;
- 70 famous AI-world identities;
- 59 published-address variants of famous AI identities;
- 70 famous non-AI identities;
- the source's repeated anonymous controls.

That produced 34,400 scientific rows. The intended scientific change was the subject only:
`openrouter/z-ai/glm-5.2` became `openrouter/z-ai/glm-5.3-flash`. GLM-5.3-Flash was pinned to
Novita FP8, high reasoning, and no provider fallback. Prompts, roster, task order, two-turn
measurement, judge, parsing, folded-confidence transform, and source aggregation were
preserved. Five-dilemma sharding and concurrent execution were infrastructure-only changes.

This is a complete replication of the released **population plain-mode confidence lane**,
not of every behavioral lane in the Transluce paper.

## Secondary results

The results are heterogeneous rather than uniformly null:

- The 59 published-address famous-AI variants had median **−0.490 points**, adjusted
  `p=0.0166`; their mean was **−0.597**, 95% CI **[−1.070, −0.124]**. This secondary condition
  shows a smaller but resolved confidence reduction.
- Famous non-AI identities shifted confidence in the opposite direction: median **+0.505
  points**, adjusted `p=0.00597`; mean **+0.456**, 95% CI **[+0.084, +0.831]**.
- Unknown AI-affiliation twins were near zero: median **+0.115 points**, adjusted `p=1.0`;
  mean **+0.011**, 95% CI **[−0.392, +0.413]**.
- Forty-three of 70 primary famous-AI person deltas were negative, and 40 of 59
  published-address variants were negative. Individual effects varied widely; these counts
  are descriptive and do not establish person-level significance.

The secondary pattern suggests that identity presentation matters, and that GLM-5.3 does not
apply one generic “famous user” response: published AI identities decreased confidence while
famous non-AI identities increased it. It does not establish that an eval/deploy representation
causes either effect.

## Verification

- 100/100 immutable shards completed; zero failed.
- 34,344/34,400 scores were valid: **99.837% completion**.
- Missingness spread across non-anonymous groups was **0.114 percentage points**.
- Every successful subject response was GLM-5.3-Flash through Novita with high reasoning and
  no fallback.
- Twenty-eight failed transport attempts were preserved and retried under the original retry
  policy; they were not new scientific samples.
- Total recorded API cost was **$28.99**, below the $50 cap.
- An independent standard-library analysis reproduced all point estimates within `1e−10`.
- A score-blind seeded manual audit passed 40/40 for binary extraction, confidence extraction,
  and on-task behavior.

The first final technical audit incorrectly counted failed transport-attempt events as model
responses. The error was corrected by a versioned audit amendment before outcome analysis;
no raw output or score was changed, and model calls were not rerun.

## Decision and limitation

The strict primary behavioral gate did not pass, so this result does not unlock a claim that
recognized AI identities recruit GLM-5.3's eval/deploy representation. The strongest supported
statement is:

> In a complete replication of Transluce's population-confidence protocol, GLM-5.3-Flash
> showed a small, directionally consistent but unresolved famous-AI confidence reduction
> (mean −0.365 points, 95% CI [−0.853, +0.121]), substantially attenuated relative to the
> released GLM-5.2 aggregate. Published-address variants retained a smaller resolved effect,
> indicating heterogeneous sensitivity to how an identity is presented.

This experiment measures stated confidence in hypothetical self-prediction. It does not show
that the model would behave differently in real deployments, recognize every identity, or use
an internal evaluation-awareness mechanism.
