# GLM-5.3 Transluce population-confidence replication: methodology parity

This run is a complete replication of the released Transluce **population plain-mode
confidence lane**, not of every experiment in the user-awareness paper. It executes the
pinned Transluce task and scorer directly from commit
`d1b9c3573470f50495202795c044bd72f72ee6e5`.

## Scientific treatment change

The intended scientific change is only the subject endpoint:

- reference: `openrouter/z-ai/glm-5.2`;
- replication: `openrouter/z-ai/glm-5.3-flash`;
- replication route: OpenRouter pinned to Novita, FP8, no provider fallback;
- reasoning effort remains `high`.

The subject checkpoint/service necessarily differs because the model differs. Provider
metadata is therefore audited on every call rather than assuming name-level equivalence.

## Preserved source methodology

- Original `evals/pmisaligned/task.py@pmisaligned` implementation and scorer.
- Original `core/personas2.json` roster and ordering.
- Five original population groups: 70 general-population, 70 unknown-AI, 70 famous-AI,
  59 real famous-AI, and 70 famous-non-AI personas.
- The source's anonymous condition repeated within each group.
- The first 100 `dailydilemmas.json` items in source order.
- `mode=plain`, `warmup=false`, `confidence=true`,
  `confidence_style=confidence`, and `seed=all`.
- Original two-turn subject interaction and confidence measurement.
- Original subject sampling contract: 8,000-token limit, with temperature, top-p, and
  generation seed omitted so the provider defaults apply.
- Original model-display-name derivation, which renders the assistant as `GLM-5`.
- Original `openrouter/openai/gpt-5.4-mini` judge, low reasoning, 2,000-token limit,
  prompts, parser, refusal handling, and missing-score behavior.
- Original folded-confidence transform, per-dilemma general-population centering, and
  person-level deltas.
- Original group median, source-indexed IQR, tie-aware Mann–Whitney comparison with the
  general-population distribution, and fixed 28-test Bonferroni correction. This final
  aggregation contract was restored prospectively in preregistration v6.2 before any
  production score was inspected.

## Infrastructure-only differences

The source's large group jobs are divided into immutable five-dilemma shards. Five shards
run concurrently with 40 connections apiece, for a fixed aggregate ceiling of 200.
Successful Inspect logs are never overwritten and are skipped on resume. Sharding changes
call scheduling, artifact boundaries, and failure recovery; it does not change prompts,
personas, dilemmas, model parameters, or scoring.

## Scope and sample count

The frozen schedule contains 100 shards and 34,400 scientific rows:

`(70 + 70 + 70 + 59 + 70 + five repeated anonymous conditions) × 100 dilemmas`.

Each row includes the original main subject turn and confidence follow-up. This is the
appropriate direct test of whether the released GLM-5.2 population-confidence result
replicates in GLM-5.3-Flash. It does not cover the paper's other behavioral lanes, such as
grading or separate verbal-awareness analyses.
