# Semantic stop record

## Decision

V11 failed the preregistered semantic control gate. This permanently blocks the
V11 RunPod and GLM stages. The binary scenarios and untouched final
counterfactual were understandable, but the control set did not reach the
frozen 90 percent acceptance threshold.

The result does not reject the confirmed GLM-5.3 behavioral interaction. It
shows that this version of the proposed eval/deploy source instrument does not
yet have a sufficiently clear semantic contract for paid activation work.

## Results

| Set | Correct or acceptable | Total | Rate | Frozen threshold | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Binary scenarios | 471 | 480 | 0.98125 | 0.90 | Pass |
| Final counterfactual | 58 | 64 | 0.90625 | 0.90 | Pass |
| All controls | 77 | 96 | 0.8020833333333334 | 0.90 | Fail |
| Neutral controls | 59 | 64 | 0.921875 | Descriptive | Pass descriptively |
| Factorial calibration | 18 | 32 | 0.5625 | Descriptive | Failure source |

Within the factorial calibration set:

- Clean evaluation rows: 6 of 6 accepted
- Clean deployment rows: 5 of 6 accepted
- No-status rows: 2 of 2 accepted
- Mixed operational and assessment rows: 5 of 18 accepted

For the 13 rejected mixed rows, the judge returned `eval` nine times and
`deploy` four times. Eight of nine mixed factor cells received the same forced
label in both formal and casual renderings. This makes random noise an unlikely
explanation.

## Diagnosis

The primary problem is a mismatch between the dataset rule and the judge rule.
The dataset labels every mixed operational-plus-assessment case as ambiguous.
The prompts state both sets of facts but do not say which purpose is primary.
The judge rubric asks for the primary purpose, so the judge often chooses one.

A secondary problem is wrapper contamination. Some failed rationales cite the
meta instruction to inspect a model-input context, or cite the downstream task,
instead of using only the scenario facts. Four of five neutral misses had judge
confidence of at least 0.95.

A later dataset version should ask the validator to recover only factual
variables such as recipient, operational consequence, model scoring, and model
comparison. Code should then derive the class from a frozen Boolean rule. The
judge wrapper should avoid model-evaluation language.

## Route and retry audit

- Expected rows: 576
- Stored rows: 576
- Missing, extra, or duplicate sample IDs: 0
- Duplicate response IDs: 0
- Request-payload or request-hash mismatches: 0
- Provider substitutions: 0
- Accepted response route: OpenAI only
- Accepted response model: `openai/gpt-5.4-mini`
- Fallbacks: disabled
- Reasoning: low, with nonzero reasoning tokens
- Accepted-row cost: `$0.68348775`
- Accepted-row tokens: 253,375 prompt and 109,657 completion

The first 27 accepted rows predate the earliest preserved attempt log by about
67 seconds. Later attempts added 83, 307, 98, and 61 missing rows. Three attempts
stopped on malformed JSON, then the fourth completed the table. The runner
validated existing request hashes and skipped existing rows, so no accepted row
was regenerated. Failed-response usage was not persisted, which prevents exact
reconstruction of total all-attempt spend.

## Immutable anchors

- Dataset rows SHA-256:
  `3efc88ee8c40de51521867bdb2896805e673b29aa470e5bf829ac33f89b7e46c`
- Semantic report SHA-256:
  `1c57cb00b19962d7f6d1565471bdfe977788f6ce44d7f00174eb3d4d13d95d9c`
- Canonical sorted semantic-row manifest SHA-256:
  `35d7887d1048c5b59b285c6dee240f0cfd202598df69d877663aa925a4f997da`
- Preserved attempt-log manifest SHA-256:
  `605e7b2dec13b51dbc1e05b565bfa26d661157eb3564da582c5b635ef220b8e1`

The full evidence builder records per-file hashes. The row-set and attempt-set
hashes above were independently recomputed before this status record was
written.

- Full semantic-stop evidence manifest SHA-256:
  `1d6d122ed7af4f34dc91ea3bd3807ffa38467fd8f04b8a1c8f58d4b2abdb467a`
- Compact semantic-stop summary SHA-256:
  `813dcd2c3ada3c14fbb33815e8a5e6b679d4d0849927bfc0fe4bf8024fc14217`
- Combined offline analysis SHA-256:
  `ec819128a06accddd259da83308126e12c6437b1ca3486990fe2393f48bee818`

The full manifest binds 18 core offline artifacts, including the combined
analysis, all 576 accepted judge rows, all four preserved attempt logs, and the
six-row supplemental packet. It scanned 598 inputs for credential patterns and
found none. The raw rows, logs, dataset, tokenizer audit, and development model
bundle are included in the Git evidence handoff.

## Authorization

All of the following remain false under V11:

- New GLM forwards
- RunPod compute
- Source activation extraction
- Local proxy scoring
- User recruitment
- Steering

Human review can explain the failure but cannot change this authorization.

## Administrative closure without human review

Two real human reviewers were unavailable before the deadline. The
post-outcome amendment does not waive or satisfy that requirement. It permits
V11 to close with the exact terminal state
`semantic_validation_failed_manual_human_review_unavailable` while keeping
every paid authorization false.

The user reported a blind AI diagnostic review of 128 primary rows and six
supplemental rows. The reported primary review matched the frozen acceptable
labels on all 128 rows. This evidence is recorded under
`status/ai_diagnostic_review/` and is explicitly ineligible for the official
human-review merge. The row-level files were not present in the workspace, so
the repository records only the user-supplied aggregate.

V12 may test a new fact-extraction validator against the unchanged 576 prompts.
That validator must pass its own preregistered automatic gate before any GLM
forward or RunPod allocation.
