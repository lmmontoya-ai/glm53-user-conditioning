# Current state

V11 has reached a preregistered offline stop. The latent-scenario dataset,
structural audit, exact-tokenizer audit, and untouched lexical holdout passed.
The frozen semantic validator failed its control-calibration threshold, so no
GLM activation extraction or RunPod compute is authorized under V11.

## Completed work

- Preregistration commit: `10f9dcc2187fa59bd50a3d45a97ac2e1f22016e6`
- Preregistration tag: `glm53-user-eval-v11-preregistered`
- Dataset: 576 rows, including 480 binary rows, 64 neutral controls, and 32
  factorial-calibration rows
- Structural audit: passed
- Exact GLM tokenizer and chat-template audit: passed for all 240 binary pairs
- Untouched lexical holdout: passed every frozen shortcut threshold
- Semantic judge: completed all 576 rows through OpenRouter's OpenAI route
- Accepted-row semantic cost: `$0.68348775`
- Paid GLM calls in V11: zero
- Combined offline analysis: written and hash-bound in the stop-evidence chain
- Raw dataset rows, exact-tokenizer audit, development model bundle, all 576
  accepted semantic rows, and all four preserved attempt logs: included in Git
  evidence rather than retained only on this workstation

The untouched lexical results were near chance, as intended:

- Character 3-to-5-gram AUROC: `0.4873046875`
- Decisive-fact-deleted character AUROC: `0.5`
- Decisive-fact-deleted word AUROC: `0.5`
- Structural metadata AUROC: `0.5`
- Word TF-IDF AUROC: `0.5`
- Keyword AUROC: `0.5`

## Semantic stop

The frozen semantic results were:

- Binary rows: `471/480 = 0.98125`, above the `0.90` threshold
- Untouched final counterfactual: `58/64 = 0.90625`, above the `0.90`
  threshold
- All controls: `77/96 = 0.8020833333333334`, below the `0.90` threshold
- Neutral controls alone: `59/64 = 0.921875`
- Factorial calibration alone: `18/32 = 0.5625`

The failure is concentrated in the mixed-purpose factorial rows. The frozen
dataset treats mixed operational and model-assessment use as `ambiguous`, but
the judge often inferred a primary purpose and returned `eval` or `deploy`.
This is a semantic-contract and judge-calibration problem, not an API route or
parser problem. The gate remains failed. We will not relabel rows, lower the
threshold, remove the factorial controls, or rerun accepted judgments.

## Provenance

All 576 accepted rows have unique sample IDs, request hashes, and response IDs.
Every accepted response used `openai/gpt-5.4-mini`, low reasoning, the OpenAI
provider, and disabled fallbacks. The preregistration tag predates every call.
No accepted row was overwritten or queried again during recovery.

The preserved retry ledger begins after the first 27 accepted rows. Failed
malformed responses also did not retain usage records. The reported cost is
therefore the exact cost of accepted rows, not a complete all-attempt bill.
These are provenance disclosures. They do not change the failed scientific
decision.

## Administrative closure

Two real human reviewers were unavailable before the application deadline. A
post-outcome administrative amendment closes V11 without claiming that this
requirement passed. The terminal decision is
`semantic_validation_failed_manual_human_review_unavailable`.

The user supplied aggregate results from a blind, manual-style AI diagnostic
review of the 128-row primary packet and six-row supplemental packet. This
diagnostic reportedly matched the frozen acceptable-label contract on all 128
primary rows. It remains nonhuman, non-gating evidence. The row-level files
were not available in this workspace, so only the reported aggregate is
preserved and marked unverified.

V11 remains failed and cannot authorize GLM or RunPod compute. A new V12 may
reuse the frozen text under a prospective preregistration that extracts the
four decisive facts and derives the class in code.

## Infrastructure and credentials

The latest read-only RunPod check found zero Pods and zero Serverless endpoints.
The 500 GB network volume `a9diryunoj` remains and is the only RunPod charge.
The live account rate was `$0.049/hour` at the check. No paid V11 worker was
created.

The obsolete credential-bearing RunPod template was deleted. The user chose to
keep the dedicated S3 credentials until formal project closure, then rotate
them. Credential rotation remains a terminal cleanup item and must not happen
earlier. No credential value is stored in project evidence.

## Current immutable analysis anchors

- Combined offline analysis SHA-256:
  `ec819128a06accddd259da83308126e12c6437b1ca3486990fe2393f48bee818`
- Full semantic-stop evidence manifest SHA-256:
  `1d6d122ed7af4f34dc91ea3bd3807ffa38467fd8f04b8a1c8f58d4b2abdb467a`
- Compact semantic-stop summary SHA-256:
  `813dcd2c3ada3c14fbb33815e8a5e6b679d4d0849927bfc0fe4bf8024fc14217`
