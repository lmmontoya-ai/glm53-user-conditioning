# v11 latent source instrument

V11 replaces the invalid explicit-framing source bank diagnosed in V10. It
does not revisit the confirmed V7 behavioral interaction and does not authorize
user-recruitment or steering work by itself.

The binary pair members contain the same task, nuisance cues, aliases, and
lexical inventory. They differ in the relations that determine who receives
the response and what the response changes. Generator families and downstream
tasks are disjoint across train, validation, test, development-counterfactual,
and final-counterfactual splits.

The offline sequence is:

```powershell
uv run python pipelines/glm53_user_eval/v11/run.py validate-prereg
uv run python pipelines/glm53_user_eval/v11/run.py plan
uv run python pipelines/glm53_user_eval/v11/run.py build-dataset
uv run python pipelines/glm53_user_eval/v11/run.py audit-structure
uv run python pipelines/glm53_user_eval/v11/run.py audit-tokenizer `
  --tokenizer-root artifacts/glm53_user_eval/v11/tokenizer_snapshot
uv run python pipelines/glm53_user_eval/v11/run.py fit-text-development
uv run python pipelines/glm53_user_eval/v11/run.py evaluate-text-final
uv run python pipelines/glm53_user_eval/v11/run.py decide-lexical
uv run python pipelines/glm53_user_eval/v11/run.py build-manual-packet
uv run python pipelines/glm53_user_eval/v11/run.py semantic-judge
uv run python pipelines/glm53_user_eval/v11/run.py analyze-semantic
uv run python pipelines/glm53_user_eval/v11/run.py build-offline-analysis
uv run python pipelines/glm53_user_eval/v11/run.py prepare-human-review `
  --packet-kind all `
  --reviewer-1-id <reviewer-1-id> `
  --reviewer-2-id <reviewer-2-id>
# Two humans complete their private primary and supplemental sheets.
uv run python pipelines/glm53_user_eval/v11/run.py merge-human-reviews `
  --packet-kind primary `
  --reviewer-1-completed <reviewer-1-primary.csv> `
  --reviewer-2-completed <reviewer-2-primary.csv>
uv run python pipelines/glm53_user_eval/v11/run.py validate-manual `
  --completed-manual-audit artifacts/glm53_user_eval/v11/offline_audit/manual_completed.csv
uv run python pipelines/glm53_user_eval/v11/run.py build-offline-analysis
uv run python pipelines/glm53_user_eval/v11/run.py verify-offline
uv run python pipelines/glm53_user_eval/v11/run.py decide-text
```

Only a passing `decide-text` artifact permits paid execution. The claim-grade
route is one load-once command:

```powershell
uv run python pipelines/glm53_user_eval/v11/run.py paid-ladder `
  --model-path <verified-local-snapshot> `
  --confirm-spend
```

## Observed offline stop

The completed V11 run passed the structural, tokenizer, lexical, binary
semantic, and final-counterfactual checks. It failed semantic control
acceptance at `77/96 = 0.80208`, below the frozen `0.90` threshold. No V11 GLM
forward or RunPod worker was started.

`build-offline-analysis` writes the preregistered combined artifact at
`artifacts/glm53_user_eval/v11/offline_audit/analysis.json`. It binds the
structural, tokenizer, text-baseline, lexical, semantic, and review artifacts.
It records human review as pending until the validated reports exist. The
combined analysis never grants paid-compute authorization. Run it again after
the primary and supplemental human reviews finish so their validated reports
replace the pending status in the same deterministic record.

Build the immutable stop evidence with:

```powershell
uv run python pipelines/glm53_user_eval/v11/build_supplemental_manual_packet.py
uv run python pipelines/glm53_user_eval/v11/build_semantic_stop_evidence.py
```

Two humans must still complete the primary 128-row packet for audit closure.
The six-row supplemental packet covers semantic disagreements outside the
primary packet and is diagnostic only. See
`status/MANUAL_REVIEW_HANDOFF.md` for the exact process. The review workflow
gives each person a separate sheet with no columns for the other person's work.
Its merge commands recheck the frozen packet hashes, exact reviewer IDs, row
order, and prompt text. A third person receives only disagreement rows when
adjudication is needed.

After both completed reviews exist, validate them, run `verify-offline`, and
run `decide-text`. A completed scientific failure now produces a machine-readable
negative decision with every paid authorization set to false.

The paid ladder loads the exact official FP8 checkpoint once. It extracts the
source features, fits the frozen development readouts, benchmarks exact
16-worker and 32-worker permutation batches, finishes all 1,000 permutations
only when the remaining ladder fits with 30 percent headroom, and opens the
activation holdout once. A passing source decision conditionally unlocks the
6,387-row local confidence proxy. A passing proxy conditionally unlocks the
2,240-row frozen user-recruitment surface. Each conditional stage has a fresh
measured throughput and budget gate. Steering and early-CoT generation are
outside V11.

The source-analysis subcommands remain available for non-paid tests and
recomputation. They are not a substitute for the load-once paid ladder.

The paid command also requires `GLM53_V11_DEADLINE_UTC` and `RUNPOD_POD_ID` in
the environment. Run it through `infra/runpod/new_glm53_v11_source_pod.ps1`,
which checks the clean preregistration tag, live price, $29.50 compute cap,
$15 reserve, 110-minute deadline, signed input bundle, and external deletion
watchdog before creating a Pod. RunPod's injected Pod-scoped API key drives a
second deadline guard inside the Pod. That guard deletes the exact Pod at the
deadline even if the workstation is unavailable. Normal terminal paths also
upload their last evidence and request self-deletion.

The launcher requires a user-set
`RUNPOD_S3_CREDENTIAL_ATTESTED_AT_UTC` timestamp no more than 24 hours old.
It performs a signed read probe and an upload/download hash round trip before
creating the Pod. The S3 key exists only in the transient Pod environment. The
launcher records that the user must rotate the key after the Pod is deleted.

Machine decisions never authorize the final recruitment claim. The paid
ladder writes a blank 85-row review template covering 40 proxy rows, 32
recruitment rows, and all 13 technical errors. A human completes a separate
JSONL file, then runs:

```powershell
uv run python pipelines/glm53_user_eval/v11/run.py authorize-downstream-claim `
  --completed-downstream-review <completed-review.jsonl>
```

The command checks every row, the human attestation, all source-row hashes,
and the source, proxy, and recruitment decisions and independent verifiers.
It writes `final_claim_authorization.json` only when all 85 checks and all
positive machine gates pass. It cannot authorize early-CoT or steering claims.
