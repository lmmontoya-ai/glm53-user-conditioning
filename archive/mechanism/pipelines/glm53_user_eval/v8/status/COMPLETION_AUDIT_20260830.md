# GLM-5.3 v8 completion audit

Audit date: 2026-08-30

This audit compares the current repository and artifacts with the full v8
specification. It does not infer completion from passing unit tests. The result
is straightforward: M0 and M1 are proven, paid execution never reached M2, and
the current M2--M8 supervisor still has several gaps that must be fixed under a
new preregistration amendment before execution resumes.

## Current proof by gate

| Gate | Status | Authoritative evidence |
| --- | --- | --- |
| M0 | Proven | `decisions/m0_decision.json`; preregistration SHA-256 `a56e70d0...`; parent commit and 100 v7 log hashes revalidated locally on 2026-08-30 |
| M1 | Proven | `m1/proxy_contract.json`; 76 passing tests; cache and codebook hashes recorded in the paused-state document |
| M2 | Not run | No `m2/runtime_report.json` or `decisions/m2_decision.json`; every Serverless attempt stopped before a model forward |
| M3 | Not run | No eval feature, direction, shuffle, holdout, or M3 decision artifact |
| M4 | Not run | No user feature, recruitment, or M4 decision artifact |
| M5 | Not run | No local proxy rows, parity report, or M5 decision artifact |
| M6 | Not run | No pilot rows, alpha selection, positive-control result, or M6 decision artifact |
| M7 | Not run | No confirmation rows, control comparison, or M7 decision artifact |
| M8 | Not run | No hardening rows, independent verification, manual audit, or final evidence bundle |

The absence of M2--M8 artifacts is expected because the user stopped execution
while the final worker was still preparing its cached model. It is not a
negative scientific result.

## Offline verification completed

The following commands passed from clean commit
`4f2a43eec89211e699ade4a6dd578b37e53514b0`:

```text
uv run pytest -q tests/glm53_user_eval_v8
76 passed

uv run ruff check pipelines/glm53_user_eval/v8 src/glm53_user_eval/v8 tests/glm53_user_eval_v8
All checks passed

uv run python pipelines/glm53_user_eval/v8/run.py validate-prereg
M0 passed; parent, 100 logs, schedule, codebooks, runtime, and direction splits passed
```

These checks prove the covered local contracts. They do not prove GPU runtime,
probe validity, recruitment, parity, intervention behavior, or the final audit.

## Implementation gaps found

### Gate decisions after M0 do not hash their inputs

`src/glm53_user_eval/v8/science.py::write_decision` calls
`decision_payload(..., inputs={}, ...)`. The v8 specification requires every
gate decision to name and hash its inputs. M2--M8 could currently write a valid
looking decision without an artifact lineage record.

Required fix: create a per-gate input manifest, hash every source artifact, and
store those hashes in the decision. The verifier must reject missing or changed
inputs.

### The M7 positive control is hard-coded

`analyze_causal_arms` currently writes:

```python
"positive_control_passed": True
```

This can let M7 pass without running the preregistered held-out eval/deploy
intervention control. That is a load-bearing error.

Required fix: run the actual held-out forced-choice positive control at the
frozen layer, direction, scope, and alpha. Persist its rows and report. Pass its
verified result into M6 and M7 rather than synthesizing a boolean.

### Pilot alpha selection does not implement the frozen rule

The supervisor chooses the largest positive target delta before it evaluates
random controls and allowed-label-mass constraints. The specification says the
eligible alpha must first pass the positive control, quality checks, and random
comparison, with smaller absolute alpha breaking ties.

Required fix: evaluate every preregistered alpha against all eligibility checks,
then apply the lexicographic rule once. Add boundary tests where the largest
target movement is ineligible but a smaller alpha is valid.

### M4 omits preregistered baselines and confound checks

The current recruitment report calculates the four-group frozen-probe
interaction and subset signs. It does not produce the required direct-prompting
baseline, random-direction distribution, prompt-length regression, or token
frequency regression.

Required fix: add immutable result tables and reports for these baselines. They
may remain secondary, but the final report must not claim they were tested until
the artifacts exist.

### M5 omits several proxy-validity outputs

The current report includes the interaction, component contrasts, codebook
effects, row-level Spearman correlation, allowed mass, and allowed-argmax rate.
It does not report person-level correlation, conditional entropy summaries,
argmax mapped confidence, or the preregistered out-of-sample calibration error.

Required fix: add the missing descriptive metrics. If isotonic calibration is
used, freeze its calibration split and keep target-condition labels out of the
fit. The uncalibrated proxy remains primary.

### The distinct-path branch is not implemented

The supervisor stops when M4, M5, or M7 fails. It never runs the preregistered
name or affiliation intervention branch even when those directions pass their
held-out validity gates and the eval direction passes its positive control.

Required fix: encode a machine decision that unlocks the distinct-path matrix
only under the preregistered conditions. Do not add new directions after target
results are visible.

### The independent verifier is incomplete

The verifier recomputes selected M3, M5, M7, and M8 quantities. It does not
independently recompute:

- the M4 recruitment bootstrap and decision;
- the M6 alpha-selection rule;
- the 20-control rank and add-one probability from immutable rows;
- all gate classifications;
- artifact hashes and decision input lineage.

Required fix: extend the verifier without importing primary analysis or decision
modules. Add import-isolation tests and fail closed on any missing required
artifact.

### M8 can pass without the required human audit

The current supervisor can write a passing M8 decision after hardening and the
partial independent verifier. There is no audit-packet builder, completed manual
audit schema, or M8 check for the review quota.

Required fix: split M8 into two phases. The GPU supervisor should write
hardening results and stop with `manual_audit_pending`. A local command should
build the seeded packet. Only a completed, schema-valid manual audit may unlock
the final M8 decision.

### Final reporting and evidence commands are absent

The CLI has no command that builds the required geometry/cross-decoding report,
manual audit packet, artifact hash audit, final evidence JSON, or final report.
There is also no final decision-to-claim mapper.

Required fix: add deterministic commands and tests for:

- post-decision geometry and cross-decoding;
- score-blind manual audit packets;
- complete SHA-256 evidence manifests;
- independent restoration checks;
- a final report that names the highest passed claim and its limits.

### The final Serverless bootstrap has not reached the handler

Template `s452pdfo5f` fixed the observed GHCR pull failure by using the exact
official Docker Hub base-image digest. Its first worker verified the image and
reached cached-model initialization, then the user requested shutdown. The
bootstrap, checkout sanitization, model load, and M2 runtime remain untested on
that transport path.

Required fix: resume with one job only, preserve the current exact pins, and
require the checkout-sanitization report plus M2 before any scientific gate.

## Required pre-resume amendment

Do not resume paid execution from preregistration v1.3 as if this audit had found
no issues. Create and tag a v1.4 amendment before the next paid model call. It
should state that no v8 target forward occurred under v1.3 and should change
only implementation and audit requirements needed to enforce the existing
scientific design.

The amendment should lock:

1. hashed inputs for every gate decision;
2. the explicit eval/deploy positive-control schedule and pass rule;
3. the corrected pilot alpha-selection algorithm;
4. the M4 baseline artifacts and M5 descriptive metrics;
5. the distinct-path unlock decision;
6. the expanded independent verifier;
7. the manual-audit-pending state and final M8 unlock;
8. the final evidence and report schemas;
9. the Docker Hub bootstrap template and exact image digest;
10. the same identities, tasks, model revision, proxy codebooks, thresholds, and
    budget limits already frozen in v1.3.

## Completion decision

The v8 goal is active and incomplete. The current repository proves the
behavioral parent, preregistration, proxy contract, and shutdown state. It does
not yet prove any white-box mechanism claim. Paid compute must remain stopped
until the user explicitly resumes it and the v1.4 audit amendment is committed
and tagged.
