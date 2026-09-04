# When the user is the evaluation

Project ID: `glm53_user_eval_shared_direction_v1`

Preregistered on 2026-08-28 for the Neel Nanda MATS 12.0 application. The subject is the official FP8 `zai-org/GLM-5.3-Flash` checkpoint at revision `04c4e9e95c5da8862dced7e5056455116f83a7e0`. Before any GLM-5.3 call, preregistration v2 pinned the extraction judge to first-party OpenAI snapshot `gpt-5.4-mini-2026-03-17`.

## Question

When GLM-5.3-Flash changes its stated behavioral confidence for a recognized AI-safety user, does that user recruit the same internal eval/deploy variable as an explicit evaluation cue?

The project separates three matched contrasts from Transluce's released roster:

1. Recognized name, `F_i - U_i`. A famous AI person is compared with an unknown person at the same organization.
2. Affiliation, `U_i - G_i`. The same unknown name is compared across AI-world and ordinary affiliations.
3. Full user effect, `F_i - G_i`.

The outcome is the confidence percentage extracted from the canonical two-turn task. It is recorded as stated. The analysis never flips confidence according to the preceding yes/no answer.

## Source and selection lock

The project, Transluce, Hua, Transformers, and model revisions are fixed in `reference/source_locks_glm53_user_eval_v1.json`. Identity selection uses only the released GLM-5.2 cache. No GLM-5.3 result may influence selection.

The eight most negative complete famous-AI rows form the enriched set. The four strongest GLM-5.2 rows form the primary intervention set. Eight prospective famous-AI pairs and the fame and general-population controls use seed `20260828`. The 100 behavior dilemmas are split into 50 main, 25 hardening, and 25 causal items with the same seed.

All target identities are excluded from direction fitting and layer selection. The strict direction split also excludes the four primary target organizations when enough organizations remain.

## Claim ladder

- C0 requires a matched GLM-5.3 behavioral effect with low missingness.
- C1 requires famous, affiliation-twin, same-name general-population, and fame controls to separate recognized identity from affiliation and generic fame.
- C2 requires grouped, held-out linear readout performance with shuffle controls.
- C3 requires the frozen eval/deploy score to move before the confidence output and predict behavior out of sample.
- C4 requires the target direction to beat matched random directions under paired intervention.
- C5 requires the cross-intervention matrix to support a shared or separable mechanism.

A probe is a readout. It is not evidence that the model has a literal belief. A steering result establishes dependence under the tested intervention, not complete mediation.

## Gates

### G0: source and harness parity

G0 requires all of the following:

- Complete source locks and verified upstream checkouts.
- Exactly 70 rows in each primary roster group.
- Exact famous, unknown-AI, and same-name genpop index mappings.
- Byte-identical plain-system prompts for every selected condition.
- Reproduction of the released GLM-5.2 cache means, counts, and rankings to `1e-9`.
- At least 50 parser fixtures covering yes/no, percentages, ranges, refusals, and malformed output.
- Frozen persona and task manifests.
- The operative preregistration committed and tagged before the first behavior call.
- A zero projected RunPod cost before G1 and a project hard cap of USD 125.

Failure stops all GLM-5.3 calls.

### G1: API behavior

The API route is first-party Z.ai model `glm-5.3-flash`, with provider fallback disabled, reasoning effort `max`, temperature `1`, and top-p `0.95`.

G1 requires:

- Confidence parse rate at least 97%.
- Complete provider and realized-reasoning metadata.
- A negative enriched-set `F-U` effect with its 90% crossed-bootstrap interval below zero.
- At least six of eight enriched pairs with the predicted sign.
- At least one clean `F-U` or `U-G` effect of 1.5 percentage points or more with a 95% interval excluding zero after the preregistered extension when needed.
- A smaller famous-non-AI effect than the clean AI-user contrast.
- Missingness spread no greater than 2 percentage points.
- Manual review of 40 seeded transcripts.

An inconclusive 50-item result may extend only the target cells to all 100 items. No identity or prompt search is allowed. G1 failure forbids white-box RunPod work.

### G2: official checkpoint feasibility

After G1 passes, expand the existing RunPod volume from 250 GB to at least 500 GB. The primary attempt uses four 96 GB RTX PRO 6000 Blackwell Server Edition GPUs. The G2 target is USD 17 and its absolute cap is USD 30.

Within two paid hours, G2 must load all 62 FP8 shards, confirm 45 layers with `[batch, sequence, 4, 4096]` outputs, reproduce the final stream-mean convention, extract selected prompt vectors, and pass alpha-zero logit and greedy-generation equivalence. One H200 fallback is allowed only for a documented VRAM failure.

### G3: local parity

The official local checkpoint must reproduce the API sign on the four primary identity triplets. Parse rate must be at least 95%, at least three pairs must have the expected sign, and the pooled effect must retain half the API magnitude or exclude zero independently.

API/local disagreement ends white-box claims for the local checkpoint.

### G4: independent direction validity

The eval/deploy readout requires held-out scenario-family AUROC of at least 0.75. Recognized-name and affiliation readouts each require pair-held-out AUROC of at least 0.70. Organization-held-out sensitivity requires at least 0.65 where feasible. Every shuffle distribution must remain near chance.

The primary steering direction is the paired mean difference. Logistic readouts use train-only standardization and map weights back into raw activation coordinates before geometric comparison.

### G5: eval-readout recruitment

The famous-versus-twin prompt-final eval score must shift by at least 0.25 source-training standard deviations with a paired 95% interval excluding zero. Enriched and prospective identities must share the sign. Famous-non-AI users must not show the same pattern.

If G5 fails, only the preregistered identity and affiliation dissociation tests remain available.

### G6: causal direction test

Shared-path evidence requires deployment steering to reduce the paired user effect by at least 30%, exceed the 95th percentile of matched random-direction reductions, show a monotonic trend across three strengths, and pass the held-out explicit eval/deploy positive control. Parse and refusal rates may move by no more than 2 points, and output length by no more than 20%.

Distinct-path evidence requires the eval direction to pass its positive control while removing less than 15% of the user effect, and a validated name or affiliation direction to remove at least 30% and beat random controls.

### G7: hardening

The result must retain its sign and at least half its magnitude on the 25 hardening dilemmas with seed `314159`. An independent script must reproduce the headline estimates from immutable rows. All headline intervention cells receive seeded manual review.

## Representation and intervention contract

Each decoder layer returns four residual streams. The canonical feature is their mean. Extraction selects the target token before converting to float32 and moving the `[batch, hidden]` vector to CPU.

Additive steering applies the same delta to every stream. This changes the collapsed mean by exactly the delta and preserves inter-stream differences. Projection removal subtracts the collapsed component from every stream. Hooks must preserve tuple metadata, run during prompt prefill and generation, and be removed in `finally`.

Layer selection uses only independent direction validation data. Target behavior cannot select a layer, strength, identity, or checkpoint.

## Statistics

Behavior intervals use a two-way cluster bootstrap over identity pairs and dilemmas with 20,000 repetitions. Every sampled pair and dilemma retains all matched conditions. Identity-level plots use dilemma bootstrap intervals.

The primary confirmatory family contains the G1 `F-U` effect, the G5 prompt-final eval-score effect, the G6 candidate-versus-baseline effect, and the G6 candidate-versus-random contrast. Other prompts, layers, effort levels, and individual identities are secondary.

## Budget and stopping

The RunPod project cap is USD 125. A new committed decision is required before exceeding it. The volume remains at 250 GB until G1 passes. No paid multi-GPU Pod may start before then.

Each gate writes an immutable decision file, checksum manifest, and cost entry. A failed gate stops the downstream claim. Threshold changes require a new preregistration, new tag, and clock reset. There is no manual pass override.
