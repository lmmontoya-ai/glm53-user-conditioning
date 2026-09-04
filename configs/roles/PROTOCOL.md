# Blind role-coding protocol for famous-AI identities

Written before any role-coding code. Its sha256 is recorded in every output of stage 4 that
uses role labels. Changing this file after coding starts invalidates those outputs.

## Purpose

Test whether the confidence shift for famous AI figures is larger for people whose public role
is scrutiny of AI systems than for people whose public role is business. The hypothesis was formed
while looking at the confirmatory run. The discovery run is therefore the held-out test set for
this hypothesis and is analyzed first.

## Units

One unit is one famous-AI identity from the roster (70 constructed-address profiles, keys
`fai2_*`; the 59 published-address profiles `fai2r_*` share a person and receive the same code).
The outcome is the twin-adjusted effect: the identity's mean centered folded confidence minus the
same quantity for its unknown twin at the same roster index, in percentage points.

## Taxonomy

Four mutually exclusive categories, frozen in `taxonomy.yaml`: `scrutiny`, `capabilities`,
`business`, `other`. Decision rules are in the same file.

## Blinding

1. Coders receive only `name` and the public role text (`affiliation`, `org`) from
   `personas2.json`. No score, effect, run, group mean, or figure is shown or available.
2. The LLM coder is called through an API with a prompt that contains only those fields and the
   taxonomy. It has no tool access and no access to this repository.
3. The human coder fills in `outputs/roles/human_coding_template.csv` before looking at any
   per-identity effect. The sheet has no effect columns.

## Sheets

- `outputs/roles/llm_coding.csv`: columns `persona_key, name, role_text, category, ambiguous,
  alternative_category, justification, model, protocol_sha256`.
- `outputs/roles/human_coding_template.csv`: columns `persona_key, name, role_text, category,
  ambiguous, alternative_category, notes`. Blank values to be filled in by the human coder.
- Merged sheet `outputs/roles/merged_coding.csv`: one row per identity with both codes, a
  `final_category`, and a `resolution` note. The merge step writes `final_category` only where
  both coders agree. Every disagreement must have a non-empty `resolution` and `final_category`
  entered by hand in the merged sheet before the contrast runs.

## Analysis order

1. Merge; refuse to compute anything until every disagreement has a recorded resolution.
2. Discovery run: mean twin-adjusted effect for `scrutiny` minus `business`, crossed
   identity-and-dilemma bootstrap 95% interval (20,000 draws, seed from `analysis.yaml`).
3. Confirmatory run: same quantity, reported second.
4. Sensitivity: drop `other`; then move every identity flagged `ambiguous` to its
   `alternative_category` and recompute both runs.
5. Report each result with the number of identities per category.

## What is not allowed

- Re-coding after seeing effects.
- Using the LLM-only sheet in any figure or contrast.
- Adding categories or changing decision rules after the first coding sheet is written.
