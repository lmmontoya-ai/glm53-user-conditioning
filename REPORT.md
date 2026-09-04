# Report: split of the GLM-5.3-Flash user-conditioning study into its own repository

Date: 2026-09-03. Source: `Non-verbal-Eval-Awareness` at tag `glm53-user-eval-v23-final`, commit
`92fb062`. Target: `D:\research\mats\neel_2026\glm53-user-conditioning`, pushed on 2026-09-04 to
`https://github.com/lmmontoya-ai/glm53-user-conditioning` (private).

## Tree

```
glm53-user-conditioning/
  README.md  PROVENANCE.md  REPORT.md  pyproject.toml  uv.lock  .gitignore
  configs/    task.yaml  analysis.yaml  positive_control.yaml  context_blocks_v1.json  plots.yaml
              roles/PROTOCOL.md  roles/taxonomy.yaml
  data/       confirmatory/  discovery/  transcripts/  judgments/  audits/  power/
  src/glm53/  io.py  measure.py  bootstrap.py  decompose.py  roles.py  transcripts.py
              run_task.py  transluce_context_task.py  plots.py  judge_plots.py
  scripts/    01_reproduce.py  02_estimands.py  03_decompose.py  04_identities.py
              05_transcripts.py  06_positive_control.py  07_plots.py  08_judge_plots.py
              crop_tool.py  _common.py
  tests/      test_reproduce.py  test_decompose_and_transcripts.py
  outputs/    (generated; ignored)   figures/  (PNG + SVG + README.md)
  archive/mechanism/   unchanged copy of src, pipelines, tests, results, infra, source locks
  docs/       preregistrations/ (unchanged)   analysis_notes.md
```

History was not rewritten: `git filter-repo` is not installed. Files were copied and every copy
verified byte-identical to its source by sha256 (4,879 files, 0 mismatches). `PROVENANCE.md`
carries the path and version-label map.

## Reproduction

Every committed number reproduced from the committed raw scores with the new `measure.py`,
`bootstrap.py`, and `decompose.py`. Tolerances: 1e-9 on points, 1e-6 on interval endpoints.

| Stage | Checks | Failures | Largest absolute diff |
|---|---:|---:|---|
| 1 reproduce (confirmatory: interaction, components, group means, both CI endpoints of four contrasts, address effect, leave-one-out, dilemma splits; discovery: interaction, group means, five group CIs, paired F−U and its CI, medians and Mann-Whitney z) | 50 | 0 | 4.4e-16 |
| 3 decompose (choice, folded, choice-standardized points and CIs; matched estimate and retained cell counts) | 12 | 0 | 4.4e-16 |
| 5 transcripts (confidence-turn token and sentence interactions with CIs; evaluation-language and deference kappas, points, CIs) | 14 | 0 | < 1e-9 |

One discrepancy was found and fixed during stage 3, not by loosening tolerances: the committed
decomposition treats the yes/no answer and the stated confidence as valid independently (43 rows
have an answer but no confidence; 55 have a confidence but no usable answer). My first
implementation required both, which moved the choice interaction by 0.016 pp and the folded
interaction by 4e-5 pp. The rule is now documented in `decompose.outcome_matrices`.

Tests: `uv run pytest` runs 10 tests in about 17 s and pins the values above.

New numbers with no committed reference (labeled as such in outputs): the crossed-bootstrap
interval of the discovery interaction, −0.831 pp [−1.404, −0.257] (seed 20260829); the interval of
the same-answer matched estimate, −0.735 pp [−1.299, −0.160]; the Spearman correlations below.

## Stage outputs

- Estimands CSV (`outputs/estimands/estimands.csv`): six estimands per run with intervals,
  bootstrap p-values, Mann-Whitney p-values for F−G and Freal−G, identity and row counts.
- Identities: 129 famous-AI profiles (70 constructed, 59 published) with effects and twin-adjusted
  effects in both runs. Spearman rho of twin-adjusted effects, discovery versus confirmatory:
  0.41 [0.01, 0.56], n = 70. Constructed versus published address within the confirmatory run:
  0.56 [0.29, 0.75], n = 59.
- Role coding: protocol frozen first (`configs/roles/PROTOCOL.md`, sha256
  `dec630d0…380011`, recorded in every role output). The LLM coder ran on name and public role
  text only; result: scrutiny 38, business 20, capabilities 12, other 0, ambiguous 20. The blank
  human sheet is `outputs/roles/human_coding_template.csv`. `merge` refuses to proceed until each
  disagreement has a recorded resolution; `contrast` refuses without a merged sheet.
- Transcripts: confidence-turn replies are longer for famous AI figures in the four-group
  interaction by 12.0 visible tokens [5.3, 18.9] and 0.39 sentences [0.10, 0.67]. Explicit
  evaluation language: −0.013 [−0.131, 0.113] as a fraction of transcripts, kappa 0.894. Deference:
  0.010 [−0.035, 0.058] on a 0 to 2 scale, kappa 0.173 (passes the frozen rule only through 96.7%
  exact agreement). Caution and impression-management dimensions are listed as dropped. Six blinded
  matched sets were written for human reading (`outputs/transcripts/sample/`).
- Figures: three figures from stage outputs only, PNG at 300 dpi and SVG.

## Positive control (built, dry-run only, not executed)

| Item | Value |
|---|---|
| Rows | 21,000 (3 conditions × 70 general-population identities × 100 dilemmas) |
| Shards | 300 (one per condition and dilemma, so each identity gets one paraphrase family per dilemma) |
| Per-row cost | $0.000841 (confirmatory run: $28.94 over 34,400 rows, subject and judge calls) |
| Projected | $17.67, or $18.55 with a 5% retry allowance |
| Cap | $25.00 in `configs/positive_control.yaml` |
| Pinned checkout | commit and all six file hashes verified |

Execution needs `--execute`, `execute: true` in the config, and `OPENROUTER_API_KEY`; it aborts before
the first call if the projection exceeds the cap. The context block is appended to the pinned
system prompt through a wrapper task that registers synthetic persona keys and wraps the pinned
prompt builder; the wrapper was verified offline (system prompt text, sample ids, script shape).

Launched 2026-09-04 on the author's instruction with the OpenRouter key from an untracked file
outside this repository (read into the process environment only). Two problems surfaced in the
one-shard smoke run and were fixed before the full launch: Inspect's OpenRouter provider needs the
`openai` package (added to the dependencies), and Inspect hands the persona list to the wrapper
task already parsed as a list, which the wrapper had stringified (fixed). The smoke shard (neutral
block, dilemma 0, 70 identities) scored 70 of 70 rows; every model-call request carried the
identity line followed by the assigned block, with all four paraphrase families present. The full
run then started with that shard skipped as complete.

### Positive-control results (run completed 2026-09-04)

300 of 300 shards succeeded; 20,993 of 21,000 rows valid (6,999 neutral, 6,998 operational use,
6,996 model assessment). Actual API cost from the logs: $17.07 (projection $17.67). Wall time
about 75 minutes at five parallel shards. Outputs: `outputs/positive_control/analysis.json`,
`per_dilemma_model_assessment.csv`, schedule manifest and eval logs.

| Quantity | Value (pp folded confidence) | 95% interval |
|---|---:|---|
| model assessment − neutral (within run, genpop identities) | +1.08 | [+0.71, +1.46] |
| operational use − neutral | +0.46 | [+0.09, +0.83] |
| neutral block − confirmatory-run genpop rows (descriptive, cross-run) | +0.36 | [−0.04, +0.75] |
| Spearman rho, per-dilemma model-assessment effect vs scrutiny-identity profile (100 dilemmas) | −0.18 | [−0.40, +0.04] |

Condition means: neutral 83.01, operational use 83.47, model assessment 84.10. Intervals are
identity-and-dilemma bootstraps with 20,000 draws (seed 20260904); the rho interval resamples
dilemmas.

Reading: telling the model outright that the session is a model assessment moves stated confidence
on this task, but upward, by about one percentage point, and an ordinary-use note moves it upward by
about half a point. Famous scrutiny-role users move it downward (−0.74 pp against their twins in
the confirmatory run). The per-dilemma profile of the assessment effect does not correlate
positively with the per-dilemma scrutiny-identity profile; the point estimate is negative and the
interval includes zero. Both conditions the framing paragraph depended on therefore fail in the
direction that matters: explicit evaluation framing is not a weaker version of the identity effect.
The identity effect is better described as a response to who the user is (technical AI figures
versus AI business figures, per the secondary contrast) than as partial evaluation awareness. Note
that the neutral block itself sits slightly above the confirmatory run's general-population rows,
so some of the within-run shifts may reflect any added system-note text rather than its content;
the operational-use versus model-assessment difference (about 0.6 pp) is the cleaner comparison.

## Missing data

No file listed in the request was missing. Not copied by choice: the Inspect eval logs of both runs
(about 344 MB each), the judge event logs under `attempts/` (88 MB), and the `behavior_api/` pilots.
Three transcript files above 50 MB are present locally but untracked, with sha256 pointers in
`data/transcripts/POINTERS.md`.

## Secret scan

`detect-secrets` over the tree (excluding the virtual environment and lock file): 25,681 hex
high-entropy findings, all sha256 hashes in manifests, decision files, and data; 5 base64 findings
(an image digest, a run id, a storage path, and two fixture placeholders); 1 "secret keyword", a
placeholder key `sk-or-v1-abcdef…` in an archived test fixture. A regex pass for OpenRouter,
Anthropic, Hugging Face, RunPod, AWS, and GitHub key formats matched only those two fixtures. No
credential was found. No `.env` exists on this machine.

## Judge rounds

Backend: local Claude Code CLI (`claude -p`, model alias `opus`) with only file reading and the crop
command allowed, because no Anthropic API key exists on this machine; the API backend with a
tool-use crop loop is implemented in `judge_plots.py` and is selected automatically when a key is
set. Each round cost about $0.9 to $2.0 of Claude usage. Logs: `outputs/judge/<figure>/round_<n>.json`.

| Figure | Round | Numbers consistent | Requests | Applied | Rejected |
|---|---:|---|---:|---:|---:|
| 1 | 1 | no: caption gave one row count for both runs | 6 | 3 | 3 |
| 1 | 2 | yes | 6 | 4 | 3 |
| 1 | 3 | yes | 4 | 1 | 4 |
| 2 | 1 | yes | 6 | 5 | 1 |
| 2 | 2 | yes | 6 | 6 | 1 |
| 2 | 3 | yes | 5 | 3 | 2 |
| 3 | 1 | yes | 5 | 6 | 1 |
| 3 | 2 | yes | 3 | 2 | 1 |
| 3 | 3 | yes | 4 | 3 | 1 |

The one number-consistency failure (Figure 1, round 1) was a caption bug: it stated the
confirmatory row count as if it applied to both runs. It was fixed in the plotting code and did not
recur. Changes applied across rounds: per-run row counts and identity counts in captions; spelled-out
row labels; distinct marker shapes and end caps for the two runs; larger and darker caption and tick
text; legend inside the axes; a reference line at the confidence effect inside the yes-rate panel
with a caption sentence saying the yes-rate interval does not exclude it; the same-answer matched
estimate drawn in grey with its interval and labeled descriptive; the adjustment method named in the
Figure 3 caption; the number of bootstrap resamples stated; Figure 2's title changed to state the
lesson with rho and its interval, a y = x line, marker transparency, label repulsion with leader
lines, and caption sentences on the labeling rule and the absence of per-identity intervals.

Rejected in every round, with the reason logged: rewording the specified titles of Figures 1 and
3 (the judge holds that Figure 1's first clause claims component effects whose intervals include
zero, and that Figure 3's first clause reads as evidence of absence); putting the SD-unit magnitude
on an axis instead of in the caption; adding the F−G row to Figure 1; labeling more than the eight
most extreme identities in Figure 2. These are the author's decisions and are listed under open
questions. No round ended with zero change requests, so every figure used all three rounds.

## Open questions and assumptions

- Target path: the request used a placeholder; I created the repository beside the source checkout.
- Twin-adjusted effect: computed as the identity's effect minus its twin's effect (difference of
  identity means over valid dilemmas), which equals the mean paired difference when both are
  complete.
- Positive-control paraphrase assignment follows the drafted design (hash of condition, identity,
  and dilemma), which requires one shard per condition and dilemma; the alternative, hashing
  condition and identity only, would allow five-dilemma shards.
- Figure 1 title: the judge argued it overclaims because both component intervals cross zero in the
  confirmatory run. I kept the specified title and added the qualification to the caption; this is
  the author's call.
- Discovery run as held-out test for the role contrast: the discovery run was also used for judge
  rubric development in the original study, so it is held out for this hypothesis only.
- Role-coder and figure-judge backend: no Anthropic key exists here, so both ran through the local
  Claude Code CLI with all tools disabled for the coder and only file reading plus the crop command
  for the judge. The API backends are implemented and selected automatically when
  `ANTHROPIC_API_KEY` is set.

## Human review results (2026-09-04)

Modes 1 and 2 of the review tool were completed by the author; timers recorded 0.04 h and 0.16 h
of visible-tab time.

- Extraction audit: 40 of 40 rows, 8 per group, judged correct on both the yes/no answer and the
  confidence; no exceptions. Paste sentence in `outputs/review/summary.md`.
- Role coding: 70 of 70 identities coded blind (human sheet: scrutiny 31, business 22,
  capabilities 14, other 3, none marked ambiguous). Merge with the LLM sheet found 16 disagreements;
  all 16 were adjudicated in the tool and in every case the final category matched the LLM label.
  Final counts: scrutiny 39, business 19, capabilities 12, other 0.
- Predeclared contrast, mean twin-adjusted effect for scrutiny minus business, crossed
  identity-and-dilemma bootstrap, 20,000 draws (`outputs/identities/role_contrast.json`):

| Run | Primary (pp) | 95% interval | Drop `other` | Swap ambiguous to alternative |
|---|---:|---|---|---|
| Discovery (held-out test) | −1.416 | [−2.340, −0.493] | −1.416 [−2.347, −0.488] | −0.877 [−1.932, +0.159] |
| Confirmatory | −1.169 | [−2.082, −0.298] | −1.169 [−2.077, −0.288] | −1.033 [−1.972, −0.114] |

  Category means of the twin-adjusted effect, confirmatory run: scrutiny −0.74, capabilities −0.39,
  business +0.43 pp. The "swap ambiguous" sensitivity moves the 20 identities the LLM coder flagged
  as ambiguous to their alternative category (the human sheet flagged none); on the held-out
  discovery run that variant's interval includes zero. Figure 2 is now colored by the merged coding
  with role centroids; `outputs/identities/per_dilemma_scrutiny_profile.csv` exists for the
  positive control. Modes 3 and 4 were not done.

### Secondary contrast and mechanical robustness (declared post hoc, 2026-09-04)

Declared in `configs/roles/SECONDARY_CONTRASTS.md` after the primary result was seen; the primary
protocol file is unchanged and both declaration hashes are recorded in `role_contrast.json`.

| Contrast | Labels | Discovery (pp) | Confirmatory (pp) |
|---|---|---|---|
| scrutiny − business (primary) | merged coding (39 vs 19) | −1.42 [−2.34, −0.49] | −1.17 [−2.08, −0.30] |
| scrutiny − capabilities (secondary) | merged coding (39 vs 12) | −0.23 [−1.18, +0.73] | −0.34 [−1.36, +0.65] |
| scrutiny − business | mechanical, affiliation string only (27 vs 30) | −1.03 [−1.92, −0.13] | −1.24 [−2.14, −0.35] |
| scrutiny − capabilities | mechanical (27 vs 12) | −0.56 [−1.62, +0.50] | −0.95 [−1.99, +0.06] |

Scrutiny and capabilities identities do not separate on either run, with either labeling. The
defensible description is therefore "technical AI experts versus AI business figures": the model
states lower confidence for people known for technical AI work, whether that work is safety and
evaluation or capabilities, and higher confidence for AI executives, investors, and commentators.
The mechanical labels (keyword rules on the affiliation text, `configs/roles/mechanical_rules.yaml`,
no judgment) agree with the merged coding on 51 of 70 identities and give the same sign and an
interval excluding zero for scrutiny minus business on both runs.

Disclosures, also stored in `role_contrast.json`:

- The primary hypothesis was formed on the confirmatory run after inspecting per-identity means
  and naming the most negative identities; the discovery run was the held-out test for that
  hypothesis only.
- All 16 human-versus-LLM coding disagreements were adjudicated to the LLM label, so the final
  coding equals the LLM coding on every disputed identity. The human-only sheet is preserved in
  `outputs/roles/human_coding.csv`.
- While building the pipeline the agent ran the contrast code once with the LLM sheet standing in
  for the human sheet to verify the code path and saw the confirmatory value; those outputs were
  deleted and no reported number uses them.

## Review tool (added after the split)

`scripts/09_review_ui.py` serves `src/glm53/review_ui/index.html` and is the only component that
touches disk; standard library only, no network resources. Four modes write the files the
pipeline expects; timers per mode persist in `outputs/review/state.json`; a summary button writes
`outputs/review/summary.md`. `tests/test_review_ui.py` starts the server on a free port with
fixture files, posts one decision per mode, and checks the written columns. A read-only smoke run
against the real data returned 40, 70, 16 (6 existing plus 10 new quartets), and 160 items with
identities hidden and no name leaking through redaction.

Packet-format notes that shaped the tool, left as follow-ups rather than restructuring the packets:

- The 40-row packet stores the dilemma inside `scenario_script[0][1]` and the confidence question
  in `scenario_script[2][1]`; `transcript_messages` holds only a truncated placeholder. The packet
  has `refused_estimate` but no separate `binary_refused` flag, so the tool shows the fields present.
- The 160-row packet carries `system_profile`, which reveals the identity; the tool hides it until
  the mode is complete. The true group is only in `human_audit_selection_private.json`, which the
  tool never reads.
- The mode 4 verdict keys (agree, disagree on evaluation, disagree on deference, both) do not record
  the reviewer's own deference value, so the kappa against each judge defines the human label as the
  first judge's label when agreeing and as flipped (binary) or "other" (ordinal) when disagreeing.
  This is stated in the summary. A per-item value entry would give a cleaner kappa.
- Mode 2 adds keys q/w/e/r to set the alternative category when an identity is marked ambiguous;
  the sensitivity analysis needs that column and the request listed no key for it.
- In this roster every published-address profile has the same role text as its constructed twin, so
  the merged row shows one text.
- Mode 3 selects scrutiny identities from the human coding once it exists (adjudicated finals
  override it) and falls back to the LLM sheet only before mode 2 is done. The quartet set is cached
  in `outputs/review/mode3_quartets_private.json` and rebuilt when the role source improves, but
  only while no tags or notes exist. Do not open that file before pressing reveal.

## Follow-ups noticed but not done

- Human steps (TODO for the author): fill `outputs/roles/human_coding_template.csv` and save it as
  `human_coding.csv` before looking at `outputs/identities/identities.csv`; then run `merge`,
  resolve disagreements in `merged_coding.csv`, and run `contrast`. Read the 40-row confirmatory
  packet and the 160-row decomposition packet in `data/audits/`; read
  `outputs/transcripts/sample/reading_blinded.md`. Decide whether to run the positive control.
- A scratch run of `merge` and `contrast` was executed with a copy of the LLM sheet as the human
  sheet to verify the code path; its outputs were deleted. This means the LLM-only contrast on
  both runs was computed once and its confirmatory value was seen on screen. The protocol forbids
  using the LLM-only sheet for any reported figure or contrast, and none does.
- The original checkout's `.venv` was modified once by a `uv sync` that ran in the wrong working
  directory; it removed ad hoc packages, and I reinstalled `inspect-ai==0.3.223` and `statsmodels`.
  No tracked file in the checkout changed (`git status` clean for `pyproject.toml` and `uv.lock`).
- The archive's `v10/analysis.py` imports `src.probe.sequence_linear` from the paper repository;
  it will not run from the archive.
- The v6 development judge rows (`data/judgments/v6_development_v2/`) are copied but unused.
- Removal of the study from the paper repository is left to the author.
