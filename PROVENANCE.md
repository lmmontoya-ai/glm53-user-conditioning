# Provenance

## Source

- Repository: `github.com/lmmontoya-ai/Non-verbal-Eval-Awareness`
- Tag: `glm53-user-eval-v23-final`
- Commit: `92fb06220d124c6165852355b3b26b781ed9fe86`
- Copied on 2026-09-03 from a local checkout of that tag whose ignored `artifacts/` tree was
  populated. `git filter-repo` was not installed, so history was not rewritten; files were copied
  and every copy was verified byte-identical to its source by sha256 (4,879 files, 0 mismatches).
  The archive copy excludes `__pycache__` directories and `.pyc` files, which were never tracked.

## Version labels

The source repository numbered stages v1 through v23. User-facing files in this repository use
plain names. The map:

| Old label | What it was | Where it lives now |
|---|---|---|
| v1 to v5 | early gates, roster pilots, API budget checks | `archive/mechanism/pipelines/glm53_user_eval/configs/`, `status/` |
| v6 (`transluce_exact_v6`) | discovery run: full roster, all 100 dilemmas, GLM-5.3-Flash | `data/discovery/` |
| v7 (`transluce_interaction_v7`) | confirmatory run: fresh calls, preregistered interaction | `data/confirmatory/` |
| v8 to v21 | mechanism work: probes, steering, local FP8 serving | `archive/mechanism/` (unchanged) |
| v22 | power analysis for the attenuation design; context blocks drafted, never run | `data/power/`, `configs/context_blocks_v1.json`, `configs/positive_control.yaml` |
| v23 | offline decomposition and blinded transcript judging | `data/judgments/`, `data/transcripts/`, `data/audits/decomposition_160/` |

## Path map

| Source path (at `92fb062`) | New path |
|---|---|
| `src/glm53_user_eval/` | `archive/mechanism/src/glm53_user_eval/` |
| `pipelines/glm53_user_eval/` | `archive/mechanism/pipelines/glm53_user_eval/` |
| `tests/glm53_user_eval*/` | `archive/mechanism/tests/` |
| `results/glm53_user_eval_20260828/` | `archive/mechanism/results/glm53_user_eval_20260828/` |
| `reference/source_locks_glm53_user_eval_v1.json` | `archive/mechanism/reference/` |
| `infra/runpod/*glm53*` | `archive/mechanism/infra/runpod/` |
| `.github/workflows/build-glm53-*.yml` | `archive/mechanism/infra/github_workflows/` |
| `docs/preregistrations/glm53_user_eval_shared_direction_v1.md` | `docs/preregistrations/` and `archive/mechanism/docs/preregistrations/` |
| `pipelines/glm53_user_eval/configs/prereg_v6_transluce_exact.yaml` | `docs/preregistrations/` |
| `pipelines/glm53_user_eval/configs/prereg_v7_interaction_confirmatory.yaml` | `docs/preregistrations/` |
| `pipelines/glm53_user_eval/configs/audit_amendment_v6_3.yaml` | `docs/preregistrations/` |
| `pipelines/glm53_user_eval/v22/configs/prereg_v22_information_substitution.yaml` | `docs/preregistrations/` |
| `pipelines/glm53_user_eval/v23/configs/prereg_v23_behavioral_decomposition.yaml` | `docs/preregistrations/` |
| `pipelines/glm53_user_eval/v22/configs/context_blocks_v1.json` | `configs/context_blocks_v1.json` |
| `pipelines/glm53_user_eval/v23/configs/{transcript_rubric_v1.md, *.schema.json}` | `data/judgments/configs/` |
| `pipelines/glm53_user_eval/reference/dilemma_split_v7.json` | `data/confirmatory/reference/` |
| `artifacts/glm53_user_eval/reports/transluce_interaction_v7/{raw_scores.jsonl, analysis.json, decision.json, verification.json, final_evidence.json, final_report.md, person_deltas.json}` | `data/confirmatory/` |
| `artifacts/glm53_user_eval/runs/transluce_interaction_v7/{schedule_manifest.json, FULL_RUN_AUDIT_PASS.json}` | `data/confirmatory/` |
| `artifacts/glm53_user_eval/runs/transluce_interaction_v7/audits/*` | `data/confirmatory/audits/` (all seven files) |
| `artifacts/glm53_user_eval/runs/transluce_interaction_v7/audits/{manual_packet.jsonl, manual_packet_manifest.json, manual_audit.json, reading_log.csv}` | also `data/audits/confirmatory_40/` |
| `artifacts/glm53_user_eval/reports/transluce_exact_v6/{raw_scores.jsonl, analysis.json, decision.json, shard_manifest.json, person_deltas.json, independent_verification.json}` | `data/discovery/` |
| `artifacts/glm53_user_eval/behavior_api/transluce_exact_v6/schedule_manifest.json` | `data/discovery/schedule_manifest.json` |
| `artifacts/glm53_user_eval/behavior_api/transluce_exact_v6/technical_audit_final*.json` | `data/discovery/audits/` |
| `artifacts/glm53_user_eval/reports/transluce_exact_v6/manual_audit/*` | `data/audits/discovery_manual/` |
| `artifacts/glm53_user_eval/v11/downstream_inputs/personas2.json` | `data/transcripts/personas2.json` |
| `artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl` (+ manifest) | `data/transcripts/` (untracked; see `POINTERS.md`) |
| `artifacts/glm53_user_eval/v23/derived/v7_transcripts.jsonl` (+ manifest) | `data/transcripts/` (untracked; see `POINTERS.md`) |
| `artifacts/glm53_user_eval/v23/derived/v7_annotation_packet.jsonl` (+ manifest) | `data/transcripts/` (untracked; see `POINTERS.md`) |
| `artifacts/glm53_user_eval/v23/derived/famous_nonai_category_packet.jsonl` (+ manifest) | `data/judgments/` |
| `artifacts/glm53_user_eval/v23/judgments/{v7_content, categories, v6_development_v2}/{luna_max, terra_high}/rows/` | `data/judgments/<same>/rows/` (per-row judge files; the `attempts/` event logs, 88 MB, were not copied) |
| `artifacts/glm53_user_eval/v23/reports/{deterministic_analysis, annotation_analysis, category_analysis, category_effects, decision, independent_verification}.json` | `data/judgments/reports/` |
| `artifacts/glm53_user_eval/v23/reports/{human_audit_packet.jsonl, human_audit_packet.manifest.json, human_audit_review_form.jsonl, human_audit_selection_private.json}` | `data/audits/decomposition_160/` |
| `artifacts/glm53_user_eval/v22/power/*` | `data/power/` |

Not copied: eval logs of both runs (`eval_logs/`, about 344 MB per run), orchestration logs,
`behavior_api/` roster pilots, `v8` to `v21` artifact trees, and the `v17/infrastructure/source_transport`
repository snapshot. The raw score files committed here were extracted from those logs by the
archived analysis scripts and are the inputs every stage uses.

## Cross-dependencies found by import scan

The only import from outside the study's own package was `src.probe.sequence_linear` in
`src/glm53_user_eval/v10/analysis.py` (and the same path listed in
`pipelines/glm53_user_eval/v10/configs/prereg_v10_offline_diagnostics.yaml`). No file in the paper
code imported the study package. The archived copy keeps the import unchanged; the paper module was
not copied.

## External resources not in this repository

- Pinned Transluce checkout: `TransluceAI/user-awareness` at
  `d1b9c3573470f50495202795c044bd72f72ee6e5`, expected at `../reference/transluce-user-awareness`
  (override with `GLM53_TRANSLUCE_ROOT`). File hashes are checked by `configs/task.yaml`.
- Provider credentials: `OPENROUTER_API_KEY` for the positive control, `ANTHROPIC_API_KEY` for the
  API backends of the role coder and figure judge. Neither is stored here.

## Absolute paths

Copied data and provenance files (manifests, audit packets, transcript rows, judge rows) still
contain the absolute paths of the machines that produced them. They were left as they are. No
user-facing document in this repository contains them.

## Positive control (run 2026-09-04)

`data/positive_control/` holds the rows extracted from the run's Inspect logs (`raw_scores.jsonl`,
21,000 rows), the analysis output, the schedule manifest, the run state, and a manifest with file
hashes. The Inspect eval logs themselves (300 shards) are kept locally under
`outputs/positive_control/eval_logs/` and are not tracked; the extractor in `src/glm53/run_task.py`
reproduces `raw_scores.jsonl` from them.
