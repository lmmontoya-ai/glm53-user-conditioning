# Completed human-review files

These are the actual files produced during the review on 2026-09-04, copied byte-identically from
the working `outputs/` tree (hashes and source modification times in `MANIFEST.json`). They are
not reconstructed from reported results.

- `roles/human_coding_template.csv`: the blank sheet the human coder filled in (no effects shown).
- `roles/human_coding.csv`: the human coding, 70 famous-AI identities, blind to all effects.
- `roles/llm_coding.csv`, `roles/llm_coding_manifest.json`: the LLM coder's sheet (name and public
  role text only), with the protocol hash it was run under.
- `roles/merged_coding.csv`: the merge of both sheets with the 16 disagreements and their
  adjudicated `final_category` and `resolution`. The `final_category` column is the label set used
  by the headline role analysis: `label_hash(final_category by persona_key)` equals
  `labels_sha256` in `identities/role_contrast.json` (checked by `tests/test_review_files.py`).
- `identities/identities.csv`: per-identity and twin-adjusted effects in both runs (the file the
  coder must not see before coding).
- `identities/role_contrast.json`: primary, secondary, mechanical-robustness contrasts and
  disclosures; `identities/per_dilemma_scrutiny_profile*.csv`: profiles used by the positive control.
- `audits/confirmatory_40_human.csv`, `audits/confirmatory_40_summary.json`: the extraction audit.
- `summary.md`: the review tool's summary with hours per mode.

To re-run the contrast from these files, copy `roles/*` to `outputs/roles/` and
`identities/identities.csv` to `outputs/identities/`, then run
`uv run python scripts/04_identities.py contrast`.
