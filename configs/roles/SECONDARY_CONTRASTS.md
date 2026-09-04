# Secondary and robustness contrasts (declared 2026-09-04, after the primary contrast was run)

The primary contrast in `PROTOCOL.md` (scrutiny minus business, discovery run first) was run on
2026-09-04 with the merged human-and-LLM coding. The two analyses below were declared after seeing
that result and are reported as secondary and post hoc. `PROTOCOL.md` is unchanged; this file's
sha256 is recorded alongside it in the outputs.

## Disclosures

- The primary hypothesis was formed on the confirmatory run after the author inspected the
  per-identity means and named the identities with the most negative effects; the discovery run was
  then used as the held-out test for that hypothesis only.
- The human coder resolved all 16 disagreements with the LLM coder in favor of the LLM label, so the
  final coding equals the LLM coding for every disputed identity.
- During construction of the pipeline the agent ran the merge-and-contrast code once with a copy of
  the LLM sheet standing in for the human sheet, to verify the code path, and saw the resulting
  confirmatory value on screen; those outputs were deleted and no reported number uses them.

## Secondary contrast: scrutiny minus capabilities

Same estimand and bootstrap as the primary contrast, with `capabilities` as the comparison group.
Purpose: if scrutiny and capabilities identities do not separate, the defensible description of the
finding is "technical AI experts versus AI business figures", not "people who evaluate AI".
Discovery run reported first, then confirmatory. No directional prediction is registered.

## Mechanical robustness: categories from the affiliation string alone

Labels assigned by fixed keyword rules on the roster's affiliation text, with no judgment and no
access to effects. Rules are in `mechanical_rules.yaml`; precedence is scrutiny, then business, then
capabilities, then other, applied to the lower-cased affiliation string. Both contrasts are
recomputed with these labels on both runs, and agreement with the merged coding is reported.
Purpose: a judgment-free labeling that gives the same sign answers the objection that the final
coding is effectively the LLM's.
