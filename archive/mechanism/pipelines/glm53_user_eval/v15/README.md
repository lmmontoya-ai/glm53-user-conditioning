# V15 fresh semantic controls

V15 is a narrow, user-authorized follow-up to V14. V14 remains immutable and
failed its frozen decision. The V15 change addresses the exact inconsistency
found in that failure audit.

All 64 old neutral controls are retired. They are replaced by 64 fresh controls,
balanced across `no_status`, `deploy`, `eval`, and `ambiguous`. Operational
controls explicitly identify an outside recipient and a live consequence;
assessment controls explicitly distinguish qualitative review, scoring, and
model comparison.

The other 512 dataset rows are byte-identical to V14. Their 1,024 prior judgments
are reused only after rechecking sample ID, scenario bytes, model, reasoning
effort, schema, prompt, request hash, and the explicit `fast_mode` disablement.
Only the 128 changed judgment cells receive fresh local Codex calls.

Fresh calls use 24 concurrent standard-tier sessions: Luna at maximum reasoning
and Terra at high reasoning. Both judges must independently pass every unchanged
threshold. V15 is terminal: it does not permit another dataset repair.
