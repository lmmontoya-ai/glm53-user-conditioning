# AI diagnostic review

This directory records a user-supplied aggregate from a blind, manual-style AI
review. It is not human evidence. It cannot satisfy the V11 preregistration,
enter the official human-review merge, change the failed semantic metric, or
authorize paid compute.

The report says the AI assigned all 128 primary rows before seeing the frozen
acceptable-label contract, then matched that contract on all 128 rows after
unblinding. It also reviewed the six supplemental disagreements. The reported
counts are in `reported_summary.json`.

The row-level files mentioned in the report were not available in this
workspace. The committed artifact therefore preserves only the reported
aggregate and marks it as unverified. V11 still closes as a failed automatic
gate with the human-review requirement unmet.
